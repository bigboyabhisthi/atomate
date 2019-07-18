# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

"""
This module provides ability to setup and parse defects on fly...
Note alot of this structure is taken from pycdt.utils.parse_calculations and pycdt.......

Requirements:
 - bulk calculation is finished (and vasprun.xml + Locpot)
 - a dielectric constant/tensor is provided
 - defect+chg calculation is finished 

 Soft requirements:
 	- Bulk and defect OUTCAR files (if charge correction by Kumagai et al. is desired)
 	- Hybrid bulk bandstructure / simple bulk structure calculation (if bandshifting is desired)
"""

import os
import itertools
import numpy as np

from monty.json import jsanitize

from pymatgen.io.vasp import Vasprun, Locpot, Poscar
from pymatgen import MPRester
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet, MVLScanRelaxSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.defects.generators import VacancyGenerator, SubstitutionGenerator, \
    InterstitialGenerator, VoronoiInterstitialGenerator, SimpleChargeGenerator

from fireworks import FiretaskBase, FWAction, explicit_serialize

from atomate.utils.utils import get_logger
from atomate.vasp.fireworks.core import TransmuterFW

from monty.serialization import dumpfn
from monty.json import MontyEncoder

logger = get_logger(__name__)

def optimize_structure_sc_scale(inp_struct, final_site_no):
    """
    A function for finding optimal supercell transformation
    by maximizing the nearest image distance with the number of
    atoms remaining less than final_site_no

    Args:
        inp_struct: input pymatgen Structure object
        final_site_no (float or int): maximum number of atoms
        for final supercell

    Returns:
        3 x 1 array for supercell transformation
    """
    if final_site_no <= len(inp_struct.sites):
        final_site_no = len(inp_struct.sites)

    dictio={}
    #consider up to a 7x7x7 supercell
    for kset in itertools.product(range(1,7), range(1,7), range(1,7)):
        num_sites = len(inp_struct) * np.product(kset)
        if num_sites > final_site_no:
            continue

        struct = inp_struct.copy()
        struct.make_supercell(kset)

        #find closest image
        min_dist = 1000.
        for image_array in itertools.product( range(-1,2), range(-1,2), range(-1,2)):
            if image_array == (0,0,0):
                continue
            distance = struct.get_distance(0, 0, image_array)
            if distance < min_dist:
                min_dist = distance

        min_dist = round(min_dist, 3)
        if min_dist in dictio.keys():
            if dictio[min_dist]['num_sites'] > num_sites:
                dictio[min_dist].update( {'num_sites': num_sites, 'supercell': kset[:]})
        else:
            dictio[min_dist] = {'num_sites': num_sites, 'supercell': kset[:]}

    if not len(dictio.keys()):
        raise RuntimeError('could not find any supercell scaling vector')

    min_dist = max( list(dictio.keys()))
    biggest = dictio[ min_dist]['supercell']

    return biggest


@explicit_serialize
class DefectSetupFiretask(FiretaskBase):
    """
    Run defect supercell setup

    Args:
        structure (Structure): input structure to have defects run on
        cellmax (int): maximum supercell size to consider for supercells
        conventional (bool):
            flag to use conventional structure (rather than primitive) for supercells,
            defaults to True.
        vasp_cmd (string):
            the vasp cmd
        db_file (string):
            the db file
        user_incar_settings (dict):
            a dictionary of incar settings specified by user for both bulk and defect supercells
            note that charges do not need to be set in this dicitionary
        user_kpoints_settings (dict or Kpoints pmg object):
            a dictionary of kpoint settings specific by user OR an Actual Kpoint set to be used for the calculation

        job_type (str): type of defect calculation that user desires to run
            default is 'normal' which runs a GGA defect calculation
            additional options are:
                'double_relaxation_run' which runs a double relaxation GGA run
                'metagga_opt_run' which runs a double relaxation with SCAN (currently
                    no way to turn off double relaxation approach with SCAN)
                'hse' which runs a relaxation step with GGA followed by a relaxation with HSE

        vacancies (list):
            If list is totally empty, all vacancies are considered (default).
            If only specific vacancies are desired then add desired Element symbol to the list
                ex. ['Ga'] in GaAs structure will only produce Galium vacancies

            if NO vacancies are desired, then just add an empty list to the list
                ex. [ [] ]  yields no vacancies

        substitutions (dict):
            If dict is totally empty, all intrinsic antisites are considered (default).
            If only specific antisites/substituions are desired then add vacant site type as key, with list of
                sub site symbol as value
                    ex 1. {'Ga': ['As'] } in GaAs structure will only produce Arsenic_on_Gallium antisites
                    ex 2. {'Ga': ['Sb'] } in GaAs structure will only produce Antimonide_on_Gallium substitutions

            if NO antisites or substitutions are desired, then just add an empty dict
                ex. {'None':{}}  yields no antisites or subs

        interstitials (list):
            If list is totally empty, NO interstitial defects are considered (default).
            Option 1 for generation: If one wants to use Pymatgen to predict interstitial
                    then list of pairs of [symbol, generation method (str)] can be provided
                        ex. ['Ga', 'Voronoi'] in GaAs structure will produce Galium interstitials from the
                            Voronoi site finding algorithm
                        NOTE: only options for interstitial generation are "Voronoi" and "InFit"
            Option 2 for generation: If user wants to add their own interstitial sites for consideration
                    the list of pairs of [symbol, Interstitial object] can be provided, where the
                    Interstitial pymatgen.analysis.defects.core object is used to describe the defect of interest
                    NOTE: use great caution with this approach. You better be sure that the supercell with Interstitial in it
                        is same as the bulk supercell...

        initial_charges (dict):
            says how to specify initial charges for each defect.
            An empty dict (DEFAULT) is to do a fairly restrictive charge generation method:
                for vacancies: use bond valence method to assign oxidation states and consider
                    negative of the vacant site's oxidation state as single charge to try
                antisites and subs: use bond valence method to assign oxidation states and consider
                    negative of the vacant site's oxidation state as single charge to try +
                    added to likely charge of substitutional site (closest to zero)
                interstitial: charge zero
            For non empty dict, charges are specified as:
                initial_charges = {'vacancies': {'Ga': [-3,2,1,0]},
                                   'substitutions': {'Ga': {'As': [0]} },
                                   'interstitials': {}}
                in the GaAs structure this makes vacancy charges in states -3,-2,-1,0; Ga_As antisites in the q=0 state,
                and all other defects will have charges generated in the restrictive automated format stated for DEFAULT

    """
    def run_task(self, fw_spec):
        if os.path.exists("POSCAR"):
            structure =  Poscar.from_file("POSCAR").structure
        else:
            structure = self.get("structure")

        if self.get("conventional", True):
            structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

        fws, parents = [], []

        #TODO: improve supercell routine
        cellmax=self.get("cellmax", 128)
        sc_scale = optimize_structure_sc_scale(structure, cellmax)

        job_type = self.get("job_type", 'normal')

        #First Firework is for bulk supercell
        bulk_supercell = structure.copy()
        bulk_supercell.make_supercell(sc_scale)
        num_atoms = len(bulk_supercell)

        user_incar_settings = self.get("user_incar_settings", {})
        user_kpoints_settings = self.get("user_kpoints_settings", {})

        bulk_incar_settings = {"EDIFF":.0001, "EDIFFG": 0.001, "ISMEAR":0, "SIGMA":0.05, "NSW": 0, "ISIF": 2,
                               "ISPIN":2,  "ISYM":2, "LVHAR":True, "LVTOT":True, "LWAVE": True}
        bulk_incar_settings.update( user_incar_settings)

        if job_type == 'metagga_opt_run':
            bulk_incar_settings['ALGO'] = "All"
            kpoints_settings = user_kpoints_settings if user_kpoints_settings else {"reciprocal_density": 100}
            vis = MVLScanRelaxSet( bulk_supercell,
                                   use_hubbard=True,
                                   user_incar_settings=bulk_incar_settings,
                                   user_kpoints_settings=kpoints_settings)
        else:
            reciprocal_density = 50 if job_type == 'hse' else 100
            kpoints_settings = user_kpoints_settings if user_kpoints_settings else {"reciprocal_density": reciprocal_density}
            vis = MPRelaxSet( bulk_supercell,
                              user_incar_settings=bulk_incar_settings,
                              user_kpoints_settings=kpoints_settings)

        supercell_size = sc_scale * np.identity(3)
        bulk_tag = "{}:{}_bulk_supercell_{}atoms".format(structure.composition.reduced_formula, job_type, num_atoms)
        stat_fw = TransmuterFW(name = bulk_tag, structure=structure,
                               transformations=['SupercellTransformation'],
                               transformation_params=[{"scaling_matrix": supercell_size}],
                               vasp_input_set=vis, copy_vasp_outputs=False, #structure already copied over...
                               vasp_cmd=self.get("vasp_cmd", ">>vasp_cmd<<"),
                               db_file=self.get("db_file", ">>db_file<<"),
                               job_type=job_type)

        fws.append(stat_fw)

        # make defect set
        vacancies = self.get("vacancies", list())
        substitutions = self.get("substitutions", dict())
        interstitials = self.get("interstitials", list())
        initial_charges  = self.get("initial_charges", dict())

        def_structs = []
        #a list with following dict structure for each entry:
        # {'defect': pymatgen defect object type,
        # 'charges': list of charges to run}

        #TODO add ability to include perturbing function to break local symmetry around defect
        #TODO: fix bug where if multiple elements are added you generate same defect multiple times..
        #TODO: add ability to do full manual insertion of defect type for subs and vacs (similar to manual insertion of defects for interstitials)
        if not vacancies:
            #default: generate all vacancies...
            b_struct = structure.copy()
            VG = VacancyGenerator( b_struct)
            for vac_ind, vac in enumerate(VG):
                vac_symbol = vac.site.specie.symbol

                charges = []
                if initial_charges:
                    if 'vacancies' in initial_charges.keys():
                        if vac_symbol in initial_charges['vacancies']:
                            #NOTE if more than one type of vacancy for a given specie, this will assign same charges to all
                            charges = initial_charges['vacancies'][vac_symbol]

                if not len(charges):
                    SCG = SimpleChargeGenerator(vac.copy())
                    charges = [v.charge for v in SCG]

                def_structs.append({'charges': charges, 'defect': vac.copy()})

        else:
            #only create vacancies of interest...
            for elt_type in vacancies:
                b_struct = structure.copy()
                VG = VacancyGenerator( b_struct)
                for vac_ind, vac in enumerate(VG):
                    vac_symbol = vac.site.specie.symbol
                    if elt_type != vac_symbol:
                        continue

                    charges = []
                    if initial_charges:
                        if 'vacancies' in initial_charges.keys():
                            if vac_symbol in initial_charges['vacancies']:
                                #NOTE if more than one type of vacancy for a given specie, this will assign same charges to all
                                charges = initial_charges['vacancies'][vac_symbol]

                    if not len(charges):
                        SCG = SimpleChargeGenerator(vac.copy())
                        charges = [v.charge for v in SCG]

                    def_structs.append({'charges': charges, 'defect': vac.copy()})


        if not substitutions:
            #default: set up all intrinsic antisites
            for sub_symbol in [elt.symbol for elt in bulk_supercell.types_of_specie]:
                b_struct = structure.copy()
                SG = SubstitutionGenerator(b_struct, sub_symbol)
                for as_ind, sub in enumerate(SG):
                    #find vac_symbol to correctly label defect
                    poss_deflist = sorted(sub.bulk_structure.get_sites_in_sphere(sub.site.coords, 2, include_index=True), key=lambda x: x[1])
                    defindex = poss_deflist[0][2]
                    vac_symbol = sub.bulk_structure[defindex].specie.symbol

                    charges = []
                    if initial_charges:
                        if 'substitutions' in initial_charges.keys():
                            if vac_symbol in initial_charges['substitutions']:
                                #NOTE if more than one type of substituion for a given specie, this will assign same charges to all
                                if sub_symbol in initial_charges['substitutions'][vac_symbol].keys():
                                    charges = initial_charges['substitutions'][vac_symbol][sub_symbol]
                    if not len(charges):
                        SCG = SimpleChargeGenerator(sub.copy())
                        charges = [v.charge for v in SCG]

                    def_structs.append({'charges': charges, 'defect': sub.copy()})
        else:
            #only set up specified antisite / substituion types
            for vac_symbol, sub_list in substitutions.items():
                for sub_symbol in sub_list:
                    b_struct = structure.copy()
                    SG = SubstitutionGenerator(b_struct, sub_symbol)
                    for as_ind, sub in enumerate(SG):
                        #find vac_symbol for this sub defect
                        poss_deflist = sorted(sub.bulk_structure.get_sites_in_sphere(sub.site.coords, 2, include_index=True), key=lambda x: x[1])
                        defindex = poss_deflist[0][2]
                        gen_vac_symbol = sub.bulk_structure[defindex].specie.symbol
                        if vac_symbol != gen_vac_symbol: #only consider subs on specfied vac_symbol site
                            continue

                        charges = []
                        if initial_charges:
                            if 'substitutions' in initial_charges.keys():
                                if vac_symbol in initial_charges['substitutions']:
                                    #NOTE if more than one type of substituion for a given specie, this will assign same charges to all
                                    if sub_symbol in initial_charges['substitutions'][vac_symbol].keys():
                                        charges = initial_charges['substitutions'][vac_symbol][sub_symbol]
                        if not len(charges):
                            SCG = SimpleChargeGenerator(sub.copy())
                            charges = [v.charge for v in SCG]

                        def_structs.append({'charges': charges, 'defect': sub.copy()})


        if interstitials:
            #default: do not include interstitial defects
            #TODO: for time savings, can reuse result of InFit intersitital finding approach since it is time consuming

            def get_charges_from_inter( inter_elt):
                inter_charges = []
                if initial_charges:
                    if 'interstitials' in initial_charges.keys():
                        if inter_elt in initial_charges['interstitials']:
                            #NOTE if more than one type of interstitial for a given specie, this will assign same charges to all
                            inter_charges = initial_charges['interstitials'][inter_elt]

                if not len(inter_charges):
                    SCG = SimpleChargeGenerator(inter_elt)
                    inter_charges = [v.charge for v in SCG]
                return inter_charges

            for elt_type, elt_val in interstitials:
                if type(elt_val) == str:
                    b_struct = structure.copy()
                    if elt_val == 'Voronoi':
                        IG = VoronoiInterstitialGenerator(b_struct, elt_type)
                    elif elt_val == 'InFit':
                        IG = InterstitialGenerator(b_struct, elt_type)
                    else:
                        raise ValueError('Interstitial finding method not recognized. '
                                         'Please choose either Voronoi or InFit.')

                    for inter_ind, inter in enumerate(IG):
                        charges = get_charges_from_inter( inter)
                        def_structs.append({'charges': charges, 'defect': inter.copy()})
                else:
                    charges = get_charges_from_inter( elt_val)
                    def_structs.append({'charges': charges, 'defect': elt_val.copy()})

        stdrd_defect_incar_settings = {"EDIFF": 0.0001, "EDIFFG": 0.001, "IBRION":2, "ISMEAR":0, "SIGMA":0.05,
                                       "ISPIN":2,  "ISYM":2, "LVHAR":True, "LVTOT":True, "NSW": 100,
                                       "NELM": 60, "ISIF": 2, "LAECHG":False, "LWAVE": True}
        stdrd_defect_incar_settings.update( user_incar_settings)

        # now that def_structs is assembled, set up Transformation FW for all defect + charge combinations
        for defcalc in def_structs:
            #get defect supercell and defect site for parsing purposes
            defect = defcalc['defect'].copy()
            defect_sc = defect.generate_defect_structure( supercell = supercell_size)
            struct_for_defect_site = Structure(defect.bulk_structure.copy().lattice,
                                               [defect.site.specie],
                                               [defect.site.frac_coords],
                                               to_unit_cell=True, coords_are_cartesian=False)
            struct_for_defect_site.make_supercell(supercell_size)
            defect_site = struct_for_defect_site[0]

            #iterate over all charges to be run
            for charge in defcalc['charges']:
                chgdstruct = defect_sc.copy()
                chgdstruct.set_charge(charge)  #NOTE that the charge will be reflected in NELECT of INCAR because use_structure_charge=True

                if job_type == 'metagga_opt_run':
                    stdrd_defect_incar_settings['ALGO'] = "All"
                    kpoints_settings = user_kpoints_settings if user_kpoints_settings else {"reciprocal_density": 100}
                    defect_input_set = MVLScanRelaxSet( chgdstruct,
                                                        use_hubbard=True,
                                                        user_incar_settings=stdrd_defect_incar_settings.copy(),
                                                        user_kpoints_settings=kpoints_settings,
                                                        use_structure_charge=True)
                else:
                    reciprocal_density = 50 if job_type == 'hse' else 100
                    kpoints_settings = user_kpoints_settings if user_kpoints_settings else {"reciprocal_density": reciprocal_density}
                    defect_input_set = MPRelaxSet( chgdstruct,
                                                   user_incar_settings=stdrd_defect_incar_settings.copy(),
                                                   user_kpoints_settings=kpoints_settings,
                                                   use_structure_charge=True)

                defect_for_trans_param = defect.copy()
                defect_for_trans_param.set_charge(charge)
                chgdef_trans = ["DefectTransformation"]
                chgdef_trans_params = [{"scaling_matrix": supercell_size,
                                        "defect": defect_for_trans_param}]
                if structure != defect_for_trans_param.bulk_structure:
                    print('\n\nERROR!\n{}\n\n{}\n'.format(structure, defect_for_trans_param.bulk_structure))
                    raise ValueError("Defect bulk_structure is not the same as input structure.")

                def_tag = "{}:{}_{}_{}_{}atoms".format(structure.composition.reduced_formula, job_type,
                                                      defect.name, charge, num_atoms)
                fw = TransmuterFW( name = def_tag, structure=structure,
                                   transformations=chgdef_trans,
                                   transformation_params=chgdef_trans_params,
                                   vasp_input_set=defect_input_set,
                                   vasp_cmd=self.get("vasp_cmd", ">>vasp_cmd<<"),
                                   copy_vasp_outputs=False,
                                   db_file=self.get("db_file", ">>db_file<<"),
                                   job_type=job_type,
                                   bandstructure_mode="auto",
                                   defect_wf_parsing=defect_site)

                fws.append(fw)

        return FWAction(detours=fws)


@explicit_serialize
class DefectAnalysisFireTask(FiretaskBase):
    """
    Use the Single Defect Analysis class (below) to loop through local folders,
        and see if additional ANalysis (base, charge, or extra) needs to be done
    """
    def run_task(self, fw_spec):
        """
        ...only meant for single job in database at a time right now...
        TODO: more computationally efficient by saving locpot data etc..
        """
        #TODO: merge this to pymatgen current capabilities and get rid of everything we have right here now?

        pat = os.getcwd()
        basepat = os.path.split(pat)[0]
        launchlistset = os.listdir(basepat)

        #assemble defect paths and job types from fw_spec (TODO: this has issues when fw_spec has multiple entries for same defect +chg?)
        globeset = {'defects':[], 'bulk': None, 'diel': None}
        for lset in fw_spec['calc_locs']:
            if "bulk_supercell" in lset['name']:
                globeset['bulk'] = lset['path']
            elif 'dielectric' in lset['name']:
                globeset['diel'] = lset['path']
            elif ('vac_' in lset['name']) or ('as_' in lset['name']) or \
                ('inter_' in lset['name']) or ('sub_' in lset['name']):
                chg = int(lset['name'].split('_')[-1])
                nom = '' #reduce name to something without charge... this is currently hacky way to do this
                nomsplitout = lset['name'].split(':')[-1].split('_')
                for tmpind, tmpnompart in enumerate(nomsplitout):
                    if tmpind != (len(nomsplitout)-1):
                        nom += tmpnompart+'_'
                def_name = nom[:-1]
                chg = int(chg)
                print ('heydan ',def_name, chg)
                # globeset['defects'].append( [lset['name'], chg, lset['path']]) #has firstentry with name_chg...
                globeset['defects'].append( [def_name, chg, lset['path']])

        #load dielectric constant
        #TODO:  should have ability to allow for manual insertion of diel value OR previously determined diel in database
        print ('\tloading diel...')
        dvr = Vasprun(os.path.join(globeset['diel'], 'vasprun.xml.gz'))
        eps_ion = dvr.epsilon_ionic
        eps_stat = dvr.epsilon_static
        eps = []
        for i in range(len(eps_ion)):
            eps.append([e[0]+e[1] for e in zip(eps_ion[i],eps_stat[i])])
        globeset['diel'] = eps
        print ('\tDielectric is ',eps)

        #run defect calculations
        print('path structure assembled. now running chg corrs etc.')
        metadata = {}
        for dset in globeset['defects']:
            dnom = dset[0] #defect name
            chg = dset[1] #charge
            dloc = dset[2] #defect location
            print('parsing',dnom,chg,' at ',dloc)
            #TODO: should have ability to skip correction step if it has already been calculated...
            sda = SingleDefectAnalysis(globeset['bulk'], dloc, globeset['diel'],
                                       dvr.final_structure.sites[0], chg) #second to last entry is defect position, right now it does nothing... so can keep at origin
            corr_details = sda.compile_all(mpid=self.get("mpid", None), print_to_file=False)

            freycorr = corr_details['freysoldt']['total_corr']
            bandfill = corr_details['bandfilling']['mbshift'] #DANNY -> this may include level shifting if not careful...

            def_entry = sda.defvr.get_computed_entry()
            cd_entry = ComputedDefect(def_entry, dvr.final_structure.sites[0], #note fake defect position...  not using true defect position yet...
                           multiplicity=None, supercell_size=[1, 1, 1],   #note fake supercell size and multiplicity... not using these values yet
                           charge=chg, charge_correction=freycorr, other_correction=bandfill, name=dnom)
            metadata[dnom+'_'+str(chg)] = {'cd': cd_entry, 'corr_details':corr_details}
            if 'entry_bulk' not in metadata.keys():
                bulk_entry = ComputedStructureEntry(sda.blkvr.final_structure, sda.blkvr.final_energy,
                                                    data={'locpot_path': os.path.join(globeset['bulk'], 'LOCPOT.gz'),
                                                          'supercell_size': [1, 1, 1]}) #note fake supercell size
                metadata['entry_bulk'] = bulk_entry

        return FWAction( update_spec=metadata)


class SingleDefectAnalysis(object):
    """
    A little unclear whether this will be a Drone or a Firetasker...
    just providing a skeleton here
    TO CONSIDER:
            > is data to be stored in database + in local file structure?


    Contents requried for inputs
        - path to bulk folder OR bulk Pymatgen objects: (vasprun.xml, Locpot
                        + path to OUTCAR file if Kumagai correction desired)
        - a dielectric constant/tensor
        - path to defect folder OR defect Pymatgen objects: (vasprun.xml, Locpot
                        + path to OUTCAR file if Kumagai correction desired)
        - information about defect: (position of defect in structure, charge of defect)

    Quantities that can be produced from this class:
        1) (base_postprocess) = minimum requirements for calculating a Defect Formation Energy
                > parsing of bandstructures for electron chemical potential
                > parsing of phase diagram for atomic chemical potentials
        2) (freysoldt) Charge Correction from Freysoldt et al. (Includes electrostatic term and potential alignment term)
        3) (kumagai) Extension of Freysoldt correction suggested by Kumagai et al. (Includes electrostatic term and potential alignment term)
                > interesting because it takes into account anisotropic dielectric response
        4) (bandfilling) Moss-Burstein filling correction
        5) (bandshifting) [developmental stages] "Shallow level" band edge shifting corrections based on Hybrid Band Structure

    """
    def __init__(self, path_to_bulk, path_to_defect, dielconst, defpos, defchg):
        self.blkpath = path_to_bulk           					                    #REQD by:  kumagai(just for OUTCAR path)
        #quicker way could be to load the follow directly to this class
        # self.blkvr = Vasprun(os.path.join(path_to_bulk, 'vasprun.xml'))			#REQD by:  base_postprocess(if no mpid provided), kumagai(just for structure)
        self.blkvr = Vasprun(os.path.join(path_to_bulk, 'vasprun.xml.gz'))			#REQD by:  base_postprocess(if no mpid provided), kumagai(just for structure)
        # self.blklp = Locpot.from_file(os.path.join(path_to_bulk, 'LOCPOT'))		#REQD by:  freysoldt
        self.blklp = Locpot.from_file(os.path.join(path_to_bulk, 'LOCPOT.gz'))		#REQD by:  freysoldt


        self.defpath = path_to_defect											#REQD by:  kumagai(just for OUTCAR path)
        #quicker way could be to load the follow directly to this class
        # self.defvr = Vasprun(os.path.join(path_to_defect, 'vasprun.xml'))		#REQD by:  freysoldt, kumagai(just for structure)
        self.defvr = Vasprun(os.path.join(path_to_defect, 'vasprun.xml.gz'))		#REQD by:  freysoldt, kumagai(just for structure)
        # self.deflp = Locpot.from_file(os.path.join(path_to_defect, 'LOCPOT'))	#REQD by:  freysoldt
        self.deflp = Locpot.from_file(os.path.join(path_to_defect, 'LOCPOT.gz'))	#REQD by:  freysoldt
        #NOTE that time can also be saved from not loading locpots for charge = 0

        self.diel = dielconst #can be either a number or a 3x3list array		#REQD by:  freysoldt, kumagai

        self.defpos = defpos													#REQD by:  none right now... should be used by freysoldt and kumagai in future
        self.defchg = defchg													#REQD by:  freysoldt, kumagai

    def compile_all(self, mpid=None, print_to_file=True):
        """
        Right now this naively accumulates a global dictionary of:
            1) base_postprocess
            2) FreysoldtCorrection
            3) KumagaiCorrection
            4) bandfilling
        TODO: implement defect level shifting method
        """
        full_compile = {}

        full_compile['base'] = self.base_postprocess(mpid=mpid)
        if not self.defchg:
            full_compile['freysoldt'] = {'electrostatic_corr': 0., 'potalign_corr': 0.,
                        'total_corr': 0., 'three_axis_data': [0.,0.,0.]}
            potalign = 0.
        else:
            full_compile['freysoldt'] = self.freysoldt()
            # full_compile['kumagai'] = self.kumagai()
            potalign = - float(full_compile['freysoldt']['potalign_corr']) / float(self.defchg)

        full_compile['bandfilling'] = self.bandfilling( potalign=potalign)

        if print_to_file:
            pat_dump = os.path.join(self.defpath, 'outfile.json')
            dumpfn(full_compile, pat_dump, cls=MontyEncoder, indent=2)
        else:
            return full_compile


    def base_postprocess(self, mpid=None):
        #if mpid provided then can use MP level data, otherwise use loaded bulk vr file data

        #first get information for electron chemical potential
        if not mpid:
            logger.warning(
                'No mp-id provided, will fetch CBM/VBM details from the '
                'bulk calculation.\nNote that it would be better to '
                'perform real band structure calculation...')
            bandgap = self.blkvr.eigenvalue_band_properties[0]
            vbm = self.blkvr.eigenvalue_band_properties[2]
        else:
            with MPRester() as mp:
                bs = mp.get_bandstructure_by_material_id(mpid)
            if not bs:
                logger.error("Could not fetch band structure!")
                raise ValueError("Could not fetch band structure!")

            vbm = bs.get_vbm()['energy']
            if not vbm:
                try:
                    vbm = bs.efermi #a hack for GGA gap errors
                except:
                    vbm = 0.
            bandgap = bs.get_band_gap()['energy']

        #second get atomic chemical potential information from phase diagram
        substitution_species = {} #TODO: make smarter approach to checking if sub species were calculated
        if mpid:
            cpa = MPChemPotAnalyzer( mpid = mpid, sub_species = substitution_species ) #, mapi_key = mapi_key)
        else:
            cpa = MPChemPotAnalyzer( bulk_ce= self.blkvr.get_computed_entry(),
                                   sub_species = substitution_species) #, mapi_key = mapi_key)

        chem_lims = cpa.analyze_GGA_chempots()
        final_base_parse = {'vbm': vbm, 'gga_bandgap':bandgap, 'chem_lims': chem_lims}

        return final_base_parse


    def freysoldt(self):
        #average the freysoldtcorrection over threeaxes (to try to eliminate potential alignment uncertainty)
        avgcorr = []
        fullcorrset = []
        #TODO: if you set title then can store potential alignment plots for freysoldt...
        for ax in range(3):
            corr_meth = FreysoldtCorrection( ax, self.diel, self.blklp, self.deflp, self.defchg)
            # valset = corr_meth.correction(title=title+'ax'+str(ax+1), partflag='AllSplit')
            valset = corr_meth.correction(partflag='AllSplit')
            avgcorr.append(valset[1])
            fullcorrset.append(valset)
            # if title: #move the file
            #     homepat = os.path.abspath('.')
            #     src = os.path.join(homepat, title+'ax'+str(ax+1)+'FreyplnravgPlot.pdf')
            #     dst = os.path.join(dpat, title+'ax'+str(ax+1)+'FreyplnravgPlot.pdf')
            #     shutil.move(src, dst)
        freycorrection = {'electrostatic_corr': valset[0], 'potalign_corr': np.mean(avgcorr),
                        'total_corr': valset[0]+np.mean(avgcorr), 'three_axis_data': fullcorrset}

        return freycorrection


    def kumagai(self):
        if os.path.exists(os.path.join( self.blkpath, 'OUTCAR')):
            bulk_outcar_path = os.path.join( self.blkpath, 'OUTCAR')
        elif os.path.exists(os.path.join( self.blkpath, 'OUTCAR.gz')):
            bulk_outcar_path = os.path.join( self.blkpath, 'OUTCAR.gz')

        if os.path.exists(os.path.join( self.blkpath, 'OUTCAR')):
            def_outcar_path = os.path.join( self.blkpath, 'OUTCAR')
        elif os.path.exists(os.path.join( self.blkpath, 'OUTCAR.gz')):
            def_outcar_path = os.path.join( self.blkpath, 'OUTCAR.gz')

        print('at Kumagai. Using outcars:\n',bulk_outcar_path,'\n',def_outcar_path)
        print('\nother inputs =\n',self.diel, self.defchg,type(self.blkvr),
              type(self.defvr),self.blkvr.final_structure.lattice.abc)

        KC=KumagaiCorrection(self.diel, self.defchg, None, None, #these Nones can be gamma and g_sum (below)
                self.blkvr.final_structure, defstructure=self.defvr.final_structure,
                energy_cutoff=520, madetol=0.0001, lengths=self.blkvr.final_structure.lattice.abc,
                bulk_outcar=bulk_outcar_path, defect_outcar=def_outcar_path)

        #substantial time can be saved if KumagaiBulk object below is reused - it only depends on the bulk structure
        KumagaiBulk = KumagaiBulkInit(self.bulkvr.final_structure, KC.dim, self.diel)

        KC.gamma = KumagaiBulk.gamma
        KC.g_sum = KumagaiBulk.g_sum

        #TODO: if you set title then can store potential alignment plots for kumagai...
        # valset = KC.correction(title=title, partflag='AllSplit')
        valset = KC.correction( partflag='AllSplit')

        # if title: #move the file
        #     homepat = os.path.abspath('.')
        #     src = os.path.join(homepat, title+'ax'+str(ax+1)+'FreyplnravgPlot.pdf')
        #     dst = os.path.join(dpat, title+'ax'+str(ax+1)+'FreyplnravgPlot.pdf')
        #     shutil.move(src, dst)

        kumagaicorrection = {'electrostatic_corr': valset[0], 'potalign_corr': valset[1],
                            'total_corr': valset[2]}

        return kumagaicorrection


    def bandfilling(self, potalign=0.):
        #potalign can be determined from either Kumagai or Freysoldt corrections
        #NOTE that if using above dictionaries, potalign corresponds to  ('potalign_corr' / (-charge))
        bfc = ExtraCorrections(self.defvr, potalign=potalign, bulk_vr=self.blkvr)

        num_hole_vbm, num_elec_cbm, bf_corr = bfc.calculate_bandfill()
        mbshift = bfc.bfcorr #note that mbshift + slshift = bf_corr above normally, but no shifts considered yet
        vbm = bfc.vbm
        gapstates = {'occupied':[[valoccu[0]-vbm, valoccu[1]] for valoccu in bfc.result_occu],
                     'unoccupied': [[valunoccu- vbm] for valunoccu in bfc.result_unocu]}

        extracorrset = {'num_hole_vbm':num_hole_vbm, 'num_elec_cbm':num_elec_cbm,
                        'mbshift': mbshift, 'gapstates': gapstates}
        return extracorrset



    def bandshifting(self):
        #several things are still in development here. Would rather wait to see how we store information before adding
        pass
