import ROOT
import math
import os
import warnings
import numpy as np
import h5py
import time

this_time = time.time()
start_time = this_time

## Specific to this script
zdir = ROOT.TVector3(0, 0, 1)

# Output array datatypes
# cc 1, nc 0
# energy, momentum in GeV
# position coordinate in cm
# lepTheta is angle wrt to the beam direction (z here) in degree
# E_reco is deposited energy in any active or passive material
# E_avail is deposited energy in 2x2 or MINERvA
# ld -> leading
# containment: 
# 1 - detector contained, 2x2 contained, 
# 2 - detector contained, MINERvA tagged, 
# 3 - detector contained, other (likely to stop in the passive material between 2x2 active and downstream MINERvA)
# 4 - detector exiting, MINERvA tagged
# 5 - detector exiting, other (exit from 2x2 directly)
event_dtype = np.dtype([("nuPDG", "<i4"), ("ccnc", "<i4"), 
                       ("Enu", "<f4"), ("q0", "<f4"), ("Q2", "<f4"), 
                       ("pnu_x", "<f4"), ("pnu_y", "<f4"), ("pnu_z", "<f4"),
                       ("Vtx_x", "<f4"), ("Vtx_y", "<f4"), ("Vtx_z", "<f4"),
                        
                       ("n_protons", "<u4"), ("n_piplus", "<u4"), ("n_piminus", "<u4"), ("n_pi0", "<u4"), 
                        
                       ("lepPDG", "<i4"), ("lep_Theta", "<f4"), ("lep_KE", "<f4"), ("lep_containment", "<u4"),
                       ("Elep", "<f4"), ("plep_x", "<f4"), ("plep_y", "<f4"), ("plep_z", "<f4"),
                       ("Elep_reco", "<f4"), ("Elep_2x2", "<f4"), ("Elep_MINERvA", "<f4"),
                       
                       ("E_had", "<f4"), ("E_had_reco", "<f4"), ("E_had_reco_2x2", "<f4"), ("E_had_reco_MINERvA", "<f4"), ("had_containment", "<u4"),

                       ("ldp_Theta", "<f4"), ("ldp_KE", "<f4"), ("ldp_containment", "<u4"),
                       ("Eldp", "<f4"), ("pldp_x", "<f4"), ("pldp_y", "<f4"), ("pldp_z", "<f4"),
                       ("Eldp_reco", "<f4"), ("Eldp_2x2", "<f4"), ("Eldp_MINERvA", "<f4"),
                        
                       ("ldpiplus_Theta", "<f4"), ("ldpiplus_KE", "<f4"), ("ldpiplus_containment", "<u4"),
                       ("Eldpiplus", "<f4"), ("pldpiplus_x", "<f4"), ("pldpiplus_y", "<f4"), ("pldpiplus_z", "<f4"),
                       ("Eldpiplus_reco", "<f4"), ("Eldpiplus_2x2", "<f4"), ("Eldpiplus_MINERvA", "<f4"),
                        
                       ("ldpiminus_Theta", "<f4"), ("ldpiminus_KE", "<f4"), ("ldpiminus_containment", "<u4"),
                       ("Eldpiminus", "<f4"), ("pldpiminus_x", "<f4"), ("pldpiminus_y", "<f4"), ("pldpiminus_z", "<f4"),
                       ("Eldpiminus_reco", "<f4"), ("Eldpiminus_2x2", "<f4"), ("Eldpiminus_MINERvA", "<f4"),
                        
                       ("ldpi0_Theta", "<f4"), ("ldpi0_KE", "<f4"), ("ldpi0_containment", "<u4"),
                       ("Eldpi0", "<f4"), ("pldpi0_x", "<f4"), ("pldpi0_y", "<f4"), ("pldpi0_z", "<f4"),
                       ("Eldpi0_reco", "<f4"), ("Eldpi0_2x2", "<f4"), ("Eldpi0_MINERvA", "<f4")], align=True)

def get_neutrino_4mom(groo_event):
    
    ## Loop over the particles in GENIE's stack
    for p in range(groo_event.StdHepN):

        ## Look for the particle status
        ## 0 is initial state, 1 is final, check the GENIE docs for others
        if groo_event.StdHepStatus[p] != 0: continue

        ## Check for a neutrino (any flavor)
        if abs(groo_event.StdHepPdg[p]) not in [12, 14, 16]: continue

        ## edep-sim uses MeV, gRooTracker uses GeV...
        ## convert genie info to MeV
        return  groo_event.StdHepPdg[p], ROOT.TLorentzVector(groo_event.StdHepP4[p*4 + 0] * 1e3,
                                   groo_event.StdHepP4[p*4 + 1] * 1e3,
                                   groo_event.StdHepP4[p*4 + 2] * 1e3,
                                   groo_event.StdHepP4[p*4 + 3] * 1e3)
    ## Should never happen...
    return None

def print_genie(groo_event):

    pdg_status = []

    ## Loop over the particles in GENIE's stack
    for p in range(groo_event.StdHepN):

        pdg_status.append((groo_event.StdHepPdg[p], groo_event.StdHepStatus[p]))

    return pdg_status

## Find the ids of primary particles with a given PDG
def get_traj_ids_for_pdg(particles, pdgs):

    ## Loop over the truth trajectories
    ## Keep track of track ids if the PDG code is the one we desire
    return tuple(x.GetTrackId() for x in particles if x.GetPDGCode() in pdgs), tuple(x.GetPDGCode() for x in particles if x.GetPDGCode() in pdgs)

def get_traj_for_pdg(particles, pdgs):
    return tuple(x for x in particles if x.GetPDGCode() in pdgs)

def is_nue_cc(event):
    
    ## Get the primary xParticle ID 
    particle_id, lep_pdg = get_traj_ids_for_pdg(event.Primaries[0].Particles, [11, -11])

    ## If there isn't a muon... this isn't CC, so default to True
    ## (because who cares where the outgoing neutrino goes in an NC event)
    ## NC
    if len(particle_id)==0: 
        return False, particle_id, lep_pdg
    ## CC
    elif len(particle_id)==1: 
        return True, particle_id, lep_pdg
    else:
        warnings.warn("Warning...........More than one leptons")    
        return True, particle_id, lep_pdg

def is_2x2_contained(pos):
    
    if abs(pos[0]) > 670: return False
    if abs(pos[1] - 430) > 670: return False
    if abs(pos[2]) > 670: return False
    return True

def is_minerva_tagged(pos):
    # ## MINERvA's maximum z value (mm)
    # ## This is very geometry specific
    # z_max = 3500
    
    # ## Radius of a cylinder that approximates MINERvA
    # ## (this is slightly smaller than a cylinder that would go through the "tips" of the MINERvA hexagon)
    approx_rad = 1870
    
    if math.sqrt(pos[0]*pos[0] + (pos[1]-430)*(pos[1]-430)) < approx_rad:
        if (pos[2]-3500)*(pos[2]-1200) <0: 
            return True

    return False

def is_detector_contained(pos):
    if abs(pos[0]) > 2000: return False
    if abs(pos[1]) > 2500: return False
    if pos[2] < -2500 or pos[2] > 4000: return False
    return True

## We want to ignore all hits produced by neutrons or their daughters
## So, make a set of all true trajectories that are neutrons or their descendants 
def get_neutron_and_daughter_ids(event):
    
    neutrons  = set()
    daughters = set()
    
    for traj in event.Trajectories:
        
        if traj.GetPDGCode() == 2112:
            neutrons .add(traj.GetTrackId())
            continue
        par_id = traj.GetParentId()
        if par_id in neutrons or par_id in daughters:
            daughters .add(traj.GetTrackId())

    return neutrons.union(daughters)

## Get a set of trajectory IDs with total energy < 10 MeV
## This is a semi-arbitrary cut-off to ignore delta rays and
## other low-energy stuff that leaks out of the detector
def get_low_energy_ids(event):
    return set(x.GetTrackId() for x in event.Trajectories if x.GetInitialMomentum().E() < 10)

def get_pi0_daughter_ids(event, pi0_id):

    daughters = set()

    for traj in event.Trajectories:
        par_id = traj.GetParentId()
        if par_id in pi0_id:
            daughters .add(traj.GetTrackId())

    return daughters

def init_evt_info(shell_evt_info):
    shell_evt_info["nuPDG"] = 0
    shell_evt_info["ccnc"] = 0
    shell_evt_info["q0"] = 0
    shell_evt_info["Q2"] = 0
    shell_evt_info["Enu"] = 0
    shell_evt_info["pnu_x"] = 0
    shell_evt_info["pnu_y"] = 0
    shell_evt_info["pnu_z"] = 0
    shell_evt_info["Vtx_x"] = 0
    shell_evt_info["Vtx_y"] = 0
    shell_evt_info["Vtx_z"] = 0
    
    shell_evt_info["n_protons"] = 0
    shell_evt_info["n_piplus"] = 0
    shell_evt_info["n_piminus"] = 0
    shell_evt_info["n_pi0"] = 0
    
    shell_evt_info["lepPDG"] = 0
    shell_evt_info["lep_Theta"] = 0
    shell_evt_info["lep_KE"] = 0
    shell_evt_info["lep_containment"] = 0
    shell_evt_info["Elep"] = 0
    shell_evt_info["plep_x"] = 0
    shell_evt_info["plep_y"] = 0
    shell_evt_info["plep_z"] = 0
    shell_evt_info["Elep_reco"] = 0
    shell_evt_info["Elep_2x2"] = 0
    shell_evt_info["Elep_MINERvA"] = 0

    shell_evt_info["E_had"] = 0
    shell_evt_info["E_had_reco"] = 0
    shell_evt_info["E_had_reco_2x2"] = 0
    shell_evt_info["E_had_reco_MINERvA"] = 0
    shell_evt_info["had_containment"] = 0
    
    shell_evt_info["ldp_Theta"] = 0
    shell_evt_info["ldp_KE"] = 0
    shell_evt_info["ldp_containment"] = 0
    shell_evt_info["Eldp"] = 0
    shell_evt_info["pldp_x"] = 0
    shell_evt_info["pldp_y"] = 0
    shell_evt_info["pldp_z"] = 0
    shell_evt_info["Eldp_reco"] = 0
    shell_evt_info["Eldp_2x2"] = 0
    shell_evt_info["Eldp_MINERvA"] = 0
     
    shell_evt_info["ldpiplus_Theta"] = 0
    shell_evt_info["ldpiplus_KE"] = 0
    shell_evt_info["ldpiplus_containment"] = 0
    shell_evt_info["Eldpiplus"] = 0
    shell_evt_info["pldpiplus_x"] = 0
    shell_evt_info["pldpiplus_y"] = 0
    shell_evt_info["pldpiplus_z"] = 0
    shell_evt_info["Eldpiplus_reco"] = 0
    shell_evt_info["Eldpiplus_2x2"] = 0
    shell_evt_info["Eldpiplus_MINERvA"] = 0
    
    shell_evt_info["ldpiminus_Theta"] = 0
    shell_evt_info["ldpiminus_KE"] = 0
    shell_evt_info["ldpiminus_containment"] = 0
    shell_evt_info["Eldpiminus"] = 0
    shell_evt_info["pldpiminus_x"] = 0
    shell_evt_info["pldpiminus_y"] = 0
    shell_evt_info["pldpiminus_z"] = 0
    shell_evt_info["Eldpiminus_reco"] = 0
    shell_evt_info["Eldpiminus_2x2"] = 0
    shell_evt_info["Eldpiminus_MINERvA"] = 0
    
    shell_evt_info["ldpi0_Theta"] = 0
    shell_evt_info["ldpi0_KE"] = 0
    shell_evt_info["ldpi0_containment"] = 0
    shell_evt_info["Eldpi0"] = 0
    shell_evt_info["pldpi0_x"] = 0
    shell_evt_info["pldpi0_y"] = 0
    shell_evt_info["pldpi0_z"] = 0
    shell_evt_info["Eldpi0_reco"] = 0
    shell_evt_info["Eldpi0_2x2"] = 0
    shell_evt_info["Eldpi0_MINERvA"] = 0
    
    return shell_evt_info

#directory = '/project/projectdirs/dune/users/2x2EventGeneration/output/NuMI_FHC_CHERRY/EDEPSIM/'
#directory = '/global/cfs/projectdirs/dune/users/2x2EventGeneration/output/NuMI_FHC_CHERRY/EDEPSIM/'
directory = '/global/cfs/projectdirs/dune/users/yifanch/2x2EventGeneration_nue/output/NuMI_RHC_CHERRY/EDEPSIM_5/'

edep_tree = ROOT.TChain("EDepSimEvents")
groo_tree = ROOT.TChain("DetSimPassThru/gRooTracker")
    
for filename in os.listdir(directory):
    fname = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(fname):        
        edep_tree.Add(fname)
        groo_tree.Add(fname)

output_file = '/global/cfs/projectdirs/dune/users/yifanch/2x2_nue/nuebar_evt_info_RHC_5.h5'
#output_file = '/project/projectdirs/dune/users/yifanch/2x2_nue/nue_evt_info_FHC.h5'
with h5py.File(output_file, 'w') as f:
    f.create_dataset('nue_info', (0,), dtype=event_dtype, maxshape=(None,))

################
## NEW
###############
# Adding containments
nEvts  = edep_tree.GetEntries()
print("nEvts: ", nEvts)

for evt in range(nEvts):
   
    #if evt < 524000: #524000: #428000: #34000:
    #    continue
    #if evt > 526000: #526000: #430000: #36000:
    #    break
    if evt%2000 == 0:
        print("--evt: ", evt)
        print("time: ", time.time() - this_time)
        this_time = time.time()
    
    #######
    # GENIE
    #######
    groo_tree.GetEntry(evt)
    nu_pdg, nu_4mom = get_neutrino_4mom(groo_tree)

    if abs(nu_pdg) != 12: continue
    
    evt_info = np.empty(1, dtype=event_dtype)
    
    evt_info = init_evt_info(evt_info)
      
    evt_info['nuPDG'] = nu_pdg
    
    # [GeV]
    evt_info['Enu'] = nu_4mom[3]/1e3
    evt_info['pnu_x'] = nu_4mom[0]/1e3
    evt_info['pnu_y'] = nu_4mom[1]/1e3
    evt_info['pnu_z'] = nu_4mom[2]/1e3
    
    # [m]
    evt_info['Vtx_x'] = groo_tree.EvtVtx[0]*100
    evt_info['Vtx_y'] = groo_tree.EvtVtx[1]*100
    evt_info['Vtx_z'] = groo_tree.EvtVtx[2]*100
    
    vtx = [groo_tree.EvtVtx[0]*100, groo_tree.EvtVtx[1]*100, groo_tree.EvtVtx[2]*100]
    if not is_2x2_contained(vtx):
        warnings.warn("Warning...........Neutrino vertex not in 2x2")

    ######  
    # G4
    ######
    edep_tree.GetEntry(evt)
    
    event = edep_tree.Event

    ## Vertex info
    # assuming one interaction here, would not be true for full spill simulation
    vertex = event.Primaries[0]

    ## Primary
    primary_pars = vertex.Particles
   
    ccnc, lep_trkid, lep_pdgcode = is_nue_cc(event) 
    if len(lep_trkid) > 1:
        print("evt: ", evt)
        print("lep_trkid: ", lep_trkid)
        print("lep_pdgcode: ", lep_pdgcode)
        print("StdHepPdg, StdHepStatus: ", print_genie(groo_tree))

    # lepton number conservation, if there's no neutrino, charged leptons come in pairs
    nu2_id, nu2_pdg = get_traj_ids_for_pdg(event.Primaries[0].Particles, [12, -12]) 
    if len(lep_trkid) % 2 == 0 and len(lep_trkid) != 0 and len(nu2_pdg) % 2 == 0 and len(nu2_pdg) != 0:
        print("nu2_id: ", nu2_id)
        print("nu2_pdg: ", nu2_pdg)
        evt_info['ccnc'] = 0

        with h5py.File(output_file, 'a') as f:
            if len(evt_info):
                n_nue = len(f['nue_info'])
                f['nue_info'].resize((n_nue+len(evt_info),))
                f['nue_info'][n_nue:] = evt_info

        continue
    if len(lep_trkid) % 2 == 1 and len(nu2_pdg) % 2 == 1:
        print("evt: ", evt)
        print("lep_trkid: ", lep_trkid)
        print("lep_pdgcode: ", lep_pdgcode)
        print("StdHepPdg, StdHepStatus: ", print_genie(groo_tree))
        print("nu2_id: ", nu2_id)
        print("nu2_pdg: ", nu2_pdg)
        evt_info['ccnc'] = 0

        with h5py.File(output_file, 'a') as f:
            if len(evt_info):
                n_nue = len(f['nue_info'])
                f['nue_info'].resize((n_nue+len(evt_info),))
                f['nue_info'][n_nue:] = evt_info

        continue

    if ccnc: 
        evt_info['ccnc'] = 1
    else:
        evt_info['ccnc'] = 0

        with h5py.File(output_file, 'a') as f:
            if len(evt_info):
                n_nue = len(f['nue_info'])
                f['nue_info'].resize((n_nue+len(evt_info),))
                f['nue_info'][n_nue:] = evt_info

        continue

    lep_trkID = []
    ldp_trkID = []
    ldpiplus_trkID = []
    ldpiminus_trkID = []
    ldpi0_trkID = []

    n_protons = 0
    n_piplus = 0
    n_piminus = 0
    n_pi0 = 0
    
    for primary in primary_pars:   
        ## electron
        ## in case there are multiple electron, pick the first one which is likely to be actually from the vertex
        if abs(primary.GetPDGCode()) == 11 and evt_info['Elep'] == 0:  
            if primary.GetPDGCode() == 11:
                evt_info['lepPDG'] = 11
            elif primary.GetPDGCode() == -11:
                evt_info['lepPDG'] = -11

            primary_mom = primary.GetMomentum()
            primary_th  = primary_mom.Vect().Angle(zdir)*180/math.pi
            primary_ek  = (primary_mom.E() - primary_mom.M())/1e3 #GeV

            evt_info['Elep'] = primary_mom[3]/1e3
            evt_info['plep_x'] = primary_mom[0]/1e3
            evt_info['plep_y'] = primary_mom[1]/1e3
            evt_info['plep_z'] = primary_mom[2]/1e3
            
            evt_info['lep_Theta'] = primary_th
            evt_info['lep_KE'] = primary_ek
            
            lep_trkID.append(primary.GetTrackId())
            
            # q2, q0
            evt_info['Q2'] = -1 *(primary_mom - nu_4mom).Mag2()/1e6 #GeV
            evt_info['q0'] = evt_info['Enu'] -  evt_info['Elep'] #GeV
            
        ## proton
        if primary.GetPDGCode() == 2212:  
            
            n_protons += 1
            
            primary_mom = primary.GetMomentum()
            
            if primary_mom[3] > evt_info['Eldp']:               
                primary_th  = primary_mom.Vect().Angle(zdir)*180/math.pi
                primary_ek  = (primary_mom.E() - primary_mom.M())/1e3 #GeV

                evt_info['Eldp'] = primary_mom[3]/1e3
                evt_info['pldp_x'] = primary_mom[0]/1e3
                evt_info['pldp_y'] = primary_mom[1]/1e3
                evt_info['pldp_z'] = primary_mom[2]/1e3

                evt_info['ldp_Theta'] = primary_th
                evt_info['ldp_KE'] = primary_ek
                
                if len(ldp_trkID) == 0:
                    ldp_trkID.append(primary.GetTrackId())
                else:
                    ldp_trkID[-1] = primary.GetTrackId()
                               
        ## piplus
        if primary.GetPDGCode() == 211:  

            n_piplus += 1
            
            primary_mom = primary.GetMomentum()
            
            if primary_mom[3] > evt_info['Eldpiplus']:               
                primary_th  = primary_mom.Vect().Angle(zdir)*180/math.pi
                primary_ek  = (primary_mom.E() - primary_mom.M())/1e3 #GeV

                evt_info['Eldpiplus'] = primary_mom[3]/1e3
                evt_info['pldpiplus_x'] = primary_mom[0]/1e3
                evt_info['pldpiplus_y'] = primary_mom[1]/1e3
                evt_info['pldpiplus_z'] = primary_mom[2]/1e3

                evt_info['ldpiplus_Theta'] = primary_th
                evt_info['ldpiplus_KE'] = primary_ek
                
                if len(ldpiplus_trkID) == 0:
                    ldpiplus_trkID.append(primary.GetTrackId())
                else:
                    ldpiplus_trkID[-1] = primary.GetTrackId()
                
        ## piminus
        if primary.GetPDGCode() == -211: 
            
            n_piminus += 1

            primary_mom = primary.GetMomentum()
                       
            if primary_mom[3] > evt_info['Eldpiminus']:               
                primary_th  = primary_mom.Vect().Angle(zdir)*180/math.pi
                primary_ek  = (primary_mom.E() - primary_mom.M())/1e3 #GeV

                evt_info['Eldpiminus'] = primary_mom[3]/1e3
                evt_info['pldpiminus_x'] = primary_mom[0]/1e3
                evt_info['pldpiminus_y'] = primary_mom[1]/1e3
                evt_info['pldpiminus_z'] = primary_mom[2]/1e3

                evt_info['ldpiminus_Theta'] = primary_th
                evt_info['ldpiminus_KE'] = primary_ek
                
                if len(ldpiminus_trkID) == 0:
                    ldpiminus_trkID.append(primary.GetTrackId())
                else:
                    ldpiminus_trkID[-1] = primary.GetTrackId()
                    
                
        ## pi0
        if primary.GetPDGCode() == 111:  
            
            n_pi0 += 1

            primary_mom = primary.GetMomentum()
              
            if primary_mom[3] > evt_info['Eldpi0']:               
                primary_th  = primary_mom.Vect().Angle(zdir)*180/math.pi
                primary_ek  = (primary_mom.E() - primary_mom.M())/1e3 #GeV

                evt_info['Eldpi0'] = primary_mom[3]/1e3
                evt_info['pldpi0_x'] = primary_mom[0]/1e3
                evt_info['pldpi0_y'] = primary_mom[1]/1e3
                evt_info['pldpi0_z'] = primary_mom[2]/1e3

                evt_info['ldpi0_Theta'] = primary_th
                evt_info['ldpi0_KE'] = primary_ek
                
                if len(ldpi0_trkID) == 0:
                    ldpi0_trkID.append(primary.GetTrackId())
                else:
                    ldpi0_trkID[-1] = primary.GetTrackId()
        
    evt_info['n_protons'] = n_protons
    evt_info['n_piplus'] = n_piplus
    evt_info['n_piminus'] = n_piminus
    evt_info['n_pi0'] = n_pi0
    
    ## electron / positron    
    e_reco_energy = 0
    e_reco_energy_2x2 = 0
    e_reco_energy_minerva = 0
    
    e_detector_exiting = False
    e_2x2_exiting = False
    e_minerva_tagged = False

    ## hadronic system
    E_had = 0 # sum of all non leptonic and non lowE energy deposition
    E_had_reco = 0 # sum of all non leptonic, non lowE and non neutron energy deposition
    E_had_reco_2x2 = 0 # E_had_reco in 2x2
    E_had_reco_MINERvA = 0 # E_had_reco in MINERvA

    had_detector_exiting = False
    had_2x2_exiting = False
    had_minerva_tagged = False

    ## proton    
    p_reco_energy = 0
    p_reco_energy_2x2 = 0
    p_reco_energy_minerva = 0
    
    p_detector_exiting = False
    p_2x2_exiting = False
    p_minerva_tagged = False
    
    ## piplus    
    piplus_reco_energy = 0
    piplus_reco_energy_2x2 = 0
    piplus_reco_energy_minerva = 0
    
    piplus_detector_exiting = False
    piplus_2x2_exiting = False
    piplus_minerva_tagged = False
    
    ## piminus   
    piminus_reco_energy = 0
    piminus_reco_energy_2x2 = 0
    piminus_reco_energy_minerva = 0
    
    piminus_detector_exiting = False
    piminus_2x2_exiting = False
    piminus_minerva_tagged = False
    
    ## pi0   
    pi0_reco_energy = 0
    pi0_reco_energy_2x2 = 0
    pi0_reco_energy_minerva = 0
    
    pi0_detector_exiting = False
    pi0_2x2_exiting = False
    pi0_minerva_tagged = False
    
    
    ## Get all neutrons and neutron descendents in the event
    neutron_ids = get_neutron_and_daughter_ids(event)
    
    ## Get a list of low energy truth trajectories (may be quite long)
    low_energy_ids = get_low_energy_ids(event)
   
    ## Get a list of pi0 daughter ids
    pi0_daughter_ids = get_pi0_daughter_ids(event, ldpi0_trkID)
    
    ## Loop over the detector segments (see description elsewhere in this file)
    for seg in event.SegmentDetectors:
        
        ## Loop over the segments in the volume
        nChunks = len(seg[1])
        for n in range(nChunks):
            
            ## Get the primary id that is associated with this segment
            key_contrib = seg[1][n].GetContributors()[0]
            par_contrib = seg[1][n].GetPrimaryId()

            ## Skip anything which is very low energy (delta rays often escape the volume and distort the containment numbers)
            if key_contrib in low_energy_ids: continue


            ## hadronic system
            if par_contrib not in lep_trkID:

                pos = seg[1][n].GetStop()

                E_had += seg[1][n].GetEnergyDeposit()

                if key_contrib not in neutron_ids:

                    ##### exclude neutrons in containment
                    ## As soon as we find something uncontained, we can just leave
                    if is_minerva_tagged(pos):
                        had_minerva_tagged = True

                    ## As soon as we find something uncontained, we can just leave
                    if not is_2x2_contained(pos):
                        had_2x2_exiting = True

                    ## As soon as we find something uncontained, we can just leave
                    if not is_detector_contained(pos):
                        had_detector_exiting = True

                    E_had_reco += seg[1][n].GetEnergyDeposit()

                    if is_2x2_contained(pos):
                        E_had_reco_2x2 += seg[1][n].GetEnergyDeposit()
                    if is_minerva_tagged(pos):
                        E_had_reco_MINERvA += seg[1][n].GetEnergyDeposit()


            ## Did this segment come (mostly) from a neutron or a descendant from a neutron?
            if key_contrib in neutron_ids: continue
    
            #############
            ## Only consider contributions that can be tracked back to the target
            ## electron
            if par_contrib in lep_trkID:
            
                pos = seg[1][n].GetStop()

                ## As soon as we find something uncontained, we can just leave
                if is_minerva_tagged(pos): 
                    e_minerva_tagged = True

                ## As soon as we find something uncontained, we can just leave
                if not is_2x2_contained(pos): 
                    e_2x2_exiting = True

                ## As soon as we find something uncontained, we can just leave
                if not is_detector_contained(pos): 
                    e_detector_exiting = True

                e_reco_energy += seg[1][n].GetEnergyDeposit()
                if is_2x2_contained(pos):
                    e_reco_energy_2x2 += seg[1][n].GetEnergyDeposit()
                if is_minerva_tagged(pos):
                    e_reco_energy_minerva += seg[1][n].GetEnergyDeposit()

            ## proton
            if par_contrib in ldp_trkID: 
            
                pos = seg[1][n].GetStop()

                ## As soon as we find something uncontained, we can just leave
                if is_minerva_tagged(pos): 
                    p_minerva_tagged = True

                ## As soon as we find something uncontained, we can just leave
                if not is_2x2_contained(pos): 
                    p_2x2_exiting = True

                ## As soon as we find something uncontained, we can just leave
                if not is_detector_contained(pos): 
                    p_detector_exiting = True

                p_reco_energy += seg[1][n].GetEnergyDeposit()
                if is_2x2_contained(pos):
                    p_reco_energy_2x2 += seg[1][n].GetEnergyDeposit()
                if is_minerva_tagged(pos):
                    p_reco_energy_minerva += seg[1][n].GetEnergyDeposit()
                    
            ## piplus
            if par_contrib in ldpiplus_trkID: 
            
                pos = seg[1][n].GetStop()

                ## As soon as we find something uncontained, we can just leave
                if is_minerva_tagged(pos): 
                    piplus_minerva_tagged = True

                ## As soon as we find something uncontained, we can just leave
                if not is_2x2_contained(pos): 
                    piplus_2x2_exiting = True

                ## As soon as we find something uncontained, we can just leave
                if not is_detector_contained(pos): 
                    piplus_detector_exiting = True

                piplus_reco_energy += seg[1][n].GetEnergyDeposit()
                if is_2x2_contained(pos):
                    piplus_reco_energy_2x2 += seg[1][n].GetEnergyDeposit()
                if is_minerva_tagged(pos):
                    piplus_reco_energy_minerva += seg[1][n].GetEnergyDeposit()
                    
            ## piminus
            if par_contrib in ldpiminus_trkID: 
            
                pos = seg[1][n].GetStop()

                ## As soon as we find something uncontained, we can just leave
                if is_minerva_tagged(pos): 
                    piminus_minerva_tagged = True

                ## As soon as we find something uncontained, we can just leave
                if not is_2x2_contained(pos): 
                    piminus_2x2_exiting = True

                ## As soon as we find something uncontained, we can just leave
                if not is_detector_contained(pos): 
                    piminus_detector_exiting = True

                piminus_reco_energy += seg[1][n].GetEnergyDeposit()
                if is_2x2_contained(pos):
                    piminus_reco_energy_2x2 += seg[1][n].GetEnergyDeposit()
                if is_minerva_tagged(pos):
                    piminus_reco_energy_minerva += seg[1][n].GetEnergyDeposit()
                    
            ## pi0
            if par_contrib in pi0_daughter_ids: 
            #if par_contrib in ldpi0_trkID: 
            
                pos = seg[1][n].GetStop()

                ## As soon as we find something uncontained, we can just leave
                if is_minerva_tagged(pos): 
                    pi0_minerva_tagged = True

                ## As soon as we find something uncontained, we can just leave
                if not is_2x2_contained(pos): 
                    pi0_2x2_exiting = True

                ## As soon as we find something uncontained, we can just leave
                if not is_detector_contained(pos): 
                    pi0_detector_exiting = True

                pi0_reco_energy += seg[1][n].GetEnergyDeposit()
                if is_2x2_contained(pos):
                    pi0_reco_energy_2x2 += seg[1][n].GetEnergyDeposit()
                if is_minerva_tagged(pos):
                    pi0_reco_energy_minerva += seg[1][n].GetEnergyDeposit()

    evt_info['E_had'] = E_had/1e3
    evt_info['E_had_reco'] = E_had_reco/1e3
    evt_info['E_had_reco_2x2'] = E_had_reco_2x2/1e3
    evt_info['E_had_reco_MINERvA'] = E_had_reco_MINERvA/1e3

    evt_info['Elep_reco'] = e_reco_energy/1e3
    evt_info['Elep_2x2'] = e_reco_energy_2x2/1e3
    evt_info['Elep_MINERvA'] = e_reco_energy_minerva/1e3

    evt_info['Eldp_reco'] = p_reco_energy/1e3
    evt_info['Eldp_2x2'] = p_reco_energy_2x2/1e3
    evt_info['Eldp_MINERvA'] = p_reco_energy_minerva/1e3
    
    evt_info['Eldpiplus_reco'] = piplus_reco_energy/1e3
    evt_info['Eldpiplus_2x2'] = piplus_reco_energy_2x2/1e3
    evt_info['Eldpiplus_MINERvA'] = piplus_reco_energy_minerva/1e3
    
    evt_info['Eldpiminus_reco'] = piminus_reco_energy/1e3
    evt_info['Eldpiminus_2x2'] = piminus_reco_energy_2x2/1e3
    evt_info['Eldpiminus_MINERvA'] = piminus_reco_energy_minerva/1e3
    
    evt_info['Eldpi0_reco'] = pi0_reco_energy/1e3
    evt_info['Eldpi0_2x2'] = pi0_reco_energy_2x2/1e3
    evt_info['Eldpi0_MINERvA'] = pi0_reco_energy_minerva/1e3
    
    ## electron
    if not e_2x2_exiting and e_minerva_tagged:
        warnings.warn("Warning...........the electron cannot be contained in 2x2 and tagged by minerva")
        
    if not e_detector_exiting:
        if not e_2x2_exiting:
            evt_info['lep_containment'] = 1
        elif e_minerva_tagged:
            evt_info['lep_containment'] = 2
        else:
            evt_info['lep_containment'] = 3
            
    else:
        if e_minerva_tagged:
            evt_info['lep_containment'] = 4
        else:
            evt_info['lep_containment'] = 5

    ## hadronic system
    if not had_2x2_exiting and had_minerva_tagged:
        warnings.warn("Warning...........the hadronic outcome cannot be contained in 2x2 and tagged by minerva")

    if not had_detector_exiting:
        if not had_2x2_exiting:
            evt_info['had_containment'] = 1
        elif had_minerva_tagged:
            evt_info['had_containment'] = 2
        else:
            evt_info['had_containment'] = 3

    else:
        if had_minerva_tagged:
            evt_info['had_containment'] = 4
        else:
            evt_info['had_containment'] = 5
            
    ## proton
    if not p_2x2_exiting and p_minerva_tagged:
        warnings.warn("Warning...........the leading proton cannot be contained in 2x2 and tagged by minerva")
        
    if not p_detector_exiting:
        if not p_2x2_exiting:
            evt_info['ldp_containment'] = 1
        elif p_minerva_tagged:
            evt_info['ldp_containment'] = 2
        else:
            evt_info['ldp_containment'] = 3
            
    else:
        if p_minerva_tagged:
            evt_info['ldp_containment'] = 4
        else:
            evt_info['ldp_containment'] = 5
                        
    ## piplus
    if not piplus_2x2_exiting and piplus_minerva_tagged:
        warnings.warn("Warning...........the leading piplus cannot be contained in 2x2 and tagged by minerva")
        
    if not piplus_detector_exiting:
        if not piplus_2x2_exiting:
            evt_info['ldpiplus_containment'] = 1
        elif piplus_minerva_tagged:
            evt_info['ldpiplus_containment'] = 2
        else:
            evt_info['ldpiplus_containment'] = 3
            
    else:
        if piplus_minerva_tagged:
            evt_info['ldpiplus_containment'] = 4
        else:
            evt_info['ldpiplus_containment'] = 5
            
    ## piminus
    if not piminus_2x2_exiting and piminus_minerva_tagged:
        warnings.warn("Warning...........the leading piminus cannot be contained in 2x2 and tagged by minerva")
        
    if not piminus_detector_exiting:
        if not piminus_2x2_exiting:
            evt_info['ldpiminus_containment'] = 1
        elif piminus_minerva_tagged:
            evt_info['ldpiminus_containment'] = 2
        else:
            evt_info['ldpiminus_containment'] = 3
            
    else:
        if piminus_minerva_tagged:
            evt_info['ldpiminus_containment'] = 4
        else:
            evt_info['ldpiminus_containment'] = 5
            
    ## pi0
    if not pi0_2x2_exiting and pi0_minerva_tagged:
        warnings.warn("Warning...........the leading pi0 cannot be contained in 2x2 and tagged by minerva")
        
    if not pi0_detector_exiting:
        if not pi0_2x2_exiting:
            evt_info['ldpi0_containment'] = 1
        elif pi0_minerva_tagged:
            evt_info['ldpi0_containment'] = 2
        else:
            evt_info['ldpi0_containment'] = 3
            
    else:
        if pi0_minerva_tagged:
            evt_info['ldpi0_containment'] = 4
        else:
            evt_info['ldpi0_containment'] = 5

    with h5py.File(output_file, 'a') as f:
        if len(evt_info):
            n_nue = len(f['nue_info'])
            f['nue_info'].resize((n_nue+len(evt_info),))
            f['nue_info'][n_nue:] = evt_info
         

print("total run time [hr]: ", (time.time() - start_time)/3600.)
print("----fin----")
