import ROOT # type: ignore
import numpy as np # type: ignore
from collections import Counter # type: ignore

#Get hit data from root files
#Ouput is a list of events. In each event is another list with the x,y,z hit position and energy 
def extract_hits(chain): 
    event_data_hcal = [] 
    event_data_ecal = []  
    event_data_particles = []

    for event in chain:
        MCParticles_vec = event.MCParticles
        hcal_hits = event.HcalFarForwardZDCHits
        ecal_hits = event.EcalFarForwardZDCHits
        temp_list = []  
        hit_listh = []
        hit_liste = []

        for i in range(MCParticles_vec.size()):
            particle = MCParticles_vec.at(i)
            mom_vec = [particle.momentum.x, particle.momentum.y, particle.momentum.z]
            mag_mom_vec = np.linalg.norm(mom_vec)
            vertex_vec = [particle.vertex.x, particle.vertex.y, particle.vertex.z]
            end_vec = [particle.endpoint.x, particle.endpoint.y, particle.endpoint.z]
            temp_list.append((particle.PDG, mom_vec, vertex_vec, end_vec))
            
        for hit in hcal_hits:
                hit_listh.append((hit.position.x, hit.position.y, hit.position.z, hit.energy, "h"))

        for hit in ecal_hits:
                hit_liste.append((hit.position.x, hit.position.y, hit.position.z, hit.energy, "e"))

        event_data_hcal.append(hit_listh)
        event_data_ecal.append(hit_liste)
        event_data_particles.append(temp_list)

    return (event_data_hcal, event_data_ecal, event_data_particles)

#Create a TChain and add ROOT files
chain = ROOT.TChain("events")
chain.Add("/Users/gursimran/eic/Lambda/singlePhotonData/photon_angle_5to70GeV.edm4hep.root")

event_data_hcal, event_data_ecal, event_data_particles = extract_hits(chain)

#Convert the lists to numpy arrays
event_data_hcal = np.array(event_data_hcal, dtype=object)
event_data_ecal = np.array(event_data_ecal, dtype=object)
event_data_particles = np.array(event_data_particles, dtype=object)

#Save the arrays to a .npz file
np.savez('singlephotondata_angle.npz', event_data_hcal=event_data_hcal, event_data_ecal=event_data_ecal, event_data_particles=event_data_particles)
print("Extraction complete. Data saved in singlephotondata_angle.npz")