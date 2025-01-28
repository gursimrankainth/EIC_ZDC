import ROOT # type: ignore
import numpy as np # type: ignore

#Get hit data from root files
#Ouput is a list of events. In each event is another list with the x,y,z hit position and energy 
def extract_hits(chain): 
    e_num = 0 #Counter for event tracking
    event_data_hits = [] 
    event_data_particles = []
    good_evt_list = []

    for event in chain:
        MCParticles_vec = event.MCParticles
        hcal_hits = event.HcalFarForwardZDCHits
        ecal_hits = event.EcalFarForwardZDCHits
        temp_list = []  
        hit_list = []

        for i in range(MCParticles_vec.size()):
            particle = MCParticles_vec.at(i)
            pid = particle.PDG
            mom_vec = [particle.momentum.x, particle.momentum.y, particle.momentum.z]
            mag_mom_vec = np.linalg.norm(mom_vec)
            vertex_vec = [particle.vertex.x, particle.vertex.y, particle.vertex.z]
            end_vec = [particle.endpoint.x, particle.endpoint.y, particle.endpoint.z]
            if (pid == 22 or pid == 2112 or pid == 3122): #ignore pi0 information
                temp_list.append((particle.PDG, mom_vec, vertex_vec, end_vec, mom_vec))
 
        for hit in hcal_hits:
            hit_list.append((hit.position.x, hit.position.y, hit.position.z, hit.energy, "h"))

        for hit in ecal_hits:
            hit_list.append((hit.position.x, hit.position.y, hit.position.z, hit.energy, "e"))

        event_data_hits.append(hit_list)
        event_data_particles.append(temp_list)

        e_num += 1   
    return (event_data_hits, event_data_particles)

#Create a TChain and add ROOT files
chain = ROOT.TChain("events")
chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r100.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r101.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r200.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r201.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r202.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r203.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r204.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r206.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r207.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r208.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r209.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r210.root")

event_data_hits, event_data_particles = extract_hits(chain)

#Convert the lists to numpy arrays
event_data_hits = np.array(event_data_hits, dtype=object)
event_data_particles = np.array(event_data_particles, dtype=object)

#Save the arrays to a .npz file
np.savez('extracted_data_all.npz', event_data_hits=event_data_hits, event_data_particles=event_data_particles)
print("Extraction complete. Data saved in extracted_data_all.npz")