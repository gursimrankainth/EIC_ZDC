import ROOT # type: ignore
import numpy as np # type: ignore
from collections import Counter # type: ignore

#Open ROOT files and get the 'events' tree
#file = ROOT.TFile("/Users/gursimran/eic/Lambda/lambda_40_200GeV_ZDCv1_1e3_1.edm4hep.root", "READ")
chain = ROOT.TChain("events")
chain.Add("/Users/gursimran/eic/Lambda/lambda_40_200GeV_ZDCv1_1e3_1.edm4hep.root")
chain.Add("/Users/gursimran/eic/Lambda/lambda_40_200GeV_ZDCv1_1e3_2.edm4hep.root")
chain.Add("/Users/gursimran/eic/Lambda/lambda_40_200GeV_ZDCv1_1e3_3.edm4hep.root")
chain.Add("/Users/gursimran/eic/Lambda/lambda_40_200GeV_ZDCv1_1e3_4.edm4hep.root")

#Make the canvas and set its properties
canvas = ROOT.TCanvas("canvas", "Momentum_Study", 800, 600)
canvas.Divide(3, 1)

#Make the Histograms
histograms = []
histpi0 = ROOT.TH2F("momentum_pi0_lambda", "Momentum Pi0 vs. Lambda; Lambda Momentum [GeV]; Pi0 Momentum [GeV]", 100, 40, 200, 100, 0, 250)
histn = ROOT.TH2F("momentum_n_lambda", "Momentum Neutron vs. Lambda; Lambda Momentum [GeV]; Neutron Momentum [GeV]", 100, 40, 200, 100, 0, 250)
histgg = ROOT.TH2F("momentum_gg_lambda", "Momentum Photons vs. Lambda; Lambda Momentum [GeV]; Photon Momentum [GeV]", 100, 40, 200, 100, 0, 250)
histograms.extend([histpi0, histn, histgg])

#Function that isolates events of interest
def interesting_event(lst):
    required_pids = {22, 111, 2112, 3122}
    lst_set = set(lst)
    return required_pids.issubset(lst_set)

#Loop over the events and get the pi0 and neutron momentum information
e_num = 0 #Counter for event tracking
event_list = [] 
ignore = [] 
single_hit = []
for event in chain:
    MCParticles_vec = event.MCParticles
    pid_list = []  
    temp_list = [e_num] 

    for i in range(MCParticles_vec.size()):
        particle = MCParticles_vec.at(i)
        pid = particle.PDG
        mom_vec = [particle.momentum.x, particle.momentum.y, particle.momentum.z]
        mag_mom_vec = np.linalg.norm(mom_vec)

        #PID: photon 22, pi0 111, neutron 2112, lambda 3122
        if pid == 22 or pid == 111 or pid == 2112 or pid == 3122:
            temp_list.extend([pid, mag_mom_vec])
        else: 
            ignore.append(e_num)
    e_num += 1   

    #Fill the Histograms with the momentum data
    check = interesting_event(temp_list)
    if check == True: 
        event_list.append(temp_list)
        #Isolate events where all daughter particles hit the ZDC
        if len(temp_list) == 11: 
            histpi0.Fill(temp_list[2],temp_list[4])
            histn.Fill(temp_list[2],temp_list[6])
            histgg.Fill(temp_list[2],temp_list[8])
            histgg.Fill(temp_list[2],temp_list[10])
        else: 
            single_hit.append(temp_list[0])

    else: 
        None
#print(event_list[0])
single_hit_frac = len(single_hit)/len(event_list)
print("Percentage of events where only a single photon hits the ZDC:", round(single_hit_frac * 100 ,2))

#Draw the Histograms and format the canvas
for i, hist in enumerate(histograms):
    pad = canvas.cd(i + 1)
    pad.SetLeftMargin(0.15)
    pad.SetRightMargin(0.15)
    hist.Draw("COLZ")
    hist.GetXaxis().SetLabelSize(0.04)
    hist.GetYaxis().SetLabelSize(0.04)
    hist.GetXaxis().SetTitleSize(0.04)
    hist.GetYaxis().SetTitleSize(0.04)

canvas.Update()

#Keep the canvas open
input("Press Enter to exit")

#No need to close the chain as its not a single file