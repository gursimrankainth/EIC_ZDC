import ROOT # type: ignore
import numpy as np # type: ignore

# Open the ROOT files and access the tree named 'events'
chain = ROOT.TChain("events")
chain.Add("/Users/gursimran/eic/Lambda/lambda_40_200GeV_ZDCv1_1e3_1.edm4hep.root")
chain.Add("/Users/gursimran/eic/Lambda/lambda_40_200GeV_ZDCv1_1e3_2.edm4hep.root")
chain.Add("/Users/gursimran/eic/Lambda/lambda_40_200GeV_ZDCv1_1e3_3.edm4hep.root")
chain.Add("/Users/gursimran/eic/Lambda/lambda_40_200GeV_ZDCv1_1e3_4.edm4hep.root")

# Create a canvas to draw histograms on
canvas = ROOT.TCanvas("canvas", "Hitmap", 800, 600)

# Create histogram and make bin settings
bins = 25
x_min, x_max = -1240, -590
y_min, y_max = -300, 300
hist = ROOT.TH2F("HitPositionXYHist","ZDC 2D Hitmap 40-200GeV; Position X [mm]; Position Y [mm]" , bins, x_min, x_max, bins, y_min, y_max)

# Fill histograms with data from the tree
e_num = 0 
event_list = [] 

for event in chain:
    ecal_hits = event.EcalFarForwardZDCHits
    hcal_hits = event.HcalFarForwardZDCHits
    MCParticles_vec = event.MCParticles
    temp_list = [e_num]

    for hit in hcal_hits:
        h_pos_x = hit.position.x
        h_pos_y = hit.position.y
        h_pos_z = hit.position.z
        hist.Fill(h_pos_x, h_pos_y)

    for hit in ecal_hits:
        e_pos_x = hit.position.x
        e_pos_y = hit.position.y
        e_pos_z = hit.position.z
        hist.Fill(e_pos_x, e_pos_y)   

# Draw the histogram and format it
hist.Draw("COLZ")
canvas.SetLeftMargin(0.15)
canvas.SetRightMargin(0.15)
hist.GetXaxis().SetLabelSize(0.04)
hist.GetYaxis().SetLabelSize(0.04)
hist.GetXaxis().SetTitleSize(0.04)
hist.GetYaxis().SetTitleSize(0.04)

canvas.Update()

# Keep the canvas open
input("Press Enter to exit")

#No need to close file since using TChain 
