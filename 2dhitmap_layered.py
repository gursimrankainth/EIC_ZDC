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
canvas.Divide(3, 3)

# Create histogram and make bin settings
bins = 25
x_min, x_max = -1240, -590
y_min, y_max = -300, 300
#x_min, x_max = -910, -850
#y_min, y_max = -200, 200
histograms = [] 
hist_e0 = ROOT.TH2F("HitPositionXYHist_E0","ECal 2D Hitmap 40-200GeV; Position X [mm]; Position Y [mm]" , bins, x_min, x_max, bins, y_min, y_max)
#hist_h0 = ROOT.TH2F("HitPositionXYHist_H0","HCal 2D Hitmap 40-200GeV; Position X [mm]; Position Y [mm]" , bins, x_min, x_max, bins, y_min, y_max)
hist_h8 = ROOT.TH2F("HitPositionXYHist_H8","HCal L8 2D Hitmap 40-200GeV; Position X [mm]; Position Y [mm]" , bins, x_min, x_max, bins, y_min, y_max)
hist_h16 = ROOT.TH2F("HitPositionXYHist_H16","HCal L16 2D Hitmap 40-200GeV; Position X [mm]; Position Y [mm]" , bins, x_min, x_max, bins, y_min, y_max)
hist_h24 = ROOT.TH2F("HitPositionXYHist_H24","HCal L24 2D Hitmap 40-200GeV; Position X [mm]; Position Y [mm]" , bins, x_min, x_max, bins, y_min, y_max)
hist_h32 = ROOT.TH2F("HitPositionXYHist_H32","HCal L32 2D Hitmap 40-200GeV; Position X [mm]; Position Y [mm]" , bins, x_min, x_max, bins, y_min, y_max)
hist_h40 = ROOT.TH2F("HitPositionXYHist_H40","HCal L40 2D Hitmap 40-200GeV; Position X [mm]; Position Y [mm]" , bins, x_min, x_max, bins, y_min, y_max)
hist_h48 = ROOT.TH2F("HitPositionXYHist_H48","HCal L48 2D Hitmap 40-200GeV; Position X [mm]; Position Y [mm]" , bins, x_min, x_max, bins, y_min, y_max)
hist_h56 = ROOT.TH2F("HitPositionXYHist_H56","HCal L56 2D Hitmap 40-200GeV; Position X [mm]; Position Y [mm]" , bins, x_min, x_max, bins, y_min, y_max)
hist_h64 = ROOT.TH2F("HitPositionXYHist_H64","HCal L64 2D Hitmap 40-200GeV; Position X [mm]; Position Y [mm]" , bins, x_min, x_max, bins, y_min, y_max)
histograms.extend([hist_e0, hist_h8, hist_h16, hist_h24, hist_h32, hist_h40, hist_h48, hist_h56, hist_h64])

#Define the ZDC Planes 
ZDC_norm = np.array([np.sin(-0.025), 0., np.cos(-0.025)])
ecal_pt = np.array([35500*np.sin(-0.025), 0., 35500*np.cos(-0.025)])
hcal_pt0 = np.array([36607.5*np.sin(-0.025), 0., 36607.5*np.cos(-0.025)])
hcal_pts = [hcal_pt0] 
for i in range(1, 64):
    lay_thick = 24.9 
    r0 = 36607.5 
    lay_loc = ZDC_norm * (lay_thick*i + r0)
    hcal_pts.append(lay_loc)

"""
print(hcal_pts[0][-1])
for i in range(7, len(hcal_pts), 8):
    print(hcal_pts[i][-1])
print(len(hcal_pts))
"""

#Function that determines the intersection between the momentum vector and ZDC Planes
def intersection(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = np.dot(planeNormal,rayDirection)
    if abs(ndotu) > epsilon:
        w = rayPoint - planePoint
        t = -np.dot(planeNormal,w) / ndotu
        u = rayDirection * t
        return u
    return None 

# Fill histograms with data from the tree
e_num = 0 
event_list = [] 

for event in chain:
    MCParticles_vec = event.MCParticles
    pid_list = [] 
    temp_list = [e_num]

    for i in range(MCParticles_vec.size()):
        particle = MCParticles_vec.at(i)
        pid = particle.PDG
        pid_list.append(pid)
        mom_vec = [particle.momentum.x, particle.momentum.y, particle.momentum.z]
        mom_vec = mom_vec / np.linalg.norm(mom_vec)
        vertex_vec = [particle.vertex.x, particle.vertex.y, particle.vertex.z]
        end_vec = [particle.endpoint.x, particle.endpoint.y, particle.endpoint.y]

        #Find the intersection point on the specific ZDC layer
        inter_pt_e = intersection(ZDC_norm, ecal_pt, np.array(mom_vec), np.array(vertex_vec))
        #inter_pt_h0 = intersection(ZDC_norm, hcal_pts[0], np.array(mom_vec), np.array(vertex_vec))
        inter_pt_h8 = intersection(ZDC_norm, hcal_pts[7], np.array(mom_vec), np.array(vertex_vec))
        inter_pt_h16 = intersection(ZDC_norm, hcal_pts[15], np.array(mom_vec), np.array(vertex_vec))
        inter_pt_h24 = intersection(ZDC_norm, hcal_pts[23], np.array(mom_vec), np.array(vertex_vec))
        inter_pt_h32 = intersection(ZDC_norm, hcal_pts[31], np.array(mom_vec), np.array(vertex_vec))
        inter_pt_h40 = intersection(ZDC_norm, hcal_pts[39], np.array(mom_vec), np.array(vertex_vec))
        inter_pt_h48 = intersection(ZDC_norm, hcal_pts[47], np.array(mom_vec), np.array(vertex_vec))
        inter_pt_h56 = intersection(ZDC_norm, hcal_pts[57], np.array(mom_vec), np.array(vertex_vec))
        inter_pt_h64 = intersection(ZDC_norm, hcal_pts[63], np.array(mom_vec), np.array(vertex_vec))
        hist_e0.Fill(inter_pt_e[0],inter_pt_e[1])
        hist_h8.Fill(inter_pt_h8[0], inter_pt_h8[1])
        hist_h16.Fill(inter_pt_h16[0], inter_pt_h16[1])
        hist_h24.Fill(inter_pt_h24[0], inter_pt_h24[1])
        hist_h32.Fill(inter_pt_h32[0], inter_pt_h32[1])
        hist_h40.Fill(inter_pt_h40[0], inter_pt_h40[1])
        hist_h48.Fill(inter_pt_h48[0], inter_pt_h48[1])
        hist_h56.Fill(inter_pt_h56[0], inter_pt_h56[1])
        hist_h64.Fill(inter_pt_h64[0], inter_pt_h64[1])

        temp_list.extend([pid, inter_pt_e, inter_pt_h8, inter_pt_h16, inter_pt_h24, inter_pt_h32, inter_pt_h40, inter_pt_h48, inter_pt_h56, inter_pt_h64])
    
    event_list.append(temp_list)
    e_num += 1    

#print(event_list[5])

#Draw the Histograms and format the canvas
for i, hist in enumerate(histograms):
    pad = canvas.cd(i + 1)
    pad.SetLeftMargin(0.15)
    pad.SetRightMargin(0.15)
    #pad.SetTopMargin(0.1)
    #pad.SetBottomMargin(0.15)
    hist.Draw("COLZ")
    hist.GetXaxis().SetLabelSize(0.04)
    hist.GetYaxis().SetLabelSize(0.04)
    hist.GetXaxis().SetTitleSize(0.04)
    hist.GetYaxis().SetTitleSize(0.04)

"""
# Draw the histogram and format it
hist_e0.Draw("COLZ")
canvas.SetLeftMargin(0.15)
canvas.SetRightMargin(0.15)
hist_e0.GetXaxis().SetLabelSize(0.04)
hist_e0.GetYaxis().SetLabelSize(0.04)
hist_e0.GetXaxis().SetTitleSize(0.04)
hist_e0.GetYaxis().SetTitleSize(0.04)
"""

canvas.Update()

# Keep the canvas open
input("Press Enter to exit")

#No need to close file since using TChain 
