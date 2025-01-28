import ROOT # type: ignore
import numpy as np # type: ignore

def main():
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

    #Make the canvas 
    canvas = ROOT.TCanvas("canvas", "Hitmap", 800, 600)

    #Create a 3D histogram for x, y, z hits in the Hcal 
    bins = 45
    xmin = -1240
    xmax = -590
    hist2D_Ecal = ROOT.TH2F("EcalPositionXYHist", "ZDC ECal 2D Hit Position; X [mm]; Y [mm]; Z [mm]", 
                            bins, xmin, xmax, bins, -300, 300)

    #Loop over all entries in the tree and collect the hit information
    event_data = []
    for event in chain:
        temp_lst = []
        ecal_hits = event.EcalFarForwardZDCHits
        for hit in ecal_hits:
            e_pos_x = hit.position.x
            e_pos_y = hit.position.y
            e_pos_z = hit.position.z
            energy = hit.energy
            temp_lst.append((e_pos_x, e_pos_y, e_pos_z, energy))
        event_data.append(temp_lst)

    #Define a function to rotate the hit data to the particle's reference frame 
    def rotate(event_data, angle=0.025): 
        rot_lst = []  
        for hit in event_data:  
            x = hit[0] * np.cos(angle) + hit[2] * np.sin(angle)
            z = -hit[0] * np.sin(angle) + hit[2] * np.cos(angle)
            rot_lst.append((x, hit[1], z, hit[3]))
        return rot_lst
    
    #Define a function to plot the hit
    def plot_event(event_index, rotate_data=True):
        hist2D_Ecal.Reset()
        event = event_data[event_index]
        if rotate_data:
            event = rotate(event)
            hist2D_Ecal.GetXaxis().SetLimits(-300, 300)
        else: 
            hist2D_Ecal.GetXaxis().SetLimits(-1240, -590)
        for hit in event:
            hist2D_Ecal.Fill(hit[0], hit[1], hit[3])

        #Format the canvas and histogram
        hist2D_Ecal.Draw("COLZ")
        hist2D_Ecal.GetYaxis().SetTitleOffset(2.5)
        hist2D_Ecal.GetZaxis().SetTitleOffset(2.5)
        hist2D_Ecal.GetXaxis().SetLabelSize(0.02)
        hist2D_Ecal.GetYaxis().SetLabelSize(0.02)
        hist2D_Ecal.GetZaxis().SetLabelSize(0.02)
        hist2D_Ecal.GetXaxis().SetTitleSize(0.02)
        hist2D_Ecal.GetYaxis().SetTitleSize(0.02)
        hist2D_Ecal.GetZaxis().SetTitleSize(0.02)    
        canvas.SetRightMargin(0.15)
        canvas.Update()

    #Allow for continuous user input 
    while True:
        user_input = input(f"Enter event number (0 to {len(event_data) - 1}), or 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break
        try:
            event_index = int(user_input)
            if 0 <= event_index < len(event_data):
                plot_event(event_index)
            else:
                print(f"Invalid event number. Please enter a number between 0 and {len(event_data) - 1}.")
        except ValueError:
            print("Invalid input. Please enter a valid event number or 'exit' to quit.")

if __name__ == "__main__":
    main()