# author: Rey Cruz Torres

import pandas as pd
import os
import ROOT
import numpy as np
import uproot
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------
# Class to analyze synrad hits
# ------------------------------------------------------------------------------------------------
class hits_from_synrad:
    def __init__(self,nevents,int_window=100.e-09,path_to_photons=None,path_to_hits=None):
        if path_to_photons==None or path_to_hits==None:
            print("need both 'path_to_photons' and 'path_to_hits' arguments")
            exit()

        self.nevents = nevents
        self.int_window = int_window
        self.path_to_photons = path_to_photons
        self.path_to_hits = path_to_hits

        self.number_seeded_geant_files = 25

        print('')
        print('requested number of events:',self.nevents)
        print('time integration window:',self.int_window)
        print('path to synrad photons:',self.path_to_photons)
        print('path to geant hits:',self.path_to_hits)
        print('')

        self.df_photons = pd.read_csv(os.path.join(self.path_to_photons,'combined_data.csv'))
        self.df_photons = self.df_photons.drop('E'    ,axis=1)
        self.df_photons = self.df_photons.drop('P'    ,axis=1)
        self.df_photons = self.df_photons.drop('Fs'   ,axis=1)
        self.df_photons = self.df_photons.drop('rho'  ,axis=1)
        self.df_photons = self.df_photons.drop('theta',axis=1)
        self.df_photons = self.df_photons.drop('phi'  ,axis=1)
        print(self.df_photons.head())

        n_entries = len(self.df_photons)
        self.h1_df = ROOT.TH1D('h1_df',';entry;W [#gamma/sec]',n_entries,0,n_entries)
        for i in range(n_entries):
            self.h1_df.SetBinContent(i+1,self.df_photons['NormFact'].iloc[i])

        self.detectors = ["VertexBarrelHits","SiBarrelHits","TaggerTracker1Hits"]

        self.hits = {}
        for detector in self.detectors:
            self.hits[detector] = {}
            for var in ['x','y','z']:
                self.hits[detector][var] = []

        '''
        "DRICHHits","EcalEndcapNHits","EcalEndcapPHits","VertexBarrelHits","SiBarrelHits","MPGDBarrelHits"
        "TrackerEndcapHits","MRICHHits","ZDC_PbSi_Hits","ZDC_WSi_Hits","ZDCHcalHits","TaggerTracker1Hits",
        "ForwardRomanPotHits","EcalBarrelHits","HcalEndcapPHits","HcalEndcapNHits","HcalEndcapPInsertHits",
        "HcalBarrelHits","B0PreshowerHits","B0TrackerHits","ForwardOffMTrackerHits","ZDC_SiliconPix_Hits",
        "ZDCEcalHits","TaggerCalorimeter1Hits","TaggerTracker2Hits","TaggerCalorimeter2Hits"
        '''

    # --------------------------------------------
    def generate_an_event(self):
        event = []
        integrated_so_far = 0.

        while integrated_so_far < self.int_window:
            x = self.h1_df.FindBin(self.h1_df.GetRandom())

            if x >= 1800000:
                continue
            
            photon = self.df_photons.iloc[x]
            current_facet = photon['facet']
            subdf = self.df_photons[self.df_photons['facet']==current_facet]

            sub_index = -100

            for i in range(len(subdf)):
                if subdf['x'].iloc[i] == photon['x'] and subdf['y'].iloc[i] == photon['y'] and subdf['z'].iloc[i] == photon['z'] \
                and subdf['px'].iloc[i] == photon['px'] and subdf['py'].iloc[i] == photon['py'] and subdf['pz'].iloc[i] == photon['pz'] \
                and subdf['NormFact'].iloc[i] == photon['NormFact']:
                    sub_index = i
                    break

            if sub_index == -100:
                print('There was an error')

            integrated_so_far += 1./photon['NormFact']
            event.append([(int)(current_facet),sub_index])
        
        return event

    # --------------------------------------------
    def generate(self):
        for i in range(self.nevents):
            if i%50==0:
                print('Event',i,'out of',self.nevents)

            event = self.generate_an_event()
            
            for photon in event:
                self.load_hits(photon[0],photon[1])

        for detector in self.detectors:
            self.plot_x_y_z_distributions(detector)

    # --------------------------------------------
    def load_hits(self,facet,idx):
        rnum = np.random.randint(1,self.number_seeded_geant_files)
        fname = os.path.join(self.path_to_hits,'geant_out_{}_seed_{}.edm4hep.root'.format(facet,rnum))
        F = uproot.open(fname)

        for detector in self.detectors:
            try:
                x = F['events/'+detector+'.position.x'].array(library="pd")[idx]
                y = F['events/'+detector+'.position.y'].array(library="pd")[idx]
                z = F['events/'+detector+'.position.z'].array(library="pd")[idx]

                for i in range(len(x)):
                    self.hits[detector]['x'].append(x.iloc[i])
                    self.hits[detector]['y'].append(y.iloc[i])
                    self.hits[detector]['z'].append(z.iloc[i])

            except:
                pass

    # --------------------------------------------
    def plot_x_y_z_distributions(self,detector):
        x = self.hits[detector]['x']
        y = self.hits[detector]['y']
        z = self.hits[detector]['z']

        print(x)

        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.scatter(x,y)
        plt.subplot(1,2,2)
        plt.scatter(z,x)

        plt.savefig('results_xyz_hits_'+detector+'.png',dpi=600)

# ------------------------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Creating an instance of hits_from_synrad')
    path_to_photons = '../220516_event_generator/'
    path_to_hits = 'geant_data/'
    nevents = 1000
    hits = hits_from_synrad(nevents,10.e-09,path_to_photons,path_to_hits) # argument is integration window in sec
    hits.generate()
