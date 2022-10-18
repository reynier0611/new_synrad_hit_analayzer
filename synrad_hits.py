# author: Rey Cruz Torres

import pandas as pd
import os
import ROOT
import numpy as np
import uproot
import matplotlib.pyplot as plt
import time

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

        self.df_photons = pd.read_csv(os.path.join(self.path_to_photons,'normalization_file.txt'))
        self.df_photons.columns = ['facet','subidx','NormFact']
        print(self.df_photons.info())
        print(self.df_photons.describe())

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
            x -= 1

            if x >= 1800000:
                continue
            
            photon = self.df_photons.iloc[x]
            current_facet = photon['facet']
            sub_index = photon['subidx']

            if photon['NormFact'] == 0.:
                continue

            integrated_so_far += 1./photon['NormFact']
            event.append([(int)(current_facet),sub_index])
        
        return event

    # --------------------------------------------
    def generate(self):
        multiplicity = []

        # Loop over and generate events the user requested
        for i in range(self.nevents):
            if i%50==0:
                print('Event',i,'out of',self.nevents)

            event = self.generate_an_event()
            multiplicity.append(len(event))
            
            for photon in event:
                self.load_hits(photon[0],photon[1])

        # -----------------
        # Multiplicity plot
        plt.figure(figsize=(10,8))
        plt.hist(multiplicity)
        plt.savefig('multiplicity.png',dpi=600)

        # -----------------
        # Hit plots
        for detector in self.detectors:
            self.plot_x_y_z_distributions(detector)

    # --------------------------------------------
    def load_hits(self,facet,idx):
        # We need to speed up this function. The code takes 91% of the time here.
        # We may want to try loading all these dataframes earlier in the code and keep them in memory, if they fit.
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
    t0 = time.time()
    print('Creating an instance of hits_from_synrad')
    path_to_photons = './'
    path_to_hits = 'geant_data/'
    nevents = 5
    hits = hits_from_synrad(nevents,100.e-09,path_to_photons,path_to_hits) # argument is integration window in sec
    hits.generate()
    print('Overall running time:',np.round((time.time()-t0)/60.,2),'min')