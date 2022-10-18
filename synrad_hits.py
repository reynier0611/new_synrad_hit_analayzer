# author: Rey Cruz Torres

import pandas as pd
import os
import ROOT
import numpy as np
import uproot
import matplotlib.pyplot as plt
import time
from silx.io.dictdump import dicttoh5, h5todict
import argparse

# ------------------------------------------------------------------------------------------------
# Class to analyze synrad hits
# ------------------------------------------------------------------------------------------------
class hits_from_synrad:
    def __init__(self,nevents,int_window=100.e-09,path_to_photons=None,path_to_hits=None,preprocess_g4_hits=True):
        if path_to_photons==None or path_to_hits==None:
            print("need both 'path_to_photons' and 'path_to_hits' arguments")
            exit()

        self.nevents = nevents
        self.int_window = int_window
        self.path_to_photons = path_to_photons
        self.path_to_hits = path_to_hits

        self.number_seeded_geant_files = 25
        self.facets = ['25098','25113','25119','25099','25114','25120','25100',
        '25115','25130','25101','25116','25131','25111','25117','25132','25112',
        '25118','25133']

        print('')
        print('requested number of events:',self.nevents)
        print('time integration window:',self.int_window)
        print('path to synrad photons:',self.path_to_photons)
        print('path to geant hits:',self.path_to_hits)
        print('')

        self.df_photons = pd.read_csv(os.path.join(self.path_to_photons,'normalization_file.txt'))
        self.df_photons.columns = ['facet','subidx','NormFact']

        n_entries = len(self.df_photons)
        self.h1_df = ROOT.TH1D('h1_df',';entry;W [#gamma/sec]',n_entries,0,n_entries)
        for i in range(n_entries):
            self.h1_df.SetBinContent(i+1,self.df_photons['NormFact'].iloc[i])

        self.detectors =["VertexBarrelHits","SiBarrelHits","TaggerTracker1Hits"]
        self.detectors = ["DRICHHits","EcalEndcapNHits","EcalEndcapPHits","VertexBarrelHits","SiBarrelHits","MPGDBarrelHits",
        "TrackerEndcapHits","MRICHHits","ZDC_PbSi_Hits","ZDC_WSi_Hits","ZDCHcalHits","TaggerTracker1Hits",
        "ForwardRomanPotHits","EcalBarrelHits","HcalEndcapPHits","HcalEndcapNHits","HcalEndcapPInsertHits",
        "HcalBarrelHits","B0PreshowerHits","B0TrackerHits","ForwardOffMTrackerHits","ZDC_SiliconPix_Hits",
        "ZDCEcalHits","TaggerCalorimeter1Hits","TaggerTracker2Hits","TaggerCalorimeter2Hits"]

        self.hits = {}
        for detector in self.detectors:
            self.hits[detector] = {}
            for var in ['x','y','z']:
                self.hits[detector][var] = []

        if preprocess_g4_hits:
            self.preprocess_hits()
        else:
            self.load_hits()

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

        print('Beginning loop over events')

        # Loop over and generate events the user requested
        for i in range(self.nevents):
            if i%100==0:
                print('Event',i,'out of',self.nevents)

            event = self.generate_an_event()
            multiplicity.append(len(event))
            
            for photon in event:
                self.locate_hits((str)(photon[0]),photon[1])

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
    def preprocess_hits(self):
        self.hit_container = {}
        for facet in self.facets:
            self.hit_container[facet] = {}
            for seed in range(1,self.number_seeded_geant_files):
                self.hit_container[facet][(str)(seed)] = {}

                fname = os.path.join(self.path_to_hits,'geant_out_{}_seed_{}.edm4hep.root'.format(facet,seed))
                F = uproot.open(fname)

                for detector in self.detectors:
                    self.hit_container[facet][(str)(seed)][detector] = {}
                    try:
                        subdf_x = F['events/'+detector+'.position.x'].array(library="pd")
                        subdf_y = F['events/'+detector+'.position.y'].array(library="pd")
                        subdf_z = F['events/'+detector+'.position.z'].array(library="pd")

                        idxs = []
                        for i in range(len(subdf_x.index)):
                            if i == 0:
                                idxs.append(subdf_x.index[i][0])
                            elif subdf_x.index[i][0] != subdf_x.index[i-1][0]:
                                idxs.append(subdf_x.index[i][0])

                        for idx in idxs:
                            self.hit_container[facet][(str)(seed)][detector][(str)(idx)] = []
                            for sh in range(len(subdf_x[idx])):
                                self.hit_container[facet][(str)(seed)][detector][(str)(idx)].append([sh,subdf_x[idx][sh],subdf_y[idx][sh],subdf_z[idx][sh]])
                    except:
                        pass

        filename = 'preprocessed_G4_hits_seeds_{}.h5'.format(self.number_seeded_geant_files)
        print(f'Writing results to {filename}')
        dicttoh5(self.hit_container,filename, overwrite_data=True)

    # --------------------------------------------
    def load_hits(self):
        '''
        Load preprocessed dictionary with Geant hits including seed information.
        '''
        filename = 'preprocessed_G4_hits_seeds_{}.h5'.format(self.number_seeded_geant_files)
        self.hit_container = h5todict(filename)

    # --------------------------------------------
    def locate_hits(self,facet,idx):
        '''
        given a facet number and photon index, generate a random integer in [1,number of seeded geant files] and
        determine if there were hits for the given detectors
        '''
        seed = np.random.randint(1,self.number_seeded_geant_files)

        for detector in self.detectors:
            try:
                elm = self.hit_container[facet][(str)(seed)][detector][(str)((int)(idx))]
                for i in range(len(elm)):
                    self.hits[detector]['x'].append(elm[i][1])
                    self.hits[detector]['y'].append(elm[i][2])
                    self.hits[detector]['z'].append(elm[i][3])

            except:
                pass

    # --------------------------------------------
    def plot_x_y_z_distributions(self,detector):
        x = self.hits[detector]['x']
        y = self.hits[detector]['y']
        z = self.hits[detector]['z']

        fig = plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.scatter(x,y)
        plt.subplot(1,2,2)
        plt.scatter(z,x)

        plt.savefig('output_plots/results_xyz_hits_'+detector+'.png',dpi=600)
        del fig
        plt.close()

# ------------------------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    t0 = time.time()

    parser = argparse.ArgumentParser(description='Analyzing hits from synchrotron radiation')
    parser.add_argument('--process_g4_hits', 
                        help='process geant hits and save dictionary to hdf5', 
                        action='store_true', default=False)
    parser.add_argument('--analyze', 
                        help='load hits from dictionary and analyze', 
                        action='store_true', default=False)
    args = parser.parse_args()

    print('Creating an instance of hits_from_synrad')
    path_to_photons = './'
    path_to_hits = 'geant_data/'
    nevents = 1000
    hits = hits_from_synrad(nevents,100.e-09,path_to_photons,path_to_hits,args.process_g4_hits) # argument is integration window in sec
    if args.analyze:
        hits.generate()

    print()
    print('Overall running time:',np.round((time.time()-t0)/60.,2),'min')