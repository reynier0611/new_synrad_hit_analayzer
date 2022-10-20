# author: Rey Cruz Torres

from xml.etree.ElementTree import C14NWriterTarget
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
    def __init__(self,nevents,int_window=100.e-09,gold_thick=5.0,path_to_photons=None,path_to_hits=None,preprocess_g4_hits=True):
        if path_to_photons==None or path_to_hits==None:
            print("need both 'path_to_photons' and 'path_to_hits' arguments")
            exit()

        # -----------------
        # Some parameters that may need modification down the line
        self.number_seeded_geant_files = 25
        self.facets = ['25098','25113','25119','25099','25114','25120','25100',
        '25115','25130','25101','25116','25131','25111','25117','25132','25112',
        '25118','25133']
        # -----------------

        self.nevents = nevents
        self.int_window = int_window
        self.gold_thick = gold_thick
        self.path_to_photons = path_to_photons
        self.total_time = (float)(self.nevents)*self.int_window
        self.path_to_hits = os.path.join(path_to_hits,'gold_coating_{}um'.format((int)(self.gold_thick)))
        self.dict_hdf5_fname = 'preprocessed_G4_hits/hits_gold_{}um_seeds_{}.h5'.format((int)(self.gold_thick),self.number_seeded_geant_files)
        self.output_plots = 'output_plots'
        self.common_outname = 'results_events_{}_int_window_{}_sec_gold_{}um_{}_seeded_G4_hits_'.format(nevents,
        int_window,(int)(gold_thick),self.number_seeded_geant_files)

        print('')
        print('requested number of events:',self.nevents)
        print('time integration window:',self.int_window,'sec')
        print('beampipe gold coating',self.gold_thick,'um')
        print('path to synrad photons:',self.path_to_photons)
        print('')

        self.df_photons = pd.read_csv(os.path.join(self.path_to_photons,'normalization_file.txt'))
        self.df_photons.columns = ['facet','subidx','NormFact']

        n_entries = len(self.df_photons)
        self.h1_df = ROOT.TH1D('h1_df',';entry;W [#gamma/sec]',n_entries,0,n_entries)
        for i in range(n_entries):
            self.h1_df.SetBinContent(i+1,self.df_photons['NormFact'].iloc[i])

        '''
        self.detectors = ["DRICHHits","EcalEndcapNHits","EcalEndcapPHits","VertexBarrelHits","SiBarrelHits","MPGDBarrelHits",
        "TrackerEndcapHits","MRICHHits","ZDC_PbSi_Hits","ZDC_WSi_Hits","ZDCHcalHits","TaggerTracker1Hits",
        "ForwardRomanPotHits","EcalBarrelHits","HcalEndcapPHits","HcalEndcapNHits","HcalEndcapPInsertHits",
        "HcalBarrelHits","B0PreshowerHits","B0TrackerHits","ForwardOffMTrackerHits","ZDC_SiliconPix_Hits",
        "ZDCEcalHits","TaggerCalorimeter1Hits","TaggerTracker2Hits","TaggerCalorimeter2Hits"]
        '''

        # Reduced list
        self.detectors = ["EcalEndcapNHits","EcalEndcapPHits","VertexBarrelHits","SiBarrelHits","MPGDBarrelHits",
        "TrackerEndcapHits","TaggerTracker1Hits","EcalBarrelHits","HcalEndcapPHits","HcalEndcapNHits","HcalBarrelHits",
        "TaggerCalorimeter1Hits","TaggerTracker2Hits","TaggerCalorimeter2Hits"]

        self.hits = {}
        for detector in self.detectors:
            self.hits[detector] = {}
            for var in ['x','y','z']:
                self.hits[detector][var] = []

        if preprocess_g4_hits:
            print('Will look for geant files in:',self.path_to_hits)
            self.preprocess_hits()
        else:
            self.hit_container = h5todict(self.dict_hdf5_fname)

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
            if i%1000==0:
                print('Event',i,'out of',self.nevents)

            event = self.generate_an_event()
            multiplicity.append(len(event))
            
            for photon in event:
                self.locate_hits((str)(photon[0]),photon[1])

        print('')
        print('Making plots')
        # -----------------
        # Multiplicity plot
        plt.figure(figsize=(10,8))
        plt.hist(multiplicity)
        fname = self.common_outname + 'multiplicity.png'
        plt.savefig(os.path.join(self.output_plots,fname),dpi=600)
        plt.close()

        # -----------------
        # Hit plots
        for detector in self.detectors:
            self.plot_x_y_z_distributions(detector)

        # -----------------
        # Summary plot
        plt.figure(figsize=(15,8))
        labels = []
        num_hits = []
        for detector in self.detectors:
            labels.append(detector)
            num_hits.append((float)(len(self.hits[detector]['x']))/self.total_time)
        plt.bar(labels,num_hits)
        plt.xticks(rotation = 90)
        plt.yscale('log')
        plt.tight_layout()
        plt.ylabel('Hz')
        fname = self.common_outname + 'summary.png'
        plt.savefig(os.path.join(self.output_plots,fname),dpi=600)
        plt.close()

    # --------------------------------------------
    def preprocess_hits(self):
        self.hit_container = {}
        for facet in self.facets:
            self.hit_container[facet] = {}
            for seed in range(1,self.number_seeded_geant_files+1):
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

        print(f'Writing results to {self.dict_hdf5_fname}')
        dicttoh5(self.hit_container,self.dict_hdf5_fname, overwrite_data=True)

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
        plt.scatter(x,y,s=10,alpha=0.3)
        plt.title(detector+' integration window = {} sec'.format(self.int_window))
        plt.xlabel('$x$ [mm]')
        plt.ylabel('$y$ [mm]')

        self.circles(detector)

        plt.subplot(1,2,2)
        plt.scatter(z,x,s=10,alpha=0.3)
        plt.xlabel('$z$ [mm]')
        plt.ylabel('$x$ [mm]')

        self.lines(detector)

        plt.tight_layout()
        fname = self.common_outname + 'results_xyz_hits_' + detector + '.png'
        plt.savefig(os.path.join(self.output_plots,fname),dpi=600)
        plt.close()

    # --------------------------------------------
    def circles(self,detector):
        if detector == 'VertexBarrelHits':
            c1_x, c1_y = self.circle(36.); self.draw_circle(c1_x,c1_y)
            c2_x, c2_y = self.circle(48.); self.draw_circle(c2_x,c2_y)
            c3_x, c3_y = self.circle(120); self.draw_circle(c3_x,c3_y)
        elif detector == 'SiBarrelHits':
            c1_x, c1_y = self.circle(239);  self.draw_circle(c1_x,c1_y)
            c2_x, c2_y = self.circle(430);  self.draw_circle(c2_x,c2_y)
            c3_x, c3_y = self.circle(270);  self.draw_circle(c3_x,c3_y)
            c4_x, c4_y = self.circle(420);  self.draw_circle(c4_x,c4_y)
            
    # --------------------------------------------
    def circle(self,radius):
        phi = np.linspace(0,2.*np.pi,100)
        return radius*np.cos(phi), radius*np.sin(phi)

    # --------------------------------------------
    def draw_circle(self,x,y):
        plt.plot(x,y,color='black',linestyle='--',alpha=0.3)

    # --------------------------------------------
    def lines(self,detector):
        if detector == 'VertexBarrelHits':
            plt.plot([-135,135],[36,36],color='black',linestyle='--',alpha=0.3)
            plt.plot([-135,135],[48,48],color='black',linestyle='--',alpha=0.3)
            plt.plot([-135,135],[120,120],color='black',linestyle='--',alpha=0.3)
            plt.plot([-135,135],[-36,-36],color='black',linestyle='--',alpha=0.3)
            plt.plot([-135,135],[-48,-48],color='black',linestyle='--',alpha=0.3)
            plt.plot([-135,135],[-120,-120],color='black',linestyle='--',alpha=0.3)

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
    parser.add_argument('--nevents', action='store', type=int, default=0,
                        help='number of events')
    parser.add_argument('--int_window', action='store', type=float, default=100.e-09,
                        help='time integration window in seconds. Default = 100.e-09')
    parser.add_argument('--gold_thick', action='store', type=float, default=5.0,
                        help='thickness of beampipe gold coating in um')
    args = parser.parse_args()

    print('******************************************************')
    print('Creating an instance of hits_from_synrad')
    print('******************************************************')
    path_to_photons = './'
    path_to_hits = 'geant_data/'
    hits = hits_from_synrad(args.nevents,args.int_window,args.gold_thick,path_to_photons,path_to_hits,args.process_g4_hits)
    if args.analyze:
        hits.generate()

    print()
    print('Overall running time:',np.round((time.time()-t0)/60.,2),'min')

    '''
    Notes:
    -  10,000 events ->  2.44 min
    - 100,000 events -> 15.56 min
    '''