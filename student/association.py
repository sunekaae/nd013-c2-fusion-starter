# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        
        # the following only works for at most one track and one measurement
        N = len(track_list) # N is number of tracks
        M = len(meas_list) # M is number of measurements
        self.association_matrix = np.inf*np.ones((N,M))  # np.matrix([]) # reset matrix
        self.unassigned_tracks = [] # reset lists
        self.unassigned_meas = []
        
        for i in range(len(meas_list)):
            self.unassigned_meas.append(i)
        for i in range(len(track_list)):
            self.unassigned_tracks.append(i)
        if len(meas_list) > 0 and len(track_list) > 0: 
            for i in range(N):
                track = track_list[i]
                for j in range(M):
                    measurement = meas_list[j]
                    distance = self.MHD(track, measurement, KF)
                    self.association_matrix[i,j] = distance
        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # the following only works for at most one track and one measurement
        if self.association_matrix.size == 0:
            return np.nan, np.nan
        
        lowest_distance = np.inf
        i = 0
        j = 0
        for i in range(np.shape(self.association_matrix)[0]):
            for j in range(np.shape(self.association_matrix)[1]):
                distance = self.association_matrix[i,j]
                if distance < lowest_distance:
                    lowest_distance = distance
        update_track = self.unassigned_tracks[i]
        update_meas = self.unassigned_meas[j]
        if (update_track==np.inf and update_meas==np.inf):
            return np.nan, np.nan 
        
        # remove from list
        self.unassigned_tracks.pop(i) 
        self.unassigned_meas.pop(j)
        # self.association_matrix = np.matrix([])
        self.association_matrix = np.delete(self.association_matrix, i, axis=0)
        self.association_matrix = np.delete(self.association_matrix, j, axis=1)
            
        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # Step 3: return True if measurement lies inside gate, otherwise False
        ############
        # check if measurement lies inside gate
        dof = 3 if sensor.name == 'lidar' else 2        
        if MHD < chi2.ppf(params.gating_threshold, df=dof):
            return True
        else:
            return False
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # Step 3: calculate and return Mahalanobis distance
        ############
#        H = np.matrix([[1, 0, 0, 0, 0, 0],
#                       [0, 1, 0, 0, 0, 0]]) 
        sensor = meas.sensor.name.lower()
        n = track.x.shape[0]

        if sensor == "lidar":  
            assert meas.z.shape == (3,1), f"lidar z must be (3,1), got {meas.z.shape}"
            assert meas.R.shape == (3,3), f"lidar R must be (3,3), got {meas.R.shape}"
            
            H = np.zeros((3, n))  # (3,6)
            H[0,0] = 1.0  # px
            H[1,1] = 1.0  # py
            H[2,2] = 1.0  # pz
            z_pred = H @ track.x
#            
            gamma = meas.z - z_pred
            S = H*track.P*H.transpose() + meas.R
            MHD = gamma.transpose()*np.linalg.inv(S)*gamma # Mahalanobis distance formula
            MHD = MHD[0,0]
            if self.gating(MHD, meas.sensor):
                return MHD
            else:
                return np.inf
#
        elif sensor == "camera":
            assert meas.z.shape == (2,1), f"camera z must be (2,1), got {meas.z.shape}"
            assert meas.R.shape == (2,2), f"camera R must be (2,2), got {meas.R.shape}"
            H = meas.sensor.get_H(track.x)
            z_pred = meas.sensor.get_hx(track.x)
#            
            gamma = meas.z - z_pred
            S = H*track.P*H.transpose() + meas.R
            MHD = gamma.transpose()*np.linalg.inv(S)*gamma # Mahalanobis distance formula
            MHD = MHD[0,0]
            if self.gating(MHD, meas.sensor):
                return MHD
            else:
                return np.inf
#
            assert H.shape == (2, n), f"camera H must be (2,{n}), got {H.shape}"
        else:
            raise ValueError(f"Unknown sensor type: {meas.sensor.name}")


        
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)