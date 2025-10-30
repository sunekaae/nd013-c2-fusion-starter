# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.q=0.1 # process noise variable for Kalman filter Q

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        delta_t = params.dt
        F = np.array([
            [1, 0, 0, delta_t, 0, 0],
            [0, 1, 0, 0, delta_t, 0],
            [0, 0, 1, 0, 0, delta_t],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        return F
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        dt = params.dt
        q4 = (dt**4)*self.q/4
        q3 = (dt**3)*self.q/2
        q2 = dt**2 * self.q
        # where q = \sigma_a^2.
        Q = np.array([
            [q4, 0, 0, q3, 0, 0],
            [0, q4, 0, 0, q3, 0],
            [0, 0, q4, 0, 0, q3],
            [q3, 0, 0, q2, 0, 0],
            [0, q3, 0, 0, q2, 0],
            [0, 0, q3, 0, 0, q2]
        ])
        return Q
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        track.set_x(self.F() * track.x)
        track.set_P(self.F() * track.P * np.transpose(self.F()) + self.Q())
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############

        return 0
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        return 0
        
        ############
        # END student code
        ############ 