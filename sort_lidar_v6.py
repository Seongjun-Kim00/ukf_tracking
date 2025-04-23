# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import numpy as np
import random
import time
from math import *
from assoication_dets_to_trks_v6 import *
from parameter_ukf import *
from ukf_box_tracker_v4 import *
from track_management import *

'''
v006 : track_management 모듈화
'''


# ========================
# INPUT DATA
# ========================
# LIDAR_DETECTION_REL_POS_X = 0
# LIDAR_DETECTION_REL_POS_Y = 1
# LIDAR_DETECTION_WIDTH = 2
# LIDAR_DETECTION_HEIGHT = 3
# LIDAR_DETECTION_LENGTH = 4
# LIDAR_DETECTION_YAW = 5
# LIDAR_DETECTION_CLASS = 6
# LIDAR_DETECTION_SCORE = 7

# ========================
# OUTPUT DATA
# ========================
# LIDAR_TRACKING_REL_POS_X = 0
# LIDAR_TRACKING_REL_POS_Y = 1
# LIDAR_TRACKING_REL_VEL = 2
# LIDAR_TRACKING_REL_VEL_X = 3
# LIDAR_TRACKING_REL_VEL_Y = 4
# LIDAR_TRACKING_YAW = 8
# LIDAR_TRACKING_YAW_RATE = 9
# LIDAR_TRACKING_WIDTH = 10
# LIDAR_TRACKING_HEIGHT = 11
# LIDAR_TRACKING_LENGTH = 12
# LIDAR_TRACKING_LIFE_TIME = 13
# LIDAR_TRACKING_CLASS = 14
# LIDAR_TRACKING_SCORE = 15
# LIDAR_TRACKING_UPDATE_TIME = 16
# LIDAR_TRACKING_ID = 17

def sorting_ID(track_states):
    
    indices= np.argsort(track_states[:, LIDAR_TRACKING_ID])
    sorted_track = track_states[indices]

    return sorted_track


class Sort(object):
    def __init__(self, max_update_time=1, iou_threshold=0.3, init_track_id=np.zeros(32)):
        self.max_update_time = max_update_time*2      # Update rate is 10Hz, and Prediction rate is 20Hz
        self.iou_threshold   = iou_threshold          
        self.using_track_id  = init_track_id  
        self.trackers        = []                     # Tracker list

         # Init Covariance ( Gaussian Noise )
        self.P                  = np.array([[INIT_POS_NOISE,              0,              0,              0,                   0],
                                            [0             , INIT_POS_NOISE,              0,              0,                   0],
                                            [0             ,              0, INIT_VEL_NOISE,              0,                   0],
                                            [0             ,              0,              0, INIT_YAW_NOISE,                   0],
                                            [0             ,              0,              0,              0, INIT_YAW_RATE_NOISE]])      
        # Process : [rel_x, rel_y, vel, yaw, yaw_rate]
        self.process_noise      =  np.array([[PROCESS_POS_NOISE,                 0,                 0,                 0,                      0],
                                             [0                , PROCESS_POS_NOISE,                 0,                 0,                      0],
                                             [0                ,                 0, PROCESS_VEL_NOISE,                 0,                      0],
                                             [0                ,                 0,                 0, PROCESS_YAW_NOISE,                      0],
                                             [0                ,                 0,                 0,                 0, PROCESS_YAW_RATE_NOISE]])
        
        # Measure : [rel_x, rel_y, yaw]
        self.measurement_noise  = np.array([[MEASUREMENT_POS_NOISE,                     0,                     0],
                                            [0                    , MEASUREMENT_POS_NOISE,                     0],
                                            [0                    ,                     0, MEASUREMENT_YAW_NOISE]])
        

        # Unscented Kalman Filter(UKF)'s parameters
        self.kappa               = KAPPA
        self.alpha               = ALPHA
        self.beta                = BETA
        self.state_dim           = 5
        self.measurement_dim     = 3
        # Track ID : (32,)
        self.id_list             = init_track_id
        # Publish only matched trk
        self.valid_tracker_range = 0

    # Give Tracking ID : Only new trackers
    def give_track_id(self):
        # ID array : (32,)
        # If Tracking ID is ON, then the array[i] = 1
        new_track_id = 0
        if self.id_list.sum() == 0:
            new_track_id = 0
        else:
            new_track_id = np.where(self.id_list == 0)[0][0]

        self.id_list[new_track_id] = 1
        return new_track_id + 1

    # Init tracker, if the number of trackers in before frame is zero
    def init_tracker(self, dets):
        new_trackers = []
        new_trackers_state = np.zeros((len(dets), LIDAR_TRACKING_STATE_NUMBER))

        # All detections have trackers
        for d, dets in enumerate(dets):
            trk = np.zeros((LIDAR_TRACKING_STATE_NUMBER,1))
            trk[LIDAR_TRACKING_REL_POS_X] = dets[LIDAR_DETECTION_REL_POS_X]
            trk[LIDAR_TRACKING_REL_POS_Y] = dets[LIDAR_DETECTION_REL_POS_Y]
            trk[    LIDAR_TRACKING_WIDTH] = dets[    LIDAR_DETECTION_WIDTH]
            trk[   LIDAR_TRACKING_HEIGHT] = dets[   LIDAR_DETECTION_HEIGHT]
            trk[   LIDAR_TRACKING_LENGTH] = dets[   LIDAR_DETECTION_LENGTH]
            trk[      LIDAR_TRACKING_YAW] = dets[      LIDAR_DETECTION_YAW]
            trk[    LIDAR_TRACKING_CLASS] = dets[    LIDAR_DETECTION_CLASS]
            trk[    LIDAR_TRACKING_SCORE] = dets[    LIDAR_DETECTION_SCORE]
            trk[       LIDAR_TRACKING_ID] = self.give_track_id()

            new_tracker = UnscentedKalman_box_tracker(tracking = trk                        , P = self.P                        , state_dim = self.state_dim,
                                                      measurement_dim = self.measurement_dim, process_noise = self.process_noise, measurement_noise = self.measurement_noise,
                                                      kappa = self.kappa                , alpha = self.alpha                , beta = self.beta)
            # trackers_state vs trackers
            # trackers mean literally trackers and trackers_state means output data ( rel_x, rel_y, ... )
            new_trackers.append(new_tracker)
            new_trackers_state[d, :] = trk.squeeze()

        self.trackers = new_trackers

        return new_trackers_state

    # Predict Module ( not correction )
    def predict(self):
        # If the number of trackers in before frame is zero, then can't predict
        if len(self.trackers) == 0:
            new_trackers_state = np.empty((0, LIDAR_TRACKING_STATE_NUMBER))

            return new_trackers_state
        
        # Predict by UKF_BOX_Tracker
        Pred_trackers_state = np.zeros((self.valid_tracker_range, LIDAR_TRACKING_STATE_NUMBER))
        for t, trk in enumerate(self.trackers):
            trk.predict()
            if t < self.valid_tracker_range:
                Pred_trackers_state[t,:] = trk.getPredState()
            

        return Pred_trackers_state
    
    '''
    Update
    1. Predict
    2. Association
    3. Correct
    4. Management
    '''
    def update(self, dets):   
        # If the number of trackers in before frame is zero, All detections to Trackers
        if len(self.trackers) == 0:
            new_trackers_state = self.init_tracker(dets)   
            return new_trackers_state
        
        # If not detections, then only predict
        if len(dets) == 0:
            Pred_trackers_state = np.zeros((len(self.trackers), LIDAR_TRACKING_STATE_NUMBER))
            for t, trk in enumerate(self.trackers):

                trk.predict()
                Pred_trackers_state[t,:] = trk.getPredState()

            return

        # Predict    
        Pred_trackers_state = np.zeros((len(self.trackers), LIDAR_TRACKING_STATE_NUMBER))
        for t, trk in enumerate(self.trackers):
            trk.predict()
            Pred_trackers_state[t,:] = trk.getPredState()

        # Association
        matched_trk_indices, unmatched_trk_indices, matched_det_indices, unmatched_det_indices = associate_detections_to_trackers(dets, Pred_trackers_state, self.iou_threshold)


        new_trackers_state = np.zeros((len(unmatched_det_indices), LIDAR_TRACKING_STATE_NUMBER))
        new_trackers = []
            
        # Correct
        Update_trackers_state = np.zeros((len(matched_trk_indices), LIDAR_TRACKING_STATE_NUMBER))

        for t, matched_trk_idx in enumerate(matched_trk_indices):
            self.trackers[matched_trk_idx].correct(dets[matched_det_indices[t], :])

            Update_tracker_state       = self.trackers[matched_trk_idx].getFinalState()
            Update_trackers_state[t,:] = Update_tracker_state
            new_trackers.append(self.trackers[matched_trk_idx])

        trackers_state = np.vstack([Update_trackers_state, new_trackers_state])

        # Management 
        track_number               = len(new_trackers) + len(unmatched_det_indices)
        new_trackers, self.id_list = track_management(unmatched_trk_indices, Pred_trackers_state, self.trackers,
                                                      new_trackers         , track_number       , self.id_list ,
                                                      self.max_update_time)

        # New Trackers by Unmatched Detection
        for ud, unmatched_det_idx in enumerate(unmatched_det_indices):
            trk = np.zeros(LIDAR_TRACKING_STATE_NUMBER)
            trk[LIDAR_TRACKING_REL_POS_X] = dets[ unmatched_det_idx, LIDAR_DETECTION_REL_POS_X]
            trk[LIDAR_TRACKING_REL_POS_Y] = dets[ unmatched_det_idx, LIDAR_DETECTION_REL_POS_Y]
            trk[    LIDAR_TRACKING_WIDTH] = dets[ unmatched_det_idx,     LIDAR_DETECTION_WIDTH]
            trk[   LIDAR_TRACKING_HEIGHT] = dets[ unmatched_det_idx,    LIDAR_DETECTION_HEIGHT]
            trk[   LIDAR_TRACKING_LENGTH] = dets[ unmatched_det_idx,    LIDAR_DETECTION_LENGTH]
            trk[      LIDAR_TRACKING_YAW] = dets[ unmatched_det_idx,       LIDAR_DETECTION_YAW]
            trk[    LIDAR_TRACKING_CLASS] = dets[ unmatched_det_idx,     LIDAR_DETECTION_CLASS]
            trk[       LIDAR_TRACKING_ID] = self.give_track_id()

            new_tracker = UnscentedKalman_box_tracker(tracking = trk                        , P = self.P                        , state_dim = self.state_dim,
                                                      measurement_dim = self.measurement_dim, process_noise = self.process_noise, measurement_noise = self.measurement_noise,
                                                      kappa = self.kappa                  , alpha = self.alpha                , beta = self.beta)
            new_trackers.append(new_tracker)
            new_trackers_state[ud, :] = trk


        self.trackers            = new_trackers
        self.valid_tracker_range = len(matched_trk_indices) + len(unmatched_det_indices)


        if trackers_state.shape[0] > 0:
            sorted_state = sorting_ID(trackers_state)
            return sorted_state
        else:
            return np.empty((0, LIDAR_TRACKING_STATE_NUMBER))
