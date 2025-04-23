
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import numpy as np
import random
import glob
import time
import argparse
from math import *
from parameter_ukf import *
from calculateBevIOU import *

'''
======================================
Input
======================================
detections : (n, LIDAR_DETECTION_STATE_NUMBER)
trackers   : (m,  LIDAR_TRACKING_STATE_NUMBER)

======================================
Output
======================================
matched_tracks     : (i, LIDAR_TRACKING_STATE_NUMBER)
unmatched_tracks   : (j,  LIDAR_TRACKING_STATE_NUMBER)

matched_dets       : (i, LIDAR_DETECTION_STATE_NUMBER)
unmatched_dets     : (k,  LIDAR_DETECTION_STATE_NUMBER)
i + j = n 
i + k = m
'''

def euclid_filtering(detections, trackers):
    dist_map            = np.zeros((len(detections), len(trackers)))
    det_association_map = np.zeros((len(detections), len(trackers)))
    trk_association_map = np.zeros((len(detections), len(trackers)))
    association_map     = np.zeros((len(detections), len(trackers)))

    dets = detections[:,[LIDAR_DETECTION_REL_POS_X, LIDAR_DETECTION_REL_POS_Y]]
    trks =   trackers[:,[ LIDAR_TRACKING_REL_POS_X,  LIDAR_TRACKING_REL_POS_Y]]

    # calculate distanceMap
    for d, det in enumerate(dets):
        diff_x          = det[0] - trks[:,0].T
        diff_y          = det[1] - trks[:,1].T
        dist_map[d,:]   = np.sqrt(diff_x**2 + diff_y**2)

    # find nearest tracker in detectionMap
    for i, dist in enumerate(dist_map):
        min_idx = np.argmin(dist)
        det_association_map[i, min_idx] = 1

    # find nearest detection in trackerMap
    for i, dist in enumerate(dist_map.T):
        min_idx = np.argmin(dist)
        trk_association_map[min_idx, i] = 1

    # find association by (nearest_detection == nearest_tracker)
    association_map = det_association_map + trk_association_map
    rows, cols      = np.where(association_map == 2)

    a = [[row, col] for row, col in zip(rows, cols)]

    return np.array(a)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):

    # If the number of trackes is zero, then return all detections
    if (len(trackers) == 0):
        matched_trk_indices     = np.empty((0, LIDAR_DETECTION_STATE_NUMBER), dtype=int)
        unmatched_trk_indices   = np.empty((0, LIDAR_DETECTION_STATE_NUMBER), dtype=int)
        matched_det_indices     = np.empty((0, LIDAR_DETECTION_STATE_NUMBER), dtype=int)
        unmatched_det_indices   = np.array(detections)
        
        return matched_trk_indices, unmatched_trk_indices, matched_det_indices, unmatched_det_indices
    
    matched_trk_indices   = []
    unmatched_trk_indices = np.arange(len(trackers))
    matched_det_indices   = []
    unmatched_det_indices = np.arange(len(detections))

    # Association 1 : Distance filtering
    a = euclid_filtering(detections, trackers)

    # Association 2 : IOU filtering
    for row, col in a:
    
        det = detections[row,:]
        trk =   trackers[col,:]

        # If tracking Life time is zero, velocity is zero. So, use only Association 1 ( Distance filtering )
        if trk[LIDAR_TRACKING_LIFE_TIME]     == 0:
            matched_det_indices.append(row)
            matched_trk_indices.append(col)

            unmatched_det_indices[row] = -1
            unmatched_trk_indices[col] = -1
        
        # Filter 2D IOU by BEV
        else:
            
            iou_2d = get_iou_bbox(trk, det)
            if iou_2d >= iou_threshold:
                
                matched_det_indices.append(row)
                matched_trk_indices.append(col)

                unmatched_det_indices[row] = -1
                unmatched_trk_indices[col] = -1
    
    del_idx               = np.where(unmatched_det_indices == -1)
    unmatched_det_indices = np.delete(unmatched_det_indices, del_idx)

    del_idx               = np.where(unmatched_trk_indices == -1)
    unmatched_trk_indices = np.delete(unmatched_trk_indices, del_idx)


    return np.array(matched_trk_indices), np.array(unmatched_trk_indices), np.array(matched_det_indices), np.array(unmatched_det_indices)
