#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import time 
from std_msgs.msg import Header
from autoware_msgs.msg import DetectedObjectArray, DetectedObject
from tracking_msg.msg import Tracking, TrackingArray
from sort_lidar_v6 import *
from math import *
from calculateBevIOU import *
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from parameter_ukf import *
from ros_topic_tool import clear_markers, create_trackingBEV


def inRange_ROI(x,y):
    is_inRange = False

    if (REAR_RANGE <= x <= FRONT_RANGE) and ( RIGHT_RANGE <= y <= LEFT_RANGE):
        is_inRange = True

    return True
'''
1. Thresholding by max_update_time & range
2. Limit total tracking number by priority ( priority : 1. Update_time, 2. Distance, 3. Life_time. Des)
'''
def track_management(unmatched_trk_indices, Pred_trackers_state, trackers, new_trackers, number_trackers, id_list, max_update_time):
    # Thresholding by max_update_time & range
    update_unmatched_trk_indices = []
    update_unmatched_trackers    = []
    new_idx_list                 = []

    for unmatched_trk_idx in unmatched_trk_indices:
        if Pred_trackers_state[unmatched_trk_idx, LIDAR_TRACKING_UPDATE_TIME] <= max_update_time:     
            rel_x = Pred_trackers_state[unmatched_trk_idx, LIDAR_TRACKING_REL_POS_X]
            rel_y = Pred_trackers_state[unmatched_trk_idx, LIDAR_TRACKING_REL_POS_Y]
            
            if inRange_ROI(rel_x, rel_y):

                update_unmatched_trackers.append(trackers[unmatched_trk_idx])
                update_unmatched_trk_indices.append(unmatched_trk_idx)

    update_unmatched_trk_indices = np.array(update_unmatched_trk_indices)

    # Limit total tracking number
    number_trackers = number_trackers + len(update_unmatched_trackers)
    number_trash    = number_trackers - LIDAR_TRACKING_TRACK_NUMBER

    # Total trackers > LIDAR_TRACKING_TRACK_NUMBER
    if number_trash > 0:
        unmatched_trackers = np.zeros((len(update_unmatched_trk_indices), 3))

        for i, unmatched_trk_idx in enumerate(update_unmatched_trk_indices):
            unmatched_trackers[i, [LIDAR_TRACKING_SORTING_UPDATE_TIME, LIDAR_TRACKING_SORTING_LIFE_TIME]] =  \
                Pred_trackers_state[unmatched_trk_idx, [LIDAR_TRACKING_UPDATE_TIME,LIDAR_TRACKING_LIFE_TIME]]
            
            rel_x = Pred_trackers_state[unmatched_trk_idx, LIDAR_TRACKING_REL_POS_X]
            rel_y = Pred_trackers_state[unmatched_trk_idx, LIDAR_TRACKING_REL_POS_Y]

            unmatched_trackers[i, LIDAR_TRACKING_SORTING_DISTANCE] = np.sqrt(rel_x** + rel_y**2)
        
        # Sorting priority : 1. Update_time, 2. Distance, 3. Life_time. Des
        sort_priority = (unmatched_trackers[:,   LIDAR_TRACKING_SORTING_LIFE_TIME],
                         unmatched_trackers[:,    LIDAR_TRACKING_SORTING_DISTANCE],
                         unmatched_trackers[:, LIDAR_TRACKING_SORTING_UPDATE_TIME])
        
        indices            = np.lexsort(sort_priority)
        sorted_trk_indices = update_unmatched_trk_indices[indices]

        for i, idx in enumerate(sorted_trk_indices):
            if i < number_trash:

                continue
            else:
                new_trackers.append(trackers[idx])
                new_idx_list.append(idx)

    else:
        for t, update_unmatched_trk_idx in enumerate(update_unmatched_trk_indices):
            new_trackers.append(update_unmatched_trackers[t])
            new_idx_list.append(update_unmatched_trk_idx)

    # Erase filtered trackers
    for unmatched_trk_idx in unmatched_trk_indices:
        if unmatched_trk_idx in new_idx_list:
            continue
        else:
            del_track_id              = int(Pred_trackers_state[unmatched_trk_idx, LIDAR_TRACKING_ID])
            id_list[del_track_id - 1] = 0
            

    return new_trackers, id_list