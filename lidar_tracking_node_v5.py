#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
V5 : visualization
'''
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
from ros_topic_tool import clear_markers, create_trackingBEV, create_tracking_topic
from numpy import pi

# ROI Filter
def inRange_ROI(x,y):
    is_inRange = False

    if (REAR_RANGE <= x <= FRONT_RANGE) and ( RIGHT_RANGE <= y <= LEFT_RANGE):
        is_inRange = True
        

        if (abs(x) + EGO_X_OFFSET <= EGO_X_RANGE) and (abs(y) <= EGO_Y_RANGE):
            is_inRange = False

    return is_inRange

def normalize_heading(heading):
    while not (-pi/2 < heading <= pi/2):
        if heading > pi/2:
            heading -= pi
        elif heading <= -pi/2:
            heading += pi

    return heading




class Lidar_Tracking:
    def __init__(self):
        rospy.init_node("tracking_node")

        sub_detecion_topic_name = '/detection/lidar_objects'
        pub_tracking_topic_name = '/tracking/lidar_tracking_results'



        rospy.Subscriber(sub_detecion_topic_name, DetectedObjectArray, self.lidar_objects_callback)

        self.tracking_results_pub = rospy.Publisher(pub_tracking_topic_name, TrackingArray, queue_size=10)
        self.trackMarkers_pub     = rospy.Publisher('/trackBEV', MarkerArray, queue_size=1)
        self.textMarkers_pub      = rospy.Publisher('/trackText', MarkerArray, queue_size=1)

        self.max_age              = MAX_UPDATE_TIME            # UDPATE max AGE
        self.iou_threshold        = IOU_THRESHOLD

        self.Lidar_Detection      = None
        self.update_trigger       = False        # Lidar update rate is 10 Hz, and Tracking update rate is 20 Hz. Then Kalman update rate is 10Hz by Lidar det. 
        
        frame_Rate                = FRAME_RATE # Hz        
        rate                      = rospy.Rate(frame_Rate)


        self.mot_tracker = Sort(max_update_time = self.max_age,
                                iou_threshold = self.iou_threshold,
                                )
        
        # TRACKING_ALL_CLASS means that tracking not only cars, but Pedestrian and Cyclist
        if TRACKING_ALL_CLASS == ON:
            self.mot_people_tracker = Sort(max_update_time = self.max_age,
                                           iou_threshold = self.iou_threshold,
                                           )

        while not rospy.is_shutdown():
            cycle_time = 0
            trackers   = np.empty((0, LIDAR_TRACKING_STATE_NUMBER))

            if self.Lidar_Detection is not None:

                start_time = time.time()
                
                if self.update_trigger:
                    # Only prediction
                    trackers            = self.mot_tracker.update(self.Lidar_Detection)
                    self.update_trigger = False

                else:
                    # Update ( Prediction & Correction )
                    trackers            = self.mot_tracker.predict()

                cycle_time = time.time() - start_time

            if TRACKING_ALL_CLASS == ON:
                if self.Lidar_Detection_PEOPLE is not None:
                    start_time = time.time()
                    
                    if self.update_trigger:
                        trackers_people     = self.mot_people_tracker.update(self.Lidar_Detection_PEOPLE)
                        self.update_trigger = False
                    else:
                        trackers_people     = self.mot_tracker.predict()


                    cycle_time += time.time() - start_time
                    trackers    = np.vstack([trackers, trackers_people])

            print("processing time : {:.2f}[ms]".format(cycle_time*1000))
            self.publishing_trackingData(trackers)

            rate.sleep()


        
    def lidar_objects_callback(self,msg):
        Lidar_Detection        = []
        Lidar_Detection_PEOPLE = []
        

        for object in msg.objects:
            det = self.extractDetInfo(object)

            is_inRange = inRange_ROI(det[LIDAR_DETECTION_REL_POS_X], det[LIDAR_DETECTION_REL_POS_Y]) # x, y

            if is_inRange:
                if det[LIDAR_DETECTION_CLASS] == VEHICLE_CLASS:

                    Lidar_Detection.append(det)
                else:
                    if TRACKING_ALL_CLASS == ON:
                        self.Lidar_Detection_PEOPLE.append(det)

        self.Lidar_Detection        = np.array(Lidar_Detection)
        self.Lidar_Detection_PEOPLE = np.array(Lidar_Detection_PEOPLE)
        self.update_trigger         = True


    def publishing_trackingData(self, trackers):
        print("-------------------------------------------------------------------------")
        
        # Visualization init
        self.trackMarkers_pub.publish(clear_markers())
        self.textMarkers_pub.publish(clear_markers())
        
        track_markers = MarkerArray()
        text_markers  = MarkerArray()
        track_array   = TrackingArray()
        
        # Track_array is empty
        if trackers is None:
            self.trackMarkers_pub.publish(track_markers)
            self.textMarkers_pub.publish(text_markers)
            self.tracking_results_pub.publish(track_array)
            return

        # Publish ROS TOPIC
        for track in trackers:
            
            track_marker, text_marker = create_trackingBEV(track)
            tracking_result           = create_tracking_topic(track)

            track_markers.markers.append(track_marker)
            text_markers.markers.append(text_marker)
            track_array.tracks.append(tracking_result)
            

        self.trackMarkers_pub.publish(track_markers)
        self.textMarkers_pub.publish(text_markers)
        self.tracking_results_pub.publish(track_array)

    # Extract detection information (ROS.MSG -> NUMPY.ARRAY) 
    def extractDetInfo(self, object):
        score = object.score

        x = object.pose.position.x
        y = object.pose.position.y
        z = object.pose.position.z

        
        l = object.dimensions.x
        w = object.dimensions.y
        h = object.dimensions.z

        yaw = object.angle # -PI/2 - PI/2 [rad]
        yaw = normalize_heading(yaw)



        classification = object.indicator_state

        return [x, y, z, yaw, w, h, l, classification, score]    

if __name__ == "__main__":
    
    try:
        Tracking_node = Lidar_Tracking()
        print("Tracking Start!!!!")    
        
    except rospy.ROSInterruptException:
        pass

