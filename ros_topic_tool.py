import rospy
import numpy as np
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from tracking_msg.msg import Tracking, TrackingArray
from geometry_msgs.msg import Point
from parameter_ukf import *
from calculateBevIOU import *

def clear_markers():
    clear_marker        = Marker()
    clear_marker.action = Marker.DELETEALL
    clear_marker_array  = MarkerArray()
    clear_marker_array.markers.append(clear_marker)
    return clear_marker_array

def create_tracking_topic(track):
    track_msg = Tracking()

    track_msg.header.frame_id = "world"
    track_msg.header.stamp = rospy.Time.now()

    track_msg.Rel_Pos_X = track[LIDAR_TRACKING_REL_POS_X]
    track_msg.Rel_Pos_Y = track[LIDAR_TRACKING_REL_POS_Y]
    track_msg.Width     = track[    LIDAR_TRACKING_WIDTH]
    track_msg.Length    = track[   LIDAR_TRACKING_LENGTH]
    track_msg.Height    = track[   LIDAR_TRACKING_HEIGHT]
    track_msg.Yaw       = track[      LIDAR_TRACKING_YAW]
    track_msg.ID        = track[       LIDAR_TRACKING_ID]
    track_msg.Rel_Vel_X = track[LIDAR_TRACKING_REL_VEL_X]
    track_msg.Rel_Vel_Y = track[LIDAR_TRACKING_REL_VEL_Y]
    track_msg.Life_Time = track[LIDAR_TRACKING_LIFE_TIME]
    track_msg.Class     = track[    LIDAR_TRACKING_CLASS]

    return track_msg


def create_trackingBEV(track):

    # BEV Square Visualaization
    trk_center      = [track[LIDAR_TRACKING_REL_POS_X], track[LIDAR_TRACKING_REL_POS_Y], 0]
    trk_yaw         = -track[LIDAR_TRACKING_YAW]
    trk_size        = [track[LIDAR_TRACKING_LENGTH], track[LIDAR_TRACKING_WIDTH], track[LIDAR_TRACKING_HEIGHT]]
    corner_3d_trk   = get_3d_box(trk_center, trk_yaw, trk_size)


    corners_x = corner_3d_trk[:,0].squeeze()
    corners_y = corner_3d_trk[:,1].squeeze()

    # print(track[LIDAR_TRACKING_ID],trk_yaw*180/np.pi, track[LIDAR_TRACKING_YAW_RATE]*180/np.pi)

    marker                 = Marker()
    marker.header.frame_id = "world"  # 적절한 프레임 ID로 변경
    marker.header.stamp = rospy.Time.now()
    marker.ns              = "square"
    marker.id              = int(track[LIDAR_TRACKING_ID])
    marker.type            = Marker.LINE_STRIP
    marker.action          = Marker.ADD

    points = [
    Point(corners_x[0], corners_y[0], 0),
    Point(corners_x[1], corners_y[1], 0),
    Point(corners_x[2], corners_y[2], 0),
    Point(corners_x[3], corners_y[3], 0),
    Point(corners_x[0], corners_y[0], 0)
    ]

    marker.points  = points

    marker.scale.x = 0.05
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 1.0

    track_marker   = marker

    # BEV Text Visualization
    marker                 = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.type            = marker.TEXT_VIEW_FACING
    marker.text            = "ID : " + str(track[LIDAR_TRACKING_ID])
    marker.pose.position.x = track[LIDAR_TRACKING_REL_POS_X]
    marker.pose.position.y = track[LIDAR_TRACKING_REL_POS_Y]
    marker.pose.position.z = 0
    marker.scale.x         = 0.8
    marker.scale.y         = 0.8
    marker.scale.z         = 0.8
    marker.color.a         = 1.0
    marker.color.r         = 0.0
    marker.color.g         = 1.0
    marker.color.b         = 0.0
    marker.id              = int(track[LIDAR_TRACKING_ID])

    text_marker            = marker

    return track_marker, text_marker
