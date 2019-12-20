#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import PointStamped, Point, Transform, TransformStamped, Quaternion
from visualization_msgs.msg import Marker
from tf2_ros import TransformListener, Buffer, ExtrapolationException
from std_srvs.srv import Empty
from sensor_msgs.msg import PointCloud2, PointField
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from sawyer_planner.srv import SetGoal
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import struct
from tf.transformations import decompose_matrix, quaternion_from_euler
from scipy.stats import multivariate_normal
from copy import deepcopy
from sensor_msgs.msg import CameraInfo, Image
from matplotlib.path import Path
import message_filters
from numpy.linalg import svd
import cv2
from cv_bridge import CvBridge
from sklearn.linear_model import LinearRegression
from skimage.draw import line as draw_line
from ros_numpy import numpify

numpy_ver = [int(x) for x in np.version.version.split('.')]

class PointGravitation(object):
    def __init__(self):
        self.goal = None
        self.last_update = None
        self.active = False
        self.max_speed = 0.02     # Goal cannot move more than x m per second

        # Pubs, subs
        self.goal_pub = rospy.Publisher('/update_goal_point', Point, queue_size=1)
        self.cloud_sub = rospy.Subscriber(rospy.get_param('inhand_camera_cloud'), PointCloud2, self.process_pc)

        self.base_frame = rospy.get_param('base_frame')
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer)

        rospy.Service('point_tracker_set_goal', SetGoal, self.add_goal)
        rospy.Service('activate_point_tracker', Empty, self.activate)
        rospy.Service('deactivate_point_tracker', Empty, self.deactivate)

    def add_goal(self, goal_msg):
        pt = goal_msg.goal.goal
        self.goal = np.array([pt.x, pt.y, pt.z])

        return []


    def activate(self, *_, **__):
        self.active = True
        self.last_update = rospy.Time.now()
        return []

    def deactivate(self, *_, **__):
        self.active = False
        self.last_update = None
        self.goal = None
        return []

    def get_tf(self, target, source, stamp):

        success = self.buffer.can_transform(target, source, stamp, rospy.Duration(1.5))

        if not success:
            raise Exception('Something went wrong with TF')
        tf = self.buffer.lookup_transform(target, source, stamp)
        return tf



    def process_pc(self, pc_msg):

        if not self.active:

            return

        goal = deepcopy(self.goal)
        last_update = deepcopy(self.last_update)

        now = pc_msg.header.stamp
        base_to_cam_tf = self.get_tf(pc_msg.header.frame_id, self.base_frame, now)
        cam_to_base_tf = self.get_tf(self.base_frame, pc_msg.header.frame_id, now)

        # Extracts the 3D array of points - needs to do some management of structured arrays for efficiency
        pts_struct = numpify(pc_msg)[['x', 'y', 'z']]
        if numpy_ver[1] >= 15:
            from numpy.lib.recfunctions import repack_fields
            pts_struct = repack_fields(pts_struct)
        pts = pts_struct.view((pts_struct.dtype[0], 3))
        if len(pts.shape) == 3:
            pts = pts.transpose(2, 0, 1).reshape(3, -1).T

        stamped_goal = PointStamped()
        stamped_goal.header.frame_id = self.base_frame
        stamped_goal.point = Point(*goal)
        stamped_goal_cam = do_transform_point(stamped_goal, base_to_cam_tf).point
        stamped_goal = np.array([stamped_goal_cam.x, stamped_goal_cam.y, stamped_goal_cam.z])

        dist = np.linalg.norm(pts - stamped_goal, axis=1)
        inv_sq_dist = 1 / (dist ** 2)
        wgt = inv_sq_dist / inv_sq_dist.sum()



        new_goal = wgt.dot(pts)
        dgoal = new_goal - stamped_goal
        dt = (now - last_update).to_sec()
        speed = np.abs(np.linalg.norm(dgoal) / dt)
        if speed > self.max_speed:
            new_goal = stamped_goal + (dgoal / np.linalg.norm(dgoal)) * self.max_speed * dt

        stamped_new_goal_cam = PointStamped()
        stamped_new_goal_cam.header.frame_id = pc_msg.header.frame_id
        stamped_new_goal_cam.point = Point(*new_goal)

        stamped_new_goal = do_transform_point(stamped_new_goal_cam, cam_to_base_tf).point
        new_goal = np.array([stamped_new_goal.x, stamped_new_goal.y, stamped_new_goal.z])

        print('Old goal:{}\nNew goal:{}'.format(goal, new_goal))
        self.goal = new_goal
        self.goal_pub.publish(Point(*new_goal))
        self.last_update = now




if __name__ == '__main__':

    rospy.init_node('point_gravitation')
    tracker = PointGravitation()

    # rate = rospy.Rate(10)
    rospy.spin()
    # while not rospy.is_shutdown():
    #     if tracker.active:
    #         new_goal_pub.publish(tracker.goal.point)
    #         tracker.rviz_publish_goals()
    #     rate.sleep()



