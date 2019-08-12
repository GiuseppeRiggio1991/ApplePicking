#!/usr/bin/env python

import rospy
from std_msgs.msg import String

rospy.init_node('freedrive_node')

pub = rospy.Publisher('/ur_driver/URScript', String)
rate = rospy.Rate(3)

rospy.loginfo('Initialized URScript freedrive node. Please set the ROS Param /freedrive to true to enable Freedrive')

while not rospy.is_shutdown():
    if rospy.get_param('/freedrive', False):
        pub.publish('freedrive_mode()')
    rate.sleep()