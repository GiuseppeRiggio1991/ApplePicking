#!/usr/bin/env python

import rospy
import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
import os
import sys
from rgb_segmentation.srv import GetCutPoint
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped
from sawyer_planner.msg import GoalUpdate

def callback(_):

    pass

if __name__ == '__main__':
    rospy.init_node('goal_manager_visual')

    cut_service = rospy.ServiceProxy('cut_point_srv', GetCutPoint)
    pub = rospy.Publisher('/update_goal_point', GoalUpdate, queue_size=1)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():

        if not rospy.get_param('/going_to_goal', False) or not rospy.get_param('/use_servoing', False):
            rate.sleep()
            continue
        rospy.loginfo_throttle(0.1, 'Trying to update goal...')
        current_pose = rospy.wait_for_message('manipulator_pose', PoseStamped)
        try:
            response = cut_service().cut_points
        except Exception as e:
            # Really hacky way to see if the error returned is the generic one with no message
            if not str(e).split('error:')[1].strip():
                continue
            raise e

        msg = GoalUpdate()
        msg.points = response
        msg.pose = current_pose

        pub.publish(msg)




