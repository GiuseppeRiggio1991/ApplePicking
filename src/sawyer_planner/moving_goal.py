#!/usr/bin/env python

import rospy
import numpy

from geometry_msgs.msg import Point

from visualization_msgs.msg import Marker

class MovingGoal:

    def __init__(self):

        self.position_pub = rospy.Publisher("/sawyer_planner/goal", Point, queue_size = 1)
        self.marker_pub = rospy.Publisher("/sawyer/goal_marker", Marker, queue_size = 1)

        # default parameters
        self.DEFAULT_POSITION = [0.0, 0.0, 2.0] # if we will forget the parameters, the robot will go up
        self.DEFAULT_SIGMA = 0.001

        position_param_name = rospy.search_param('position')
        sigma_param_name = rospy.search_param('sigma')

        if (position_param_name == None or sigma_param_name == None):
            rospy.logwarn('Setting default parameters!')
            self.position = numpy.array(self.DEFAULT_POSITION)
            self.sigma = numpy.array(self.DEFAULT_SIGMA)
        else:
            self.position = numpy.array(rospy.get_param(position_param_name))
            self.sigma = numpy.array(rospy.get_param(sigma_param_name))

        if self.sigma.size == 1.0:
            self.sigma = numpy.ones(3) * self.sigma

        # marker
        self.create_sphere()

    def create_sphere(self):

        self.marker = Marker()
        
        self.marker.header.frame_id = "world"
        self.marker.header.stamp = rospy.Time.now()

        self.marker.id = 0
        self.marker.type = Marker.SPHERE
        self.marker.action = Marker.ADD

        self.marker.pose.position.x = self.position[0]
        self.marker.pose.position.y = self.position[1]
        self.marker.pose.position.z = self.position[2]

        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0

        self.marker.scale.x = 0.1
        self.marker.scale.y = 0.1
        self.marker.scale.z = 0.1

        self.marker.color.a = 1.0
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 1.0

        self.marker_pub.publish(self.marker)


    def update_position(self):

        covariance = numpy.diag(self.sigma**2)

        new_position = numpy.random.multivariate_normal(self.position, covariance)

        new_position_point = Point(new_position[0], new_position[1], new_position[2])
        self.position_pub.publish(new_position_point)
        
        # update marker position
        self.marker.header.stamp = rospy.Time.now()
        self.marker.action = Marker.MODIFY

        self.marker.pose.position.x = new_position[0]
        self.marker.pose.position.y = new_position[1]
        self.marker.pose.position.z = new_position[2]

        self.marker_pub.publish(self.marker)

    def clean(self):

        # on shutdown

        position_param_name = rospy.search_param('position')
        sigma_param_name = rospy.search_param('sigma')
        
        if (position_param_name != None):
            rospy.delete_param(position_param_name)

        if (sigma_param_name != None):
            rospy.delete_param(sigma_param_name)
       
