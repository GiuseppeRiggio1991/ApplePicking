#!/usr/bin/env python

import time
import rospy
import enum
import numpy
from dynamixel_workbench_msgs.srv import WheelCommand
from dynamixel_workbench_msgs.msg import DynamixelStateList
from std_srvs.srv import SetBool


class Gripper:

    def __init__(self):

        self.VELOCITY_DEFAULT = 4.0
        self.STATE = enum.Enum('STATE', 'OPEN CLOSED')
        self.MAX_REVOLUTIONS = 12
        self.gripper_actuation_time = 5.0

        velocity_param_name = rospy.search_param('velocity')

        if velocity_param_name == None:
            rospy.logwarn('Setting default velocity = 0.3 rad/s!')
            self.velocity = self.VELOCITY_DEFAULT
        else:
            self.velocity = numpy.array(rospy.get_param(velocity_param_name))

        self.velocity_client = rospy.ServiceProxy('/wheel_command', WheelCommand)
        self.velocity_server = rospy.Service('/gripper_action', SetBool, self.move_gripper)

        self.first_cycle = True
        self.motor_position = 0.0
        self.revolutions = 0
        self.gripper_state = self.STATE.OPEN

        rospy.Subscriber("/dynamixel_state", DynamixelStateList, self.get_motor_state, queue_size = 1)


    def move_gripper(self, req):

        print(req.data)
        if req.data:
            return self.open(), ""
        else:
            return self.close(), ""


    def get_motor_state(self, msg):

        if self.first_cycle:
            self.first_cycle = False
        else:
            if self.gripper_state == self.STATE.OPEN:
                if self.motor_position - msg.dynamixel_state[0].present_position > 100:
                    self.revolutions += 1
            else:
                if self.motor_position - msg.dynamixel_state[0].present_position < -100:
                    self.revolutions += 1

        self.motor_position = msg.dynamixel_state[0].present_position

        print "rev: ", self.revolutions, " pos: ", self.motor_position
        
    def open(self):

        if self.gripper_state == self.STATE.OPEN:
            rospy.logwarn('The gripper is already open!')
            return False

        left_vel = - self.velocity
        right_vel = - self.velocity

        called = False

        time_start = rospy.Time.now()
        # while self.revolutions < self.MAX_REVOLUTIONS:
        while (rospy.Time.now() - time_start).to_sec() < self.gripper_actuation_time:
            # print("debug 1")
            if (not called) and self.velocity_client(right_vel, left_vel):
                # print("debug 2")
                called = True

        while not self.velocity_client(0, 0):
            # print("debug 3")
            pass

        time.sleep(0.1)

        self.revolutions = 0
        self.gripper_state = self.STATE.OPEN

        return True
  

    def close(self):

        if self.gripper_state == self.STATE.CLOSED:
            rospy.logwarn('The gripper is already closed!')
            return False

        left_vel = self.velocity
        right_vel = self.velocity

        called = False

        time_start = rospy.Time.now()
        # while self.revolutions < self.MAX_REVOLUTIONS:
        while (rospy.Time.now() - time_start).to_sec() < self.gripper_actuation_time:
            print("debug 1")
            if (not called) and self.velocity_client(right_vel, left_vel):
                print("debug 2")
                called = True

        while not self.velocity_client(0, 0):
            print("debug 3")
            pass

        time.sleep(0.1)

        self.revolutions = 0
        self.gripper_state = self.STATE.CLOSED

        return True


    def clean(self):

        # on shutdown

        velocity_param_name = rospy.search_param('velocity')

        if velocity_param_name != None:
            rospy.delete_param(velocity_param_name)
