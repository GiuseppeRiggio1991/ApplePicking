#!/usr/bin/env python

import time
import enum
import openravepy
import numpy
import prpy
import sys
from openravepy.misc import InitOpenRAVELogging
from prpy.planning.cbirrt import CBiRRTPlanner
from copy import *

import rospy
import rospkg
from intera_core_msgs.msg import EndpointState, JointLimits
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger
from sawyer_planner.srv import AppleCheck, AppleCheckRequest, AppleCheckResponse
from online_planner.srv import *

import intera_interface

import pyquaternion
import socket

class SawyerPlanner:

    def __init__(self):

        self.environment_setup()
        self.STATE = enum.Enum('STATE', 'SEARCH TO_NEXT APPROACH GRAB CHECK_GRASPING TO_DROP DROP')

        self.goal = [None]
        self.ee_position = None
        self.starting_position_offset = 0.4

        rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState, self.get_robot_ee_position, queue_size = 1)
        rospy.wait_for_message("/robot/limb/right/endpoint_state", EndpointState)
        rospy.Subscriber("/sawyer_planner/goal", Point, self.get_goal, queue_size = 1)
        rospy.Subscriber("/robot/joint_limits", JointLimits, self.get_joint_limits, queue_size = 1)
        self.enable_bridge_pub = rospy.Publisher("/sawyer_planner/enable_bridge", Bool, queue_size = 1)

        self.gripper_client = rospy.ServiceProxy('/gripper_action', SetBool)
        self.apple_check_client = rospy.ServiceProxy("/sawyer_planner/apple_check", AppleCheck)
        self.start_pipeline_client = rospy.ServiceProxy("/sawyer_planner/start_pipeline", Trigger)
        self.plan_pose_client = rospy.ServiceProxy("plan_pose_srv", PlanPose)

        time.sleep(0.5)
        
        #self.enable_bridge_pub.publish(Bool(True))

        self.arm = intera_interface.Limb("right")

        # trasform from the real sawyer EE to openrave camera frame 
        self.T_EE2C = numpy.array([
                    [1.0, 0.0, 0.0, 0.0026],
                    [0.0, 1.0, 0.0, 0.0525],
                    [0.0, 0.0, 1.0, -0.0899],
                    [0.0, 0.0, 0.0, 1.0]
                    ])

        # transformation from the new camera frame (mounted on the gripper) and the EE
        # self.T_G2EE = numpy.array([
        #             [0.9961947, -0.0871557, 0.0, 0.0],
        #             [0.0871557, 0.9961947, 0.0, -0.075],
        #             [0.0, 0.0, 1.0, 0.12],
        #             [0.0, 0.0, 0.0, 1.0]
        #             ])
        self.T_G2EE = numpy.array([
                    [0.9961947, -0.0871557, 0.0, 0.0],
                    [0.0871557, 0.9961947, 0.0, 0.075],
                    [0.0, 0.0, 1.0, -0.12],
                    [0.0, 0.0, 0.0, 1.0]
                    ])

        #self.T_G2EE = numpy.array([
        #            [0.0, -1.0, 0.0, 0.075],
        #            [1.0, 0.0, 0.0, 0.0],
        #            [0.0, 0.0, 1.0, 0.14],
        #            [0.0, 0.0, 0.0, 1.0]
        #            ])

        self.K_V = 0.3
        self.K_VQ = 2.1
        self.CONFIG_UP = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, -numpy.pi/2, 0.0])
        self.MIN_MANIPULABILITY = 0.03

        #self.enable_bridge_pub.publish(Bool(False))

        # open the TCP socket to the planner
        # I will parametrize all this stuff
        # TCP_IP = '192.168.1.103' 
        # TCP_PORT = 5007
        # self.BUFFER_SIZE = 1024
        # self.socket_handler = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.socket_handler.settimeout(2.0)
        # self.socket_handler.connect((TCP_IP, TCP_PORT))


        # rospy.Timer(rospy.Duration(0.05), self.socket_handshake)

        self.state = self.STATE.SEARCH

    def environment_setup(self):

        self.env = openravepy.Environment()
        InitOpenRAVELogging()
        module = openravepy.RaveCreateModule(self.env, 'urdf')
        rospack = rospkg.RosPack()
        with self.env:
            # name = module.SendCommand('load /home/peppe/python_test/sawyer.urdf /home/peppe/python_test/sawyer_base.srdf')
            name = module.SendCommand(
                'loadURI ' + rospack.get_path('fredsmp_utils') + '/robots/sawyer/sawyer.urdf'
                + ' ' + rospack.get_path('fredsmp_utils') + '/robots/sawyer/sawyer.srdf')
            self.robot = self.env.GetRobot(name)

        time.sleep(0.5)

        self.ikmodel = openravepy.databases.inversekinematics.InverseKinematicsModel(self.robot,iktype=openravepy.IkParameterization.Type.Transform6D)
        if not self.ikmodel.load():
            self.ikmodel.autogenerate()


        manip = self.robot.GetActiveManipulator()
        self.robot.SetActiveDOFs(manip.GetArmIndices())

    # def socket_handshake(self, *args):
    #     self.socket_handler.send("\n")

    def computeManipulability(self):

        current_joints_pos = self.arm.joint_angles()
        current_joints_pos = current_joints_pos.values()

        with self.robot:
            self.robot.SetDOFValues(current_joints_pos[::-1], self.robot.GetActiveManipulator().GetArmIndices())
            J_t = self.robot.GetActiveManipulator().CalculateJacobian()
            J_r = self.robot.GetActiveManipulator().CalculateAngularVelocityJacobian()
            J = numpy.concatenate((J_t, J_r), axis = 0)

        u, s, v = numpy.linalg.svd(J, full_matrices = False) # here you can try to use just J_t instead of J

        assert numpy.allclose(J, numpy.dot(u, numpy.dot(numpy.diag(s), v) ))

        return numpy.prod(s)


    def update(self):

        if self.state == self.STATE.SEARCH:

            rospy.loginfo("SEARCH")
        
            # call the hydra_bridge server
            #pipeline_srv = Trigger()
            #resp = self.start_pipeline_client.call(pipeline_srv)
            #while resp.success == False:
            #    resp = self.start_pipeline_client.call(pipeline_srv)

            # wait???

            self.state = self.STATE.TO_NEXT
        
        elif self.state == self.STATE.TO_NEXT:

            rospy.loginfo("TO_NEXT")

            # self.enable_bridge_pub.publish(Bool(True)) 

            rospy.sleep(1.0)  # avoid exiting before subscriber updates new apple goal, 666 should replace with something more elegant
            if self.goal[0] == None:
                rospy.logerr("There are no apples to pick!")
                sys.exit()
            print("starting position and direction: ")
            print(self.starting_position, self.starting_direction)
            self.plan_to_goal(self.starting_position, self.starting_direction, 0.0)

            self.state = self.STATE.APPROACH

        elif self.state == self.STATE.APPROACH:

            rospy.loginfo("APPROACH")

            #start = time.time()

            #while (self.goal[0] == None) and (time.time() - start < 20.0) :
            #    pass
            self.enable_bridge_pub.publish(Bool(True)) 
            rospy.sleep(1.0)

            self.K_VQ = 2.5

            apple_check_srv = AppleCheckRequest()
            apple_check_srv.apple_pose.x = self.goal[0];
            apple_check_srv.apple_pose.y = self.goal[1];
            apple_check_srv.apple_pose.z = self.goal[2];

            # resp = self.apple_check_client.call(apple_check_srv)

            # if resp.apple_is_there:
            if 1:
                print("apple is there, going to grab")
                self.go_to_goal()
                self.state = self.STATE.GRAB
                # self.state = self.STATE.TO_DROP
            else:
                print("apple is NOT there, going to next apple")
                #rospy.logerr("There are no apples to pick!")
                #sys.exit()
                self.goal[0] = None
                print("self.goal: " + str(self.goal))
                self.state = self.STATE.TO_NEXT

        elif self.state == self.STATE.GRAB:

            rospy.loginfo("GRAB")

            # self.grab()

            #self.enable_bridge_pub.publish(Bool(False))

            self.state = self.STATE.CHECK_GRASPING

        elif self.state == self.STATE.CHECK_GRASPING:

            rospy.loginfo("CHECK_GRASPING")
            print ("start2: ", self.starting_position, " ", self.starting_direction)

            self.go_to_goal(self.starting_position, self.starting_direction, 0.0)

            apple_check_srv = AppleCheckRequest()
            apple_check_srv.apple_pose.x = self.goal[0];
            apple_check_srv.apple_pose.y = self.goal[1];
            apple_check_srv.apple_pose.z = self.goal[2];

            resp = self.apple_check_client.call(apple_check_srv)

            # if resp.apple_is_there:
            if 0:
                print("apple is still there, trying to grab again")
                # self.drop()
                self.go_to_goal()
                self.state = self.STATE.GRAB
            else:
                print("apple successfully picked, going to drop")
                self.goal = [None]
                self.state = self.STATE.TO_DROP
                self.enable_bridge_pub.publish(Bool(False)) 
                rospy.sleep(1.0)

        elif self.state == self.STATE.TO_DROP:

            rospy.loginfo("TO_DROP")


            self.K_VQ = 0.5
            # goal = deepcopy(self.goal)
            # goal[0] -= 0.35
            # # to_goal = numpy.dot( self.ee_orientation.rotation_matrix, numpy.array([0.0, 0.0, 1.0]) )
            # print("starting position and direction: ")
            # print(self.starting_position, self.starting_direction)
            # self.go_to_goal(self.starting_position, self.starting_direction, 0.0)
            # self.go_to_goal(goal)

            # self.go_to_place()

            # self.state = self.STATE.DROP
            self.state = self.STATE.TO_NEXT

        elif self.state == self.STATE.DROP:

            rospy.loginfo("DROP")

            self.drop()

            self.state = self.STATE.TO_NEXT

        else:
            pass


    def get_robot_ee_position(self, msg):

        self.ee_orientation = pyquaternion.Quaternion(msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z)
        self.ee_position = numpy.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def get_joint_limits(self, msg):
        
        lower = numpy.array(msg.position_lower)
        upper = numpy.array(msg.position_upper)

        self.robot.SetDOFLimits(lower[:7], upper[:7], self.robot.GetActiveManipulator().GetArmIndices())


    def get_goal(self, msg, offset = 0.08):

        self.goal = numpy.array([msg.x, msg.y, msg.z])
        self.goal_off = self.goal - offset * self.normalize(self.goal - self.ee_position)

        self.starting_position = self.goal - numpy.array([self.starting_position_offset, 0.0, 0.0]);
        self.starting_direction = numpy.array([1.0, 0.0, 0.0])
        

    def go_to_start(self):
        
        joints = self.arm.joint_angles()

        if ( numpy.linalg.norm(joints.values() - self.qstart ) < 4.0 ):
            cmd = dict(zip(joints.keys(), self.qstart))
            self.arm.move_to_joint_positions(cmd)
        else:
            # if too far, go up first
            rospy.logwarn("The starting position is too far, I'm going up first!")
            
            cmd = dict(zip(joints.keys(), self.CONFIG_UP))
            self.arm.move_to_joint_positions(cmd)
            
            time.sleep(1)
            
            cmd = dict(zip(joints.keys(), self.qstart))
            self.arm.move_to_joint_positions(cmd)
                           
        
    def go_to_goal(self, goal = [None], to_goal = [None], offset = 0.05):

        if goal[0] == None:
            goal = self.goal
            self_goal = True
            print("goal: " + str(goal))
            goal_off = self.goal_off
        else:
            if to_goal[0] == None:
                goal_off = goal - offset * self.normalize(goal - self.ee_position)
            else:
                goal_off = goal - offset * self.normalize(to_goal)

        while numpy.linalg.norm(goal_off - self.ee_position) > 0.01:
            # print("goal_off: " + str(goal_off))
            if (self_goal):
                goal = self.goal
                goal_off = self.goal_off

            des_vel_t = self.K_V * (goal_off - self.ee_position)

            if to_goal[0] != None:
                des_omega = - self.K_VQ * self.get_angular_velocity([None], to_goal)
            else:
                des_omega = - self.K_VQ * self.get_angular_velocity(goal)

            des_vel = numpy.append(des_vel_t, des_omega)

            #print "goal: ", self.goal, " off: ", self.goal_off
            #print "des_vel: ", des_vel

            #singularity check

            manipulability = self.computeManipulability()
            if (manipulability < self.MIN_MANIPULABILITY):
            # if 0:

                rospy.loginfo("detected low manipulability, exiting: " + str(manipulability))
                # singularity detected, send zero velocity to the robot and exit

                joint_vel = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                cmd = self.arm.joint_velocities()
                cmd = dict(zip(cmd.keys(), joint_vel[::-1]))
                self.arm.set_joint_velocities(cmd)
                sys.exit()

            else:

                joint_vel = self.compute_joint_vel(des_vel)
                cmd = self.arm.joint_velocities()
                cmd = dict(zip(cmd.keys(), joint_vel[::-1]))
                self.arm.set_joint_velocities(cmd)


    def plan_to_goal(self, goal = [None], to_goal = [None], offset = 0.13, ignore_trellis=False):

        if goal[0] == None:
            goal = self.goal
            goal_off = self.goal_off
        else:
            if to_goal[0] == None:
                goal_off = goal - offset * self.normalize(goal - self.ee_position)
            else:
                goal_off = goal - offset * self.normalize(to_goal)

        #orientation_q = pyquaternion.Quaternion(axis = to_goal, angle = 0)
        #orientation = [orientation_q[0], orientation_q[1], orientation_q[2], orientation_q[3]]

        q = pyquaternion.Quaternion(0.5, -0.5, 0.5, -0.5)

        # compute camera transform
        T_EE = numpy.zeros((4, 4))
        T_EE[:3, :3] = q.rotation_matrix
        T_EE[:3, 3] = goal_off.transpose()
        T_EE[3, 3] = 1.0

        # T_C = numpy.dot(T_EE, numpy.linalg.inv(self.T_G2EE))
        T_C = numpy.dot(T_EE, numpy.linalg.inv(self.T_EE2C))

        goal_off_camera = T_C[:3, 3]
        
        plan_pose_msg = Pose()
        plan_pose_msg.position.x = goal_off_camera[0]
        plan_pose_msg.position.y = goal_off_camera[1]
        plan_pose_msg.position.z = goal_off_camera[2]
        plan_pose_msg.orientation.x = -0.5
        plan_pose_msg.orientation.y = 0.5
        plan_pose_msg.orientation.z = -0.5
        plan_pose_msg.orientation.w = 0.5
        resp = self.plan_pose_client(plan_pose_msg, ignore_trellis)

        if not resp.success:
            rospy.logwarn("planning to next target failed")
        else:
        # message = "moveArm," + ",".join(map(str, goal_off_camera)) + "," + ",".join(map(str, [-0.5, 0.5, -0.5, 0.5])) + "\n"
        
        # resp = ""
        # while resp == "":
    
        #     self.socket_handler.send(message)
        #     print("sent moveArm")
        #     resp = self.socket_handler.recv(self.BUFFER_SIZE) # string -> 0 = success 1 = fail

        # message = "sendArm\n"

        # stopped = ""
        # while stopped == "":
        #     self.socket_handler.send(message)
        #     print("Sent SendArm")
        #     stopped = self.socket_handler.recv(self.BUFFER_SIZE)   

        # print("Starting moving")

            while numpy.linalg.norm(goal_off - self.ee_position) > 0.04:
                print("waiting for arm to reach goal_off pose")
                print("goal_off_camera: " + str(goal_off_camera))
                print("goal_off: " + str(goal_off))
                print("self.ee_position: " + str(self.ee_position))
                rospy.sleep(0.1)
                pass 

        return resp.success

    def get_angular_velocity(self, goal = [None], to_goal = [None]):
        
        # rotation between the vector pointing to the goal and the z-axis
        if goal[0] == None:
            goal == self.goal

        z = numpy.dot( self.ee_orientation.rotation_matrix, numpy.array([0.0, 0.0, 1.0]) )

        if to_goal[0] == None:
            to_goal = goal - self.ee_position

        # normalize vectors
        z = self.normalize(z)
        to_goal = self.normalize(to_goal)

        axis = numpy.cross(to_goal, z)
        angle = numpy.arccos(numpy.dot(to_goal, z))

        if (angle < 0.75*numpy.pi):
            return angle * axis
        else:
            rospy.logwarn("The orientation error is too big!")
            return 0.0 * axis

    def normalize(self, v):
        
        norm = numpy.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm


    def compute_joint_vel(self, des_vel):

        joints = self.arm.joint_angles()
        joints = joints.values()

        with self.robot:
            self.robot.SetDOFValues(joints[::-1], self.robot.GetActiveManipulator().GetArmIndices())
            J_t = self.robot.GetActiveManipulator().CalculateJacobian()
            J_r = self.robot.GetActiveManipulator().CalculateAngularVelocityJacobian()
            J = numpy.concatenate((J_t, J_r), axis = 0)

        return numpy.dot( numpy.linalg.pinv(J), des_vel)

    def grab(self):

        resp = self.gripper_client.call(False)

        while not resp.success:
            resp = self.gripper_client.call(False)

        #self.go_up_and_back() 

    def go_up_and_back(self):

        # first go up

        goal = deepcopy(self.goal)
        goal[2] += 0.10

        to_goal = numpy.dot( self.ee_orientation.rotation_matrix, numpy.array([0.0, 0.0, 1.0]) )

        self.go_to_goal(goal, to_goal)

        # then go back
        goal[0] -= 0.25
        self.go_to_goal(goal, to_goal)

    def drop(self):

        self.goal = [None]

        resp = self.gripper_client.call(True)

        while not resp.success:
            resp = self.gripper_client.call(True)

    def go_to_place(self):

        to_goal = self.normalize(numpy.array([0.87758256189, 0.0, -0.4794255386]))

        self.plan_to_goal(self.starting_position, to_goal, 0.0)


    def clean(self):

        # on shutdown

        # self.socket_handler.close()

        qstart_param_name = rospy.search_param('qstart')

        if qstart_param_name != None:
            rospy.delete_param(qstart_param_name)
