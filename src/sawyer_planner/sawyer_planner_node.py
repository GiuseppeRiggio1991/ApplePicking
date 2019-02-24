#!/usr/bin/env python

import time
import enum
import openravepy
import numpy
import prpy
import sys
import os
import random
from openravepy.misc import InitOpenRAVELogging
from prpy.planning.cbirrt import CBiRRTPlanner
from copy import *
import csv
import json

import rospy
import rospkg
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger, Empty
from sawyer_planner.srv import AppleCheck, AppleCheckRequest, AppleCheckResponse
from online_planner.srv import *
from task_planner.srv import *
# from task_planner.msgs import *

import pyquaternion
import socket

LOGGING = True

class SawyerPlanner:

    def __init__(self, metric, sim=False, goal_array=[], noise_array=[]):

        self.environment_setup()
        self.STATE = enum.Enum('STATE', 'SEARCH TO_NEXT APPROACH GRAB CHECK_GRASPING TO_DROP DROP RECOVER')

        self.goal = [None]
        self.goal_array = []
        self.sequenced_goals = []
        self.sequenced_trajectories = []
        self.manipulator_joints = []
        self.num_goals_history = 0  # length of sequenced goals
        self.ee_position = None
        self.starting_position_offset = 0.35
        self.apple_offset = [0.0, 0.0, 0.0]
        # self.go_to_goal_offset = [0.05, 0.0, 0.0]
        self.go_to_goal_offset = 0.12  # offset from apple centre to sawyer end effector frame (not gripper)
        self.sim = sim
        self.limits_epsilon = 0.01
        self.K_V = 0.3
        self.K_VQ = 2.1
        self.CONFIG_UP = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, -numpy.pi/2, 0.0])
        self.MIN_MANIPULABILITY = 0.025
        self.MIN_MANIPULABILITY_RECOVER = 0.02
        self.MAX_RECOVERY_DURATION = 1.0
        self.recovery_flag = False
        self.start_recovery_time = None
        self.recovery_trajectory = []

        self.sequencing_metric = metric

        self.joint_limits_lower = numpy.array([-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124]) + self.limits_epsilon
        self.joint_limits_upper = numpy.array([3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124]) - self.limits_epsilon
        self.joint_limits_lower_recover = numpy.array([-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124])
        self.joint_limits_upper_recover = numpy.array([3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124])

        self.joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']

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
                    [0.9961947, 0.0871557, 0.0, 0.0],
                    [-0.0871557, 0.9961947, 0.0, 0.075],
                    [0.0, 0.0, 1.0, -0.12],
                    [0.0, 0.0, 0.0, 1.0]
                    ])

        #self.T_G2EE = numpy.array([
        #            [0.0, -1.0, 0.0, 0.075],
        #            [1.0, 0.0, 0.0, 0.0],
        #            [0.0, 0.0, 1.0, 0.14],
        #            [0.0, 0.0, 0.0, 1.0]
        #            ])
        # self.robot.SetDOFLimits(self.joint_limits_lower, self.joint_limits_upper, self.robot.GetActiveManipulator().GetArmIndices())

        # rospy.Subscriber("/sawyer_planner/goal", Point, self.get_goal, queue_size = 1)
        rospy.Subscriber("/sawyer_planner/goal_array", Float32MultiArray, self.get_goal_array, queue_size = 1)
        self.enable_bridge_pub = rospy.Publisher("/sawyer_planner/enable_bridge", Bool, queue_size = 1)
        self.sim_joint_velocities_pub = rospy.Publisher('sim_joint_velocities', JointTrajectoryPoint, queue_size=1)

        self.gripper_client = rospy.ServiceProxy('/gripper_action', SetBool)
        self.apple_check_client = rospy.ServiceProxy("/sawyer_planner/apple_check", AppleCheck)
        self.start_pipeline_client = rospy.ServiceProxy("/sawyer_planner/start_pipeline", Trigger)
        self.plan_pose_client = rospy.ServiceProxy("/plan_pose_srv", PlanPose)
        self.optimise_offset_client = rospy.ServiceProxy("/optimise_offset_srv", OptimiseTrajectory)
        self.optimise_trajectory_client = rospy.ServiceProxy("/optimise_trajectory_srv", OptimiseTrajectory)
        self.sequencer_client = rospy.ServiceProxy("/sequence_tasks_srv", SequenceTasks)

        time.sleep(0.5)
        
        #self.enable_bridge_pub.publish(Bool(True))

        rospy.Subscriber('manipulator_pose', PoseStamped, self.get_robot_ee_position, queue_size = 1)
        rospy.Subscriber('manipulator_joints', JointState, self.get_robot_joints, queue_size = 1)
        or_joint_states = rospy.wait_for_message('manipulator_joints', JointState)
        or_joints_pos = numpy.array(or_joint_states.position)

        if self.sim:
            self.set_home_client = rospy.ServiceProxy('set_home_position', Empty)
            self.set_home_client.call()
            rospy.sleep(1.0)
            # self.apple_offset = [0.5, 0.0, 0.0]
            # self.goal_array = [[0.8, 0.3, 0.5], [0.8, -0.3, 0.5]]
            # self.goal_array = [[0.8, 0.3, 0.5]]  # bad run low manip
            # self.goal_array = [[0.8, 0.1, 0.5]]  # semi okay run
            #self.goal_array = [[0.8, 0.3, 0.2]]  # bad run joint limits
            #self.goal_array = [[0.7, -0.3, 0.8]]  # good one
            # self.goal_array = [[0.8, -0.4, 0.8]]0.77170295 -0.1971278   0.14966555
            # self.goal_array = [[0.914682, -0.218761, 0.694819]]
            # self.goal_array = [[0.9, 0.2,   0.7]]  # low manip
            # self.goal_array= [ [ 0.9,         0.12397271,  0.77931632]]  # wobbly
            #self.goal_array = [[0.735962,-0.204364,0.555817]]  # joint limits
            # self.goal_array = [[0.8, 0.0, 0.2]] # planner fails
            # self.goal_array = [[ 0.95,        0.30172234,  0.6943676 ]]  # joint limits

            # random.seed(time.time())
            # # numpy.random.seed(time.time())
            # self.goal_array = []
            # for i in range(8):
            #     rand_x = random.uniform(0.8, 0.9)
            #     rand_y = random.uniform(-0.35, 0.35)
            #     rand_z = random.uniform(0.2, 0.7)
            #     self.goal_array.append([rand_x, rand_y, rand_z])

            # # add Gaussian noise
            # x_noise = numpy.random.normal(0.0, 0.02, len(self.goal_array))
            # y_noise = numpy.random.normal(0.0, 0.05, len(self.goal_array))
            # z_noise = numpy.random.normal(0.0, 0.05, len(self.goal_array))
            # self.noise_array = numpy.vstack((x_noise, y_noise, z_noise)).transpose()
            # rospy.loginfo("self.noise_array: ")
            # rospy.loginfo(str(self.noise_array))

            self.goal_array = copy(goal_array)
            self.noise_array = copy(noise_array)

            # self.goal_array = []
            # x_array = [0.9]
            # y_array = [-0.35, 0.0, 0.35]
            # z_array = [0.2, 0.45, 0.7]
            # for x in x_array:
            #     for y in y_array:
            #         for z in z_array:
            #             self.goal_array.append([x, y, z])


            #self.goal_array = []
            #for index in range (-6, 7):
            #    self.goal_array.append([0.8, index/20.0, 0.4])


        # if not self.sim:
        else:
            # from intera_core_msgs.msg import EndpointState, JointLimits
            import intera_interface
            # rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState, self.get_robot_ee_position, queue_size = 1)
            # rospy.wait_for_message("/robot/limb/right/endpoint_state", EndpointState)
            # rospy.Subscriber("/robot/joint_limits", JointLimits, self.get_joint_limits, queue_size = 1)
            self.arm = intera_interface.Limb("right")
            current_joints_pos = self.arm.joint_angles()
            current_joints_pos = numpy.array(current_joints_pos.values()[::-1])
            # joint_states = rospy.wait_for_message("/robot/joint_states", JointState)
            # joints_pos = numpy.array(joint_states.position)
            if numpy.linalg.norm(or_joints_pos - current_joints_pos) > 0.1:
                rospy.logerr("it seems like you should be running the node in sim mode, exiting.")
                sys.exit()

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

        if LOGGING:
            import rospkg
            rospack = rospkg.RosPack()
            self.directory = rospack.get_path('sawyer_planner') + '/results'
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            self.directory_metric = self.directory + '/' + metric
            if not os.path.exists(self.directory_metric):
                os.makedirs(self.directory_metric)
            # self.results_filename = self.directory_metric + 'results_' + time.strftime("%Y%m%d-%H%M%S")
            # self.fails_filename = self.directory_metric + 'fails_' + time.strftime("%Y%m%d-%H%M%S")
            # self.results_table = []
        self.results_dict = {'Sequencing Time':[], 'Planner Computation Time':[], 'Planner Execution Time':[], 'Approach Time':[], 'Num Apples':0}
        self.failures_dict = {'Joint Limits':[], 'Low Manip':[], 'Planner': [], 'Grasp Misalignment':[], 'Grasp Obstructed':[]}

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

        # current_joints_pos = self.arm.joint_angles()
        # current_joints_pos = current_joints_pos.values()
        current_joints_pos = copy(self.manipulator_joints)

        with self.robot:
            # self.robot.SetDOFValues(current_joints_pos[::-1], self.robot.GetActiveManipulator().GetArmIndices())
            self.robot.SetDOFValues(current_joints_pos, self.robot.GetActiveManipulator().GetArmIndices())
            J_t = self.robot.GetActiveManipulator().CalculateJacobian()
            J_r = self.robot.GetActiveManipulator().CalculateAngularVelocityJacobian()
            J = numpy.concatenate((J_t, J_r), axis = 0)

        u, s, v = numpy.linalg.svd(J, full_matrices = False) # here you can try to use just J_t instead of J

        assert numpy.allclose(J, numpy.dot(u, numpy.dot(numpy.diag(s), v) ))

        return numpy.prod(s)

    def computeReciprocalConditionNumber(self):

        # current_joints_pos = self.arm.joint_angles()
        # current_joints_pos = current_joints_pos.values()
        current_joints_pos = copy(self.manipulator_joints)

        with self.robot:
            # self.robot.SetDOFValues(current_joints_pos[::-1], self.robot.GetActiveManipulator().GetArmIndices())
            self.robot.SetDOFValues(current_joints_pos, self.robot.GetActiveManipulator().GetArmIndices())
            J_t = self.robot.GetActiveManipulator().CalculateJacobian()
            J_r = self.robot.GetActiveManipulator().CalculateAngularVelocityJacobian()
            J = numpy.concatenate((J_t, J_r), axis = 0)

        u, s, v = numpy.linalg.svd(J, full_matrices = False) # here you can try to use just J_t instead of J

        assert numpy.allclose(J, numpy.dot(u, numpy.dot(numpy.diag(s), v) ))

        return numpy.min(s)/numpy.max(s)

    def remove_current_apple(self):
        if not self.sim:
            apple_check_srv = AppleCheckRequest()
            apple_check_srv.apple_pose.x = self.goal[0];
            apple_check_srv.apple_pose.y = self.goal[1];
            apple_check_srv.apple_pose.z = self.goal[2];

            rospy.loginfo('checking apple: ' + str(self.goal))
            resp = self.apple_check_client.call(apple_check_srv)
        else:
            self.remove_from_goal_array(self.goal)

    def save_logs(self):
        if LOGGING:
            results_filename = self.directory_metric + '/results_' + time.strftime("%Y%m%d-%H%M%S")
            fails_filename = self.directory_metric + '/fails_' + time.strftime("%Y%m%d-%H%M%S")

            with open(results_filename, 'w') as outfile:
                json.dump(self.results_dict, outfile)
            with open(fails_filename, 'w') as outfile:
                json.dump(self.failures_dict, outfile)
            # w = csv.writer(open(results_filename, "w"))
            # for key, val in self.results_dict.items():
            #     w.writerow([key, val])
            # w = csv.writer(open(fails_filename, "w"))
            # for key, val in self.failures_dict.items():
            #     w.writerow([key, val])

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
            time_start = rospy.get_time()
            self.sequence_goals()
            elapsed_time = rospy.get_time() - time_start

            rospy.sleep(1.0)  # avoid exiting before subscriber updates new apple goal, 666 should replace with something more elegant
            if self.goal[0] == None:
                self.save_logs()
                rospy.logerr("There are no apples to pick!")
                # sys.exit()
                return False

            self.results_dict['Sequencing Time'].append(elapsed_time)
            self.results_dict['Num Apples'] += 1
            self.failures_dict['Grasp Misalignment'].append(0)
            self.failures_dict['Grasp Obstructed'].append(0)

            # if self.sim:
            #     resp = self.optimise_trajectory_client(self.sequenced_trajectories[0])
            #     sys.exit()
                
            print("starting position and direction: ")
            print(self.starting_position, self.starting_direction)
            # time_start = rospy.get_time()
            plan_success, plan_duration, traj_duration = self.plan_to_goal(self.starting_position, self.starting_direction, 0.0)
            # elapsed_time = rospy.get_time() - time_start
            # print("self.goal: " + str(self.goal))
            # raw_input('press enter to continue...')

            if plan_success:
                self.state = self.STATE.APPROACH
                # self.results_dict['Planner Time'].append(elapsed_time)
                self.results_dict['Planner Computation Time'].append(plan_duration)
                self.results_dict['Planner Execution Time'].append(traj_duration)
                self.failures_dict['Planner'].append(0)
            else:
                self.failures_dict['Planner'].append(1)
                # self.results_dict['Planner Time'].append(numpy.nan)
                self.results_dict['Planner Computation Time'].append(plan_duration)
                self.results_dict['Planner Execution Time'].append(numpy.nan)
                rospy.logerr('plan failed, going to next...')
                self.remove_current_apple()
                self.state = self.STATE.TO_NEXT
                # blacklist_goal_srv(self.goal)
                # sys.exit()

        elif self.state == self.STATE.APPROACH:

            rospy.loginfo("APPROACH")

            #start = time.time()

            #while (self.goal[0] == None) and (time.time() - start < 20.0) :
            #    pass
            if not self.sim:
                self.enable_bridge_pub.publish(Bool(True)) 
            rospy.sleep(1.0)

            self.K_VQ = 2.5

            # apple_check_srv = AppleCheckRequest()
            # apple_check_srv.apple_pose.x = self.goal[0];
            # apple_check_srv.apple_pose.y = self.goal[1];
            # apple_check_srv.apple_pose.z = self.goal[2];

            # resp = self.apple_check_client.call(apple_check_srv)

            # if resp.apple_is_there:
            # if 1:
            print("apple is there, going to grab")

            self.recovery_trajectory = []

            time_start = rospy.get_time()
            status = self.go_to_goal([None], numpy.array([1.0, 0.0, 0.0]), self.go_to_goal_offset)
            elapsed_time = rospy.get_time() - time_start
            if status == 0:
                self.state = self.STATE.GRAB
                self.results_dict['Approach Time'].append(elapsed_time)
                self.failures_dict['Joint Limits'].append(0)
                self.failures_dict['Low Manip'].append(0)
            elif status == 1:
                self.results_dict['Approach Time'].append(numpy.nan)
                self.failures_dict['Joint Limits'].append(1)
                self.failures_dict['Low Manip'].append(0)
                self.remove_current_apple()
                self.state = self.STATE.RECOVER
            elif status == 2:
                self.results_dict['Approach Time'].append(numpy.nan)
                self.failures_dict['Low Manip'].append(1)
                self.failures_dict['Joint Limits'].append(0)
                self.remove_current_apple()
                self.state = self.STATE.RECOVER
                # blacklist_goal_srv(self.goal)

                # self.state = self.STATE.TO_DROP
            # else:
            #     print("apple is NOT there, going to next apple")
            #     #rospy.logerr("There are no apples to pick!")
            #     #sys.exit()
            #     self.goal[0] = None
            #     print("self.goal: " + str(self.goal))
            #     self.state = self.STATE.TO_NEXT

        elif self.state == self.STATE.GRAB:

            rospy.loginfo("GRAB")

            # if 0:
            if not self.sim:
                self.grab()

            #self.enable_bridge_pub.publish(Bool(False))

            self.state = self.STATE.CHECK_GRASPING

        elif self.state == self.STATE.CHECK_GRASPING:

            rospy.loginfo("CHECK_GRASPING")

            self.remove_current_apple()
                # del self.goal_array[0]

            # if resp.apple_is_there:
            # if 0:
            #     print("apple is still there, trying to grab again")
            #     # self.drop()
            #     #self.go_to_goal()
            #     self.go_to_goal([None], numpy.array([1.0, 0.0, 0.0]), self.go_to_goal_offset)
            #     self.state = self.STATE.GRAB
            # else:
            print("apple successfully picked, going to drop")
            self.state = self.STATE.TO_DROP
            rospy.sleep(1.0)
            if not self.sim:
                self.enable_bridge_pub.publish(Bool(False)) 
                # rospy.sleep(1.0)

        elif self.state == self.STATE.RECOVER:

            rospy.loginfo("RECOVER")
            self.recovery_flag = True
            self.start_recovery_time = rospy.get_time()

            rospy.sleep(0.1)
            if numpy.linalg.norm(numpy.asarray(self.recovery_trajectory[-1]) - numpy.asarray(self.recovery_trajectory[0])) > 0.01:
                self.recovery_trajectory.append(self.manipulator_joints)

                print("self.recovery_trajectory[-1]: " + str(self.recovery_trajectory[-1]))
                print("self.recovery_trajectory[0]: " + str(self.recovery_trajectory[0]))
                traj_msg = self.list_trajectory_msg(self.recovery_trajectory[::-1])
                resp = self.optimise_trajectory_client.call(traj_msg, LOGGING)
                if not resp.success:
                # print ("start2: ", self.starting_position, " ", self.starting_direction)

                # if not self.go_to_goal(self.starting_position, self.starting_direction, 0.0):
                    # self.save_logs()
                    rospy.logerr("could not move back to drop, don't know how to recover, exiting...")
                    # sys.exit()
                    return False
            else:
                rospy.logwarn("arm didn't appear to move much before recovery state, continuing without recovery procedure")

            self.goal = [None]

            self.recovery_flag = False

            self.state = self.STATE.TO_NEXT


        elif self.state == self.STATE.TO_DROP:

            rospy.loginfo("TO_DROP")

            rospy.sleep(0.1)
            if numpy.linalg.norm(numpy.asarray(self.recovery_trajectory[-1]) - numpy.asarray(self.recovery_trajectory[0])) > 0.01:
                self.recovery_trajectory.append(self.manipulator_joints)

                print("self.recovery_trajectory[-1]: " + str(self.recovery_trajectory[-1]))
                print("self.recovery_trajectory[0]: " + str(self.recovery_trajectory[0]))
                traj_msg = self.list_trajectory_msg(self.recovery_trajectory[::-1])
                resp = self.optimise_trajectory_client.call(traj_msg, LOGGING)
                if not resp.success:
                # print ("start2: ", self.starting_position, " ", self.starting_direction)

                # if not self.go_to_goal(self.starting_position, self.starting_direction, 0.0):
                    # self.save_logs()
                    rospy.logerr("could not move back to drop, don't know how to recover, exiting...")
                    # sys.exit()
                    return False
            else:
                # self.save_logs()
                rospy.logwarn("arm didn't appear to move much before approaching state, shouldn't of happened when moving to drop...exiting")
                # sys.exit()
                return False

            # print ("start2: ", self.starting_position, " ", self.starting_direction)

            # if not self.go_to_goal(self.starting_position, self.starting_direction, 0.0):
            #     rospy.logerr("could not move back to drop, don't know how to recover, exiting...")
            #     sys.exit()

            self.goal = [None]

            # self.K_VQ = 0.5
            # goal = deepcopy(self.goal)
            # goal[0] -= 0.35
            # # to_goal = numpy.dot( self.ee_orientation.rotation_matrix, numpy.array([0.0, 0.0, 1.0]) )
            # print("starting position and direction: ")
            # print(self.starting_position, self.starting_direction)
            # self.go_to_goal(self.starting_position, self.starting_direction, 0.0)
            # self.go_to_goal(goal)

            # self.go_to_place()

            self.state = self.STATE.DROP
            # self.state = self.STATE.TO_NEXT

        elif self.state == self.STATE.DROP:

            rospy.loginfo("DROP")

            # if 0:
            if not self.sim:
                self.drop()

            self.state = self.STATE.TO_NEXT

        else:
            pass
        return True

    def list_trajectory_msg(self, traj):
        traj_msg = JointTrajectory()
        for positions in traj:
            point = JointTrajectoryPoint()
            point.positions = positions
            traj_msg.points.append(point)
        return traj_msg

    def remove_from_goal_array(self, goal):
        idx = numpy.linalg.norm(self.goal_array - goal, axis=1).argmin()
        # print (self.goal_array)
        # print (numpy.linalg.norm(self.goal_array - goal))
        del self.goal_array[idx]
        if self.sim:
            self.noise_array = numpy.delete(self.noise_array, idx, 0)
        # raw_input('press_enter')

    def sequence_goals(self):
        if not len(self.goal_array):
            self.goal = [None]
            return
        goals = numpy.array(deepcopy(self.goal_array))
        tasks_msg = PoseArray()
        for goal in goals:
            T = openravepy.transformLookat(goal + [0.1, 0.0, 0.0], goal - [self.starting_position_offset, 0.0, 0.0], [0, 0, -1])
            T = numpy.dot(T, numpy.linalg.inv(self.T_G2EE))
            pose = openravepy.poseFromMatrix(T)
            pose_msg = Pose()
            pose_msg.orientation.w = pose[0]
            pose_msg.orientation.x = pose[1]
            pose_msg.orientation.y = pose[2]
            pose_msg.orientation.z = pose[3]
            pose_msg.position.x = pose[4]
            pose_msg.position.y = pose[5]
            pose_msg.position.z = pose[6]
            tasks_msg.poses.append(pose_msg)
        resp = self.sequencer_client.call(tasks_msg, self.sequencing_metric)
        self.sequenced_goals = [goals[i] for i in resp.sequence]
        self.sequenced_trajectories = resp.database_trajectories
        self.num_goals_history = len(self.sequenced_goals)
        print("sequenced goals: " + str(self.sequenced_goals))
        print("resp.sequence: " + str(resp.sequence))
        if self.sim:
            self.sequenced_noise = [self.noise_array[i] for i in resp.sequence] # necessary so that the noise maps correctly no matter the sequence (only needed for logging)
            self.goal = numpy.array(self.sequenced_goals[0])
            self.goal_off = self.goal - self.go_to_goal_offset * self.normalize(self.goal - self.ee_position)
            self.noise = numpy.asarray(self.sequenced_noise[0])

            self.starting_position = self.goal - numpy.array([self.starting_position_offset, 0.0, 0.0]);
            self.starting_direction = numpy.array([1.0, 0.0, 0.0])

    def get_robot_ee_position(self, msg):

        # ee_orientation = pyquaternion.Quaternion(msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z)
        # ee_position = numpy.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        pose = [msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z]
        pose = openravepy.matrixFromPose(pose)
        ee_pose = numpy.dot(pose, self.T_G2EE)
        ee_pose = openravepy.poseFromMatrix(ee_pose)
        self.ee_orientation = pyquaternion.Quaternion(ee_pose[:4])
        self.ee_position = numpy.array(ee_pose[4:])


    # def get_joint_limits(self, msg):
        
        # self.lower_limit = numpy.array(msg.position_lower)
        # self.upper_limit = numpy.array(msg.position_upper)

        # self.robot.SetDOFLimits(self.joint_limits_lower, self.joint_limits_upper, self.robot.GetActiveManipulator().GetArmIndices())

    def get_robot_joints(self, msg):
        self.manipulator_joints = numpy.array(msg.position)

    def get_goal_array(self, msg):

        if not self.sim:
            goal_array = []
            goal_array_1D = msg.data
            if len(goal_array_1D):
                goal_array_1D = list(msg.data)
                for i in range(len(goal_array_1D) / 3):
                    goal_array.append(numpy.array(goal_array_1D[3 * i: 3 * i + 3]) - self.apple_offset)
            self.goal_array = goal_array
            rospy.loginfo_throttle(1, "goal_array: " + str(self.goal_array))

            if len(self.sequenced_goals):
                min_dist = numpy.inf
                current_goal = self.sequenced_goals[0]
                min_index = len(self.goal_array)
                for it, goal in enumerate(self.goal_array):
                    dist = sum([(x - y)**2 for x, y in zip(goal, current_goal)])
                    if dist < min_dist:
                        min_dist = dist
                        min_index = it

                if min_index != len(self.goal_array):
                    # offset = 0.08
                    self.goal = numpy.array(self.goal_array[min_index])
                    self.goal_off = self.goal - self.go_to_goal_offset * self.normalize(self.goal - self.ee_position)

                    self.starting_position = self.goal - numpy.array([self.starting_position_offset, 0.0, 0.0]);
                    self.starting_direction = numpy.array([1.0, 0.0, 0.0])
                rospy.loginfo_throttle(1, "goal: " + str(self.goal))

    def get_goal(self, msg, offset = 0.08):

        self.goal = numpy.array([msg.x, msg.y, msg.z])
        self.goal_off = self.goal - offset * self.normalize(self.goal - self.ee_position)

        self.starting_position = self.goal - numpy.array([self.starting_position_offset, 0.0, 0.0]);
        self.starting_direction = numpy.array([1.0, 0.0, 0.0])
        

    # def go_to_start(self):
        
    #     joints = self.arm.joint_angles()

    #     if ( numpy.linalg.norm(joints.values() - self.qstart ) < 4.0 ):
    #         cmd = dict(zip(joints.keys(), self.qstart))
    #         self.arm.move_to_joint_positions(cmd)
    #     else:
    #         # if too far, go up first
    #         rospy.logwarn("The starting position is too far, I'm going up first!")
            
    #         cmd = dict(zip(joints.keys(), self.CONFIG_UP))
    #         self.arm.move_to_joint_positions(cmd)
            
    #         time.sleep(1)
            
    #         cmd = dict(zip(joints.keys(), self.qstart))
    #         self.arm.move_to_joint_positions(cmd)
                           
    def stop_arm(self):
        joint_vel = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if self.sim:
            msg = JointTrajectoryPoint()
            msg.velocities = joint_vel
            self.sim_joint_velocities_pub.publish(msg)
        else:
            cmd = self.arm.joint_velocities()
            cmd = dict(zip(cmd.keys(), joint_vel[::-1]))
            self.arm.set_joint_velocities(cmd)

    def is_in_joint_limits(self):
        joints = copy(self.manipulator_joints)
        if not self.recovery_flag:
            if (joints <= self.joint_limits_lower).any() or (joints >= self.joint_limits_upper).any():
                rospy.loginfo("joints: " + str(joints))
                rospy.logwarn('joints are about to go outside of limits, exiting...')
                self.stop_arm()
                return False
            else:
                return True
        else:
            rospy.logwarn(str(rospy.get_time() - self.start_recovery_time))
            if rospy.get_time() - self.start_recovery_time < self.MAX_RECOVERY_DURATION:
                return True
            else:
                rospy.loginfo("joints: " + str(joints))
                rospy.logerr("took too long to recover, exiting...")
                self.stop_arm()
                return False
            # if (joints <= self.joint_limits_lower_recover).any() or (joints >= self.joint_limits_upper_recover).any():
            #     rospy.loginfo("joints: " + str(joints))
            #     rospy.logwarn('joints are about to go outside of recover limits, exiting...')
            #     self.stop_arm()
            #     return False
            # else:
            #     return True

    def is_greater_min_manipulability(self):
        #manipulability = self.computeManipulability()
        manipulability = self.computeReciprocalConditionNumber()
        if not self.recovery_flag:
            if (manipulability < self.MIN_MANIPULABILITY):
            # if 0:
                rospy.logwarn("detected low manipulability, exiting: " + str(manipulability))
                # singularity detected, send zero velocity to the robot and exit
                self.stop_arm()
                # sys.exit()
                return False
            else:
                return True
        else:
            if (manipulability < self.MIN_MANIPULABILITY_RECOVER):
            # if 0:
                rospy.logwarn("detected low manipulability in recovery, exiting: " + str(manipulability))
                # singularity detected, send zero velocity to the robot and exit
                self.stop_arm()
                # sys.exit()
                return False
            else:
                return True

        
    def go_to_goal(self, goal = [None], to_goal = [None], offset = 0.05):

        if goal[0] == None:
            # x_noise = numpy.random.normal(0.0, 0.03)
            # y_noise = numpy.random.normal(0.0, 0.05)
            # z_noise = numpy.random.normal(0.0, 0.05)
            goal = deepcopy(self.goal)
            self_goal = True
            rospy.loginfo("goal: " + str(goal))
            goal_off = deepcopy(self.goal_off)
            if self.sim:
                # update goal_off because ee_position changes
                goal_off = goal - offset * self.normalize(goal - self.ee_position)

        else:
            self_goal = False
            if to_goal[0] == None:
                goal_off = goal - offset * self.normalize(goal - self.ee_position)
            else:
                goal_off = goal - offset * self.normalize(to_goal)

        while numpy.linalg.norm(goal_off - self.ee_position) > 0.01 and not rospy.is_shutdown():
            rospy.loginfo_throttle(0.5, "goal_off: " + str(goal_off))
            # print("ee_position: " + str(self.ee_position))
            #print("ee_orientation: " + str(self.ee_orientation))
            if (self_goal):  # means servo'ing to a dynamic target
                goal = deepcopy(self.goal)
                goal_off = deepcopy(self.goal_off)
                if self.sim:
                    # update goal_off because ee_position changes
                    goal += self.noise
                    goal_off = goal - offset * self.normalize(goal - self.ee_position)
                    rospy.loginfo_throttle(0.5, "adding noise" + str(self.noise))
                    rospy.loginfo_throttle(0.5, "self.noise_array" + str(self.noise_array))
                self.recovery_trajectory.append(copy(self.manipulator_joints))

            rospy.loginfo_throttle(0.5, "ee distance from apple: " + str(numpy.linalg.norm(self.ee_position - goal)))
            rospy.loginfo_throttle(0.5, "[distance calc] ee_position: " + str(self.ee_position))
            rospy.loginfo_throttle(0.5, "[distance calc] goal: " + str(goal))

            if numpy.linalg.norm(self.ee_position - self.goal) < 0.3:  # 0.2m min range on sr300 and +0.1 to account for camera frame offset from EE
                rospy.loginfo_throttle(0.5, "disabling updating of apple position because too close")
                if not self.sim:
                    self.enable_bridge_pub.publish(Bool(False))
            else:
                if not self.sim:
                    self.enable_bridge_pub.publish(Bool(True))

            # check if joints are outside the limits
            # joints = self.arm.joint_angles()
            # joints = joints.values()[::-1]  # need to reverse because method's ordering is j6-j0

            if not self.is_in_joint_limits():
                return 1
                # sys.exit()


            des_vel_t = self.K_V * (goal_off - self.ee_position)

            if to_goal[0] != None:
                des_omega = - self.K_VQ * self.get_angular_velocity([None], to_goal)
            else:
                des_omega = - self.K_VQ * self.get_angular_velocity(goal)

            #print("omega: " + str(des_omega))

            #des_omega = numpy.array([0.0, 0.0, 0.0])

            des_vel = numpy.append(des_vel_t, des_omega)

            #print "goal: ", self.goal, " off: ", self.goal_off
            #print "des_vel: ", des_vel

            #singularity check
            if not self.is_greater_min_manipulability():
                return 2
            else:
                joint_vel = self.compute_joint_vel(des_vel)
                # print("joint_vel: " + str(joint_vel))
                if self.sim:
                    msg = JointTrajectoryPoint()
                    msg.velocities = joint_vel
                    self.sim_joint_velocities_pub.publish(msg)
                else:
                    cmd = self.arm.joint_velocities()
                    cmd = dict(zip(cmd.keys(), joint_vel[::-1]))
                    self.arm.set_joint_velocities(cmd)

            # rospy.sleep(0.1)

        self.stop_arm()
        return 0


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

        T_C = numpy.dot(T_EE, numpy.linalg.inv(self.T_G2EE))
        # T_C = numpy.dot(T_EE, numpy.linalg.inv(self.T_EE2C))

        goal_off_camera = T_C[:3, 3]
        
        plan_pose_msg = Pose()
        plan_pose_msg.position.x = goal_off_camera[0]
        plan_pose_msg.position.y = goal_off_camera[1]
        plan_pose_msg.position.z = goal_off_camera[2]
        plan_pose_msg.orientation.x = -0.5
        plan_pose_msg.orientation.y = 0.5
        plan_pose_msg.orientation.z = -0.5
        plan_pose_msg.orientation.w = 0.5
        if self.sequencing_metric == 'fredsmp':
            resp = self.optimise_offset_client(self.sequenced_trajectories[0], LOGGING)
        elif self.sequencing_metric == 'euclidean':
            resp = self.plan_pose_client(plan_pose_msg, ignore_trellis, LOGGING)

        if not resp.success:
            rospy.logwarn("planning to next target failed")
        else:
            # T = openravepy.transformLookat(self.sequenced_goals[0] + [0.1, 0.0, 0.0], self.sequenced_goals[0] - [self.starting_position_offset, 0.0, 0.0], [0, 0, -1])
            # T = numpy.dot(T, numpy.linalg.inv(self.T_G2EE))
            goal_off = self.sequenced_goals[0] - [self.starting_position_offset, 0.0, 0.0]
            # goal_off = T[:3, 3]
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

            # while numpy.linalg.norm(goal_off - self.ee_position) > 0.01 and not rospy.is_shutdown():
            #     print("waiting for arm to reach goal_off pose")
            #     print("goal_off_camera: " + str(goal_off_camera))
            #     print("goal_off: " + str(goal_off))
            #     print("self.ee_position: " + str(self.ee_position))
            #     rospy.sleep(0.1)
            #     pass 
            rospy.sleep(1.0)

        return resp.success, resp.plan_duration, resp.traj_duration

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

        # joints = self.arm.joint_angles()
        # joints = joints.values()
        joints = copy(self.manipulator_joints)

        with self.robot:
            # self.robot.SetDOFValues(joints[::-1], self.robot.GetActiveManipulator().GetArmIndices())
            self.robot.SetDOFValues(joints, self.robot.GetActiveManipulator().GetArmIndices())
            J_t = self.robot.GetActiveManipulator().CalculateJacobian()
            J_r = self.robot.GetActiveManipulator().CalculateAngularVelocityJacobian()
            J = numpy.concatenate((J_t, J_r), axis = 0)

        # add joint limit repulsive potential
        mid_joint_limit = (self.joint_limits_lower + self.joint_limits_upper) / 2.0
        Q_star = (self.joint_limits_upper - self.joint_limits_lower) / 20.0

        q_dot = numpy.zeros((7, 1))

        max_joint_speed = 1.0
        K = 1.0
        weight_vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        for i in range(7):
            #q_dot[i] = - (K * weight_vector[i]) * (joints[i] - mid_joint_limit[i]) / ( (self.joint_limits_upper[i] - self.joint_limits_lower[i])**2)
            if (abs(joints[i] - self.joint_limits_upper[i]) <= Q_star[i]):
               q_dot[i] = - (K * weight_vector[i] / (self.joint_limits_upper[i] - self.joint_limits_lower[i])) * ( 1.0/Q_star[i] - 1.0/abs(joints[i] - self.joint_limits_upper[i]) ) * (1.0/(joints[i] - self.joint_limits_upper[i])**2) * abs(joints[i] - self.joint_limits_upper[i]) / (joints[i] - self.joint_limits_upper[i])
            else:
                q_dot[i] = 0

            if (abs(joints[i] - self.joint_limits_lower[i]) <= Q_star[i]):
                q_dot[i] = q_dot[i] - (K * weight_vector[i] / (self.joint_limits_upper[i] - self.joint_limits_lower[i])) * ( 1.0/Q_star[i] - 1.0/abs(joints[i] - self.joint_limits_lower[i]) ) * (1.0/(joints[i] - self.joint_limits_lower[i])**2) * abs(joints[i] - self.joint_limits_lower[i]) / (joints[i] - self.joint_limits_lower[i])

            if (abs(q_dot[i]) > max_joint_speed):
                q_dot[i] = max_joint_speed * self.normalize(q_dot[i])


        # print(numpy.linalg.pinv(J).shape)
        # print(des_vel.shape)
        # print(q_dot.shape)
        # print("joints: " + str(self.manipulator_joints))
        # print ("q: " + str(numpy.dot( numpy.linalg.pinv(J), des_vel.reshape(6,1))))
        # print ("q_dot: " + str(q_dot))
        # print ("qdot_proj: " + str(numpy.dot( (numpy.eye(7) - numpy.dot( numpy.linalg.pinv(J) , J )), q_dot)))
        return numpy.dot( numpy.linalg.pinv(J), des_vel.reshape(6,1)) + numpy.dot( (numpy.eye(7) - numpy.dot( numpy.linalg.pinv(J) , J )), q_dot)
        #return numpy.dot( (numpy.eye(7) - numpy.dot( numpy.linalg.pinv(J) , J )), q_dot)

        # return numpy.dot( numpy.linalg.pinv(J), des_vel.reshape(6,1))

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

        self.stop_arm()

        qstart_param_name = rospy.search_param('qstart')

        if qstart_param_name != None:
            rospy.delete_param(qstart_param_name)
