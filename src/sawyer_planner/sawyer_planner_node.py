#!/usr/bin/env python

import time
import enum
import openravepy
import numpy
import numpy as np
import os
from openravepy.misc import InitOpenRAVELogging
from copy import *
import json
import math

import rospy
import rospkg
from geometry_msgs.msg import Point, Quaternion, PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, PointCloud2
from trajectory_msgs.msg import JointTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger, Empty
from sawyer_planner.srv import CheckBranchOrientation, SetGoal
from online_planner.srv import *
from task_planner.srv import *
from localisation.srv import *
from rgb_segmentation.srv import *
from tf2_ros import TransformListener as TransformListener2, Buffer
from tf2_geometry_msgs import do_transform_point
from collections import defaultdict
from pandas import DataFrame

import pyquaternion
import socket

LOGGING = True
HOME_POSE = True

class SawyerPlanner:

    def __init__(self, metric, goals, sim=True, starting_joints=None, robot_name=None):

        """
        Main planner which is tasked with moving to the goals.
        """

        self.robot_name = robot_name
        if self.robot_name is None:
            self.robot_name = rospy.get_param('robot_name')

        print self.robot_name
        self.STATE = enum.Enum('STATE', 'SEARCH TO_NEXT APPROACH GRAB CHECK_GRASPING TO_DROP DROP RECOVER')
        self.sim = sim

        self.goals = goals

        self.current_goal = [None]          # The coordinates of the point being targeted - can be volatile!
        self.current_iteration = -1
        self.goal_point_camera_pose = []    # Records the pose originally used for goal - ONLY FOR SIMULATION

        self.sequence = []

        self.manipulator_joints = []
        self.ee_position = None
        self.ee_pose = []
        self.apple_offset = [0.0, 0.0, 0.0]
        self.go_to_goal_offset = 0.25  # How far should the EE be from the target for the approach position?
        self.limits_epsilon = 0.01
        self.K_V = 0.3
        self.K_VQ = 2.5
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
        self.environment_setup()
        self.robot_arm_indices = self.robot.GetActiveManipulator().GetArmIndices()
        self.num_joints = len(self.robot_arm_indices)

        # Segmentation logic
        self.use_camera = rospy.get_param('/use_camera', False)
        self.target_color = rospy.get_param('/target_color', False)
        self.rgb_seg = self.use_camera and bool(self.target_color)

        if self.robot_name == "sawyer":
            self.T_G2EE = numpy.array([
                        [0.9961947, 0.0871557, 0.0, 0.0],
                        [-0.0871557, 0.9961947, 0.0, 0.075],
                        [0.0, 0.0, 1.0, -0.12],
                        [0.0, 0.0, 0.0, 1.0]
                        ])
        else:
            self.T_G2EE = numpy.array([
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -0.12],
                        [0.0, 0.0, 0.0, 1.0]
                        ])

        self.enable_bridge_pub = rospy.Publisher("/sawyer_planner/enable_bridge", Bool, queue_size = 1)
        self.sim_joint_velocities_pub = rospy.Publisher('sim_joint_velocities', JointTrajectoryPoint, queue_size=1)

        self.gripper_client = rospy.ServiceProxy('/gripper_action', SetBool)
        self.start_pipeline_client = rospy.ServiceProxy("/sawyer_planner/start_pipeline", Trigger)
        self.plan_pose_client = rospy.ServiceProxy("/plan_pose_srv", PlanPose)
        self.plan_joints_client = rospy.ServiceProxy("/plan_joints_srv", PlanJoints)
        self.optimise_offset_client = rospy.ServiceProxy("/optimise_offset_srv", OptimiseTrajectory)
        self.optimise_trajectory_client = rospy.ServiceProxy("/optimise_trajectory_srv", OptimiseTrajectory)
        self.sequencer_client = rospy.ServiceProxy("/sequence_tasks_srv", SequenceTasks)
        self.set_robot_joints = rospy.ServiceProxy('set_robot_joints', SetRobotJoints)
        self.find_ik_solutions_srv = rospy.ServiceProxy('find_ik_solutions', FindIKSolutions)
        self.draw_point_pub = rospy.Publisher('draw_point', Point, queue_size=1)
        self.draw_branch_srv = rospy.ServiceProxy('draw_branch', DrawBranch)
        self.clear_point_srv = rospy.ServiceProxy('clear_point', Empty)
        self.check_ray_srv = rospy.ServiceProxy('test_check_ray', CheckRay)
        self.cut_point_srv = rospy.ServiceProxy('cut_point_srv', GetCutPoint)
        self.get_orientation_srv = rospy.ServiceProxy('branch_orientation', CheckBranchOrientation)

        self.load_octomap = rospy.ServiceProxy('/load_octomap', LoadOctomap)
        self.update_octomap = rospy.ServiceProxy('/update_octomap', Empty)
        self.update_octomap_filtered = rospy.ServiceProxy('/update_octomap_filtered', Empty)

        if self.use_camera and not self.target_color:
            self.activate_point_tracker = rospy.ServiceProxy('/activate_point_tracker', Empty)
            self.deactivate_point_tracker = rospy.ServiceProxy('/deactivate_point_tracker', Empty)
            self.point_tracker_set_goal = rospy.ServiceProxy('point_tracker_set_goal', SetGoal)
        else:
            self.activate_point_tracker = dummy_function
            self.deactivate_point_tracker = dummy_function
            self.point_tracker_set_goal = dummy_function
        
        #self.enable_bridge_pub.publish(Bool(True))
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener2(self.tf_buffer)

        rospy.Subscriber('manipulator_pose', PoseStamped, self.get_robot_ee_position, queue_size = 1)
        rospy.Subscriber('manipulator_joints', JointState, self.get_robot_joints, queue_size = 1)
        or_joint_states = rospy.wait_for_message('manipulator_joints', JointState)
        or_joints_pos = numpy.array(or_joint_states.position)

        self.initial_joints = rospy.wait_for_message('manipulator_joints', JointState)
        self.camera_offset_matrix = self.get_transform_matrix('manipulator', 'camera_color_optical_frame')

        # Get the offset between the manipulator target and the point that actually does the cutting
        tf = self.retrieve_tf('manipulator', 'cutpoint')
        trans = tf.transform.translation
        self.manipulator_cutpoint_offset = PointStamped()
        self.manipulator_cutpoint_offset.header.frame_id = 'manipulator'
        self.manipulator_cutpoint_offset.point = Point(trans.x, trans.y, trans.z)

        rospy.Subscriber('/update_goal_point', Point, self.update_goal, queue_size=1)

        self.stop_update_threshold = rospy.get_param('/stop_update_threshold', 0.03)

        if self.sim:
            self.stop_arm()

            if HOME_POSE:
                if starting_joints is not None:
                    joint_msg = JointState()
                    joint_msg.position = starting_joints
                    self.set_robot_joints(joint_msg)
                else:
                    self.set_home_client = rospy.ServiceProxy('set_home_position', Empty)
                    self.set_home_client.call()
                rospy.sleep(1.0)
        else:
            if rospy.get_param('/robot_name') == "sawyer":
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
            elif rospy.get_param('/robot_name').startswith("ur5") or rospy.get_param('/robot_name') == "ur10":
                import socket
                HOST = rospy.get_param('/robot_ip')   # The remote host
                PORT = rospy.get_param('/robot_socket', 30002)              # The same port as used by the server
                self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.s.connect((HOST, PORT))
                current_joints_pos = copy(self.manipulator_joints)
            if numpy.linalg.norm(or_joints_pos - current_joints_pos) > 0.1:
                rospy.logerr("it seems like you should be running the node in sim mode, exiting.")
                sys.exit()


        self.logger = Logger()
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
        self.results_dict = {'Sequencing Time':[], 'Planner Time':[], 'Planner Computation Time':[], 'Planner Execution Time':[], 'Approach Time':[], 'Num Apples':0}
        self.failures_dict = {'Joint Limits':[], 'Low Manip':[], 'Planner': [], 'Grasp Misalignment':[], 'Grasp Obstructed':[]}

        self.state = self.STATE.SEARCH

    @property
    def current_sequence_index(self):
        return self.sequence[self.current_iteration]

    @property
    def current_goal_array(self):
        return point_as_array(self.goals[self.current_sequence_index].goal)

    @property
    def current_orientation_ref_array(self):
        return point_as_array(self.goals[self.current_sequence_index].orientation)


    def retrieve_tf(self, base_frame, target_frame, stamp = rospy.Time()):

        # Retrieves a TransformStamped with a child frame of base_frame and a target frame of target_frame
        # This transform can be applied to a point in the base frame to transform it to the target frame

        success = self.tf_buffer.can_transform(target_frame, base_frame, stamp, rospy.Duration(0.5))
        if not success:
            rospy.logerr("Couldn't look up transform between {} and {}!".format(target_frame, base_frame))

        tf = self.tf_buffer.lookup_transform(target_frame, base_frame, stamp)
        return tf


    def go_to_start(self):

        self.plan_joints_client(self.initial_joints, False, True)

    def environment_setup(self):

        self.env = openravepy.Environment()
        InitOpenRAVELogging()
        module = openravepy.RaveCreateModule(self.env, 'urdf')
        rospack = rospkg.RosPack()
        with self.env:
            # name = module.SendCommand('load /home/peppe/python_test/sawyer.urdf /home/peppe/python_test/sawyer_base.srdf')
            if self.robot_name == "sawyer":
                name = module.SendCommand(
                    'loadURI ' + rospack.get_path('fredsmp_utils') + '/robots/sawyer/sawyer.urdf'
                    + ' ' + rospack.get_path('fredsmp_utils') + '/robots/sawyer/sawyer.srdf')
            elif self.robot_name == "ur10":
                name = module.SendCommand(
                    'loadURI ' + rospack.get_path('fredsmp_utils') + '/robots/ur10/ur10.urdf'
                    + ' ' + rospack.get_path('fredsmp_utils') + '/robots/ur10/ur10.srdf')               
            elif self.robot_name == "ur5":
                name = module.SendCommand(
                    'loadURI ' + rospack.get_path('fredsmp_utils') + '/robots/ur5/ur5.urdf'
                    + ' ' + rospack.get_path('fredsmp_utils') + '/robots/ur5/ur5.srdf')                
            elif self.robot_name == "ur5_cutter":
                name = module.SendCommand(
                    'loadURI ' + rospack.get_path('fredsmp_utils') + '/robots/ur5/ur5_cutter.urdf'
                    + ' ' + rospack.get_path('fredsmp_utils') + '/robots/ur5/ur5_cutter.srdf')
            elif self.robot_name == "ur5_cutter_test":
                name = module.SendCommand(
                    'loadURI ' + rospack.get_path('fredsmp_utils') + '/robots/ur5/ur5_cutter_test.urdf'
                    + ' ' + rospack.get_path('fredsmp_utils') + '/robots/ur5/ur5_cutter_test.srdf')
            else:
                rospy.logerr("invalid robot name, exiting...")
                sys.exit()              
            self.robot = self.env.GetRobot(name)

        time.sleep(0.5)

        self.ikmodel = openravepy.databases.inversekinematics.InverseKinematicsModel(self.robot,iktype=openravepy.IkParameterization.Type.Transform6D)
        if not self.ikmodel.load():
            self.ikmodel.autogenerate()


        manip = self.robot.GetActiveManipulator()
        self.robot.SetActiveDOFs(manip.GetArmIndices())
        if 1:
        # if self.robot_name == "ur10" or self.robot_name.startswith("ur5"):
            self.joint_limits_lower = self.robot.GetActiveDOFLimits()[0] + self.limits_epsilon
            self.joint_limits_upper = self.robot.GetActiveDOFLimits()[1] - self.limits_epsilon
            print "limits_lower:"
            print self.joint_limits_lower
            print "limits_upper:"
            print self.joint_limits_upper

    # def socket_handshake(self, *args):
    #     self.socket_handler.send("\n")

    def transform_point(self, point, tf_matrix=None, reverse=False):
        """
        Transforms a point from a source frame to a target frame as indicated by the tf_matrix.
        If tf_matrix is None, the current camera frame will be used.
        If reverse is True, will perform the reverse operation. Note that this causes the inverse to be repeatedly
        calculated, in which it may be better to pre-invert the matrix.
        """

        if tf_matrix is None:
            tf_matrix = self.get_camera_pose(reverse=reverse)

        if isinstance(point, Point):
            point = [point.x, point.y, point.z]

        pose_mat = np.identity(4)
        pose_mat[:3, 3] = point
        pose_mat_world = numpy.dot(tf_matrix, pose_mat)
        return pose_mat_world[:3, 3]

    def get_transform_matrix(self, base_frame, target_frame, t = None):

        """
        Retrieves transform from base_frame to target_frame.
        Note that this means that applying this matrix to a point will transform it from target_frame to base_frame
        """
        if t is None:
            t = rospy.Time(0)
        tf = self.retrieve_tf(target_frame, base_frame, t)
        trans = tf.transform.translation
        rot = tf.transform.rotation

        or_pose = [rot.w, rot.x, rot.y, rot.z, trans.x, trans.y, trans.z]
        return openravepy.matrixFromPose(or_pose)


    def get_camera_pose(self, ee_pose=None, reverse=False):
        """
        Retrieves the pose matrix of the camera which can be used to transform points in the frame of the camera
        to the world frame.
        If reverse is set to True, will return the pose matrix of the world relative to the camera, which can be used
        to transform world frame points to the camera frame.
        """

        if ee_pose is None:
            ee_pose = self.ee_pose

        camera_pose = numpy.dot(ee_pose, self.camera_offset_matrix)
        if reverse:
            return np.linalg.inv(camera_pose)
        else:
            return camera_pose


    def rgb_segment_goal(self, pose_array = None, ee_pose = None, retrieve_cloud = False):

        cut_points = []
        cut_points_msg = None
        rospy.set_param('segment_color', self.target_color)

        # Use RGB segmentation to retrieve the camera-frame points to cut
        if pose_array is None:

            cut_points_msg = self.cut_point_srv.call(Bool(retrieve_cloud))

            rospy.loginfo('{} points of interest found!'.format(len(cut_points_msg.cut_points.poses)))
            pose_array = cut_points_msg.cut_points
            pose_array = self.refine_points(pose_array)
            if len(pose_array.poses) != len(cut_points_msg.cut_points.poses):
                rospy.loginfo('(Condensed to {} points)'.format(len(pose_array.poses)))

        # Transform the camera-frame points to the global frame
        if ee_pose is None:
            ee_pose = self.ee_pose
        camera_pose = self.get_camera_pose(ee_pose=ee_pose)
        for cut_pose in pose_array.poses:
            transformed_point = self.transform_point(cut_pose.position, tf_matrix=camera_pose)
            cut_points.append(transformed_point)

        return cut_points, camera_pose, cut_points_msg

    # def rgb_segment_set_goal(self):
    #
    #     # Retrieve the cut points as well as the cloud info, necessary for orientation retrieval
    #     cut_points, camera_pose, srv_response = self.rgb_segment_goal(retrieve_cloud=True)
    #     self.goal_array = cut_points
    #     self.goal_point_camera_pose = camera_pose
    #     cloud = srv_response.pointcloud
    #
    #     inverse_camera_pose = np.linalg.inv(camera_pose)
    #     orientation_markers = []
    #
    #     for point in cut_points:
    #
    #         # Global points must be transformed back into camera frame to be processed by cloud
    #         point_camera_frame = self.transform_point(point, tf_matrix=inverse_camera_pose)
    #         res = self.get_orientation_srv(Point(*point_camera_frame), cloud)
    #         orientation_markers.append(self.transform_point(res.point_1, tf_matrix=camera_pose))
    #
    #     self.orientation_array = orientation_markers
    #

    def get_cutter_goal_orientation(self, goal=None, reference=None, camera_inverse=None):

        if camera_inverse is None:
            camera_inverse = self.get_camera_pose(reverse=True)

        # Convert the goal and the reference point to the camera frame and use their camera-frame XY projection
        # to pick the angle

        if goal is None:
            goal = self.current_goal_array
        if reference is None:
            reference = self.current_orientation_ref_array

        goal = self.transform_point(goal, tf_matrix=camera_inverse)
        reference = self.transform_point(reference, tf_matrix=camera_inverse)

        diff = goal - reference

        branch_angle = np.arctan(diff[1] / diff[0])
        # The cutter frame is already rotated 90 degrees relative to the camera frame and as such there's no need to
        # add a 90 degree (e.g. measuring a horizontal branch is 0 degrees, which corresponds with a vertical cutter
        # orientation

        target = (branch_angle + np.pi / 2) % np.pi - np.pi / 2
        return target

    def refine_points(self, pose_array):
        """
        Clusters detected points with some given threshold together,
        and condenses them into a point representing the centroid.
        """

        refine_threshold = rospy.get_param('segment_refine_threshold', 0.03)

        new_pose_array = PoseArray()
        positions = np.array([[p.position.x, p.position.y, p.position.z] for p in pose_array.poses])

        all_rows = set(range(positions.shape[0]))
        assigned = set()

        for row_index in all_rows:

            if row_index in assigned:
                continue

            cluster_points = []

            frontier = [row_index]
            while frontier:
                cluster_points.extend(frontier)
                assigned.update(frontier)
                new_frontier = []
                for base_index in frontier:
                    for compare_index in all_rows.difference(assigned):
                        if compare_index == row_index:
                            continue

                        # Compare two row magnitudes, see if they're in refine threshold
                        # If they are, add to the new frontier
                        if np.sqrt(np.sum((positions[base_index,:] - positions[compare_index,:]) ** 2)) < refine_threshold:
                            new_frontier.append(compare_index)

                frontier = new_frontier

            centroid = positions[cluster_points, :].mean(axis=0)

            new_pose = Pose()
            new_pose.position.x = centroid[0]
            new_pose.position.y = centroid[1]
            new_pose.position.z = centroid[2]

            new_pose_array.poses.append(new_pose)

        return new_pose_array

    def update_goal(self, pt):

        if self.state != self.STATE.APPROACH:
            return

        self.current_goal = np.array([pt.x, pt.y, pt.z])
        rospy.loginfo_throttle(2, 'Current goal: {:.3f}, {:.3f}, {:.3f}'.format(*self.current_goal))

    def get_pose_ik(self, pose, joints=None):

        if joints is None:
            joints = self.manipulator_joints
        configs = self.ikmodel.manip.FindIKSolutions(pose, openravepy.IkFilterOptions.CheckEnvCollisions)
        if not configs.size:
            return None
        min_index = np.argmin(np.abs(configs - joints).sum(axis=1))

        return configs[min_index,:]


    def computeReciprocalConditionNumber(self, pose=None):

        if pose is None:
            pose = copy(self.manipulator_joints)

        # TODO: I suspect this is not setting the joints correctly as the robot is still driving itself into bad positions

        with self.robot:
            # self.robot.SetDOFValues(current_joints_pos[::-1], self.robot.GetActiveManipulator().GetArmIndices())
            self.robot.SetDOFValues(pose, self.robot.GetActiveManipulator().GetArmIndices())
            J_t = self.robot.GetActiveManipulator().CalculateJacobian()
            J_r = self.robot.GetActiveManipulator().CalculateAngularVelocityJacobian()
            J = numpy.concatenate((J_t, J_r), axis = 0)

        u, s, v = numpy.linalg.svd(J, full_matrices = False) # here you can try to use just J_t instead of J

        assert numpy.allclose(J, numpy.dot(u, numpy.dot(numpy.diag(s), v) ))

        # return numpy.prod(s)  # Old manipulability measure
        return numpy.min(s)/numpy.max(s)

    def remove_current_apple(self):
        # Is there really a need to remove the current target? Just advance the sequencer
        pass


    def save_logs(self):
        if LOGGING:

            file_name = 'results_{}_{}'.format(self.sequencing_metric, time.strftime("%Y%m%d-%H%M%S"))
            file_path = os.path.join(self.directory, file_name)

            df = self.logger.output_df()
            df.to_pickle(file_path + '.pickle')
            df.to_csv(file_path + '.csv')

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

            self.logger.start_timer('octomap_time')
            rospy.loginfo("SEARCH")
            self.refresh_octomap()
            self.logger.end_timer('octomap_time')

            # Sequence the goals here

            self.logger.start_timer('sequencing_time')
            self.sequence_goals()
            elapsed = self.logger.end_timer('sequencing_time')

            self.results_dict['Sequencing Time'].append(elapsed)

            self.state = self.STATE.TO_NEXT
        
        elif self.state == self.STATE.TO_NEXT:

            rospy.loginfo("TO_NEXT")
            self.current_iteration += 1
            self.logger.increment_iteration()

            rospy.loginfo("current apple ind: " + str(self.current_iteration))

            # rospy.sleep(1.0)  # TODO: avoid exiting before subscriber updates new apple goal, 666 should replace with something more elegant

            if self.current_iteration >= len(self.sequence):
                self.save_logs()
                rospy.logerr("There are no apples to pick!")
                return False
            else:
                self.current_goal = self.current_goal_array

            # Draw the branch and cut point onto the OpenRAVE simulation
            self.results_dict['Num Apples'] += 1
            self.failures_dict['Grasp Misalignment'].append(0)
            self.failures_dict['Grasp Obstructed'].append(0)

            self.openrave_draw_branch()

            plan_success, plan_duration, traj_duration = self.plan_to_goal()

            self.logger['planning_time'] = plan_duration
            self.logger['traj_time'] = traj_duration
            self.logger[['gx', 'gy', 'gz']] = self.current_goal_array

            rospy.logwarn("TRAJ DURATION: " + str(traj_duration))
            if plan_success:
                self.state = self.STATE.APPROACH
                # self.state = self.STATE.TO_NEXT
                # self.remove_current_apple()
                # self.results_dict['Planner Time'].append(elapsed_time - plan_duration)
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

            # When running on a real arm, a new position means new occlusions will be present!

            if not self.sim:
                self.enable_bridge_pub.publish(Bool(True)) 

            print("apple is there, going to grab")

            self.recovery_trajectory = []

            self.logger.start_timer('servoing_time')
            try:
                status = self.servo_to_goal()
            finally:
                self.deactivate_point_tracker()
                self.stop_arm()
                rospy.set_param('/going_to_goal', False)
            elapsed_time = self.logger.end_timer('servoing_time')

            final_goal = deepcopy(self.ee_position)
            deviation = np.linalg.norm(final_goal - self.current_goal_array)
            rospy.loginfo('Final approach position was {:.1f} cm off from the original point'.format(deviation))

            self.logger['status'] = status
            self.logger.set_value(['fgx', 'fgy', 'fgz'], final_goal)


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
            try:

                # We want to move the manipulator so that the branch is in the cut point
                # Do this by taking the offset between the endpoint and cutter, treating it as
                # a point in the manipulator's frame of view, and adjusting the world goal accordingly

                tf = self.retrieve_tf('manipulator', 'base_link')
                pt = do_transform_point(self.manipulator_cutpoint_offset, tf)
                self.current_goal = point_as_array(pt)

                self.logger.start_timer('cutting_time')
                self.servo_to_goal(rotate=False, use_visual=False)
                self.logger.end_timer('cutting_time')

            finally:
                self.stop_arm()

            #self.enable_bridge_pub.publish(Bool(False))

            self.state = self.STATE.CHECK_GRASPING

        elif self.state == self.STATE.CHECK_GRASPING:

            rospy.loginfo("CHECK_GRASPING")

            self.remove_current_apple()

            print("apple successfully picked, going to drop")
            self.state = self.STATE.TO_DROP
            rospy.sleep(1.0)
            if not self.sim:
                self.enable_bridge_pub.publish(Bool(False)) 

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

                self.logger.start_timer('recovery_time')
                resp = self.optimise_trajectory_client.call(traj_msg, LOGGING)
                self.logger.end_timer('recovery_time')
                if not resp.success:
                    rospy.logerr("could not move back to drop, don't know how to recover, exiting...")
                    return False
            else:
                rospy.logwarn("arm didn't appear to move much before recovery state, continuing without recovery procedure")

            self.current_goal = [None]

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
                self.logger.start_timer('recovery_time')
                resp = self.optimise_trajectory_client.call(traj_msg, LOGGING)
                self.logger.end_timer('recovery_time')

                self.stop_arm()
                if not resp.success:
                    rospy.logerr("could not move back to drop, don't know how to recover, exiting...")
                    return False
            else:
                # self.save_logs()
                rospy.logwarn("arm didn't appear to move much before approaching state, shouldn't of happened when moving to drop...exiting")
                # sys.exit()
                return False

            self.current_goal = [None]
            self.state = self.STATE.DROP

        elif self.state == self.STATE.DROP:

            rospy.loginfo("DROP")

            if 0:
            # if not self.sim:
                self.drop()

            self.state = self.STATE.TO_NEXT

        else:
            pass
        return True

    def openrave_draw_branch(self):

        goal = self.current_goal_array
        branch_orientation_reference = self.current_orientation_ref_array

        vec = branch_orientation_reference - goal
        vec = vec / np.sqrt((vec ** 2).sum())

        start = Point(*goal - 0.5 * vec)
        end = Point(*goal + 0.5 * vec)

        draw_point_msg = Point(self.current_goal[0], self.current_goal[1], self.current_goal[2])
        self.draw_branch_srv(start, end)
        self.draw_point_pub.publish(draw_point_msg)


    def pose_to_ros_msg(self, pose):
        pose_msg = Pose()
        pose_msg.orientation.w = pose[0]
        pose_msg.orientation.x = pose[1]
        pose_msg.orientation.y = pose[2]
        pose_msg.orientation.z = pose[3]
        pose_msg.position.x = pose[4]
        pose_msg.position.y = pose[5]
        pose_msg.position.z = pose[6]
        return pose_msg

    def list_trajectory_msg(self, traj):
        traj_msg = JointTrajectory()
        for positions in traj:
            point = JointTrajectoryPoint()
            point.positions = positions
            traj_msg.points.append(point)
        return traj_msg

    def sequence_goals(self):

        if not len(self.goals):
            self.current_goal = [None]
            return
        if not self.sequence:
            tasks_msg = PoseArray()

            for planner_goal in self.goals:
                goal = point_as_array(planner_goal.goal)
                orientation_reference = point_as_array(planner_goal.orientation)

                angle = self.generate_feasible_approach_angles(samples=1, goal=goal, orientation_reference=orientation_reference)
                _, pose, _ = self.get_goal_approach_pose(goal, self.go_to_goal_offset, angle, orientation_reference)
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
            if not len(resp.sequence):
                raise Exception('The sequencer failed to return a proper sequence!')

            if len(resp.sequence) != len(self.goals):
                rospy.logwarn('The sequencer only returned {} out of {} goal points!'.format(len(resp.sequence), len(self.goals)))

            self.sequence = resp.sequence
            print("resp.sequence: " + str(resp.sequence))
        else:
            rospy.logwarn("Called sequence again, but there is already an existing sequence. Doing nothing...")


    def get_robot_ee_position(self, msg, return_values=False):

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
        ee_pose = openravepy.poseFromMatrix(pose)
        orientation = pyquaternion.Quaternion(ee_pose[:4])
        position = numpy.array(ee_pose[4:])

        if not return_values:
            self.ee_pose = copy(pose)
            self.ee_orientation = orientation
            self.ee_position = position
        else:
            return pose, orientation, position

    # def get_joint_limits(self, msg):
        
        # self.lower_limit = numpy.array(msg.position_lower)
        # self.upper_limit = numpy.array(msg.position_upper)

        # self.robot.SetDOFLimits(self.joint_limits_lower, self.joint_limits_upper, self.robot.GetActiveManipulator().GetArmIndices())

    def get_robot_joints(self, msg):
        self.manipulator_joints = numpy.array(msg.position)
                           
    def stop_arm(self):
        # joint_vel = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        joint_vel = numpy.zeros(self.num_joints)
        if self.sim:
            msg = JointTrajectoryPoint()
            msg.velocities = joint_vel
            self.sim_joint_velocities_pub.publish(msg)
        else:
            if self.robot_name == "sawyer":
                cmd = self.arm.joint_velocities()
                cmd = dict(zip(cmd.keys(), joint_vel[::-1]))
                self.arm.set_joint_velocities(cmd)
            elif self.robot_name.startswith("ur5") or self.robot_name == "ur10":
                self.s.send("stopj(2)" + "\n")

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


    def servo_to_goal(self, rotate=True, use_visual=True):

        goal = deepcopy(self.current_goal)

        rospy.loginfo("Servoing to goal: {:.3f}, {:.3f}, {:.3f}".format(*goal))

        if use_visual:

            # TODO: A bit hacky, service requires PlannerGoal point, but self.current_goal can be numpy array which differs from PlannerGoal
            self.point_tracker_set_goal(self.goals[self.current_sequence_index])
            self.activate_point_tracker()

        # Computes position of arm relative to goal so that we can know when we've moved past goal
        starting_ee_goal_vector = (np.array(goal) - self.ee_position)[:2]

        # Main loop which servos towards the goal
        while not rospy.is_shutdown():

            goal = deepcopy(self.current_goal)      # self.goal is updated based off Subscriber
            ee_goal_vector = (np.array(goal) - self.ee_position)[:2]

            # Termination condition
            if numpy.linalg.norm(goal - self.ee_position) < 0.001 or np.dot(ee_goal_vector, starting_ee_goal_vector) < 0:
                break

            self.recovery_trajectory.append(copy(self.manipulator_joints))

            draw_point_msg = Point(*goal)
            self.draw_point_pub.publish(draw_point_msg)

            rospy.loginfo_throttle(0.5, "ee distance from apple: " + str(numpy.linalg.norm(self.ee_position - goal)))

            if numpy.linalg.norm(self.ee_position - self.current_goal) < self.stop_update_threshold:
                # rospy.loginfo_throttle(1.0, "disabling updating of apple position because too close")
                rotate = False
                self.deactivate_point_tracker()
                rospy.loginfo_throttle(5, 'Point tracking disabled!')
                if not self.sim:
                    self.enable_bridge_pub.publish(Bool(False))
            else:
                if not self.sim:
                    self.enable_bridge_pub.publish(Bool(True))

            if not self.is_in_joint_limits():
                return 1

            # Computes the velocity of linear movement
            MIN_VEL = 0.015
            des_vel_t = self.K_V * (goal - self.ee_position)
            total_vel = np.linalg.norm(des_vel_t)
            if total_vel < MIN_VEL:
                des_vel_t = des_vel_t * (MIN_VEL / total_vel)

            if not rotate:
                des_omega = np.zeros(3)
            else:
                des_omega = - self.K_VQ * self.get_angular_velocity(goal)

            des_vel = numpy.append(des_vel_t, des_omega)

            #singularity check
            if not self.is_greater_min_manipulability():
                return 2
            else:
                joint_vel = self.compute_joint_vel(des_vel)
                self.send_velocity_command(joint_vel)

        self.deactivate_point_tracker()
        self.stop_arm()

        if self.sim:
            self.clear_point_srv()
        return 0


    def send_velocity_command(self, joint_vel):
        if self.sim:
            msg = JointTrajectoryPoint()
            msg.velocities = joint_vel
            self.sim_joint_velocities_pub.publish(msg)
        else:
            if self.robot_name == "sawyer":
                cmd = self.arm.joint_velocities()
                cmd = dict(zip(cmd.keys(), joint_vel[::-1]))
                self.arm.set_joint_velocities(cmd)
            elif self.robot_name.startswith("ur5") or self.robot_name == "ur10":
                cmd = str("speedj([%.5f,%.5f,%.5f,%.5f,%.5f,%.5f],5.0,0.05)" % tuple(joint_vel)) + "\n"
                # print ("CMD: " + cmd)
                self.s.send(cmd)


    def get_goal_approach_pose(self, goal, offset, angle, orientation_reference=None):

        if orientation_reference is not None:

            _, _, base_pose_mat = self.get_goal_approach_pose(goal, offset, angle, None)
            assumed_camera_inverse = self.get_camera_pose(ee_pose=base_pose_mat, reverse=True)
            target = self.get_cutter_goal_orientation(goal, orientation_reference, assumed_camera_inverse)

            # Up axis should be defined in world frame
            # Thus we transform a vector in the camera's XY frame to one in the world frame
            camera_frame_up_axis = [-np.sin(target), np.cos(target), 0, 0]  # Last 0 is dummy for matrix mult
            up_axis = np.linalg.inv(assumed_camera_inverse).dot(camera_frame_up_axis)[:3]

        else:
            up_axis = [0, 0, -1]

        goal = np.array(goal)
        view_offset = np.array([offset * math.cos(angle), offset * math.sin(angle), 0.0])

        goal_off_pose_mat = openravepy.transformLookat(goal, goal + view_offset, up_axis)
        goal_off_pose = openravepy.poseFromMatrix(goal_off_pose_mat)

        tool_position = goal_off_pose[4:]
        tool_pose = goal_off_pose

        return tool_position, tool_pose, goal_off_pose_mat

    def evaluate_goal_approach_manipulability(self, angle, offset):

        goal = self.current_goal_array
        reference = self.current_orientation_ref_array

        _, pose, _ = self.get_goal_approach_pose(goal, offset, angle, reference)
        joints = self.get_pose_ik(pose)
        if joints is None:
            return -1
        manip = self.computeReciprocalConditionNumber(joints)
        return manip

    def generate_feasible_approach_angles(self, angle_from_center=np.pi/4, samples=9, goal=None, orientation_reference=None):

        # First, check if the branch looks sufficiently vertical, subject to some threshold.
        # If so, any approach angle should be fine, so we choose ones in line with the base

        VERTICAL_ANGLE_REF = math.radians(15.0)

        if goal is None:
            goal = self.current_goal_array

        if orientation_reference is None:
            ref = self.current_orientation_ref_array
        else:
            ref = orientation_reference


        goal_xy = goal[:2]
        ref_xy = ref[:2]

        diff = goal - ref
        normalized_diff = diff / np.linalg.norm(diff)

        if np.abs(normalized_diff[2]) > math.cos(VERTICAL_ANGLE_REF):
            rospy.loginfo('This branch looks vertical, choosing angles appropriately...')
            perp_vec = goal_xy

        else:
            diff_vec = goal_xy - ref_xy
            perp_vec = np.array([-diff_vec[1], diff_vec[0]])

            # Orient the perpendicular vector so that it faces outwards from the robot
            # This is used so we don't try to cut branches from the other side of the branch
            if perp_vec.dot(goal_xy) < 0:
                perp_vec = -perp_vec

        # Base angle is the perpendicular approach vector in the world frame
        # Note that the vector is negated due to how to get_goal_approach_pose() defines the angle
        base_angle = math.atan2(-perp_vec[1], -perp_vec[0])
        if samples == 1:
            return base_angle

        angle_deviations = np.linspace(-angle_from_center, angle_from_center, num=samples, endpoint=True)
        ordering = np.argsort(np.abs(angle_deviations))

        return (base_angle + angle_deviations)[ordering]

    def plan_to_goal(self, offset = 0.25, angle = None, ignore_trellis=False):


        if angle is not None:
            angles = [angle]
        else:
            angles = self.generate_feasible_approach_angles()

        reference = self.current_orientation_ref_array

        for angle in angles:

            if self.evaluate_goal_approach_manipulability(angle, offset) < self.MIN_MANIPULABILITY_RECOVER:
                continue

            position, pose, _ = self.get_goal_approach_pose(self.current_goal, offset, angle, orientation_reference=reference)
            resp = self.check_ray_srv(self.pose_to_ros_msg(pose))

            if not resp.collision:

                plan_pose_msg = Pose()
                plan_pose_msg.position.x = position[0]
                plan_pose_msg.position.y = position[1]
                plan_pose_msg.position.z = position[2]
                plan_pose_msg.orientation.x = pose[1]
                plan_pose_msg.orientation.y = pose[2]
                plan_pose_msg.orientation.z = pose[3]
                plan_pose_msg.orientation.w = pose[0]

                if self.sequencing_metric == 'fredsmp' or self.sequencing_metric == 'hybrid':
                    tasks_msg = PoseArray()
                    tasks_msg.poses.append(plan_pose_msg)

                    resp_sequencer = self.sequencer_client.call(tasks_msg, self.sequencing_metric)
                    if len(resp_sequencer.database_trajectories):
                        joint_msg = JointState()
                        joint_msg.position = resp_sequencer.database_trajectories[0].points[-1].positions
                        # resp = self.plan_joints_client(joint_msg, ignore_trellis, True)
                        # resp = self.plan_pose_client(plan_pose_msg, ignore_trellis, True)
                        resp = self.optimise_offset_client(resp_sequencer.database_trajectories[0], True)
                    else:
                        continue
                elif self.sequencing_metric == 'euclidean':
                    # resp = self.plan_pose_client(plan_pose_msg, ignore_trellis, self.sim)
                    resp = self.plan_pose_client(plan_pose_msg, ignore_trellis, True)   # Moves it into place
                else:
                    raise ValueError()

                if not resp.success:
                    rospy.logwarn("planning to next target failed")
                else:
                    rospy.sleep(1.0)
                    return resp.success, resp.plan_duration, resp.traj_duration

        return False, numpy.nan, numpy.nan

    def plan_to_joints(self, joints):

        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = self.initial_joints.name
        joint_msg.position = joints

        self.plan_joints_client(joint_msg, False, True)

    def get_angular_velocity(self, goal = [None], to_goal = [None]):
        
        # rotation between the vector pointing to the goal and the z-axis
        if goal[0] == None:
            goal == self.current_goal

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

        q_dot = numpy.zeros((self.num_joints, 1))

        max_joint_speed = 5.0
        K = 1.0
        weight_vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        for i in range(self.num_joints):
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
        # print ("qdot_proj: " + str(numpy.dot( (numpy.eye(self.num_joints) - numpy.dot( numpy.linalg.pinv(J) , J )), q_dot)))
        return numpy.dot( numpy.linalg.pinv(J), des_vel.reshape(6,1)) + numpy.dot( (numpy.eye(self.num_joints) - numpy.dot( numpy.linalg.pinv(J) , J )), q_dot)
        #return numpy.dot( (numpy.eye(self.num_joints) - numpy.dot( numpy.linalg.pinv(J) , J )), q_dot)

        # return numpy.dot( numpy.linalg.pinv(J), des_vel.reshape(6,1))

    def grab(self):

        resp = self.gripper_client.call(False)

        while not resp.success:
            resp = self.gripper_client.call(False)


    def drop(self):

        self.current_goal = [None]

        resp = self.gripper_client.call(True)

        while not resp.success:
            resp = self.gripper_client.call(True)


    def refresh_octomap(self, point = None, camera_frame=False):

        if not self.use_camera:
            return

        if point is None:
            self.update_octomap()
            self.load_octomap()     # Currently assumes arm is still while loading octomap
        else:

            # If camera_frame is False, point should be expressed in terms of the global coordinates
            # (or the robot base if the robot is not moving)
            if not camera_frame:
                if self.sim:
                    # Even though the arm moves before approaching the target, for our sim, the camera stays still
                    # Hence we record the original position of the camera and pretend the camera stays fixed
                    # and filter out the goal points in the original frame
                    camera_pose = self.goal_point_camera_pose

                else:
                    # Otherwise, in a real implementation, the camera moves with the arm
                    # Therefore we need to express the goal points in the view of the new position of the camera
                    # so that we can properly filter out the octomap
                    camera_pose = self.get_camera_pose()

                goal_point_world_mat = numpy.identity(4)
                goal_point_world_mat[0:3, 3] = point
                goal_point_camera_mat = numpy.dot(numpy.linalg.inv(camera_pose), goal_point_world_mat)
                goal_point_camera = goal_point_camera_mat[0:3, 3]
            else:
                goal_point_camera = point

            rospy.set_param('/goal_x', goal_point_camera[0].item())
            rospy.set_param('/goal_y', goal_point_camera[1].item())
            rospy.set_param('/goal_z', goal_point_camera[2].item())

            self.update_octomap_filtered()

        # TODO: BUG: Load_octomap for simulation will refresh the octomap in reference to the current camera position,
        # however for simulation it should refresh it with respect to the original camera position
        self.load_octomap()
        rospy.set_param('/goal_x', False)
        rospy.set_param('/goal_y', False)
        rospy.set_param('/goal_z', False)


    def clean(self):

        # on shutdown

        # self.socket_handler.close()

        self.stop_arm()
        if not rospy.get_param('sim'):
            self.s.close()
        qstart_param_name = rospy.search_param('qstart')

        if qstart_param_name != None:
            rospy.delete_param(qstart_param_name)


def point_as_array(pt):
    if isinstance(pt, PointStamped):
        pt = pt.point
    return np.array([pt.x, pt.y, pt.z])


def dummy_function(*_, **__):
    pass


class Logger(object):
    def __init__(self):

        self.active_timers = {}
        self.iteration = 0
        self.log = defaultdict(dict)

        self.fields = [
            'gx', 'gy', 'gz',           # What are the goal coordinates?
            'a', 'r',                   # What angle did it approach relative to the base frame? What was the roll
            'fgx', 'fgy', 'fgz',        # During the visual update phase, what goal did the cutter eventually settle on?

            # Moving to next approach
            'planning_time', 'traj_time',
            'traj_dist',

            # Related to servoing
            'servoing_time',
            'cutting_time',
            'recovery_time',
            'status',                   # What sort of error did we run into, if any?

            'sequencing_time', 'octomap_time',  # One-time costs incurred in the first iteration
        ]

    def start_timer(self, event):
        assert event in self.fields
        if event in self.active_timers:
            raise Exception('Attempting to start timing an event which is still running')
        self.active_timers[event] = time.time()

    def end_timer(self, event):
        end = time.time()
        start = self.active_timers.pop(event)
        self.log[self.iteration][event] = end - start
        return end - start

    def increment_iteration(self):
        if self.active_timers:
            raise Exception('Attempting to increment iteration while timers are running')

        self.iteration += 1

    def set_value(self, field, value):

        if isinstance(field, list):
            for key, val in zip(field, value):
                self.set_value(key, val)
            return

        assert field in self.fields

        self.log[self.iteration][field] = value

    def __setitem__(self, key, value):
        self.set_value(key, value)

    def output_df(self):
        return DataFrame(self.log).T.reindex(columns=self.fields)