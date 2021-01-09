#!/usr/bin/env python

import openravepy
import os

import rospy
import rospkg
from geometry_msgs.msg import Pose, Point, Quaternion
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header
from std_srvs.srv import Empty
from sawyer_planner.srv import LoadTrajectory, AppendJoints, AppendPose
from online_planner.srv import *
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from copy import deepcopy
import cPickle

import datetime


class TrajectoryConfig(object):

    def __init__(self, name, ns=None):
        self.name = name
        self.waypoints = []
        self.trajectories = []
        self.current_waypoint = 0

        self.plan_pose_client = rospy.ServiceProxy("/plan_pose_srv", PlanPose)
        self.plan_joints_client = rospy.ServiceProxy("/plan_joints_srv", PlanJoints)
        self.execute_traj_client = rospy.ServiceProxy("/execute_traj_srv", ExecuteTraj)

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer)

        prefix = ''
        if ns:
            prefix = '{}/'.format(ns)

        self.save_srv = rospy.Service("{}traj_recorder/save_current_traj".format(prefix), Empty, self.save)
        self.load_srv = rospy.Service("{}traj_recorder/load_traj".format(prefix), LoadTrajectory, self.load)
        self.append_joints_srv = rospy.Service("{}traj_recorder/append_joints".format(prefix), AppendJoints, self.append_joints)
        self.append_pose_srv = rospy.Service("{}traj_recorder/append_pose".format(prefix), AppendPose, self.append_pose)
        self.plan_srv = rospy.Service("{}traj_recorder/plan_all".format(prefix), Empty, self.plan_all)
        self.playback_srv = rospy.Service("{}traj_recorder/playback".format(prefix), Empty, self.playback)
        self.plan_to_start_srv = rospy.Service("{}traj_recorder/plan_to_start".format(prefix), Empty, self.plan_to_start)

    @classmethod
    def get_file_path(cls, name):
        rospack = rospkg.RosPack()
        file_path = os.path.join(rospack.get_path('sawyer_planner'), 'configs', '{}.config'.format(name))
        return file_path

    def load(self, name, *_, **__):
        if not isinstance(name, str):   # Is a ROS message
            name = name.name.data
        name = name.replace('.config', '')
        file_path = self.get_file_path(name)
        try:
            with open(file_path, 'rb') as fh:
                config = cPickle.load(fh)
        except IOError:
            rospy.logerr('Could not find config named {}!'.format(name))
            return []

        self.name = name
        self.waypoints = config['waypoints']
        self.trajectories = config['trajectories']

        return []

    def save(self, *_, **__):
        file_path = self.get_file_path(self.name)
        to_save = {
            'waypoints': self.waypoints,
            'trajectories': self.trajectories
        }
        with open(file_path, 'wb') as fh:
            cPickle.dump(to_save, fh)

        print('Config file saved to {}'.format(file_path))

        return []

    def append_joints(self, append_msg):
        joints = append_msg.joints
        if not len(joints.positions):
            joints = self.construct_joint_message()
        self.waypoints.append(joints)

        return []

    def append_pose(self, append_msg):
        if not self.waypoints:
            raise Exception('First point of a trajectory must be a joint configuration, not a pose!')
        pose = append_msg.pose
        if not any([pose.position.x, pose.position.y, pose.position.z]):
            pose = rospy.wait_for_message('/manipulator_pose', PoseStamped, 1.0).pose
        self.waypoints.append(pose)

        return []


    @staticmethod
    def construct_joint_message(*angles):
        msg = rospy.wait_for_message('/manipulator_joints', JointState, 1.0)
        msg.header = Header()
        if len(angles):
            msg.position = angles

        return msg

    @staticmethod
    def construct_pose_lookat(source, target, up_axis=None):

        if up_axis is None:
            up_axis = [0, 0, -1]

        source = convert_pt_to_array(source)
        target = convert_pt_to_array(target)

        goal_off_pose_mat = openravepy.transformLookat(target, source, up_axis)
        goal_off_pose = openravepy.poseFromMatrix(goal_off_pose_mat)

        return or_pose_to_ros_msg(goal_off_pose)

    def plan(self, waypoint_index):


        start = self.waypoints[waypoint_index]
        if not isinstance(start, JointState):
            start = JointState()
            previous_traj = self.trajectories[(waypoint_index - 1) % len(self.waypoints)]
            start.name = previous_traj.joint_names
            start.position = previous_traj.points[-1].positions

        goal = self.waypoints[(waypoint_index + 1) % len(self.waypoints)]
        if isinstance(goal, Pose):
            resp = self.plan_pose_client(goal, False, False, start)
        else:
            resp = self.plan_joints_client(goal, False, False, start)

        traj = resp.traj
        self.trajectories.append(traj)

    def plan_to_start(self, *_, **__):
        self.plan_joints_client(self.waypoints[0], False, True, JointState())
        self.current_waypoint = 0
        return []

    def plan_all(self, *_, **__):
        self.trajectories = []
        for i in range(len(self.waypoints)):
            self.plan(i)

        return []


    def execute(self, waypoint_index=None, set_start=False):
        if waypoint_index is None:
            waypoint_index = self.current_waypoint

        traj = self.trajectories[waypoint_index]
        self.execute_traj_client(traj, False, set_start)

        self.current_waypoint = (waypoint_index + 1) % len(self.waypoints)

        return []

    def playback(self):
        for i in range(len(self.waypoints)):
            self.execute(i, set_start=True)
            rospy.sleep(0.5)

    def get_manipulator_tf(self):

        now = rospy.Time.now()
        success = self.buffer.can_transform(rospy.get_param('base_frame'), 'manipulator', now, rospy.Duration(1.0))
        if not success:
            rospy.logerr("Couldn't look up manip transform! Try again")
            raise Exception()

        tf = self.buffer.lookup_transform(rospy.get_param('base_frame'), 'manipulator', now)
        return tf



def or_pose_to_ros_msg(pose):
    pose_msg = Pose()
    pose_msg.orientation.w = pose[0]
    pose_msg.orientation.x = pose[1]
    pose_msg.orientation.y = pose[2]
    pose_msg.orientation.z = pose[3]
    pose_msg.position.x = pose[4]
    pose_msg.position.y = pose[5]
    pose_msg.position.z = pose[6]
    return pose_msg

def convert_pt_to_array(pt):
    if isinstance(pt, PointStamped):
        pt = pt.point

    if isinstance(pt, Point):
        return np.array([pt.x, pt.y, pt.z])

    return pt





if __name__ == '__main__':

    rospy.init_node('trajectory_playback')

    traj = TrajectoryConfig(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), ns='traj')
    mode = ''
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == 'wizard':

        def input_to_array(prompt):
            input_str = raw_input(prompt).strip()
            if not input_str:
                return []
            return [float(x.strip()) for x in input_str.split(',')]

        while True:
            print('What would you like to do?')
            action = raw_input('j: Add joints\np: Add pose\nnr: Add current pose, no-roll\ns: Save\nl: Load\nv: View\nplan: Plan\nplay: Playback\ne: Execute next\nstart: Move to start\nq: Quit\n').strip()
            if action == 'j':
                print('Enter joints as comma-separated values, or leave empty to use current')
                joints = input_to_array('Joints: ')
                traj.waypoints.append(traj.construct_joint_message(*joints))

            elif action == 'p':
                print("What's the look-at point?")
                pt_1 = input_to_array('Point 1: ')

                print("What's the source point?")
                pt_2 = input_to_array('Point 2: ')

                traj.waypoints.append(traj.construct_pose_lookat(pt_2, pt_1))

            elif action == 'nr':
                # Horizon - takes the current pose and aligns it to the horizon
                try:

                    tf = traj.get_manipulator_tf()

                    pt_1 = PointStamped()
                    pt_1.header.frame_id = 'manipulator'
                    pt_1.point = Point(0, 0, 1)

                    pt_2 = deepcopy(pt_1)
                    pt_2.point = Point(0, 0, 0)

                    pt_1_tfed = do_transform_point(pt_1, tf)
                    pt_2_tfed = do_transform_point(pt_2, tf)

                    traj.waypoints.append(traj.construct_pose_lookat(pt_2_tfed, pt_1_tfed))
                    
                except Exception:
                    rospy.logwarn('Something went wrong, try again...')

            elif action == 's':
                traj.save()

            elif action == 'l':
                to_load = raw_input('Config to load: ').strip()
                traj.load(to_load)
            elif action == 'v':
                print('Current number of waypoints: {}'.format(len(traj.waypoints)))
                print('Current number of computed trajectories: {}'.format(len(traj.trajectories)))
                print('Current status: {}'.format(traj.current_waypoint))
            elif action == 'plan':
                traj.plan_all()

            elif action == 'play':
                traj.playback()
            elif action == 'e':
                traj.execute()
            elif action == 'start':
                traj.plan_to_start()
            elif action == 'q':
                break
            else:
                print('Unknown action {}'.format(action))
                continue

    else:
        rospy.spin()
















