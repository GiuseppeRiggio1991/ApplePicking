#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PointStamped, Point
from sensor_msgs.msg import PointCloud2
from sawyer_planner.srv import CheckBranchOrientation
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from tf2_ros import TransformListener, Buffer
import numpy as np
from sawyer_planner.msg import PlannerGoal
from copy import deepcopy



class PointSelector(object):

    def __init__(self, base_frame, point_cloud_topic):

        self.base_frame = base_frame
        self.point_cloud_topic = point_cloud_topic
        self.goals = []

        rospy.Subscriber('/clicked_point', PointStamped, self.add_goal, queue_size=1)
        self.orientation_srv = rospy.ServiceProxy('branch_orientation', CheckBranchOrientation)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

    def add_goal(self, ps):

        new_goal = PlannerGoal()

        pc = rospy.wait_for_message(self.point_cloud_topic, PointCloud2, 0.5)

        if ps.header.frame_id != pc.header.frame_id:
            tf_camera = self.get_tf(pc.header.frame_id, ps.header.frame_id, ps.header.stamp)
            ps = do_transform_point(ps, tf_camera)

        camera_to_base_tf = self.get_tf(self.base_frame, pc.header.frame_id, ps.header.stamp)

        resp = self.orientation_srv(ps.point, pc)

        reference_point = PointStamped()
        reference_point.point = resp.point_1
        reference_point.header.frame_id = ps.header.frame_id

        principal_point = deepcopy(ps)
        principal_point.point = Point(0,0,0)

        new_goal.goal = do_transform_point(ps, camera_to_base_tf).point
        new_goal.orientation = do_transform_point(reference_point, camera_to_base_tf).point
        new_goal.camera = do_transform_point(principal_point, camera_to_base_tf).point

        self.goals.append(new_goal)

        rospy.loginfo('Successfully recorded new point at {:.3f}, {:.3f}, {:.3f}'.format(ps.point.x, ps.point.y, ps.point.z))

    def get_tf(self, target_frame, source_frame, stamp=rospy.Time()):
        success = self.tf_buffer.can_transform(target_frame, source_frame, stamp, timeout=rospy.Duration(0.5))
        if not success:
            raise Exception("Couldn't retrieve tf between frames {} and {}!".format(target_frame, source_frame))

        return self.tf_buffer.lookup_transform(target_frame, source_frame, stamp)

    def wait_for_inputs(self):
        self.goals = []
        rospy.set_param('freedrive', True)
        while True:
            myinput = raw_input("Select your cut points from RViz, then hit Enter when you're done. Or type u to remove the last added goal.")
            if myinput.strip() == 'u':
                if self.goals:
                    self.goals = self.goals[:-1]
                    print('Removed last entry')
                else:
                    print('Goals were empty, nothing to remove!')
            elif not myinput:
                break

        rospy.set_param('freedrive', False)

if __name__ == '__main__':
    # Testing the point selector functionality
    rospy.init_node('point_selector')
    selector = PointSelector('camera_link', "/point_cloud_inhand")
    rospy.spin()
