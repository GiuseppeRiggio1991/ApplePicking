#!/usr/bin/env python

import rospy
from sawyer_planner.srv import CheckBranchOrientationRequest, CheckBranchOrientation, CheckBranchOrientationResponse
from sensor_msgs.point_cloud2 import read_points
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
import numpy as np
from numpy.linalg import svd
from visualization_msgs.msg import Marker

rospy.init_node('branch_orientation')

# PARAMS
min_points = rospy.get_param('min_points', 25)      # How many points should be in the point cloud for us to attempt to find an angle
radius = rospy.get_param('radius', 0.04)           # How far from the requested points should we consider the points
# clustering_threshold = rospy.get_param('clustering_threshold', 0.0025)      # For the clustering algorithm, what counts as a neighbor?
point_centroid_dist = rospy.get_param('point_centroid_dist', 0.5)           # For the return points, how far off the centroid do we want them to be (mostly cosmetic)


def rviz_publish_point(target, frame_id):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.id = 0

    marker.pose.position = target
    marker.color.a = 1.0
    marker.color.b = 1.0

    marker.scale.x = marker.scale.y = marker.scale.z = 0.03
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    rviz_pub.publish(marker)

def rviz_publish_line(point_1, point_2, frame_id):

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.id = 1

    marker.color.a = 0.5
    marker.color.g = 1.0

    marker.scale.x = 0.015
    marker.points.append(point_1)
    marker.points.append(point_2)

    rviz_pub.publish(marker)

def process_orientation(msg):


    target = msg.target
    target = np.array([target.x, target.y, target.z])
    cloud = msg.cloud
    response = CheckBranchOrientationResponse()

    point_generator = read_points(cloud, skip_nans=True, field_names=('x', 'y', 'z'))
    all_points = np.array(list(point_generator))

    # Temporary, the point cloud should be prefiltered?
    all_points = all_points[all_points[:,2] < 1.0]
    all_points = all_points[all_points[:,2] > 0.30]

    if not target.sum():
        target = all_points[np.random.randint(0, all_points.shape[0]), :]


    rviz_publish_point(Point(*target), cloud.header.frame_id)

    # Filter out points with a higher radius
    indexer = ((all_points - target) ** 2).sum(axis=1) < radius ** 2
    all_points = all_points[indexer]
    if all_points.shape[0] < min_points:
        return response



    # Demean points for fitting with SVD
    centroid = all_points.mean(axis=0)
    points_demeaned = all_points - centroid

    _, _, v = svd(points_demeaned)
    component_vector = v[0,:]
    point_1 = Point(*(centroid + point_centroid_dist * component_vector))
    point_2 = Point(*(centroid - point_centroid_dist * component_vector))

    rviz_publish_line(point_1, point_2, cloud.header.frame_id)
    response.point_1 = point_1
    response.point_2 = point_2

    return response

rospy.Service('branch_orientation', CheckBranchOrientation, process_orientation)
rviz_pub = rospy.Publisher("visualization_marker", Marker, queue_size=1)

rospy.spin()

# while not rospy.is_shutdown():
#     req = CheckBranchOrientationRequest()
#     cloud = rospy.wait_for_message('/camera/depth_registered/points_z_filtered', PointCloud2)
#     point = Point()
#
#     req.cloud = cloud
#     req.target = point
#
#     process_orientation(req)