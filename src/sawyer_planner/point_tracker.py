#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import PointStamped, Point, Transform, TransformStamped, Quaternion
from visualization_msgs.msg import Marker
from tf2_ros import TransformListener, Buffer, ExtrapolationException
from std_srvs.srv import Empty
from sensor_msgs.msg import PointCloud2, PointField
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from sawyer_planner.srv import SetGoal
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import struct
from tf.transformations import decompose_matrix, quaternion_from_euler
from scipy.stats import multivariate_normal
from copy import deepcopy
from sensor_msgs.msg import CameraInfo, Image
from matplotlib.path import Path
import message_filters
from numpy.linalg import svd
import cv2
from cv_bridge import CvBridge
from sklearn.linear_model import LinearRegression
from skimage.draw import line as draw_line
from ros_numpy import numpify

class PointTracker(object):
    def __init__(self, debug=True):


        self.goal = None            # Keeps track of the moving target to be followed
        self.camera_base = None     # For the current goal, records the position of the camera's principal point
        self.goal_anchor = None     # Keeps track of the original estimate in the event the tracking gets lost

        self.base_frame = 'base_link'
        self.initial_snap = False


        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.bridge = CvBridge()

        self.active = False

        self.image_occlusion_zone = Path(np.array([
            [539, -1],
            [537, 30],
            [493, 88],
            [478, 182],
            [485, 238],
            [511, 275],
            [640, 275],
            [640, -1],
        ]))     # Note that 639 and 0 are inflated to 640 and -1 due to edge detection ambiguities

        self.debug = debug
        self.mutex = False

        self.point_cloud_topic = '/camera/depth_registered/points_z_filtered'
        # self.endpoint_frame = endpoint_frame     # This should be changed to whatever endpoint you care about!

        self.rviz_pub = rospy.Publisher("visualization_marker", Marker, queue_size=1)
        self.goal_publisher = rospy.Publisher('update_goal_point', Point, queue_size=1)

        pc_subscriber = message_filters.Subscriber(self.point_cloud_topic, PointCloud2)
        foreground = message_filters.Subscriber('/camera/color/foreground', Image)
        cam_info = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)
        sync = message_filters.ApproximateTimeSynchronizer([pc_subscriber, foreground, cam_info], 1, slop=0.2)
        sync.registerCallback(self.update_goal)

        rospy.Service('point_tracker_set_goal', SetGoal, self.add_goal)
        rospy.Service('activate_point_tracker', Empty, self.activate)
        rospy.Service('deactivate_point_tracker', Empty, self.deactivate)

        # Diagnostic
        self.diagnostic_pub = rospy.Publisher('point_tracking_diagnostics', Image, queue_size=1)

    def activate(self, *_):
        self.active = True
        return []

    def deactivate(self, *_):
        self.active = False
        self.initial_snap = False
        return []

    def add_goal(self, set_goal_msg):

        self.goal = create_stamped_point(set_goal_msg.goal.goal)
        self.goal_anchor = deepcopy(self.goal)
        self.camera_base = create_stamped_point(set_goal_msg.goal.camera)

        p = self.goal.point

        rospy.loginfo('Set goal point ({:.3f}, {:.3f}, {:.3f})'.format(p.x, p.y, p.z))

        return []

    def get_message_tf(self, msg, new_frame=None, msg_frame_target = True):

        if new_frame is None:
            new_frame = self.base_frame

        timestamp = msg.header.stamp
        if msg_frame_target:
            target = msg.header.frame_id
            source = new_frame
        else:
            target = new_frame
            source = msg.header.frame_id

        success = self.tf_buffer.can_transform(target, source, timestamp, rospy.Duration(0.5))
        if not success:
            rospy.logerr("Couldn't look up transform between {} and {}!".format(target, source))

        tf = self.tf_buffer.lookup_transform(target, source, timestamp)
        return tf

    def update_goal(self, pc, fg_mask, camera_info):

        if not self.active or self.mutex or self.goal is None:
            return

        self.mutex = True

        try:
            self.update_goal_core(pc, fg_mask, camera_info)
        finally:
            self.mutex = False

    def update_goal_core(self, pc, fg_mask, camera_info):

        goal = deepcopy(self.goal)
        goal.header.stamp = fg_mask.header.stamp

        # Convert the current goal to the camera frame
        camera_tf = self.get_message_tf(goal, camera_info.header.frame_id, msg_frame_target=False)
        inv_camera_tf = self.get_message_tf(goal, camera_info.header.frame_id)

        goal_camera = do_transform_point(goal, camera_tf)
        goal_camera_array = point_to_array(goal_camera)

        goal_in_image = project_to_pixel(goal_camera_array, camera_info)
        if self.image_occlusion_zone.contains_point(goal_in_image):
            return

        # Convert foreground mask to image
        mask = np.asarray(self.bridge.imgmsg_to_cv2(fg_mask, desired_encoding="passthrough")).T  # Index as [x,y]
        positive_pixel_points = np.array(np.where(mask == 255)).T

        goal_in_image, line_info = self.reconcile_goal_with_branch_points(goal, mask == 255, camera_info, max_jump=0.05)
        new_goal_in_image = snap_reference_to_2d_points(goal_in_image, positive_pixel_points, 40, min_density=0.05)

        # Take the point cloud points, filter them to points within a certain distance of the previous goal
        assert pc.header.frame_id == camera_info.header.frame_id

        # Extracts the 3D array of points - needs to do some management of structured arrays for efficiency
        pts_struct = numpify(pc)[['x', 'y', 'z']]
        pts = pts_struct.view((pts_struct.dtype[0], 3))

        # Using the old z value, get the points in the point cloud near your new estimate of the goal
        new_goal = deproject_pixel(new_goal_in_image[0], new_goal_in_image[1], goal_camera.point.z, camera_info)
        dists = np.linalg.norm(pts - new_goal, axis=1)
        filter = dists < 0.03
        pts = pts[filter]
        if pts.shape[0] > 10:  # If not enough points, we just assume the z is unchanged from before (in plane)

            dists = dists[filter]

            # Project the nearby points to the XY plane and run a linear regression weighted by inverse exponential
            # distance
            decay_per_centimeter = 0.5
            weights = decay_per_centimeter ** (dists / 0.01)

            pix_xy = project_to_pixel(pts, camera_info)
            model = LinearRegression().fit(pix_xy, pts[:, 2], weights)
            new_z = model.predict([new_goal_in_image])[0]
            new_goal[2] = new_z

        goal.point = Point(*new_goal)

        self.goal = do_transform_point(goal, inv_camera_tf)

        self.mutex = False

        # Diagnostic: Plot the old and new
        rgb_frame = cv2.cvtColor(self.bridge.imgmsg_to_cv2(fg_mask), cv2.COLOR_GRAY2BGR)
        cv2.line(rgb_frame, line_info[0], line_info[1], (255, 0, 0), 2)
        cv2.circle(rgb_frame, tuple(goal_in_image.astype(int)), 10, (0, 0, 255), 3)
        cv2.circle(rgb_frame, tuple(new_goal_in_image.astype(int)), 7, (0, 255, 0), 3)
        diagnostic_image = self.bridge.cv2_to_imgmsg(rgb_frame)
        self.diagnostic_pub.publish(diagnostic_image)


    def reconcile_goal_with_branch_points(self, global_goal, pixel_mask, camera_info, max_jump=0.25):
        """
        Tries to reconcile an initial global estimate of a goal from a certain position with a set of candidate pixels
        :param global_goal: A StampedPoint object in the base frame
        :param pixel_mask: A Boolean array of pixels which should be considered for projection
        :return:
        """

        camera_tf = self.get_message_tf(global_goal, camera_info.header.frame_id, msg_frame_target=False)
        goal_in_camera = do_transform_point(global_goal, camera_tf)
        original_viewpoint_in_camera = do_transform_point(self.camera_base, camera_tf)

        goal_in_camera_array = point_to_array(goal_in_camera)
        original_viewpoint_in_camera_array = point_to_array(original_viewpoint_in_camera)

        goal_in_pixels = project_to_pixel(goal_in_camera_array, camera_info)
        orig_viewpoint_in_pixels = project_to_pixel(original_viewpoint_in_camera_array, camera_info)

        diff = orig_viewpoint_in_pixels - goal_in_pixels
        if all(diff == 0):
            return goal_in_pixels, None

        max_dim = np.argmax(np.abs(diff))
        if max_dim == 0:
            slope = diff[1] / diff[0]
            left_x = 0
            left_y = goal_in_pixels[1] - slope * goal_in_pixels[0]
            right_x = camera_info.width - 1
            right_y = left_y + right_x * slope
        else:
            slope = diff[0] / diff[1]
            left_y = 0
            left_x = goal_in_pixels[0] - slope * goal_in_pixels[1]
            right_y = camera_info.height - 1
            right_x = left_x + right_y * slope

        line = np.array(draw_line(int(left_x), int(left_y), int(right_x), int(right_y))).T
        line_info = ((int(left_x), int(left_y)), (int(right_x), int(right_y)))

        # Find intersection of mask and line
        # Note that the clean way to do this would be to find the line intersections with the box and only draw those
        # pixels, so this filtering method is just a hack for now

        common = []
        for pixel in line:
            in_mask = (0 <= pixel[0] < pixel_mask.shape[0]) and (0 <= pixel[1] < pixel_mask.shape[1])
            if in_mask and pixel_mask[pixel[0], pixel[1]]:
                common.append(pixel)

        if not common:
            return goal_in_pixels, line_info

        principal_point = deproject_pixel(0, 0, 0, camera_info)

        current_best_dist = np.inf
        current_best_point = goal_in_pixels

        for pixel_tuple in common:
            pixel = np.array(pixel_tuple)
            deprojected_point = deproject_pixel(pixel[0], pixel[1], 1, camera_info)
            skew_point, _ = get_closest_points_skew_lines(principal_point, deprojected_point, original_viewpoint_in_camera_array,
                                                          goal_in_camera_array)

            camera_skew_dist_from_goal = np.linalg.norm(skew_point - goal_in_camera_array)
            if camera_skew_dist_from_goal > max_jump:
                continue
            if camera_skew_dist_from_goal < current_best_dist:
                current_best_dist = camera_skew_dist_from_goal
                current_best_point = pixel

        rospy.loginfo_throttle(0.5, 'Best match found was {:.1f} cm away from original estimate'.format(current_best_dist * 100))

        return current_best_point, line_info


    #
    # def get_latest_cloud(self):
    #     pc = rospy.wait_for_message(self.point_cloud_topic, PointCloud2)
    #     success = self.tf_buffer.can_transform(pc.header.frame_id, self.base_frame, pc.header.stamp, timeout=rospy.Duration(0.5))
    #     if not success:
    #         rospy.logerr("Couldn't find the transform between the point cloud frame and the base frame!")
    #         return []
    #
    #     tf = self.get_message_tf(pc, msg_frame_target=False)
    #     return do_transform_cloud(pc, tf)


    def rviz_publish_goals(self):

        # Temp
        if self.goal is None:
            return

        marker = Marker()
        marker.ns = 'point_tracker'
        marker.header.frame_id = self.base_frame
        marker.type = Marker.SPHERE_LIST
        marker.header.stamp = rospy.Time.now()
        marker.id = 0

        marker.points = [self.goal.point, self.goal_anchor.point]

        marker.color.a = 0.5
        marker.color.b = 1.0

        marker.scale.x = marker.scale.y = marker.scale.z = 0.02
        marker.action = Marker.ADD

        self.rviz_pub.publish(marker)

def get_closest_points_skew_lines(line_1_start, line_1_end, line_2_start, line_2_end):
    # https://en.wikipedia.org/wiki/Skew_lines#Distance
    p1 = line_1_start
    d1 = (line_1_end - line_1_start)
    p2 = line_2_start
    d2 = (line_2_end - line_2_start)

    n1 = np.cross(d1, np.cross(d2, d1))
    n2 = np.cross(d2, np.cross(d1, d2))

    c1 = p1 + np.dot(p2-p1, n2) / np.dot(d1, n2) * d1
    c2 = p2 + np.dot(p1-p2, n1) / np.dot(d2, n1) * d2

    return c1, c2

# Camera deprojection method
def project_to_pixel(points, camera_info):
    """
    Uses the camera matrix to turn a point in the depth frame to the image frame
    :param point: A numpy array with 3 elements
    :param camera_info:
    :return:
    """

    fx, _, ppx, _, fy, ppy, _, _, _ = camera_info.K
    coeffs = np.array(camera_info.D)
    if not np.all(coeffs == 0):
        raise NotImplementedError('Point projection currently doesn\'t work with this camera model')

    if len(points.shape) == 1:
        x, y, z = points
    else:
        assert points.shape[1] == 3
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

    pixel_x = x / z * fx + ppx
    pixel_y = y / z * fy + ppy

    return np.array([pixel_x, pixel_y]).T


def deproject_pixel(pixel_x, pixel_y, depth, camera_info_msg):

    # Extract camera parameters
    fx, _, ppx, _, fy, ppy, _, _, _ = camera_info_msg.K
    coeffs = np.array(camera_info_msg.D)

    x = (pixel_x - ppx) / fx
    y = (pixel_y - ppy) / fy

    if not np.all(coeffs == 0):
        raise NotImplementedError("This part isn't tested, make sure to confirm it first!")
        r2 = x*x + y*y
        f = 1 + coeffs[0] * r2 + coeffs[1] * r2 ** 2 + coeffs[4] * r2 ** 3
        ux = x * f + 2 * coeffs[2] * x * y + coeffs[3] * (r2 + 2 * x * x)
        uy = y * f + 2 * coeffs[3] * x * y + coeffs[2] * (r2 + 2 * y * y)
        x = ux
        y = uy

    return np.array([depth * x, depth * y, depth])


def filter_cloud_by_point(pc, point, distance):

    all_pts = np.array(list(pc2.read_points(pc, field_names=('x', 'y', 'z'))))
    goal_pt = np.array([point.x, point.y, point.z])
    dists = np.linalg.norm(all_pts - goal_pt, axis=1)
    filtered_pts = (all_pts[dists < distance]).tolist()
    return convert_points_to_pointcloud(filtered_pts, pc.header.frame_id)


def convert_points_to_pointcloud(points, frame_id):

    # points is a Python list with [x, y, z, (r, g, b)]  - Colors will be detected based on first element length
    # can also be a numpy array
    # colors is either None if you don't want RGB, or an Nx3 numpy array with the corresponding RGB values as integers from 0 to 255

    # Heavily sourced from: https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
    # And: https://answers.ros.org/question/289576/understanding-the-bytes-in-a-pcl2-message/

    dim = len(points[0])
    if dim == 3:
        colors = False
    elif dim == 6:
        colors = True
    else:
        raise ValueError('Unsure how to interpret dimension {} input'.format(dim))

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]
    if colors:
        fields.append(PointField('rgb', 12, PointField.UINT32, 1))
        formatted_points = []
        for row in points:
            xyz = row[:3]
            r, g, b = row[3:]
            a = 255
            rgba = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            xyz.append(rgba)
            formatted_points.append(xyz)

        points = formatted_points

    pc = pc2.create_cloud(header, fields, points)
    return pc

def point_to_array(pt):
    if isinstance(pt, PointStamped):
        pt = pt.point

    return np.array([pt.x, pt.y, pt.z])

def snap_reference_to_2d_points(reference, points, selection_radius=40, min_density=0.0):
    dist = np.linalg.norm(points - reference, axis=1)
    points = points[dist < selection_radius]
    if not points.shape[0] or points.shape[0] / (selection_radius ** 2 * np.pi) < min_density:
        return reference

    centroid = points.mean(axis=0)
    points_demeaned = points - centroid

    _, _, v = svd(points_demeaned)
    component_vector = v[0, :]
    orthogonal_vector = np.array([-component_vector[1], component_vector[0]])

    x1, y1 = centroid
    x2, y2 = centroid + orthogonal_vector

    px = points[:,0]
    py = points[:,1]

    # Help, what's linear algebra
    # http://paulbourke.net/geometry/pointlineplane/

    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / np.linalg.norm(orthogonal_vector)

    x_proj = x1 + u * (x2 - x1)
    y_proj = y1 + u * (y2 - y1)

    return np.array([x_proj.mean(), y_proj.mean()])

def create_stamped_point(point, frame_id='base_link', stamp=None):
    if stamp is None:
        stamp = rospy.Time()
    ps = PointStamped()
    ps.header.frame_id = frame_id
    ps.header.stamp = stamp
    ps.point = point

    return ps

if __name__ == '__main__':

    rospy.init_node('point_tracker')
    tracker = PointTracker()

    new_goal_pub = rospy.Publisher('/update_goal_point', Point)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if tracker.active:
            new_goal_pub.publish(tracker.goal.point)
            tracker.rviz_publish_goals()
        rate.sleep()



