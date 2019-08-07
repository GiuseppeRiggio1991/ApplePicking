#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge
from copy import deepcopy

class BackgroundProcessor(object):
    def __init__(self):

        self.bridge = CvBridge()
        self.image_topic = '/camera/color/image_raw'
        self.repub_topic = '/camera/color/foreground'

        self.subtractor = cv2.createBackgroundSubtractorKNN()

        self.mutex = False
        rospy.Subscriber(self.image_topic, Image, self.process_frame)
        self.fg_pub = rospy.Publisher(self.repub_topic, Image, queue_size=1)

    def process_frame(self, img_ros):

        if self.mutex:
            return

        self.mutex = True

        img = self.bridge.imgmsg_to_cv2(img_ros, desired_encoding="passthrough")
        mask = self.subtractor.apply(img)
        img_msg = self.bridge.cv2_to_imgmsg(mask, encoding="passthrough")
        img_msg.header = deepcopy(img_ros.header)

        self.mutex = False
        self.fg_pub.publish(img_msg)


if __name__ == '__main__':
    rospy.init_node('background_processor')

    tracker = BackgroundProcessor()

    rospy.spin()