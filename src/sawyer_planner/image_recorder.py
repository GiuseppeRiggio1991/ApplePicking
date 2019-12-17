#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber as MsgSubscriber
import cPickle
import os
import datetime

mutex = False
cooldown = rospy.get_param('/cooldown', 2.0)
last_time = rospy.Time()

output_loc = os.path.join(os.path.expanduser('~'), 'data', 'tree_data')

def image_callback(*args):

    global mutex
    global last_time

    now = rospy.Time.now()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if mutex or (now - last_time).to_sec() < cooldown:
        return

    mutex = True
    try:
        output = dict(zip(topics, args))
        file_path = os.path.join(output_loc, '{}.pickle'.format(ts))
        with open(file_path, 'wb') as fh:
            cPickle.dump(output, fh)
        rospy.loginfo('All topics successfully saved to {}'.format(file_path))
    finally:
        mutex = False
        last_time = now

if __name__ == '__main__':
    rospy.init_node('image_recorder')
    # Topic recording callback
    subs = [
        MsgSubscriber('/camera/color/image_raw', Image),
        MsgSubscriber('/theia/left/image_raw', Image),
        MsgSubscriber('/theia/right/image_raw', Image),
        MsgSubscriber('/camera/depth_registered/points', PointCloud2),
    ]
    topics = [s.topic for s in subs]
    sync = ApproximateTimeSynchronizer(subs, 1, 0.5)
    sync.registerCallback(image_callback)

    rospy.spin()