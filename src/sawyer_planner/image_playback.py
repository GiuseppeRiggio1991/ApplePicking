#!/usr/bin/env python

import rospy
import cPickle
import os
import sys

last_file = ''
last_obj = {}
data_loc = os.path.join(os.path.expanduser('~'), 'data', 'tree_data')

publishers = {}

rospy.init_node('image_playback')

if len(sys.argv) > 1:
    rospy.set_param('playback_file', sys.argv[1])

while not rospy.is_shutdown():

    now = rospy.Time.now()
    if last_obj:

        for k, v in last_obj.iteritems():
            if k not in publishers:
                publishers[k] = rospy.Publisher(k, type(v), queue_size=1)
            v.header.stamp = now
            publishers[k].publish(v)

    new_file = rospy.get_param('playback_file', '')

    if new_file != last_file:
        last_file = new_file
        file_name = os.path.join(data_loc, '{}.pickle'.format(new_file.replace('.pickle', '')))
        try:
            with open(file_name, 'rb') as fh:
                last_obj = cPickle.load(fh)
            rospy.loginfo('Loaded file {}!'.format(new_file))
        except IOError:
            rospy.logerr('No file named {}!'.format(new_file))

    rospy.sleep(0.1)