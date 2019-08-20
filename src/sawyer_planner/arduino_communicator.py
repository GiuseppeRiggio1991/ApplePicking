#!/usr/bin/env python

import rospy
import serial
import sys
from std_srvs.srv import Empty

PORT = '/dev/ttyACM1'
msg = 'a'

def send_arduino_message(_):
    arduino.write(msg)
    while True:
        data = arduino.readline()
        if data:
            arduino.flush()
            break

    return []

if __name__ == '__main__':
    rospy.init_node('arduino_communicator')
    try:
        arduino = serial.Serial('/dev/ttyACM1', 115200, timeout=0.1)
    except serial.SerialException as e:
        rospy.logerr('Could not connect to serial! Error message: {}'.format(e.message))
        rospy.logerr("If it's a permission error, consider the following command: sudo chmod a+rw {}".format(PORT))
        sys.exit(1)

    rospy.loginfo('Successfully connected to Arduino!')

    rospy.Service('activate_arduino', Empty, send_arduino_message)
    rospy.spin()

    arduino.close()