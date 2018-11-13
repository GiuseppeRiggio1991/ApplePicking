#!/usr/bin/env python

import rospy
import numpy
import intera_interface

if __name__ == "__main__":

    rospy.init_node('test_rosnode')
    rate = rospy.Rate(1000)
    
    limb = intera_interface.Limb('right')

    limb.set_command_timeout(0.1)

    #des_pos = numpy.array([-2.14003109, -0.76870472, 1.94924496, -1.29428104, 0.80476166, -0.44328272, 1.0416117])

    # starting from j6
    des_pos = numpy.array([0.8706, -0.5798, -0.0301, 1.5812, 0.10396, -0.9397, 1.56914])

    while not rospy.is_shutdown():
          
        pos = limb.joint_angles()
        #vel = limb.joint_velocities()

        cmd_values = 0.1 * (des_pos - numpy.array(pos.values()))
        #cmd_values = 10.0 * (des_pos - numpy.array(pos.values())) - 1.0 * numpy.array(vel.values())
        cmd = dict(zip(pos.keys(), cmd_values))       

        print cmd_values

        limb.set_joint_velocities(cmd)
        #limb.set_joint_torques(cmd)
        rate.sleep()

