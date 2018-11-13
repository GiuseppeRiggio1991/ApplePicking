#!/usr/bin/env python

import rospy
import numpy
import intera_interface
import time

def saturate (val, minval, maxval):
    if val < minval: return minval
    if val > maxval: return maxval
    return val


if __name__ == "__main__":

    rospy.init_node('test_rosnode')
    rate = rospy.Rate(1000)
    
    limb = intera_interface.Limb('right')

    limb.set_command_timeout(0.1)

    #des_pos = numpy.array([-2.14003109, -0.76870472, 1.94924496, -1.29428104, 0.80476166, -0.44328272, 1.0416117])

    # starting from j6
    des_pos = numpy.array([0.8706, -0.5798, -0.0301, 1.5812, 0.10396, -0.9397, 1.56914])

    current_time = time.time()
    cycle_time = None

    MAX_VEL = 3.0

    while not rospy.is_shutdown():

        if cycle_time is None:
            cycle_time = 0.02 #500 Hz
        else:
            cycle_time = time.time() - current_time

        current_time = time.time()

        if cycle_time > 0.1: # safety
            cycle_time = 0.02


        pos = limb.joint_angles()
        #vel = limb.joint_velocities()

        # position control
        vel = 15.0 * (des_pos - numpy.array(pos.values()))

        for i in range(0, len(vel)):
            vel[i] = saturate(vel[i], -MAX_VEL, MAX_VEL)            

        cmd_values = numpy.add(pos.values(), vel * cycle_time)
        
        cmd = dict(zip(pos.keys(), cmd_values))       

        print vel
        #print cycle_time

        limb.set_joint_positions(cmd)
        rate.sleep()

