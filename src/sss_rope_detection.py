#!/usr/bin/python

import rospy
import numpy as np
from smarc_msgs import Sidescan


class sss_detection:
    def __init__(self, width=1000, height=2000):
        self.width = width
        self.height = height
        self.sss_img = np.zeros((self.width, self.height), dtype=np.ubyte)
        self.sss_sub = rospy.Subscriber("/sam/payload/sidescan", Sidescan,
                                        self.callback)
        self.effective_height = 0

    def callback(self, msg):
        port = np.array(bytearray(msg.port_channel), dtype=np.ubyte)
        stbd = np.array(bytearray(msg.starboard_channel), dtype=np.ubyte)
        meas = np.concatenate([np.flip(port), stbd])
        self.sss_img[1:, :] = self.sss_img[:-1, :]
        self.sss_img[0, :] = meas

        self.effective_height = min(self.effective_height + 1, self.height)


def main():
    rospy.init_node('sss_rope_detection', anonymous=True)
    r = rospy.Rate(5)  # 10hz

    print('Node sss_rope_detection running...')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down')


if __name__ == '__main__':
    main()
