#!/usr/bin/python

import cv2
import math
import rospy
import numpy as np
from smarc_msgs.msg import Sidescan
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class sss_detection:
    def __init__(self, width=1000, height=2000):
        self.width = width
        self.height = height
        self.effective_height = 0
        self.sss_img = np.zeros((self.width, self.height), dtype=np.ubyte)
        self.sss_sub = rospy.Subscriber("/sam/payload/sidescan", Sidescan,
                                        self.callback)
        self.detection_pub = rospy.Publisher("/sam/detection/rope",
                                             Image,
                                             queue_size=2)
        self.bridge = CvBridge()

    def callback(self, msg):
        port = np.array(bytearray(msg.port_channel), dtype=np.ubyte)
        stbd = np.array(bytearray(msg.starboard_channel), dtype=np.ubyte)
        meas = np.concatenate([np.flip(port), stbd])
        self.sss_img[1:, :] = self.sss_img[:-1, :]
        self.sss_img[0, :] = meas

        self.effective_height = min(self.effective_height + 1, self.height)

        self.detect_rope()

    def detect_rope(self):
        image = np.copy(self.sss_img[:self.effective_height, :])
        image = cv2.Canny(image, 100, 150)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        lines = cv2.HoughLines(image,
                               1,
                               np.pi / 180,
                               50,
                               min_theta=0,
                               max_theta=np.pi / 180 * 10)

        if lines is not None:
            for line in lines:
                rho = line[0][0]
                theta = line[0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 - 1000 * b), int(y0 + 1000 * a))
                pt2 = (int(x0 + 1000 * b), int(y0 - 1000 * a))
                cv2.line(image_bgr, pt1, pt2, (0, 255, 0), 3, cv2.LINE_AA)
            print(f"Number of detected lines: {len(lines)}")

        try:
            image_bgr = self.bridge.cv2_to_imgmsg(image_bgr, "bgr8")
            self.detection_pub.publish(image_bgr)

        except CvBridgeError as e:
            print(e)


def main():
    rospy.init_node('sss_rope_detection', anonymous=True)
    r = rospy.Rate(5)  # 10hz

    sss_rope_detector = sss_detection()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down')


if __name__ == '__main__':
    main()
