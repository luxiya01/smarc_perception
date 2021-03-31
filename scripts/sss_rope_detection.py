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
        self.nadir_pub = rospy.Publisher("/sam/detection/nadir",
                                         Image,
                                         queue_size=2)
        self.edge_pub = rospy.Publisher("/sam/detection/rope_edge",
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

        image = np.copy(self.sss_img[:self.effective_height, :])
        self.find_nadir(image)
        self.detect_rope(image)

    def find_nadir(self, image):
        image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
        image_blurred = cv2.Canny(image_blurred, 100, 250)
        image_blurred_hough_bgr = self._get_hough_lines(
            image_blurred,
            min_line_length=int(self.effective_height * .8),
            max_line_gap=100)
        self._publish_image(image_blurred_hough_bgr, self.nadir_pub)

    def detect_rope(self, image):
        image = cv2.Canny(image, 100, 150)
        image_hough_bgr = self._get_hough_lines(image)
        self._publish_image(image, self.edge_pub)
        self._publish_image(image_hough_bgr, self.detection_pub)

    def _get_hough_lines(self, image, min_line_length=20, max_line_gap=20):
        lines = cv2.HoughLinesP(image,
                                1,
                                np.pi / 180,
                                50,
                                minLineLength=min_line_length,
                                maxLineGap=max_line_gap)

        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        num_non_horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Ignore close to horizontal lines
                if abs((y2 - y1) / (x2 - x1)) < 2:
                    continue
                num_non_horizontal_lines += 1
                cv2.line(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3,
                         cv2.LINE_AA)
            #if num_non_horizontal_lines > 0:
            #    print(f"Number of detected lines: {num_non_horizontal_lines}")
        return image_bgr

    def _publish_image(self, image, publisher):
        try:
            publisher.publish(self.bridge.cv2_to_imgmsg(image, "passthrough"))

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
