#!/usr/bin/python

import rospy

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
