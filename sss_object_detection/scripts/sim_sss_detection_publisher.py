#!/usr/bin/env python

import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Odometry
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2DArray, Detection2D
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped, PointStamped
from tf.transformations import euler_from_quaternion
import copy
import math


class sim_sss_detector:
    """A mock SSS object detector for simulation. Only objects within the
    detection_range of the vehicle will be detectable."""
    def __init__(self,
                 robot_name,
                 detection_range=8,
                 buoy_radius=0.20,
                 noise_sigma=.1):
        self.robot_name = robot_name
        self.detection_range = detection_range
        self.buoy_radius = buoy_radius
        self.noise_sigma = noise_sigma
        self.published_frame_id = '{}/base_link'.format(self.robot_name)
        self.gt_frame_id = 'gt/{}/base_link'.format(self.robot_name)
        self.current_pose = None
        self.marked_positions = {}

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.odom_sub = rospy.Subscriber(
            '/{}/sim/odom'.format(self.robot_name), Odometry,
            self._update_pose)
        self.marked_pos_sub = rospy.Subscriber(
            '/{}/sim/marked_positions'.format(robot_name), MarkerArray,
            self._update_marked_positions)
        self.pub = rospy.Publisher(
            '/{}/payload/sidescan/detection_hypothesis'.format(robot_name),
            Detection2DArray,
            queue_size=2)
        self.pub_detected_markers = rospy.Publisher(
            '/{}/payload/sidescan/detected_markers'.format(robot_name),
            Marker,
            queue_size=2)

    def _update_marked_positions(self, msg):
        """Update marked_positions based on the MarkerArray msg received."""
        if len(self.marked_positions) > 0:
            return
        for marker in msg.markers:
            self.marked_positions['{}/{}'.format(marker.ns,
                                                 marker.id)] = marker
        print('There are {} number of marked positions'.format(
            len(self.marked_positions)))

    def _update_pose(self, msg):
        """Update pose based on msg from simulated groundtruth odom at /{robot_name}/sim/odom"""

        self.current_pose = self._transform_pose(
            msg.pose, from_frame=msg.header.frame_id).pose

        markers_in_range = self.get_markers_in_detection_range()
        heading = self.calculate_heading()

        for marker in markers_in_range:
            cos_sim = self.calculate_marker_cosine_angle(heading, marker)
            detectable = cos_sim <= self.buoy_radius

            if detectable:
                print('Detected {}, cos = {}'.format(marker, cos_sim))
                self._publish_marker_detection(self.marked_positions[marker],
                                               cos_sim)

    def _publish_marker_detection(self, marker, cos_sim):
        """Publish detected marker"""
        distance = self._get_distance_to_marker(marker)

        object_hypothesis = ObjectHypothesisWithPose()
        object_hypothesis.id = 1
        # the higher the cos_sim (further away from 90 degree angle between current_pose
        # and the marker), the lower the score
        object_hypothesis.score = (-cos_sim + (self.buoy_radius * 2)) / (
            self.buoy_radius * 2)

        marker_transformed = self._transform_pose(
            marker, from_frame=marker.header.frame_id)
        object_hypothesis.pose.pose = marker_transformed.pose
        # Add random noise to pose of object
        object_hypothesis.pose.pose.position.x += np.random.randn(
        ) * self.noise_sigma
        object_hypothesis.pose.pose.position.y += np.random.randn(
        ) * self.noise_sigma
        object_hypothesis.pose.pose.position.z += np.random.randn(
        ) * self.noise_sigma
        object_hypothesis.pose.pose.orientation.x += np.random.randn(
        ) * self.noise_sigma
        object_hypothesis.pose.pose.orientation.y += np.random.randn(
        ) * self.noise_sigma
        object_hypothesis.pose.pose.orientation.z += np.random.randn(
        ) * self.noise_sigma
        object_hypothesis.pose.pose.orientation.w += np.random.randn(
        ) * self.noise_sigma

        # Wrap ObjectHypothesisWithPose msg into Detection2D msg
        detection_msg = Detection2D()
        detection_msg.header.frame_id = self.published_frame_id
        detection_msg.header.stamp = rospy.Time.now()
        detection_msg.results.append(object_hypothesis)

        # Wrap Detection2D msg into Detection2DArray msg
        detection_array_msg = Detection2DArray()
        detection_array_msg.header = detection_msg.header
        detection_array_msg.detections.append(detection_msg)
        self.pub.publish(detection_array_msg)

        # Publish detection as a Marker for visualization in rviz
        detected_marker = copy.deepcopy(marker)
        detected_marker.header.stamp = rospy.Time.now()
        detected_marker.ns = 'detected_{}'.format(detected_marker.ns)
        detected_marker.color = ColorRGBA(0, 1, 0, 1)
        detected_marker.lifetime.secs = 1
        self.pub_detected_markers.publish(detected_marker)

    def _get_position_differences(self, position1, position2):
        dx = position1.x - position2.x
        dy = position1.y - position2.y
        dz = position1.z - position2.z
        return dx, dy, dz

    def _normalize_vector(self, position_array):
        """Given an np.ndarray, return the normalized equivalent"""
        norm = np.linalg.norm(position_array)
        if norm > 0:
            position_array = position_array / norm
        return position_array

    def calculate_heading(self):
        """Calculate a normalized heading vector using current orientation"""
        quaternion = self.current_pose.orientation
        (_, pitch, yaw) = euler_from_quaternion(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        heading = np.array([
            math.cos(yaw) * math.cos(pitch),
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch)
        ]).reshape(-1, 1)
        heading = self._normalize_vector(heading)
        return heading

    def _calculate_distance_to_position(self, position):
        """Calculate the distance between current_pose.position and the given position"""
        dx, dy, dz = self._get_position_differences(position,
                                                    self.current_pose.position)
        return (dx**2 + dy**2 + dz**2)**.5

    def _get_vec_to_position(self, position, normalized=True):
        """Return vector from current_pose.position to the given position"""
        dx, dy, dz = self._get_position_differences(position,
                                                    self.current_pose.position)
        vec_to_position = np.array([dx, dy, dz]).reshape(-1, 1)

        if normalized:
            vec_to_position = self._normalize_vector(
                position_array=vec_to_position)
        return vec_to_position

    def _wait_for_transform(self, from_frame, to_frame):
        """Wait for transform from from_frame to to_frame"""
        trans = None
        while trans is None:
            try:
                trans = self.tf_buffer.lookup_transform(
                    to_frame, from_frame, rospy.Time(0), rospy.Duration(3.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as error:
                print('Failed to transform. Error: {}'.format(error))
        return trans

    def _transform_pose(self, pose, from_frame):
        trans = self._wait_for_transform(from_frame=from_frame,
                                         to_frame=self.gt_frame_id)
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose, trans)
        return pose_transformed

    def _get_distance_to_marker(self, marker):
        """Return distance from the marker to current_pose"""
        marker_transformed = self._transform_pose(marker,
                                                  marker.header.frame_id)

        distance = self._calculate_distance_to_position(
            marker_transformed.pose.position)
        return distance

    def get_markers_in_detection_range(self):
        """Returns a list of markers within detection_range relative to
        self.current_pose"""
        markers_in_range = []
        for marker_name, marker in self.marked_positions.items():
            distance = self._get_distance_to_marker(marker)
            if distance < self.detection_range:
                markers_in_range.append(marker_name)
        return markers_in_range

    def calculate_marker_cosine_angle(self, heading, marker):
        """Calculate the cosine between the heading and the marker position.
        Used to determine whether the marker is observable:
        A marker is observable if the magnitude of the projection of the vector
        from self.current_pose.position onto the heading vector <= the marker's radius."""
        pose = self.marked_positions[marker]
        marker_transformed = self._transform_pose(
            pose, from_frame=pose.header.frame_id)

        vec_to_marker_position = self._get_vec_to_position(
            marker_transformed.pose.position, normalized=True)
        cos_heading_marker = np.dot(heading.reshape(1, -1),
                                    vec_to_marker_position.reshape(-1,
                                                                   1))[0][0]
        return abs(cos_heading_marker)


def main():
    rospy.init_node('sim_sss_detection_publisher', anonymous=True)
    rospy.Rate(5)  # ROS Rate at 5Hz

    robot_name_param = '~robot_name'
    if rospy.has_param(robot_name_param):
        robot_name = rospy.get_param(robot_name_param)
        print('Getting robot_name = {} from param server'.format(robot_name))
    else:
        robot_name = 'sam'
        print('{} param not found in param server.\n'.format(robot_name_param))
        print('Setting robot_name = {} default value.'.format(robot_name))

    detector = sim_sss_detector(robot_name=robot_name)

    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == '__main__':
    main()
