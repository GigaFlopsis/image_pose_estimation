#!/usr/bin/env python
# coding=utf8

import os.path
import cv2

import rospy
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf

import image_processing

bridge = CvBridge()
cv_image = Image()

topic_tf_child = "/object"
topic_tf_perent = "/base_link"

t = TransformStamped()
tf2_br = tf2_ros.TransformBroadcaster()


def image_clb(data):
    """
    get image from ros
    :type data: Image
    :return:
    """
    global bridge, cv_image, get_image_flag, topic_tf_perent

    topic_tf_perent = data.header.frame_id

    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        get_image_flag = True
    except CvBridgeError as e:
        print(e)

def pubTf(position, orientation):
    """
    publish find object to tf2
    :param position:
    :param orientation:
    :return:
    """
    global t
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = topic_tf_perent
    t.child_frame_id = topic_tf_child
    t.transform.translation.x = position[0]
    t.transform.translation.y = position[1]
    t.transform.translation.z = position[2]
    quaternion = tf.transformations.quaternion_from_euler(orientation[0], orientation[1], orientation[2])

    t.transform.rotation.x = quaternion[0]
    t.transform.rotation.y = quaternion[1]
    t.transform.rotation.z = quaternion[2]
    t.transform.rotation.w = quaternion[3]

    tf2_br.sendTransform(t)

def camera_info_clb(data):
    """
    Get cam info
    :param data:
    :return:
    """
    global image_proc
    for i in range(1,4):
        for k in range(1,4):
            image_proc.camera_parameters[0][1] = data.K[(i*k)-1]
    for i in range(5):
        image_proc.camera_distortion_param[i] = data.D[i]


############
### Main ###
############
if __name__ == '__main__':
    rospy.init_node('image_pose_estimation_node', anonymous=True)

    _rate = 10.                 # this is a voracious application, so I recommend to lower the frequency, if it is not critical

    MIN_MATCH_COUNT = 10        # the lower the value, the more sensitive the filter
    blur_threshold = 300        # the higher the value, the more sensitive the filter
    max_dist = 10.              # publish objects that are no further than the specified value

    size_image = 2.             # the width of the image in meters

    use_image = False           # uses a known image
    image_path = "image.jpg"    # path to known image

    show_image = True           # show image in window
    camera_name = "camera"      # the name of the camera in ROS


    # init params
    camera_name = rospy.get_param("~camera_name", camera_name)
    topic_tf_child = rospy.get_param("~frame_id", topic_tf_child)
    show_image = rospy.get_param("~show_image", show_image)

    use_image = rospy.get_param("~use_image", use_image)
    image_path = rospy.get_param("~image_path", image_path)
    size_image = rospy.get_param("~size_image", size_image)

    MIN_MATCH_COUNT = rospy.get_param("~min_match_count", MIN_MATCH_COUNT)
    blur_threshold = rospy.get_param("~blur_threshold", blur_threshold)
    _rate = rospy.get_param("~rate", blur_threshold)
    _rate = -1. if _rate <= 0. else _rate

    #  Check init params
    if use_image is False and show_image is False:
        rospy.logerr("image not set.\n"
                     "Solutions:\n"
                     "* Enable param: show_image = True\n"
                     "* Set path to image in param: image_path and use_image = true")
        exit()

    if use_image is True:
        if image_path == "" or os.path.isfile(image_path) is False:
            rospy.logerr("Path to image invalid.\n"
                         "Solutions:\n"
                         "* Set path to image in param: image_path\n"
                         "* Disable param: use_image = False")
            exit()

    # init params
    get_image_flag = False

    image_proc = image_processing.ImageEstimation(MIN_MATCH_COUNT, 300, use_image, size_image, image_path, show_image)
    image_proc.max_dist = max_dist

    rate = rospy.Rate(_rate)
    rospy.Subscriber(camera_name+"/image_raw", Image, image_clb)
    rospy.Subscriber(camera_name+"/camera_info", CameraInfo, camera_info_clb)

    while not rospy.is_shutdown():
        if get_image_flag:
            # read the current frame
            k = cv2.waitKey(1) & 0xFF
            frame, trans, rot = image_proc.update(cv_image, k)

            if trans is not None and rot is not None:
                pubTf(trans,rot)

            if get_image_flag and show_image:
                cv2.imshow("frame", frame)

            # break when user pressing ESC
            if k == 27:
                break
        if _rate > 0.:
            rate.sleep()
    cv2.destroyAllWindows()
