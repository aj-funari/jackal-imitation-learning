#!/usr/bin/env python

import os
import cv2
import torch
import rospy
from datetime import datetime
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist # message type for cmd_vel
from sensor_msgs.msg import Image # message type for image
from ResNet import ResNet, block 

bridge = CvBridge()
move = Twist()

img_channels = 3
num_classes = 2
ResNet50 = ResNet(block, [3,4,6,3], img_channels, num_classes)

class data_recorder(object):

    def __init__(self):
        self.data = None
        self.image = None
        self.tensor_x_z_actions = []
        self.count = 0
        # Node for Subscriber/Publisher
        self.node = rospy.init_node('listener', anonymous=True)
        self.vel = rospy.Subscriber('/jackal_velocity_controller/cmd_vel', Twist, self.cmd_callback)
        self.img = rospy.Subscriber('/d400/depth/image_rect_raw', Image, self.img_callback)
        self.rate = rospy.Rate(10)

    def format(self, string):
        msg = string.split()
        x = msg[2]
        z = msg[13]
        msg = x + '-' + z
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = msg + '-' + current_time + '-' + '.jpeg'
        return(msg)

    def cmd_callback(self, msg):
        jackal_velocity = self.format(str(msg))
        self.data = jackal_velocity  # subscribed to jackal velocity

        self.count += 1
        directory = '/home/vail/aj_ws/src/jackal/images/box_images'
        os.chdir(directory)

        cv2.imwrite(self.data, self.image)
        print(self.count,"images saved")


    def img_callback(self, image):
        try:
            # Convert ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(image)  # returns array
            self.image = cv2_img

        except CvBridgeError as e:
            pass


if __name__ =='__main__':
    print("I am in main!")
    data_recorder()
    rospy.spin()
