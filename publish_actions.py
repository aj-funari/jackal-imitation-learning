#!/usr/bin/env python

import torch
import rospy
from yaml import parse
from geometry_msgs.msg import Twist # message type for cmd_vel
from sensor_msgs.msg import Image # message type for image
from cv_bridge import CvBridge, CvBridgeError
from parse_data import parse_data
from ResNet_CPU import ResNet, block

bridge = CvBridge()
setup = parse_data()
move = Twist()

img_channels = 3
num_classes = 2
PATH = '/home/aj/catkin_ws/src/models/ResNet50.pt'

### LOAD MODEL
model = ResNet(block, [3,4,6,3], img_channels, num_classes)
model.load_state_dict(torch.load(PATH))
model.eval()

class data_recorder(object):

    def __init__(self):
        self.data = None
        self.left_image = None
        self.right_image = None
        self.tensor_x_z_actions = []
        self.count = 0
        # Node for Subscriber/Publisher
        self.node = rospy.init_node('talker', anonymous=True)
        self.img_left = rospy.Subscriber('/front/left/image_raw', Image, self.left_img_callback)
        self.pub = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=10) # definging the publisher by topic, message type
        self.rate = rospy.Rate(10)

    def left_img_callback(self, image):
        # print("I recieved an image!")
        try:
            # Convert ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')  # returns array

            # Feed image through neural network
            img_resize = setup.resize(cv2_img)
            img_tensor = torch.from_numpy(img_resize)
            img = img_tensor.reshape(1, 3, 224, 224)
            
            # move model/image to gpu 
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # ResNet50.to(device)
            # img = img.to(device)

            # feed image through neural network
            tensor_out = model(img)
            self.tensor_x_z_actions.append(tensor_out)
            print("actions sent to publisher:", tensor_out)

        except CvBridgeError as e:
            pass

    def publishMethod(self):    
        i = 0
        tmp = 1
        while not rospy.is_shutdown():
            # handle delay between subscriber and publisher
            if len(self.tensor_x_z_actions) == 0:
                pass
            else:
                if len(self.tensor_x_z_actions) >= tmp:  # publish actions only when action is sent from neural network output
                    # print("x-z actions:", self.tensor_x_z_actions[i])
                    move.linear.x = self.tensor_x_z_actions[i][0][0]
                    move.linear.z = self.tensor_x_z_actions[i][0][1]
                    rospy.loginfo("Data is being sent") 
                    self.pub.publish(move)
                    self.rate.sleep()
                    i += 1
                    tmp += 1

if __name__ =='__main__':
    data_recorder()
    pub = data_recorder()
    pub.publishMethod()
    rospy.spin()
