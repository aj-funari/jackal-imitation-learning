#!/usr/bin/env python

import cv2
import torch
import rospy
import numpy
from geometry_msgs.msg import Twist # message type for cmd_vel
from sensor_msgs.msg import Image # message type for image
from cv_bridge import CvBridge, CvBridgeError
from parse_data import parse_data
from ResNet import ResNet, block

bridge = CvBridge()
setup = parse_data()
move = Twist()

img_channels = 3
num_classes = 2
ResNet50 = ResNet(block, [3,4,6,3], img_channels, num_classes)

### LOAD MODEL
model = ResNet(block, [3,4,6,3], img_channels, num_classes)
PATH = '/home/vail/aj_ws/src/jackal/models/boxes__1800%40_ResNet50_lr_0.002_wd_0.001_loss_0.8276655283239153.pt'
model.load_state_dict(torch.load(PATH))
model.eval()
# model.to(torch.device('cuda'))

class data_recorder(object):

    def __init__(self):
        self.data = None
        self.left_image = None
        self.right_image = None
        self.tensor_x_z_actions = []
        self.count = 0
        # Node for Subscriber/Publisher
        self.node = rospy.init_node('talker', anonymous=True)
        self.img_left = rospy.Subscriber('/d400/depth/image_rect_raw', Image, self.left_img_callback)
        self.pub = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=10) # definging the publisher by topic, message type
        self.rate = rospy.Rate(10)

    def left_img_callback(self, image):
        '''
        Convert ROS Image message to OpenCV2 Image
        
        Because I am subscribing to depth camera for testing
        I need to convert the subscribed image from grey scale
        to RGB. Neural network only accepts RGB images. 
        '''
        
        # try:
        cv2_img = bridge.imgmsg_to_cv2(image)  # returns array
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
        print("Image received and converted to RGB!")

        # resize image 
        img_resize = setup.resize(img_rgb)
        img_float32 = numpy.array(img_resize, dtype=numpy.float32)
        img_tensor = torch.from_numpy(img_float32)
        image = img_tensor.reshape(1, 3, 224, 224)
        
        # move images to GPU
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # image = image.to(device)
        
        # feed image through neural network
        tensor_out = model(image)
        self.tensor_x_z_actions.append(tensor_out)
        print(tensor_out)

        # except CvBridgeError as e:
        #     pass

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
                    move.linear.x = self.tensor_x_z_actions[i][0][0]/16
                    move.linear.z = self.tensor_x_z_actions[i][0][1]/16
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
