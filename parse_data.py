#!/usr/bin/env python

import os
import cv2
import torch
import random
import matplotlib.pyplot as plt

class parse_data():
    def __init__(self):
        self.trainloader = []
        self.training_label = []
        self.rand_batch_epoch = []
        self.rand_label_epoch = []
        self.batch_size = 0

    def parse_images(self):
        DATADIR = '/home/aj/catkin_ws/src/forest+hallway_images'

        print("----------------------------------")
        print("SETTING UP DATA FOR NEURAL NETWORK")
        print("----------------------------------")

        for img in os.listdir(DATADIR):
            try:
                img_array = cv2.imread(os.path.join(DATADIR, img))  # <type 'numpy.ndarray'>
                img_resize = self.resize(img_array)
                img_tensor = torch.from_numpy(img_resize).float()  # <class 'torch.Tensor')
                tensor = img_tensor.reshape([1, 3, 224, 224])
                self.trainloader.append(tensor)
            
            except Exception as e:
                pass

    def parse_labels(self):
        DATADIR = '/home/aj/catkin_ws/src/forest+hallway_images'
        count = 0

        for label in os.listdir(DATADIR):
            label = label.split('-')

            if len(label) == 4:  # positive x and z coordinates
                x = label[0]
                z = label[1]
                action = [x,z]
        
            if len(label) == 5:  # negative x or z coordinate
                if label[0] == '': # -x
                    x = '-' + label[1]
                    z = label[2]
                if label[1] == '': # -z
                    x = label[0]
                    z = '-' + label[2]

            if len(label) == 6:  # negative x and z coordinates
                x = '-' + label[1]
                z = '-' + label[3]

            action = [x,z]
            self.training_label.append(action)
            count += 1

    def resize(self, img): # image input size = 768x1024
        # plt.imshow(img)
        # plt.show()
        dim = (224, 224) # rescale down to 224x224
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return(img)

    def randomize_data (self, num_batch):
        self.batch_size = int(len(self.trainloader) / num_batch)
        batches_tmp = []
        labels_tmp = []
        start = 0
        end = self.batch_size 
        check_rand_num = {}
 
        for i in range(num_batch):  # for each batch
            check_rand_num.clear()  # clear list for random generated numbers
            for j in range(self.batch_size):  # for each image
                while len(check_rand_num) < self.batch_size: 
                    x = random.randrange(start, end)
                    if x not in check_rand_num.keys():  # if random number is not in dictorary, add random image
                        check_rand_num[x] = x # add num to check rand num list 
                        batches_tmp.append(self.trainloader[x])  # append imgage to temporary list
                        labels_tmp.append(self.training_label[x]) # append label to temporary list
                        # print(check_rand_num)
            
            start += self.batch_size
            end += self.batch_size
            
            ### ADD BATCHES OF IMAGES TOGETHER
            self.rand_batch_epoch += batches_tmp
            batches_tmp.clear()  # clear temporary list for next batch

            ### ADD BATCHES OF LABELS TOGETHER
            self.rand_label_epoch += labels_tmp
            labels_tmp.clear()

        ###  PRINT INFORMATION
        print("\n-----------------------------------")
        print("Number of batches:", num_batch)
        print("Size of batches in batch epoch:", self.batch_size)
        print("Total images:", len(self.rand_batch_epoch))
        print("-----------------------------------\n")  

if __name__ == "__main__":
    ### CREATE CLASS INSTANCE
    DATA = parse_data()

    ### PARSE DATA
    DATA.parse_images()
    DATA.parse_labels()

    ### RANDOMIZE IMAGES
    DATA.randomize_data(25)