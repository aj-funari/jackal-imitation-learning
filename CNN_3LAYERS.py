#!/usr/bin/env python

import time
import torch
import torch.nn as nn
from parse_data import parse_data

class CNN_3LAYERS(nn.Module):  # input: 224 x 224 x 3
    def __init__(self, image_channels, num_classes):
        super(CNN_3LAYERS, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=0)
        # self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=7, stride=2, padding=0)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=7, stride=2, padding=0)
        # self.bn3 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1*16, num_classes)

    def forward(self, x): 
        x = self.conv1(x.float())
        # x = self.bn1(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)  # ???
        x = self.fc(x)
        return x

if __name__ == '__main__':
    ### CREATE CLASS INSTANCE
    DATA = parse_data()

    ### PARSE DATA
    DATA.parse_images()
    DATA.parse_labels()

    ### RANDOMIZE BATCHES
    print("\n------------------------")
    print("Enter number of batches: ")
    number = int(input())
    print("------------------------")
    DATA.randomize_data(number)

    # ### CALL MODEL
    learning_rate = 0.001
    weight_decay = 0.001

    net = CNN_3LAYERS(image_channels=3, num_classes=2)
    print(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    x = 0
    num = 1
    lst = []
    x_accuracy = 0
    z_accuracy = 0
    total_loss = 0

    for input_batch in DATA.rand_batch_epoch:  # loop through randomized epochs
        for image in input_batch:   # for each image in epoch
            total_time = time.time()
            start_time = time.time()
            image = image.reshape(1, 3, 224, 224)
            image = image / 255.0
            
            ### FEED IMAGE THROUGH NEURAL NETWORK --> USES NOT RANDOMIZED LIST FIX!!!
            output = net(image)

            ### CHECK LIST IS NOT EMPTY
            if DATA.rand_label_epoch[x][0] != "" and DATA.rand_label_epoch[x][1] != "":
               
                ### CONVERT X-Z ACTIONS TO TENSORS
                x_target = torch.as_tensor(float(DATA.rand_label_epoch[x][0]))
                z_target = torch.as_tensor(float(DATA.rand_label_epoch[x][1]))
                print(x_target, z_target)

            ### MEAN SQUARED ERROR
            x_mae_loss = nn.L1Loss()
            z_mae_loss = nn.L1Loss()
            x_out = torch.as_tensor(output[0][0])
            z_out = torch.as_tensor(output[0][1])
            x_loss = x_mae_loss(x_out, x_target)
            z_loss = z_mae_loss(z_out, z_target)
            loss = x_loss + z_loss  # backpropogation stays separate for images
            optimizer.zero_grad()  # clears the optimizer/gradient
            loss.backward()
            optimizer.step()

            ### CALCULATING TOTAL LOSS
            total_loss += float(loss)

            ### REPORT INFORMATION EACH BATCH
            x += 1
            if x == DATA.batch_size:
                # SAVE LOSS FOR FILENAME
                filename_loss = total_loss/DATA.batch_size

                print("--------------------------------")
                print("BATCH #", num)
                print("RUMTIME #", time.time() - start_time)
                print(" start time:", start_time)
                print(" end time:", time.time())
                print(" loss:", total_loss/DATA.batch_size)
                print("--------------------------------\n")
                
                time.sleep(5)
                start_time = 0
                num += 1
                x -= DATA.batch_size
                x_accuracy = 0
                z_accuracy = 0
                total_loss = 0

    # print("FINISHED TRAINING NEURAL NETWORK!")

    PATH = '/home/aj/catkin_ws/src/models/CNN_3LAYERS_lr_0.001_wd_0.001_loss_' + str(filename_loss) + '.pt'
    torch.save(net.state_dict(), PATH)
    print("------------")
    print("MODEL SAVED!")
    print("------------")