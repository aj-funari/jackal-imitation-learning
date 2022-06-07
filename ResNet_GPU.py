#!/usr/bin/env python

import time
import torch
import torch.nn as nn
from parse_data import parse_data

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4  # the number of channels after the block is always 4 times what is was when it entered
        self.conv1 = nn.Conv2d(
            in_channels, 
            intermediate_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels, 
            intermediate_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels, 
            intermediate_channels * self.expansion,
            kernel_size=1, 
            stride=1, 
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample  # only for first block
                                                        # this will be a conv layer that we're going to do to the identity 
                                                        # mapping so that it's of the same shape later on in the layers
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):  # ResNet50 = [3, 4, 6, 3]
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers 
        self.layer1 = self.make_layer(
            block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layer(
            block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layer(
            block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layer(
            block, layers[3], intermediate_channels=512, stride=2)  # 2048

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x.float())
        x = self.bn1(x)
        self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        # only case when we change the numbere of channels and stide is in the first block
        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)  # changes the number of channels
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))  # in_channels = 256; out_challens = 64: 256 -> 64, 64*4 = 256 again 

        return nn.Sequential(*layers)  # * --> will unpack list so PyTorch knows each layers come after another

def ResNet50(img_channels=3, num_classes=2):
    return(ResNet(block, [3,4,6,3], img_channels, num_classes))

def ResNet101(img_channels=3, num_classes=2):
    return(ResNet(block, [3,4,23,3], img_channels, num_classes))

def ResNet152(img_channels=3, num_classes=2):
    return(ResNet(block, [3,8,36,3], img_channels, num_classes))

def test():
    net = ResNet50(img_channels=3, num_classes=10)
    y = net(torch.randn(1, 3, 244, 244))
    print("ResNet50 shape:", y.size())
    print(("ResNet50 output:", y))

    net = ResNet101(img_channels=3, num_classes=10)
    y = net(torch.randn(1, 3, 224, 224))
    print("ResNet101 shape:", y.size())
    print(("ResNet101 output:", y))

    net = ResNet152(img_channels=3, num_classes=10)
    y = net(torch.randn(1, 3, 224, 224))
    print("ResNet152 shape:", y.size())
    print(("ResNet152 output:", y))

if __name__ == '__main__':
    ### CREATE CLASS INSTANCE
    DATA = parse_data()

    ### PARSE DATA
    DATA.parse_images()
    DATA.parse_labels()

    ### NUMBER OF IMAGES
    count = 0
    for i in DATA.trainloader:
        count += 1
    print("number of images: ", count)

    ### NUMBER OF LABELS
    count = 0
    for label in DATA.training_label:
        count += 1
    print("number of labels: ", count)

    ### BATCHES
    DATA.rand_batches_labels(10)

    print("\n-----------------------------------------------")
    print("MOVE TRAINING TO GPU")
    print("Hardware?", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Is GPU available?", torch.cuda.is_available())
    print("-----------------------------------------------")

    ### CALL MODEL
    net = ResNet50(img_channels=3, num_classes=2)

    ### MOVE MODEL/COMPUTATIONS TO GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    if torch.cuda.is_available():
        for input_batch in DATA.batch_epoch:
            input_batch = input_batch.to('cuda')
    with torch.no_grad():
        output = net(input_batch)

    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, weight_decay=0.001)

    x = 0
    num = 1
    lst = []
    x_accuracy = 0
    z_accuracy = 0
    total_loss = 0

    for input_batch in DATA.batch_epoch:  # loop through 10 batches in epoch
        for image in input_batch:   # for each image in batch
            image = image.to('cuda')
            total_time = time.time()
            start_time = time.time()
            image = image.reshape(1, 3, 224, 224)
            
            ### FEED IMAGE THROUGH NEURAL NETWORK
            output = net(image)

            if DATA.training_label[x][0] != "" and DATA.training_label[x][1] != "":
               
                x_target = float(DATA.training_label[x][0])
                x_target = torch.as_tensor(x_target)
                z_target = float(DATA.training_label[x][1])
                z_target = torch.as_tensor(z_target)

                x_target = torch.as_tensor(float(DATA.training_label[x][0]))
                z_target = torch.as_tensor(float(DATA.training_label[x][1]))
                x_target, z_target = x_target.cuda(), z_target.cuda()

            ### MEAN SQUARED ERROR
            x_mae_loss = nn.L1Loss()
            z_mae_loss = nn.L1Loss()
            x_out = torch.as_tensor(output[0][0])
            z_out = torch.as_tensor(output[0][1])
            x_loss = x_mae_loss(x_out, x_target)
            z_loss = z_mae_loss(z_out, z_target)
            # x_loss = x_mae_loss(x_out.cuda(), x_target.cuda())
            # z_loss = z_mae_loss(z_out.cuda(), z_target.cuda())
            loss = x_loss + z_loss  # backpropogation stays separate for images
            optimizer.zero_grad()  # clears the optimizer/gradient
            loss.backward()
            optimizer.step()

            ### CALCULATING TOTAL LOSS
            total_loss += float(loss)

            ### PRINTING TRAINING TIME OF NEURAL NETWORK
            x += 1
            if x == DATA.batch_size:
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

    print("Neural Network training time: ", (time.time() - total_time))
    print("FINISHED TRAINING NEURAL NETWORK!")
