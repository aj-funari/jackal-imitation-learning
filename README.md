# Jackal Imitation Learning
The purpose of this project is to explore control options a differential drive robot. Control inputs: RGB images, depth images, and LiDar.
1) Data Collection
    - Collect images with labels {linear velocty, angular velocity}.timestep
    - Images are collected when manually controlling Jackal through simulated environment
2) Train Convolution Neural Network
    - Train model using camera images as input
    - {linear velocity, angular velocity} as output
3) Test model in simulation and real world environments

Videos: https://drive.google.com/drive/folders/1i6puXClETlsWGkH8pF0k0eiL08-Jq8BF?usp=sharing
