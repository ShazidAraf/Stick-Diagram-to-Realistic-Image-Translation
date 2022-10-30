# Stick-Diagram-to-Realistic-Image-Translation

## Dataset
I recorded the video on my own. If you have few data make sure - video background is as simple as possible, no camera vibration during recording, uniform intensity of light in video .

you can also use this video. This is an ideal one

[Afrobeats Dance Workout - 20 Minutes Fat Burning Workout](https://www.youtube.com/watch?v=kyKNPPQW3bM&t=342s)



## Environments:

[Ubuntu](https://ubuntu.com/download/desktop) 16/18. (I used 18.04)

python 3.6.4 

opencv: 3.4 
```bash
pip install opencv-contrib-python==3.4.4.19
```
Pytorch 0.4.1

Torchvision 0.2.1 
```bash
conda install pytorch=0.4.1 torchvision cuda90 -c pytorch
``` 
(this command will install both pytorch and torchvision)

dominate
```bash
pip install dominate 
```
tqdm  
```bash
pip install tqdm 
```
# Overview

![Stick Figures are in on upper line, Realistic images are in lower line ](https://github.com/ShazidAraf/Stick-Diagram-to-Realistic-Image-Translation/blob/master/result/result.png)

# Usage



step-1: keep video of a person making free movements in 'data/target/video' folder

step-2: Extract Image from video. run '1.pre_processing.py'. It extracts images from video. You can control rate of extraction using frame_interval. Cropping and rotating function are also written. Output of this stage is 512 x 512 pixels images

step-3: Pose Estimation. I used OpenPose for pose estimation. run '2.image_to_stick_figure.py' to get stick diagram of folder 'data/target/images'. Stick figures are saved in 'data/train/train_label' folder.

step-4: Pix2pixHD training. run '3.train_pose2vid.py' to learn the mapping of stick figure to realistic image. The generator and discriminator trained model will be saved 'checkpoints/target' folder.

step-5: Give input stick diagrams to 'data/source' folder 

step-6: run '4.transfer.py'. It will translate stick diagrams of 'data/source' folder to realistic image in 'result/target/test_latest/images' folder

step-7: run '5.images_to_video.py' to create video of result images.


# Reference
[Pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
[Everybody Dance Now] (https://github.com/yanx27/EverybodyDanceNow_reproduce_pytorch)
