import cv2
import os
from tqdm import tqdm
import numpy as np

#Input is a video of size 1080 x 720
#Making it into images of 512 x 512 by cropping and resizing  

def getFrame(sec,save_dir,count):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(save_dir+"/{:06d}".format(count)+".png", image)
    return hasFrames



#Image Rotate
def rotate(img,angle):
    rows,cols,dim = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

frame_interval = 0.04

vid_dir = 'data/target/video/test_video.mp4'

save_dir =   'data/target/images/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


vidcap = cv2.VideoCapture(vid_dir)

start = 0
count=0
sec = start


success = getFrame(sec,save_dir,count)

	# while (success and sec<end):
while success:
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec,save_dir,count)
    count = count + 1


rot =0
if rot==1:
    for i in tqdm(range(len(os.listdir(save_dir)))):
        img_name = '{0}/{:06d}'.format(save_dir,i)+'.png'
        img = cv2.imread(img_name)
        rot_img = rotate(img,180)
        cv2.imwrite(img_name,rot_img)



for i in tqdm(range(len(os.listdir(save_dir)))):
    
#     if i>1:
#         continue

    img_name = save_dir+'/{:06d}'.format(i)+'.png'
    img = cv2.imread(img_name)
    
    cropped_img = img[:,250:970,:]
    final_img = cv2.resize(cropped_img,(512,512))
    cv2.imwrite(img_name,final_img)
