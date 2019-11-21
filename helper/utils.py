import math
import cv2
import matplotlib.cm
import numpy as np
from pathlib import Path
import os
import pickle
import copy



cmap = matplotlib.cm.get_cmap('hsv')

joint_to_limb_heatmap_relationship = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    [2, 16], [5, 17]]

colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]



NUM_JOINTS = 18
NUM_LIMBS = len(joint_to_limb_heatmap_relationship)


body_cords = np.load('cords/body_cords.npy')
parts_id = np.load('cords/parts_id.npy')
l = np.load('cords/limb_length.npy')
Fs = 25



def create_label(shape, joint_list, person_to_joint_assoc,plot_ear_to_shoulder=False):

    # print(joint_list)
    # print(person_to_joint_assoc)
    label = np.zeros(shape, dtype=np.uint8)
    which_limbs_to_plot = NUM_LIMBS if plot_ear_to_shoulder else NUM_LIMBS - 2
 
    for limb_type in range(which_limbs_to_plot):
        for person_joint_info in person_to_joint_assoc:
            joint_indices = person_joint_info[joint_to_limb_heatmap_relationship[limb_type]].astype(int)
            if -1 in joint_indices:
                continue
            joint_coords = joint_list[joint_indices, :2]
#             print(joint_coords)
            coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))

            limb_dir = joint_coords[0, :] - joint_coords[1, :]
            limb_length = np.linalg.norm(limb_dir)
            angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
            polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(label, polygon, colors[limb_type])
    return label


def create_label_bw(shape, joint_list, person_to_joint_assoc,plot_ear_to_shoulder=False):

    # print(joint_list)
    # print(person_to_joint_assoc)
    label = np.zeros(shape, dtype=np.uint8)
    which_limbs_to_plot = NUM_LIMBS if plot_ear_to_shoulder else NUM_LIMBS - 2
 
    for limb_type in range(which_limbs_to_plot):
        for person_joint_info in person_to_joint_assoc:
            joint_indices = person_joint_info[joint_to_limb_heatmap_relationship[limb_type]].astype(int)
            if -1 in joint_indices:
                continue
            joint_coords = joint_list[joint_indices, :2]
#             print(joint_coords)
            coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))

            limb_dir = joint_coords[0, :] - joint_coords[1, :]
            limb_length = np.linalg.norm(limb_dir)
            angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
            polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(label, polygon, limb_type+1)
    return label


def limb_length(joint_list, person_to_joint_assoc):

    
    which_limbs_to_plot = NUM_LIMBS - 2
    l = np.zeros(which_limbs_to_plot)
#     print(l)
 
    for limb_type in range(which_limbs_to_plot):
        for person_joint_info in person_to_joint_assoc:
            joint_indices = person_joint_info[joint_to_limb_heatmap_relationship[limb_type]].astype(int)
            if -1 in joint_indices:
                continue
            joint_coords = joint_list[joint_indices, :2]
#             print(joint_coords)
            coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))

            limb_dir = joint_coords[0, :] - joint_coords[1, :]
            limb_length = np.linalg.norm(limb_dir)
            l[limb_type] = limb_length
    return l



def plot_all(start,save_dir,new_cords, parts_id ):
    
    for i in range(len(new_cords)):
        label = create_label((512,512,3), new_cords[i] ,parts_id)
        cv2.imwrite(str(save_dir.joinpath('{:05}.png'.format(i+start))), label)


def plot_all_bw(start,save_dir,new_cords, parts_id ):
    
    for i in range(len(new_cords)):
        label = create_label_bw((512,512), new_cords[i] ,parts_id)
        cv2.imwrite(str(save_dir.joinpath('{:05}.png'.format(i+start))), label)




def image_to_video(pathIn,pathOut):
    import cv2
    import numpy as np
    import os
    from os.path import isfile, join 
    pathIn = './'+ pathIn +'/'
    # pathOut = 'video.avi'
    fps =Fs
    frame_array = []


    images = []

    for i in range(0,len(os.listdir(pathIn))):
        
        a = ('{:05}.png'.format(i))
        images.append(a)
        
    files = images

    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
