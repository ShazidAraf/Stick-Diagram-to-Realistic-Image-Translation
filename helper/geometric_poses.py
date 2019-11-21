import math
import cv2
import matplotlib.cm
import numpy as np
from pathlib import Path
import os
import pickle
import copy


Fs = 25


def get_circle_cords(center, radius , angle, k):
    
    cords=[]
    
    for i in  np.linspace(angle[0], angle[1], k):
        
        x = center[0] + radius*np.cos(np.deg2rad(i))
        y = center[1] - radius*np.sin(np.deg2rad(i))
        c = [x,y]
        cords.append(c)
    cords = np.array(cords, dtype=np.float)

    return cords

def transition(time, a , b):
    
    k = int(Fs*time)
    new_cords= np.zeros([k,b.shape[0],b.shape[1]])

    for i in range(k):
        new_cords[i] = a
    
    s1 = b-a
#     s2 = s1[:,1]/s1[:,0]
    
    
    for i in range(b.shape[0]):
        p1 = np.linspace(a[i][0],b[i][0],k)
        
        if s1[i][0] == 0:
            p2 = np.linspace(a[i][1],b[i][1],k)

        for j in range(k):
            new_cords[j][i][0] = p1[j]
            
            if s1[i][0] == 0:
                new_cords[j][i][1] = p2[j]
            else:
                m = s1[i,1]/s1[i,0]
                new_cords[j][i][1] = a[i][1] + m*(new_cords[j][i][0] - a[i][0])
#                 new_cords[j][i][1] = a[i][1] + s2[i]*(new_cords[j][i][0] - a[i][0])
                
    return new_cords
    


def combine(time,new_cords1,new_cords2,idx):
    
    if new_cords1.shape[0]> new_cords2.shape[0]:
        new_cords1 = new_cords1[0:new_cords2.shape[0]]
    else:
        new_cords2 = new_cords2[0:new_cords1.shape[0]]
    
    

    for i in range(new_cords1.shape[0]):

        for j in idx:

            new_cords1[i,j,:] = new_cords2[i,j,:]
            
            
    if new_cords1.shape[0]<int(time*Fs):
        d = int(time*Fs) -new_cords1.shape[0]
        t = copy.deepcopy(np.array(new_cords1[-1]))
        t = t[np.newaxis,:,:]

        for i in range(d):
            new_cords1 = np.concatenate((new_cords1,t),axis=0)
            

            
            
    return new_cords1




def pose0(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,0,0,0,0,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords

def pose1(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,0,0,0,1,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords


def pose2(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,0,0,1,0,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords


def pose3(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,0,0,1,1,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords


def pose4(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,0,1,0,0,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords


def pose5(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,0,1,0,1,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords


def pose6(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,0,1,1,0,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords

def pose7(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,0,1,1,1,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords
        
def pose8(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,1,0,0,0,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords


def pose9(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,1,0,0,1,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords


def pose10(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,1,0,1,0,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords


def pose11(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,1,0,1,1,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords

def pose12(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,1,1,0,0,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords


def pose13(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,1,1,0,1,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords


def pose14(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,1,1,1,0,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords

def pose15(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,1,1,1,1,angle1 =[-60,60],angle2 = [-90,-45])
    
    return new_cords




def pose16(time,cords,limb_length):
    new_cords = pose1_2(time,cords,limb_length,1,1,0,0,[60,150],[60,150],1)
    
    return new_cords

def pose17(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,1,1,0,0,[-60,-150],[-90,-45],1)
    
    return new_cords

def pose18(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,1,1,0,1,[0,90],[-90,-135],1)
    
    return new_cords

def pose19(time,cords,limb_length):
    
    new_cords = pose1_2(time,cords,limb_length,1,1,1,0,[180,90],[-90,-45],1)
    
    return new_cords





def pose20(time,cords,limb_length):    
    
    new_cords  = pose1_3(time,cords,limb_length,1,0,[-90,-180])
    return new_cords

def pose21(time,cords,limb_length):    
    
    new_cords  = pose1_3(time,cords,limb_length,0,1,[-90,-180])
    return new_cords
    
    
def pose22(time,cords,limb_length):    
    
    new_cords  = pose1_3(time,cords,limb_length,1,1,[-90,-180])
    return new_cords
    
    
def pose23(time,cords,limb_length):    
    
    new_cords1 = pose1_3(time/2,cords,limb_length,1,0,[-90,-180])
    new_cords2 = pose1_1(time/2,new_cords1[-1],limb_length,1,0)
    new_cords = np.concatenate((new_cords1 , new_cords2) ,axis= 0)
    return new_cords

def pose24(time,cords,limb_length):    
    
    new_cords1 = pose1_3(time/2,cords,limb_length,0,1,[-90,-180])
    new_cords2 = pose1_1(time/2,new_cords1[-1],limb_length,0,1)
    new_cords = np.concatenate((new_cords1 , new_cords2) ,axis= 0)
    return new_cords

def pose25(time,cords,limb_length):    
    
    new_cords1 = pose1_3(time/2,cords,limb_length,1,1,[-90,-180])
    new_cords2 = pose1_1(time/2,new_cords1[-1],limb_length,1,1)
    new_cords = np.concatenate((new_cords1 , new_cords2) ,axis= 0)
    return new_cords




def pose26(time,cords,limb_length):
    
    new_cords1 = pose1_3(time,cords,limb_length,1,0,[-270,-120])
    new_cords2 = pose1_3(time,new_cords1[-1],limb_length,1,0,[-120,-270])
    new_cords = np.concatenate((new_cords1 , new_cords2) ,axis= 0)
    
    return new_cords


def pose27(time,cords,limb_length):
    
    new_cords1 = pose1_3(time,cords,limb_length,0,1,[-270,-120])
    new_cords2 = pose1_3(time,new_cords1[-1],limb_length,0,1,[-120,-270])
    new_cords = np.concatenate((new_cords1 , new_cords2) ,axis= 0)
    
    return new_cords


def pose28(time,cords,limb_length):
    
    new_cords1 = pose1_3(time,cords,limb_length,1,1,[-270,-120])
    new_cords2 = pose1_3(time,new_cords1[-1],limb_length,1,1,[-120,-270])
    new_cords = np.concatenate((new_cords1 , new_cords2) ,axis= 0)
    
    return new_cords



def pose29(time,cords,limb_length):
    
    new_cords1 = pose1_3(time,cords,limb_length,1,0,[-300,-240])
    new_cords2 = pose1_3(time,new_cords1[-1],limb_length,1,0,[-240,-300])
    new_cords = np.concatenate((new_cords1 , new_cords2) ,axis= 0)
    
    return new_cords


def pose30(time,cords,limb_length):
    
    new_cords1 = pose1_3(time,cords,limb_length,0,1,[-300,-240])
    new_cords2 = pose1_3(time,new_cords1[-1],limb_length,0,1,[-240,-300])
    new_cords = np.concatenate((new_cords1 , new_cords2) ,axis= 0)
    
    return new_cords


def pose31(time,cords,limb_length):
    
    new_cords1 = pose1_3(time,cords,limb_length,1,1,[-300,-240])
    new_cords2 = pose1_3(time,new_cords1[-1],limb_length,1,1,[-240,-300])
    new_cords = np.concatenate((new_cords1 , new_cords2) ,axis= 0)
    
    return new_cords


def pose32(time,cords,limb_length):    
    
    
    new_cords1 = pose1_3(time/2,cords,limb_length,1,0,[-90,-240])
    
    new_cords2 = pose1_1(time/2,new_cords1[-1],limb_length,1,0,-150)
    
    new_cords = np.concatenate((new_cords1,new_cords2),axis = 0)
    return new_cords


def pose33(time,cords,limb_length):    
    
    
    new_cords1 = pose1_3(time/2,cords,limb_length,0,1,[-90,-240])
    
    new_cords2 = pose1_1(time/2,new_cords1[-1],limb_length,0,1,-150)
    
    new_cords = np.concatenate((new_cords1,new_cords2),axis = 0)
    return new_cords


def pose34(time,cords,limb_length):    
    
    
    new_cords1 = pose1_3(time/2,cords,limb_length,1,1,[-90,-240])
    
    new_cords2 = pose1_1(time/2,new_cords1[-1],limb_length,1,1,-150)
    
    new_cords = np.concatenate((new_cords1,new_cords2),axis = 0)
    return new_cords



def pose35(time,cords,l):
    new_cords1 = pose1_4(time/2,cords,l,1,0)
    new_cords2 = pose1_3(time/2,new_cords1[-1],l,1,0,[-90 ,-240])
    
    new_cords = np.concatenate((new_cords1,new_cords2),axis=0)
    
    return new_cords


def pose36(time,cords,l):
    new_cords1 = pose1_4(time/2,cords,l,1,0)
    new_cords2 = pose1_3(time/2,new_cords1[-1],l,0,1,[-90 ,-240])
    
    new_cords = np.concatenate((new_cords1,new_cords2),axis=0)
    
    return new_cords


def pose37(time,cords,l):
    new_cords1 = pose1_4(time/2,cords,l,1,0)
    new_cords2 = pose1_3(time/2,new_cords1[-1],l,1,1,[-90 ,-240])
    
    new_cords = np.concatenate((new_cords1,new_cords2),axis=0)
    
    return new_cords


def pose38(time,cords,l):
    new_cords1 = pose1_4(time/2,cords,l,0,1)
    new_cords2 = pose1_3(time/2,new_cords1[-1],l,0,1,[-90 ,-240])
    
    new_cords = np.concatenate((new_cords1,new_cords2),axis=0)
    
    return new_cords

def pose39(time,cords,l):
    new_cords1 = pose1_4(time/2,cords,l,0,1)
    new_cords2 = pose1_3(time/2,new_cords1[-1],l,1,0,[-90 ,-240])
    
    new_cords = np.concatenate((new_cords1,new_cords2),axis=0)
    
    return new_cords

def pose40(time,cords,l):
    new_cords1 = pose1_4(time/2,cords,l,0,1)
    new_cords2 = pose1_3(time/2,new_cords1[-1],l,1,1,[-90 ,-240])
    
    new_cords = np.concatenate((new_cords1,new_cords2),axis=0)
    
    return new_cords



def pose41(time,cords,l):
    
    new_cords = leg_march(time,cords,l)

                                
    return new_cords


def pose42(time,cords,l):
    
    new_cords1 = pose8(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords


def pose43(time,cords,l):
    
    new_cords1 = pose4(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords


def pose44(time,cords,l):
    
    new_cords1 = pose12(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords



def pose45(time,cords,l):
    
    new_cords1 = pose20(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords


def pose46(time,cords,l):
    
    new_cords1 = pose21(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords


def pose47(time,cords,l):
    
    new_cords1 = pose22(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords

def pose48(time,cords,l):
    
    new_cords1 = pose23(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords


def pose49(time,cords,l):
    
    new_cords1 = pose24(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords


def pose50(time,cords,l):
    
    new_cords1 = pose25(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords



def pose51(time,cords,l):
    
    new_cords1 = pose26(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords


def pose52(time,cords,l):
    
    new_cords1 = pose27(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords


def pose53(time,cords,l):
    
    new_cords1 = pose28(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords



def pose54(time,cords,l):
    
    new_cords1 = pose29(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords


def pose55(time,cords,l):
    
    new_cords1 = pose30(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords


def pose56(time,cords,l):
    
    new_cords1 = pose31(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords


def pose57(time,cords,l):
    
    new_cords1 = pose32(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords


def pose58(time,cords,l):
    
    new_cords1 = pose33(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords


def pose59(time,cords,l):
    
    new_cords1 = pose34(time,cords,l)
    new_cords2 = leg_march(time,cords,l)
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    
    idx = np.concatenate((right_leg_idx,left_leg_idx), axis =0)
    
    
    new_cords = combine(time,new_cords1,new_cords2,idx)
                              
    return new_cords




def pose1_1(time,cords,limb_length, R  , L , a=None):
    
    
    if a is None:
        a = -20
    
    k = int(Fs*time)
    new_cords= np.zeros([k,cords.shape[0],cords.shape[1]])
    
    for i in range(k):
        new_cords[i]=cords

        
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    hand_length_idx = np.array([1,4,5])
    leg_length_idx = np.array([9,10,11])
    
    
        
    if R==1:   

        s1 = cords[6]-cords[5]
        s1 = math.degrees(math.atan(s1[1]/s1[0])) 
        s2 = cords[7]-cords[6]
        s2 = math.degrees(math.atan(s2[1]/s2[0]))


        angle=np.array([-s1,a])


        center = cords[right_hand_idx[0]]

        t1 = right_hand_idx[1]
        t2 = right_hand_idx[2]

        r1 = limb_length[hand_length_idx[1]]
        r2 = limb_length[hand_length_idx[2]]

        p1 = get_circle_cords(center, r1 , angle, k )

        for i in range(len(p1)):
            new_cords[i,t1,:] = p1[i]
            new_cords[i,t2,0] = new_cords[i,t1,0] - r2*np.cos(np.deg2rad(s2))
            new_cords[i,t2,1] = new_cords[i,t1,1] - r2*np.sin(np.deg2rad(s2))
            
            
    if L==1:   

        s1 = cords[3]-cords[2]
        s1 = math.degrees(math.atan(s1[1]/s1[0])) 
        s2 = cords[4]-cords[3]
        s2 = math.degrees(math.atan(s2[1]/s2[0]))

        angle=np.array([-180-s1,-180-a])


        center = cords[left_hand_idx[0]]

        t1 = left_hand_idx[1]
        t2 = left_hand_idx[2]

        r1 = limb_length[hand_length_idx[1]]
        r2 = limb_length[hand_length_idx[2]]

        p1 = get_circle_cords(center, r1 , angle, k )

        for i in range(len(p1)):
            new_cords[i,t1,:] = p1[i]
            new_cords[i,t2,0] = new_cords[i,t1,0] + r2*np.cos(np.deg2rad(s2))
            new_cords[i,t2,1] = new_cords[i,t1,1] + r2*np.sin(np.deg2rad(s2))
        
    return new_cords
        

def pose1_2(time,cords,limb_length,rh,lh,rl,ll,angle1 = None,angle2 = None,same=None):
    
    if angle1 is None:
        angle1 = [-60,60]
    if angle2 is None:
        angle2 = [-90,-45]
    if same is None:
        same = 0
    

    
    k = int(Fs*time)

    
    new_cords= np.zeros([k,cords.shape[0],cords.shape[1]])
    
    for i in range(k):
        new_cords[i]=cords
    
    
    
    
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    hand_length_idx = np.array([1,4,5])
    leg_length_idx = np.array([9,10,11])
    
    
    if rh == 1:
        
        angle = angle1
        
        center = cords[right_hand_idx[0]]
        t1 = right_hand_idx[1]
        t2 = right_hand_idx[2]
        r1 = limb_length[hand_length_idx[1]]
        r2 = limb_length[hand_length_idx[1]]+limb_length[hand_length_idx[2]]
        
        p1 = get_circle_cords(center, r1 , angle, k )
        p2 = get_circle_cords(center, r2 , angle, k )

        
        for i in range(len(p1)):
            new_cords[i,t1,:] = p1[i]
            new_cords[i,t2,:] = p2[i]

            
            
    if lh == 1:
        
        
        angle =angle1
        
        if same == 0:
            angle[0] = 180 - angle[0]
            angle[1] = 180 - angle[1]
        
        center = cords[left_hand_idx[0]]
        t1 = left_hand_idx[1]
        t2 = left_hand_idx[2]
        r1 = limb_length[hand_length_idx[1]]
        r2 = limb_length[hand_length_idx[1]]+limb_length[hand_length_idx[2]]
        
            
        p1 = get_circle_cords(center, r1 , angle, k )
        p2 = get_circle_cords(center, r2 , angle, k )

        
        for i in range(len(p1)):
            new_cords[i,t1,:] = p1[i]
            new_cords[i,t2,:] = p2[i]

            
            
            
            
    if rl == 1:
        
        angle = angle2
        
        center = cords[right_leg_idx[0]]
        t1 = right_leg_idx[1]
        t2 = right_leg_idx[2]
        r1 = limb_length[leg_length_idx[1]]
        r2 = limb_length[leg_length_idx[1]]+limb_length[leg_length_idx[2]]
        
            
        p1 = get_circle_cords(center, r1 , angle, k )
        p2 = get_circle_cords(center, r2 , angle, k )
#         new_cords= np.zeros([len(p1),cords.shape[0],cords.shape[1]])
        
        for i in range(len(p1)):
            new_cords[i,t1,:] = p1[i]
            new_cords[i,t2,:] = p2[i]
#             new_cords[i] = cords
            
            
    if ll == 1:
        
        angle = angle2
        
        if same == 0:
            angle[0] = -180 - angle[0]
            angle[1] = -180 - angle[1]
        
        center = cords[left_leg_idx[0]]
        t1 = left_leg_idx[1]
        t2 = left_leg_idx[2]
        r1 = limb_length[leg_length_idx[1]]
        r2 = limb_length[leg_length_idx[1]]+limb_length[leg_length_idx[2]]
            
        p1 = get_circle_cords(center, r1 , angle, k )
        p2 = get_circle_cords(center, r2 , angle, k )
#         new_cords= np.zeros([len(p1),cords.shape[0],cords.shape[1]])
        
        for i in range(len(p1)):
            new_cords[i,t1,:] = p1[i]
            new_cords[i,t2,:] = p2[i]
#             new_cords[i] = cords
            
    
    return new_cords
        



def pose1_3(time,cords,limb_length,R,L,angle=None):
    
    
    if angle is None:
        angle = [-90,-180]

    
    k = int(Fs*time)
    new_cords= np.zeros([k,cords.shape[0],cords.shape[1]])
    
    for i in range(k):
        new_cords[i]=cords
        
        
    left_hand_idx = np.array([2,3,4])
    right_hand_idx = np.array([5,6,7])
    left_leg_idx = np.array([8,9,10])
    right_leg_idx = np.array([11,12,13])
    hand_length_idx = np.array([1,4,5])
    leg_length_idx = np.array([9,10,11])
    
    
#     angle=np.array([-90,-180])
    
    
    if R==1:

        center = cords[right_hand_idx[1]]
        t1 = right_hand_idx[2]
        r1 = limb_length[hand_length_idx[2]]
        p1 = get_circle_cords(center, r1 , angle, k )
        for i in range(len(p1)):
            new_cords[i,t1,:] = p1[i]

           
        
    if L==1:
        
        
        angle[0]= -180-angle[0]
        angle[1]= -180-angle[1]

        center = cords[left_hand_idx[1]]
        t1 = left_hand_idx[2]
        r1 = limb_length[hand_length_idx[2]]
        p1 = get_circle_cords(center, r1 , angle, k )
        for i in range(len(p1)):
            new_cords[i,t1,:] = p1[i]
        
    
    
    return new_cords
        

def pose1_4(time,cords,l,R=None,L=None):

    new_cords = copy.deepcopy(cords)
    if R==1:
        shift = 120
        new_cords[:,0] = new_cords[:,0] +shift
        new_cords[10,:] = cords[13,:]
        m = new_cords[8,:] - new_cords[10,:]
        m = m[1]/m[0]
    #     print(m)
        for i in range(9,11):
            t = new_cords[i,1]
            new_cords[i,0] = (t-new_cords[1,1])/m+new_cords[1,0] 
            
    if L==1:
        shift = -120
        new_cords[:,0] = new_cords[:,0] +shift
        new_cords[13,:] = cords[10,:]
        m = new_cords[11,:] - new_cords[13,:]
        m = m[1]/m[0]
    #     print(m)
        for i in range(12,14):
            t = new_cords[i,1]
            new_cords[i,0] = (t-new_cords[1,1])/m+new_cords[1,0] 
        
        
    new_cords = transition(time,cords,new_cords)
        
    return new_cords

def leg_march(time,cords,l):
    
    new_cords1 = copy.deepcopy(cords)

    a = l[7]*0.7
    b = l[7]*0.3
    
    new_cords1[12,1] = new_cords1[12,1]-a
    new_cords1[13,1] = new_cords1[13,1]-a
    new_cords1[12,0] = new_cords1[12,0]+b
#     new_cords1[13,0] = new_cords1[13,1]-b
    
    new_cords1 = transition(time/4,cords,new_cords1)
    new_cords2 = transition(time/4,new_cords1[-1],cords)
    
    
    new_cords3 = copy.deepcopy(cords)
    
    new_cords3[9,1] = new_cords3[9,1]-a
    new_cords3[10,1] = new_cords3[10,1]-a
    new_cords3[9,0] = new_cords3[9,0]-b
    
    new_cords3 = transition(time/4,cords,new_cords3)
    new_cords4 = transition(time/4,new_cords3[-1],cords)
    
    new_cords =  np.concatenate((new_cords1,new_cords2,new_cords3,new_cords4),axis=0)
                                
    return new_cords






def Geometrioc_Choreography(pose_no,time,b,l):
    
    if pose_no == 0:
        new_cords = pose0(time,b,l)
        
        
    if pose_no == 1:
        new_cords = pose1(time,b,l)
        
    if pose_no == 2:
        new_cords = pose2(time,b,l)
        
        
    if pose_no == 3:
        new_cords = pose3(time,b,l)
        
        
    if pose_no == 4:
        new_cords = pose4(time,b,l)
        
        
    if pose_no == 5:
        new_cords = pose5(time,b,l)
        
    if pose_no == 6:
        new_cords = pose6(time,b,l)
        
        
    if pose_no == 7:
        new_cords = pose7(time,b,l)
        
    if pose_no == 8:
        new_cords = pose8(time,b,l)
        
    if pose_no == 9:
        new_cords = pose9(time,b,l)
        
        
        
        
    if pose_no == 10:
        new_cords = pose10(time,b,l)
        
        
    if pose_no == 11:
        new_cords = pose11(time,b,l)
        
    if pose_no == 12:
        new_cords = pose12(time,b,l)
        
        
    if pose_no == 13:
        new_cords = pose13(time,b,l)
        
        
    if pose_no == 14:
        new_cords = pose14(time,b,l)
        
        
    if pose_no == 15:
        new_cords = pose15(time,b,l)
        
    if pose_no == 16:
        new_cords = pose16(time,b,l)
        
        
    if pose_no == 17:
        new_cords = pose17(time,b,l)
        
    if pose_no == 18:
        new_cords = pose18(time,b,l)
        
        
    if pose_no == 19:
        new_cords = pose19(time,b,l)
        

    if pose_no == 20:
        new_cords = pose20(time,b,l)
        
        
    if pose_no == 21:
        new_cords = pose21(time,b,l)
        
    if pose_no == 22:
        new_cords = pose22(time,b,l)
        
        
    if pose_no == 23:
        new_cords = pose23(time,b,l)
        
        
    if pose_no == 24:
        new_cords = pose24(time,b,l)
        
        
    if pose_no == 25:
        new_cords = pose25(time,b,l)
        
    if pose_no == 26:
        new_cords = pose26(time,b,l)
        
        
    if pose_no == 27:
        new_cords = pose27(time,b,l)
        
    if pose_no == 28:
        new_cords = pose28(time,b,l)
        
        
    if pose_no == 29:
        new_cords = pose29(time,b,l)
        
        
    if pose_no == 30:
        new_cords = pose30(time,b,l)
        
        
    if pose_no == 31:
        new_cords = pose31(time,b,l)
        
    if pose_no == 32:
        new_cords = pose32(time,b,l)
        
        
    if pose_no == 33:
        new_cords = pose33(time,b,l)
        
        
    if pose_no == 34:
        new_cords = pose34(time,b,l)
        
        
    if pose_no == 35:
        new_cords = pose35(time,b,l)
        
    if pose_no == 36:
        new_cords = pose36(time,b,l)
        
        
    if pose_no == 37:
        new_cords = pose37(time,b,l)
        
    if pose_no == 38:
        new_cords = pose38(time,b,l)
        
        
    if pose_no == 39:
        new_cords = pose39(time,b,l)
        
        
        
    if pose_no == 40:
        new_cords = pose40(time,b,l)
        
        
    if pose_no == 41:
        new_cords = pose41(time,b,l)
        
    if pose_no == 42:
        new_cords = pose42(time,b,l)
        
        
    if pose_no == 43:
        new_cords = pose43(time,b,l)
        
        
    if pose_no == 44:
        new_cords = pose44(time,b,l)
        
        
    if pose_no == 45:
        new_cords = pose45(time,b,l)
        
    if pose_no == 46:
        new_cords = pose46(time,b,l)
        
        
    if pose_no == 47:
        new_cords = pose47(time,b,l)
        
    if pose_no == 48:
        new_cords = pose48(time,b,l)
        
        
    if pose_no == 49:
        new_cords = pose49(time,b,l)
        
        
        
    if pose_no == 50:
        new_cords = pose50(time,b,l)
        
        
    if pose_no == 51:
        new_cords = pose51(time,b,l)
        
    if pose_no == 52:
        new_cords = pose52(time,b,l)
        
        
    if pose_no == 53:
        new_cords = pose53(time,b,l)
        
        
    if pose_no == 54:
        new_cords = pose54(time,b,l)
        
        
    if pose_no == 55:
        new_cords = pose55(time,b,l)
        
    if pose_no == 56:
        new_cords = pose56(time,b,l)
        
        
    if pose_no == 57:
        new_cords = pose57(time,b,l)
        
    if pose_no == 58:
        new_cords = pose58(time,b,l)
        
        
    if pose_no == 59:
        new_cords = pose59(time,b,l)
        
        
    return new_cords


