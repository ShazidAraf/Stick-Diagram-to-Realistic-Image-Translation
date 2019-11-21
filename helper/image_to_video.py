Fs =25

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
	pass

