# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:26:29 2022

@author: Anis Maysami
"""
#This class measure assymetry index in X and Y
import cv2
import numpy as np

def Assymetry(image, cnt):
    #image is binary image
    (h, w)=image.shape[:2]
    
    m=cv2.moments(cnt)
    cx=int(m['m10']/m['m00'])
    cy=int(m['m01']/m['m00'])
    ####
    ellipse=cv2.fitEllipse(cnt)
    angle=ellipse[2]
    angle=angle-90
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated=cv2.warpAffine(image, M, (w, h))
    
    #Y axis assymety
    half_horz_1=rotated[:,0:cx+1]#[:,:,2]

    half_horz_2=rotated[:,cx:]#[:,:,2]
    half_horz_2=cv2.flip(half_horz_2,1)

    width_img_1=half_horz_1.shape[1]
    width_img_2=half_horz_2.shape[1]
    if width_img_2 > width_img_1:
        half_horz_2=half_horz_2[:, width_img_2-width_img_1:]
    elif width_img_2 < width_img_1: 
        half_horz_1=half_horz_1[:, width_img_1-width_img_2:]

    else:
        pass



    counter_X1=0 #Counter for pixels that equals to 255 for union1
    counter_X2=0 #Counter for pixels that equals to 255 for intersection1
    counter_Y1=0 #Counter for pixels that equals to 255 for union2
    counter_Y2=0 #Counter for pixels that equals to 255 for intersection1
    #union of two section
    union1=cv2.bitwise_or(half_horz_1, half_horz_2)

    #Intersection of two part
    intersection1=cv2.bitwise_and(half_horz_1, half_horz_2)


    #Count numper pixels of union1 
    counter_X1 = np.count_nonzero(union1)

    #Count number pixels of intersection1
    counter_X2 = np.count_nonzero(intersection1)

    #Calculate assymetry for y-axis        
    y_assymetry=round(((counter_X1 - counter_X2)/ counter_X1), 2)           
    #X axis assymety
    half_var_1=rotated[0:cy+1,:]#[:,:,2]
    half_var_2=rotated[cy:,:]#:,:,2]
    half_var_2=cv2.flip(half_var_2,0)
    

    
    height_img_3=half_var_1.shape[0]
    height_img_4=half_var_2.shape[0]
    
    if height_img_4 > height_img_3:
        half_var_2=half_var_2[height_img_4-height_img_3:,:]
        #half_var_2=cv2.resize(half_var_2, (half_var_1.shape[1], half_var_1.shape[0]))
    elif height_img_4 < height_img_3: 
        half_var_1=half_var_1[height_img_3-height_img_4:,:]
    else:
        pass

    #union of two section
    union2=cv2.bitwise_or(half_var_1, half_var_2)

    #In tersection of two part
    intersection2=cv2.bitwise_and(half_var_1, half_var_2)

    #Count number pixels of union2
    counter_Y1=np.count_nonzero(union2)

    #Count number pixels of intersection2
    counter_Y2=np.count_nonzero(intersection2)

    #Calculate assymetry for x-axis  
    x_assymetry=round(((counter_Y1 - counter_Y2)/counter_Y1), 2)          
          
    return x_assymetry, y_assymetry