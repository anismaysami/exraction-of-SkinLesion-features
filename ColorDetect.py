# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:59:14 2022

@author: Anis
"""

import cv2
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import pandas as pd
import os



class DominantColors:
    CLUSTERS=None
    IMAGE=None
    COLORS=None
    LABELS=None
    def __init__(self, image, clusters=6):
        self.CLUSTERS=clusters
        self.BIT_IMAGE=image
        cwd=os.getcwd()
        #Reading csv file with pandas and giving names to each column
        index=["color_name","color","R","G","B"]
        self.csv = pd.read_excel(cwd+'\mod\color\color_skin.xlsx', names=index)
        
    
    def dominantcolors(self):
        # #read image
        # image_init=cv2.imread(self.IMAGE)
        # #converting to grayscale
        # gray=cv2.cvtColor(image_init, cv2.COLOR_BGR2GRAY)

        # #blurring
        # blur=cv2.GaussianBlur(gray, (5,5), 0)
        # _, binary=cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # edge_detected=cv2.Canny(binary, 0,1)

        # kernel=np.ones((5,5))
        # edge=cv2.dilate(edge_detected, kernel, iterations=3)
        # edge=cv2.erode(edge, kernel, iterations=2)

        # contours, hierarchy=cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # max_cnt=sorted(contours, key=cv2.contourArea)[-1]

        # mask=np.zeros(gray.shape, np.uint8)

        # masked=cv2.drawContours(mask, [max_cnt], -1, 255, -1)
        # bit=cv2.bitwise_or(image_init, image_init, mask=masked)
        # r,g,b=cv2.split(self.BIT_IMAGE)

        # r_change=(r==0)
        # g_change=(g==0)
        # b_change=(b==0)

        # r[r_change]=128
        # g[g_change]=255
        # b[b_change]=0

        # img=cv2.merge([r, g, b])

        width=self.BIT_IMAGE.shape[1]; height=self.BIT_IMAGE.shape[0]
        self.total_pixel=width*height
        
        #our limitation
        self.colored_pixel=self.total_pixel * 0.001
        #convert to rgb from bgr
        img=cv2.cvtColor(self.BIT_IMAGE, cv2.COLOR_BGR2RGB)
        img=img.reshape((img.shape[0]*img.shape[1], 3))
        #save image after operations
        self.IMAGE=img
        
        #using kmeans clustring to cluster pixels
        kmeans=KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors
        self.COLORS=kmeans.cluster_centers_

        #removing background color value
        boolin_matrix=(self.COLORS ==[0,255,128])#this is mask
        index=np.where(boolin_matrix)#finding index of this row: output is a tuple that is 0 indicate row and 1 indicates column
        #self.COLORS=np.delete(self.COLORS, index[0][0], 0)


        #save labels
        self.LABELS=kmeans.labels_
        #returning after converting to integer  from float
        self.COLORS=self.COLORS.astype(int)
        #removing background color value
        boolin_matrix=(self.COLORS == [128,255,0])#mask
        index=np.where(boolin_matrix)#finding index of this row: output is a tuple that is 0 indicate row and 1 indicates column
        self.COLORS=np.delete(self.COLORS, index[0][0], 0)
        print(self.COLORS)
        return self.COLORS
    
    def percentage(self):
        labels=list(self.LABELS)
        percent=[]
        for i in range(len(self.COLORS)):
            j=labels.count(i)
            j=j/(len(labels))*100
            percent.append(j)
        print(percent)
    
    def plotHistogram(self):
        #labels from 0 to no. of clusters
        Labels_sec=np.arange(0, self.CLUSTERS)
        #create frequency count tables
        (hist,_)=np.histogram(self.LABELS, bins=Labels_sec)
        hist=hist.astype('float')
        hist /=hist.sum()

        
        #appendingfrequencies to cluster centers
        colors=self.COLORS
        #descending order sorting as per frequency count
        colors=colors[(-hist).argsort()]
        hist=hist[(-hist).argsort()]
        #creating empty chart
        chart=np.zeros((50,500,3), np.uint8)
        start=0
        
        #creating color rectangle
        for i in range(len(colors)):
            end=start+hist[i]*500
            
            #getting rgb values
            r=colors[i][0]
            g=colors[i][1]
            b=colors[i][2]
            
            #using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start),0), (int(end),50), (int(r),int(g),int(b)),-1)
            start=end
            
        #display figure
        # plt.axis('off')
        # plt.imshow(chart)
        # plt.show()
    
    def color_name_detector(self, rgbval):

        report_color=[]; self.color_score=0
        self.black_num=0; self.white_num=0; self.red_num=0
        self.lbrown_num=0; self.dbrown_num=0; self.bgray_num=0

        minimum = 10000
        color_name=[]; label_num=[]
        labels=list(self.LABELS)

        for i in range(len(rgbval)):
            r=rgbval[i][0] #denotes red
            g=rgbval[i][1] #denotes green
            b=rgbval[i][2] #denotes blue

            #label count to recognize number of each cluster pixels
            color_num_pix=labels.count(i)

            if (r <= 72 and g <= 62 and b <= 62):
                cname='"Black"'
                color_name.append(cname)
                label_num.append(color_num_pix)

            elif (r >= 205 and g >= 205 and b >= 205):
                cname='"White"'
                color_name.append(cname)
                label_num.append(color_num_pix)

            elif (r >= 150 and g < 62 and b < 62):
                cname='"Red"'
                color_name.append(cname)
                label_num.append(color_num_pix)
                
            elif (150 <= r <= 240 and 50 < g <= 150 and 0 < b < 100):
                cname='"Light brown"'
                color_name.append(cname)
                label_num.append(color_num_pix)
                
            elif (62 < r < 150 and 0 <= g < 100 and 0 <= b < 100):
                cname='"Dark brown"'
                color_name.append(cname)
                label_num.append(color_num_pix)
                
            elif (0 <= r <= 150 and 100 <= g <= 125 and 125 <= b <= 150):
                cname='"Blue-gray"'
                color_name.append(cname)
                label_num.append(color_num_pix)
                
            else:

                for j in range(len(self.csv)):
                    d=abs(r- int(self.csv.loc[j,"R"])) + abs(g- int(self.csv.loc[j,"G"]))+ abs(b- int(self.csv.loc[j,"B"]))
    
                    if(d<=minimum):
                        minimum = d
                        cname =self.csv.loc[j,"color"]
                color_name.append(cname)           
                label_num.append(color_num_pix)
        # print(color_name)
        # print(label_num)

        #creating a duplicate dict
        color_dict=defaultdict(list)
        for k, v in zip(color_name, label_num):
            color_dict[k].append(v)


        #max black colored pixel
        if not color_dict['"Black"'] == []:
            self.black_num=sum(color_dict['"Black"'])

        #max white colored pixel
        if not color_dict['"White"'] == []:
            self.white_num=sum(color_dict['"White"'])

        #max red colored pixel
        if not color_dict['"Red"'] == []:
            self.red_num=sum(color_dict['"Red"'])

        #max light brown colored pixel
        if not color_dict['"Light brown"'] == []:
            self.lbrown_num=sum(color_dict['"Light brown"'])

        #max dark brown colored pixel
        if not color_dict['"Dark brown"'] == []:
            self.dbrown_num=sum(color_dict['"Dark brown"'])

        #max blue-gray colored pixel
        if not color_dict['"Blue-gray"'] == []:
            self.bgray_num=sum(color_dict['"Blue-gray"'])

        if self.black_num >= self.colored_pixel:
            report_color.append('Black')
            self.color_score +=1

        if self.white_num >= self.colored_pixel:
            report_color.append('White')
            self.color_score +=1

        if self.red_num >= self.colored_pixel:
            report_color.append('Red')
            self.color_score +=1

        if self.lbrown_num >= self.colored_pixel:
            report_color.append('Light brown')
            self.color_score +=1

        if self.dbrown_num >= self.colored_pixel:
            report_color.append('Dark brown')
            self.color_score +=1

        if self.bgray_num >=  self.colored_pixel:
            report_color.append('Blue gray')
            self.color_score +=1
        else:
            pass


        report_color_nonlist=(','.join(report_color))#Now it is not list
        #print(report_color_nonlist)
            
        return report_color_nonlist , self.color_score


                
            
                
                
  #%%          
# #running program
# from masked_image import mask_background
# path=r'F:\dars\project\optoskin software\mod\main\half_horz_1_change.jpg'
# image=cv2.imread(path)
# #converting to grayscale
# gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #blurring
# blur=cv2.GaussianBlur(gray, (5,5), 0)
# _, binary=cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# edge_detected=cv2.Canny(binary, 0,1)

# kernel=np.ones((5,5))
# edge=cv2.dilate(edge_detected, kernel, iterations=3)
# edge=cv2.erode(edge, kernel, iterations=2)

# contours, hierarchy=cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# max_cnt=sorted(contours, key=cv2.contourArea)[-1]

# IMAGE=mask_background(image, max_cnt)
# clusters=6
# dc=DominantColors(IMAGE, clusters)
# colors=dc.dominantcolors()
# dc.plotHistogram()

# result=dc.color_name_detector(colors)