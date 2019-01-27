# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:50:30 2019

@author: Kang Sun
"""

import numpy as np
# conda install -c conda-forge opencv 
import cv2

class popy(object):
    
    def __init__(self,sensor_name,grid_size=0.1,west=-180,east=180,south=-90,north=90):
        
        self.sensor_name = sensor_name
        
        if(sensor_name == "OMI"):
            k1 = 4
            k2 = 2
            k3 = 1
            error_model = "linear"
        elif(sensor_name == "IASI"):
            k1 = 2
            k2 = 2
            k3 = 9
            error_model = "square"
        elif(sensor_name == "CrIS"):
            k1 = 2
            k2 = 2
            k3 = 4
            error_model = "log"
        elif(sensor_name == "OMPS"):
            k1 = 6
            k2 = 2
            k3 = 3
            error_model = "linear"
        else:
            k1 = 2
            k2 = 2
            k3 = 1
            error_model = "linear"
        
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.error_model = error_model
        
        self.grid_size = grid_size
        self.west = west
        self.east = east
        self.south = south
        self.north = north
        
        xgrid = np.arange(west,east,grid_size)+grid_size/2
        ygrid = np.arange(south,north,grid_size)+grid_size/2
        [xmesh,ymesh] = np.meshgrid(xgrid,ygrid)
        
        xgridr = np.hstack((np.arange(west,east,grid_size),east))
        ygridr = np.hstack((np.arange(south,north,grid_size),north))
        [xmeshr,ymeshr] = np.meshgrid(xgridr,ygridr)
        
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.xmesh = xmesh
        self.ymesh = ymesh
        self.xgridr = xgridr
        self.ygridr = ygridr
        self.xmeshr = xmeshr
        self.ymeshr = ymeshr
    
    def F_generalized_SG(self,x,y,fwhmx,fwhmy):
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        wx = fwhmx/(np.log(2)**(1/k1/k3))
        wy = fwhmy/(np.log(2)**(1/k2/k3))
        sg = np.exp(-(np.abs(x/wx)**k1+np.abs(y/wy)**k2)**k3)
        return sg
    
    def F_2D_SG_rotate(self,xmesh,ymesh,x_c,y_c,fwhmx,fwhmy,angle):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),  np.cos(angle)]])
        xym1 = np.array([xmesh.flatten()-x_c,ymesh.flatten()-y_c])
        xym2 = rotation_matrix.dot(xym1)
        sg0 = self.F_generalized_SG(xym2[0,:],xym2[1,:],fwhmx,fwhmy)
        sg = sg0.reshape(xmesh.shape)
        return sg
    
    def F_2D_SG_transform(self,xmesh,ymesh,x_r,y_r,x_c,y_c):
        vList = np.column_stack((x_r-x_c,y_r-y_c))
        leftpoint = np.mean(vList[0:2,:],axis=0)
        rightpoint = np.mean(vList[2:4,:],axis=0)
        uppoint = np.mean(vList[1:3,:],axis=0)
        lowpoint = np.mean(vList[[0,3],:],axis=0)
        xvector = rightpoint-leftpoint
        yvector = uppoint-lowpoint
        
        fwhmx = np.linalg.norm(xvector)
        fwhmy = np.linalg.norm(yvector)
        
        fixedPoints = np.array([[-fwhmx,-fwhmy],[-fwhmx,fwhmy],[fwhmx,fwhmy],[fwhmx,-fwhmy]],dtype=vList.dtype)/2
        tform = cv2.getPerspectiveTransform(vList,fixedPoints)
        
        xym1 = np.column_stack((xmesh.flatten()-x_c,ymesh.flatten()-y_c))
        xym2 = np.hstack((xym1,np.ones((xmesh.size,1)))).dot(tform.T)[:,0:2]
        
        sg0 = self.F_generalized_SG(xym2[:,0],xym2[:,1],fwhmx,fwhmy)
        sg = sg0.reshape(xmesh.shape)
        return sg



### testing
omi_popy = popy(sensor_name="OMI",grid_size=0.1,west=-10,east=10,south=-10,north=10)
sg = omi_popy.F_generalized_SG(omi_popy.xmesh,omi_popy.ymesh,5,5)
sg1 = omi_popy.F_2D_SG_rotate(omi_popy.xmesh,omi_popy.ymesh,-2,3,5,5,np.pi/4)
x_r = np.float32([-2,-2,2,2])
y_r = np.float32([-2,0,1,-1])
x_c = 0;y_c = 0;
sg2 = omi_popy.F_2D_SG_transform(omi_popy.xmesh,omi_popy.ymesh,x_r,y_r,x_c,y_c)
import matplotlib.pyplot as plt
plt.contour(omi_popy.xmesh,omi_popy.ymesh,sg1)
plt.plot(x_c,y_c,'o')
plt.plot(x_r,y_r,'*')