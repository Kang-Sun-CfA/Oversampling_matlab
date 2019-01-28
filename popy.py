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
        
        if east < west:
            east = east+360
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
        
        self.nrows = len(ygrid)
        self.ncols = len(xgrid)
    
    def F_generalized_SG(self,x,y,fwhmx,fwhmy):
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        wx = fwhmx/(np.log(2)**(1/k1/k3))
        wy = fwhmy/(np.log(2)**(1/k2/k3))
        sg = np.exp(-(np.abs(x/wx)**k1+np.abs(y/wy)**k2)**k3)
        return sg
    
    def F_2D_SG_rotate(self,xmesh,ymesh,x_c,y_c,fwhmx,fwhmy,angle):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle),  np.cos(angle)]])
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
        
        fixedPoints = np.array([[-fwhmx,-fwhmy],[-fwhmx,fwhmy],[fwhmx,fwhmy],
                                [fwhmx,-fwhmy]],dtype=vList.dtype)/2
        tform = cv2.getPerspectiveTransform(vList,fixedPoints)
        
        xym1 = np.column_stack((xmesh.flatten()-x_c,ymesh.flatten()-y_c))
        xym2 = np.hstack((xym1,np.ones((xmesh.size,1)))).dot(tform.T)[:,0:2]
        
        sg0 = self.F_generalized_SG(xym2[:,0],xym2[:,1],fwhmx,fwhmy)
        sg = sg0.reshape(xmesh.shape)
        return sg
    
    def F_regrid(self,l2g_data):
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        nrows = self.nrows
        ncols = self.ncols
        tmplon = l2g_data['lonc']-west
        tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
        validmask = (tmplon >= 0) & (tmplon <= east-west) &\
        (l2g_data['latc'] >= south) & (l2g_data['latc'] <= north)
        l2g_data = {k:v[validmask,] for (k,v) in l2g_data.items()}
        nl2 = len(l2g_data['latc'])
        
        #construct a rectangle envelopes the orginal pixel
        xmargin = 3  #how many times to extend zonally
        ymargin = 2 #how many times to extend meridonally
        
        vcd_A = np.zeros((nrows,ncols))
        error_A = vcd_A
        albedo_A = vcd_A
        cloud_fraction_A = vcd_A
        cloud_height_A = vcd_A
        B = vcd_A
        D = vcd_A
        
        for il2 in range(nl2-1):
            local_l2g_data = {k:v[il2,] for (k,v) in l2g_data.items()}
            if self.sensor_name in {"OMI","OMPS","GOME","GOME2","SCIAMACHY"}:
                latc = local_l2g_data['latc']
                lonc = local_l2g_data['lonc']
                latr = local_l2g_data['latr']
                lonr = local_l2g_data['lonr']
                
                if lonc < west:
                    lonc = lonc+360
                    lonr = lonr+360
                
                lat_min = latr.min()
                lon_min = lonr.min()
                
                local_left = lonc-xmargin*(lonc-lon_min)
                local_right = lonc+xmargin*(lonc-lon_min)
                
                local_bottom = latc-ymargin*(latc-lat_min)
                local_top = latc+ymargin*(latc-lat_min)
                
                vcd = local_l2g_data['vcd']
                vcde = local_l2g_data['vcde']



### testing
omi_popy = popy(sensor_name="OMI",grid_size=0.1,west=-10,east=10,south=-10,north=10)
l2g_data = {'lonc':np.float32([0,1]),'latc':np.float32([0,1]),
            'lonr':np.float32([[-2,-2,2,2],[-1,-1,3,3]]),
            'latr':np.float32([[-1,1,1,-1],[0,2,2,0]]),
            'vcd':np.float32([0.5,1]),'vcde':np.float32([0.25,.5]),
            'albedo':np.float32([0.5,0.2]),'cloud_fraction':np.float32([0.,0.]),
            'cloud_height':np.float32([1,1]),}

sg = omi_popy.F_generalized_SG(omi_popy.xmesh,omi_popy.ymesh,5,5)
sg1 = omi_popy.F_2D_SG_rotate(omi_popy.xmesh,omi_popy.ymesh,-2,3,5,5,np.pi/4)
x_r = np.float32([-2,-2,2,2])
y_r = np.float32([-2,0,1,-1])
x_c = 0;y_c = 0;
sg2 = omi_popy.F_2D_SG_transform(omi_popy.xmesh,omi_popy.ymesh,x_r,y_r,x_c,y_c)
import matplotlib.pyplot as plt
plt.contour(omi_popy.xmesh,omi_popy.ymesh,sg2)
plt.plot(x_c,y_c,'o')
plt.plot(x_r,y_r,'*')