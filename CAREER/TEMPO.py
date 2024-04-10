import sys, os, glob
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import logging
from popy import Level3_Data, F_center2edge, Level3_List
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
from matplotlib import path 

class TEMPO():
    '''class for a TEMPO-observed region'''
    def __init__(self,geometry=None,xys=None,start_dt=None,end_dt=None,
                 west=None,east=None,south=None,north=None,grid_size=0.01,flux_grid_size=0.05):
        '''
        geometry:
            a list of tuples for the polygon, e.g., [(xarray,yarray)], or geometry in a gpd row
        start/end_dt:
            datetime objects accurate to the day
        west, east, south, north:
            boundary of the region
        grid_size:
            grid size for level 3 data
        flux_grid_size:
            grid size for directional derivatives (level 4 data)
        '''
        self.logger = logging.getLogger(__name__)
        self.start_dt = start_dt or dt.datetime(2008,1,1)
        self.end_dt = end_dt or dt.datetime.now()
        if geometry is None and xys is not None:
            geometry = xys
        tmp = False
        if isinstance(geometry,list):
            xys = geometry
            self.west = west or np.min([np.min(xy[0]) for xy in xys])
            self.east = east or np.max([np.max(xy[0]) for xy in xys])
            self.south = south or np.min([np.min(xy[1]) for xy in xys])
            self.north = north or np.max([np.max(xy[1]) for xy in xys])
            self.xys = xys
        elif isinstance(geometry,shapely.geometry.multipolygon.MultiPolygon):
            bounds = geometry.bounds
            self.west = west or bounds[0]
            self.east = east or bounds[2]
            self.south = south or bounds[1]
            self.north = north or bounds[3]
            self.xys = [g.exterior.xy for g in geometry.geoms] #zitong edit
        elif isinstance(geometry,shapely.geometry.polygon.Polygon):
            bounds = geometry.bounds
            self.west = west or bounds[0]
            self.east = east or bounds[2]
            self.south = south or bounds[1]
            self.north = north or bounds[3]
            self.xys = [geometry.exterior.xy]
        elif geometry is None:
            self.west = west
            self.east = east
            self.south = south
            self.north = north
            tmp = True
        # nudge west/south
        westmost = -180
        self.west = westmost+np.floor((self.west-westmost)/flux_grid_size)*flux_grid_size
        southmost = -90
        self.south = southmost+np.floor((self.south-southmost)/flux_grid_size)*flux_grid_size
        if tmp: 
            self.xys = [([self.west,self.west,self.east,self.east],
                         [self.south,self.north,self.north,self.south])]   