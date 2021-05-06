import math
import os
import pickle
import re
from datetime import datetime

import cv2
import deep_ga
import matplotlib.pyplot as plt
import numpy as np
import pocolog_pybind
from plyfile import PlyData, PlyElement


def euler_from_quaternion(quaternion):
    """
    Convert a quaternion into euler angles (yaw, pitch, roll) (in radians)
    """
    x=quaternion["im"][0]
    y=quaternion["im"][1]
    z=quaternion["im"][2]
    w=quaternion["re"]

    t0 = 2*(w*z + x*y)
    t1 = 1 - 2 * (x*x + y*y)
    t1 = w*w + x*x - y*y - z*z 
    yaw = math.atan2(t0, t1)
    
    t2 = 2*(w*y - z*x)
    pitch = math.asin(t2)
    
    t3 = 2*(w*x + y*z)
    t4 = w*w - x*x - y*y + z*z 
    roll = math.atan2(t3, t4)
    
    return yaw, pitch, roll # in radians

def extract_rigidbody_stream(stream, resolution=1):
    size=gps_stream.get_size()
    size//=resolution
    state = np.empty((size,7))
    for i in range(0,size):
        value = gps_stream.get_sample(i*resolution)
        py_value = value.cast(recursive=True)
        # print(py_value.keys())
        time = py_value["time"]["microseconds"]
        pos = py_value["position"]["data"]
        eul = euler_from_quaternion(py_value["orientation"])
        state[i,0]=time
        state[i,1:4]=pos
        state[i,4:]=eul
        value.destroy()
    return state

def get_slope(patch, resolution=1):
    slopeX = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)/resolution
    slopeY = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)/resolution
    slope = cv2.magnitude(slopeX, slopeY)
    return slope
    
dataset='/media/heimdal/Dataset1'

traverses={
    "Nominal start":     '9June/Traverse/20170609-1413/',
    "Nominal end":       '9June/Traverse/20170609-1450/',
    "Nominal reverse":   '10June/Traverse/20170610-1315/',      
    "Nurburing":         '10June/Traverse/20170610-1448/',      
    "Nurburing End":     '10June/Traverse/20170610-1615/',      
    "Side Track":         '9June/Traverse/20170609-1556/' ,      
    "Eight Track (Dusk)": '9June/Traverse/20170609-1909/'    
}

traverse=traverses["Eight Track (Dusk)"]

path = dataset + "/" + traverse
processed_path = dataset + "/processed/" + traverse

# get map
pickle_path=dataset+"/processed/Maps/map1.pickle"
if not os.path.isfile(pickle_path):
    ply_file=dataset+"/Maps/minas_densified_point_cloud_part_1.ply"
    dem, img, displacement = deep_ga.ply_to_image(ply_file, resolution=1)
    pickle.dump( (dem,img,displacement), open( pickle_path, "wb+" ) )
else:
    dem, img, displacement = pickle.load( open( pickle_path, "rb" ) )

gps_references = deep_ga.get_gps_references()

# create file index. Its possible to specify multiple logfiles
multi_file_index = pocolog_pybind.pocolog.MultiFileIndex()
multi_file_index.create_index([processed_path + "ga_slam.0.log",
                            path + "updated/waypoint_navigation.log"])

streams = multi_file_index.get_all_streams()
print(streams.keys())
dem_stream = streams["/ga_slam.localElevationMapMean"]
gps_stream = streams["/gps_heading.pose_samples_out"]

state = extract_rigidbody_stream(gps_stream)
plotSize = 40
resolution = 0.5
pixelSize = plotSize * resolution

p={
  "resolution" : 1,                      # resolution of map [meters per pixel]
  "mapLength"  : 40,                    # size of one side of the map [meters]
  "minSlopeThreshold"  : 0.5,            # minimum slope to be counted [proportion]
  "maxNanPercentage"   : 5/100,           # maximum percentage of NaNs in a patch [%]
  "minSlopePercentage" : 0/100,          # minimum percentage of slope in a patch [%]
  "maxSlopePercentage" : 100/100,          # maximum percentage of slope in a patch [%]
  "stdPatchShift"          : 15,          # standard deviation of shift between to patches [m]
}
p["mapLengthPixels"]=math.ceil(p["mapLength"]/p["resolution"])

index_gps = 0

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax3.imshow(np.transpose(img/255,(1, 0, 2)), origin="lower")
x=state[:,1]-displacement[0]+gps_references[0]
y=state[:,2]-displacement[1]+gps_references[1]
ax3.plot(x,y)
ax3.set_xlim(np.min(x),np.max(x))
ax3.set_ylim(np.min(y),np.max(y))

for t in range(1000,dem_stream.get_size()):
    value = dem_stream.get_sample(t)
    py_value = value.cast(recursive=True)
    # print(py_value.keys())
    value.destroy()
    local_dem = np.array(py_value["data"])
    local_dem = local_dem.reshape((py_value["height"], py_value["width"]), order="F").astype("float32")
    local_dem = local_dem[::-1,::-1].T
    
    while state[index_gps+1,0]<=py_value["time"]["microseconds"]:
        index_gps+=1
    
    
    x=state[index_gps,1]-displacement[0]+gps_references[0]
    y=state[index_gps,2]-displacement[1]+gps_references[1]



    global_dem = deep_ga.get_patch(dem, y, x, p)
    global_img = cv2.getRectSubPix(img, (p["mapLengthPixels"], p["mapLengthPixels"]), (y, x))

    # local_slope = get_slope(local_dem,p["resolution"])
    # global_slope = get_slope(global_dem,p["resolution"])

    # res = cv2.matchTemplate(np.nan_to_num(local_slope),np.nan_to_num(global_slope),cv2.TM_CCORR_NORMED)
    # _, max_val, _, max_loc = cv2.minMaxLoc(res)
    # print(max_loc)
    # print(max_val)
    
    # best_pos = (max_loc[0]+local_dem.shape[0]/2,max_loc[1]+local_dem.shape[1]/2)
    # print(best_pos)
    
    # best_dem =  cv2.getRectSubPix(global_dem, local_dem.shape, best_pos)
    # if best_dem is None:
    #     print(best_pos)
    #     print(dem.shape)

    # break
    ax1.clear()
    ax2.clear()
    ax1.imshow(np.transpose(local_dem,(1,0)),origin="lower")
    ax2.imshow(np.transpose(global_dem,(1,0)),origin="lower")
    ax3.plot([x],[y],"ro")
    print(x,y)
    plt.pause(0.05)
    
    # plt.show()
    # exit()


    # break
plt.show()
