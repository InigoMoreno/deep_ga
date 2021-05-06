import pocolog_pybind
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import math
import os, re
import deep_ga
import pickle

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
    
dataset='/media/heimdal/Dataset1'

# get map
pickle_path=dataset+"/processed/Maps/map1.pickle"
if not os.path.isfile(pickle_path):
    ply_file=dataset+"/Maps/minas_densified_point_cloud_part_1.ply"
    dem, img, displacement = deep_ga.ply_to_image(ply_file, resolution=1)
    pickle.dump( (dem,img,displacement), open( pickle_path, "wb+" ) )
else:
    dem, img, displacement = pickle.load( open( pickle_path, "rb" ) )

plt.figure()
plt.imshow(np.transpose(img/255,(1, 0, 2)), origin="lower")

# get traverses paths
p = re.compile(dataset+'/\d+June/Traverse/\d+-\d+$')
traverses = [x[0][len(dataset)+1:]+'/' for x in os.walk(dataset) if p.match(x[0])]


cloudEastingOffset = 344178.0
cloudNorthingOffset = 3127579.0
cloudElevationOffset = 2442.0
robotEastingOffset = 344043.8
robotNorthingOffset = 3127408.3
robotElevationOffset = 2543.0

eastingReference = robotEastingOffset - cloudEastingOffset
northingReference = robotNorthingOffset - cloudNorthingOffset
elevationReference = robotElevationOffset - cloudElevationOffset
print(f"eastingReference: {eastingReference}")
print(f"northingReference: {northingReference}")
print(f"elevationReference: {elevationReference}")

x_max=-10000
y_max=-10000
x_min=10000
y_min=10000

for traverse in ['9June/Traverse/20170609-1909/']:
    path = dataset + "/" + traverse
    processed_path = dataset + "/processed/" + traverse

    print(traverse)
    if not os.path.isfile(path+"updated/waypoint_navigation.log"):
        print(path+"updated/waypoint_navigation.log not found")
        continue


    # create file index. Its possible to specify multiple logfiles
    multi_file_index = pocolog_pybind.pocolog.MultiFileIndex()
    multi_file_index.create_index([processed_path + "ga_slam.0.log",
                                path + "updated/waypoint_navigation.log"])

    streams = multi_file_index.get_all_streams()
    dem_stream = streams["/ga_slam.localElevationMapMean"]
    gps_stream = streams["/gps_heading.pose_samples_out"]

    state = extract_rigidbody_stream(gps_stream,1)
    x=state[:,1]-displacement[0]+eastingReference
    y=state[:,2]-displacement[1]+northingReference
    x_min=min(np.min(x),x_min)
    x_max=max(np.max(x),x_max)
    y_min=min(np.min(y),y_min)
    y_max=max(np.max(y),y_max)
    plt.plot(x,y)
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.pause(0.05)
print("Done, close image to end")
plt.show()