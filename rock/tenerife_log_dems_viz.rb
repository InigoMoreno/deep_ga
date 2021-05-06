#! /usr/bin/env ruby

require 'orocos'
require 'orocos/log'
require 'rock/bundle'
require 'vizkit'

include Orocos

Bundles.initialize
Bundles.transformer.load_conf(
    Bundles.find_file('config', 'transforms_scripts_ga_slam.rb'))

####### Replay Logs #######
bag = Orocos::Log::Replay.open(
#       Nominal start
#        '/media/heimdal/Dataset1/9June/Traverse/20170609-1413/bb3.log',
#        '/media/heimdal/Dataset1/9June/Traverse/20170609-1413/waypoint_navigation.log',
#        '/media/heimdal/Dataset1/9June/Traverse/20170609-1413/imu.log',
#       Nominal end
#       '/media/heimdal/Dataset1/9June/Traverse/20170609-1450/bb3.log',
#       '/media/heimdal/Dataset1/9June/Traverse/20170609-1450/waypoint_navigation.log',
#       '/media/heimdal/Dataset1/9June/Traverse/20170609-1450/imu.log',
#       Nominal reverse
#       '/media/heimdal/Dataset1/10June/Traverse/20170610-1315/bb3.log',
#       '/media/heimdal/Dataset1/10June/Traverse/20170610-1315/waypoint_navigation.log',
#       '/media/heimdal/Dataset1/10June/Traverse/20170610-1315/imu.log',
#       Nurburing
#        '/media/heimdal/Dataset1/10June/Traverse/20170610-1448/bb3.log',
#        '/media/heimdal/Dataset1/10June/Traverse/20170610-1448/waypoint_navigation.log',
#        '/media/heimdal/Dataset1/10June/Traverse/20170610-1448/imu.log',
#       Nurburing End //Not used due to lack of time
#        '/media/heimdal/Dataset1/10June/Traverse/20170610-1615/bb3.log',
#        '/media/heimdal/Dataset1/10June/Traverse/20170610-1615/waypoint_navigation.log',
#        '/media/heimdal/Dataset1/10June/Traverse/20170610-1615/imu.log',
#       Side Track
#        '/media/heimdal/Dataset1/9June/Traverse/20170609-1556/bb3.log',
#        '/media/heimdal/Dataset1/9June/Traverse/20170609-1556/waypoint_navigation.log',
#        '/media/heimdal/Dataset1/9June/Traverse/20170609-1556/imu.log',
#       Eight Track (Dusk)
        '/media/heimdal/Dataset1/9June/Traverse/20170609-1909/bb3.log',
        '/media/heimdal/Dataset1/9June/Traverse/20170609-1909/waypoint_navigation.log',
        '/media/heimdal/Dataset1/9June/Traverse/20170609-1909/imu.log',
#       Valley Circle
#        '/media/heimdal/Dataset1/11June/Traverse/20170611-1407/bb3.log',
#        '/media/heimdal/Dataset1/11June/Traverse/20170611-1407/waypoint_navigation.log',
#        '/media/heimdal/Dataset1/11June/Traverse/20170611-1407/imu.log',
)
bag.use_sample_time = true

Orocos.run(
    ####### Tasks #######
    'camera_bb3::Task' => 'camera_bb3',
    'stereo::Task' => ['stereo_bb3'],
    'gps_transformer::Task' => 'gps_transformer',
    'orbiter_preprocessing::Task' => 'orbiter_preprocessing',
    'ga_slam::Task' => 'ga_slam',
    ####### Debug #######
    # :output => '%m-%p.log',
    # :gdb => ['ga_slam'],
    # :valgrind => ['ga_slam'],
    :valgrind_options => ['--track-origins=yes']) \
do
    ####### Configure Tasks #######

    camera_bb3 = TaskContext.get 'camera_bb3'
    Orocos.conf.apply(camera_bb3, ['default'], :override => true)
    camera_bb3.configure

    stereo_bb3 = TaskContext.get 'stereo_bb3'
    Orocos.conf.apply(stereo_bb3, ['hdpr_bb3_left_right'], :override => true)
    stereo_bb3.configure

    gps_transformer = TaskContext.get 'gps_transformer'
    gps_transformer.configure

    orbiter_preprocessing = TaskContext.get 'orbiter_preprocessing'
    Orocos.conf.apply(orbiter_preprocessing, ['default', 'ga_slam', 'deep_ga'], :override => true)
    # Orocos.conf.apply(orbiter_preprocessing, ['prepared'], :override => true)
    orbiter_preprocessing.configure

    ga_slam = TaskContext.get 'ga_slam'
    # Orocos.conf.apply(ga_slam, ['default'], :override => true)
    Orocos.conf.apply(ga_slam, ['default', 'test', 'deep_ga'], :override => true)
    Bundles.transformer.setup(ga_slam)
    ga_slam.configure

    # Copy parameters from ga_slam to orbiter_preprocessing
    orbiter_preprocessing.cropSize = ga_slam.orbiterMapLength
    orbiter_preprocessing.voxelSize = ga_slam.orbiterMapResolution

    ####### Connect Task Ports #######
    bag.camera_firewire_bb3.frame.connect_to        camera_bb3.frame_in

    camera_bb3.left_frame.connect_to                stereo_bb3.left_frame
    camera_bb3.right_frame.connect_to               stereo_bb3.right_frame

    stereo_bb3.point_cloud.connect_to               ga_slam.loccamCloud

    bag.gps_heading.pose_samples_out.connect_to     gps_transformer.inputPose
    bag.gps_heading.pose_samples_out.connect_to     orbiter_preprocessing.robotPose

    gps_transformer.outputPose.connect_to           ga_slam.odometryPose

    # Connect IMU (roll, pitch) + Laser Gyro (yaw)
    gps_transformer.outputPose.connect_to           ga_slam.imuOrientation

    orbiter_preprocessing.pointCloud.connect_to     ga_slam.orbiterCloud
    gps_transformer.outputPose.connect_to           ga_slam.orbiterCloudPose

    ####### Start Tasks #######
    camera_bb3.start
    stereo_bb3.start
    gps_transformer.start
    orbiter_preprocessing.start
    ga_slam.start


    ####### Vizkit Display #######
    # Vizkit.display gps_transformer.outputPose,
    #     :widget => Vizkit.default_loader.RigidBodyStateVisualization
    # Vizkit.display gps_transformer.outputPose,
    #     :widget => Vizkit.default_loader.TrajectoryVisualization
    # Vizkit.display ga_slam.estimatedPose,
    #     :widget => Vizkit.default_loader.RigidBodyStateVisualization
    # Vizkit.display ga_slam.estimatedPose,
    #     :widget => Vizkit.default_loader.TrajectoryVisualization

    # Vizkit.display camera_bb3.left_frame

    # Vizkit.display stereo_bb3.point_cloud
    # Vizkit.display ga_slam.mapCloud

    # Vizkit.display orbiter_preprocessing.pointCloud

    Vizkit.display ga_slam.localElevationMapMean
    # Vizkit.display ga_slam.localElevationMapVariance
    # Vizkit.display ga_slam.globalElevationMapMean

    Vizkit.display ga_slam.localElevationMapMean,
        :widget => Vizkit.default_loader.DistanceImageVisualization
    # Vizkit.display ga_slam.globalElevationMapMean,
    #     :widget => Vizkit.default_loader.DistanceImageVisualization

    ####### Vizkit Replay Control #######
    control = Vizkit.control bag
#    control.speed = 1.0
#    control.seek_to 13000 # Nominal
#    control.seek_to 34700 #17181 #34000 #31000 # Nurburing
#    control.seek_to 59000 # Eight Track Dusk
#    control.seek_to 4955 #24000 #15378 # Side Track
    control.bplay_clicked

    ####### Vizkit #######
    Vizkit.exec
end