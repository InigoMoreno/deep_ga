#! /usr/bin/env ruby

require 'orocos'
require 'orocos/log'
require 'rock/bundle'

include Orocos

Bundles.initialize
Bundles.transformer.load_conf(
    Bundles.find_file('config', 'transforms_scripts_ga_slam.rb'))

path = ARGV.shift
puts path
if path.nil?
    dataset='/media/heimdal/Dataset1'
    # traverse ='9June/Traverse/20170609-1413/'  #       Nominal start
    # traverse ='9June/Traverse/20170609-1450/'  #       Nominal end
    # traverse ='10June/Traverse/20170610-1315/' #       Nominal reverse
    # traverse ='10June/Traverse/20170610-1448/' #       Nurburing
    # traverse ='10June/Traverse/20170610-1615/' #       Nurburing End //Not used due to lack of time
    # traverse ='9June/Traverse/20170609-1556/'  #       Side Track
    traverse ='9June/Traverse/20170609-1909/'  #       Eight Track (Dusk)
    # traverse ='11June/Traverse/20170611-1407/' #       Valley Circle
else
    dataset = path.split("/")[0..-4].join("/")
    traverse = path.split("/")[-3..-1].join("/")
end

unless traverse[-1]=='/'
    traverse+='/'
end

path = dataset+'/'+traverse
save_to = dataset + '/processed/' + traverse #log files will be removed from default folder to here, set to nil if you don't want to do this

####### Replay Logs #######
bag = Orocos::Log::Replay.open(
        path+'/bb3.log',
        # path+'/imu.log',
        path+'/waypoint_navigation.log'
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

    # Log the output (only ga_slam:localElevationMapMean and gps:outputPose and orbiter_preprocessing:robotPoseTransformed)
    Orocos.log_all_ports(tasks:/(ga_slam|gps|orbiter)/, exclude_ports: /(global|Variance|estim|state|Drift|Delta)/, exclude_types: /(cloud|Status|double)/) 

    ####### Start Tasks #######
    camera_bb3.start
    stereo_bb3.start
    gps_transformer.start
    orbiter_preprocessing.start
    ga_slam.start


    # Run log
    bag.speed = 1

    begin
        while bag.step(true)# && bag.sample_index <= 1000
        end
    ensure
        unless save_to.nil?
            sleep(5)
            system("mkdir -p #{save_to}")
            system("cp #{Bundles.log_dir}/*.log  #{save_to}")
            system("rm -r #{Bundles.log_dir}")
        end
    end
end