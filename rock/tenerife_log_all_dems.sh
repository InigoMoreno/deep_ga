#!/bin/bash

set +e
dataset=/media/heimdal/Dataset1
for day in $dataset/*June; do
    for traverse in $day/Traverse/*; do
        save_dir=$dataset/processed/${traverse#$dataset/}
        echo $save_dir
        if [ -d $save_dir ]; then
           # if [ ! -f $dataset/processed/${traverse#$day/Traverse/}.npz ]; then
                python3 dem_to_python.py ${traverse#$dataset}/
            #fi
        fi


        # if [ ! -f $traverse/updated/waypoint_navigation.log ]; then
        #     rock-convert $traverse/waypoint_navigation.log -o $traverse/updated/
        # fi
        

        # if [ -d $save_dir ]; then
        #     duration_log=`pocolog $traverse/bb3.log | grep -P '\[[\d:.]*\]' -o`
        #     duration_processed=`pocolog $save_dir/ga_slam.0.log | grep -P '\[[\d:.]*\]' -o`
        #     if [[ "$duration_log" < "$duration_processed" ]]; then
        #         echo "Processing already done"
        #         continue
        #     else
        #         echo "Folder found but seems not complete"
        #     fi
        # fi


        # ruby tenerife_log_dems.rb $traverse
        # sleep 5
        # pkill -9 orogen
        # sudo /etc/init.d/omniorb4-nameserver stop
        # sudo rm -f /var/lib/omniorb/*
        # sudo /etc/init.d/omniorb4-nameserver start
        # sleep 5
    done
done