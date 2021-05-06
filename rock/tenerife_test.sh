#!/bin/bash
rm -f failed_logs.txt
for log in /media/heimdal/Dataset1/*/Traverse/*/*.log; do
    pocolog $log || echo $log >> failed_logs.txt
done