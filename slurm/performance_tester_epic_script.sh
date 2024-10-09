#!/bin/bash

while true; do
    # Count files in the current directory and print the result
    count=$(ls /work/cvcs2024/VisionWise/train | wc -w)
    echo -ne "Number of files: $count\r"
    
    # Wait for 1000 milliseconds (1 second)
    sleep 1
done
