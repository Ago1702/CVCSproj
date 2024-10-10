#!/bin/bash

# Check if the starting number is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <start_number>"
    exit 1
fi

# Get the starting number from the first parameter
start_number=$1

# Loop from 0 to 1023
for i in $(seq "$start_number" 1023)
do
    # Call the Python script with the current number
    python data_fetcher.py $i > /dev/null 2>&1

    # Check if the Python script was executed successfully
    if [ $? -ne 0 ]; then
        echo "Python script failed for number: $i"
        exit 1
    fi
    echo execution finished for script: $i
    echo "$i" >> ~/CVCSproj/model/pazzia_annotata
done
echo "Sono completamente pazzo"
echo "All numbers processed successfully."