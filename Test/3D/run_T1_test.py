# Import necessary libraries
import os
import time
import subprocess as sp
from datetime import datetime

# Specify the GPU ID to use
gpu_id = '0'

# Define the directory where the model weights are stored
model_dir = '../../Weights/3D/'

# Specify the path to the CSV file containing data
csv_path = 'T1_3D_CSV_PATH.csv'

# Set the acceleration factor (undersampling rate)
R = 2

# Construct the command to execute a Python script
cmd = f"python3 test_T1_3D_tf2.py --R {R} --K 5 --csv_path csv {csv_path} --model_dir={model_dir}"

# Use subprocess to run the command in the shell
sp.call(cmd, shell=True)
