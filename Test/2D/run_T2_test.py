# Import necessary libraries
import os
import time
import subprocess as sp
from datetime import datetime

# Specify the GPU to use (in this case, GPU 0)
gpu_id = '0'

# Define the directory where model weights are stored
model_dir = '../../Weights/2D/'

# Define the path to the CSV file containing data (assuming it contains T2 data)
csv_path = 'T2_2D_CSV_PATH.csv'

# Set the undersampling factor (R)
R = 2

# Construct the command to run a Python script with arguments
# This command runs 'test_T2_2D_tf2.py' with specified arguments
cmd = f"python3 test_T2_2D_tf2.py --R {R} --K 10 --csv_path csv {csv_path} --model_dir={model_dir} "

# Execute the command in the shell
sp.call(cmd, shell=True)
