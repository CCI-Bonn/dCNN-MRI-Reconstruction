import os,time
import subprocess as sp
from datetime import datetime

gpu_id = '2'
model_dir='/Test/Weights/3D/'
dir_recon = '/Test/3D'
csv_path ='T1_3D_CSV_PATH.csv'
R = 2

cmd = f"python3 test_T1_3D_tf2.py --R {R} --K 5 --csv_path csv {csv_path} --model_dir={model_dir} --dir_recon {dir_recon}"
sp.call(cmd,shell =True)
