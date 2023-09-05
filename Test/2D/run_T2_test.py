import os,time
import subprocess as sp
from datetime import datetime



import os,time
import subprocess as sp
from datetime import datetime

gpu_id = '2'
model_dir='/Test/Weights/2D/'
dir_recon = '/Test/2D/'
csv_path ='T2_2D_CSV_PATH.csv'
R = 2

cmd = f"python3 Test_T2_2D_tf2.py --R {R} --K 5 --csv_path csv {csv_path} --model_dir={model_dir} --dir_recon {dir_recon}"
sp.call(cmd,shell =True)
