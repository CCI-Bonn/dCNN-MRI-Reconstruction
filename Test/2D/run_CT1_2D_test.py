import os,time
import subprocess as sp
from datetime import datetime



import os,time
import subprocess as sp
from datetime import datetime

gpu_id = '2'
model_dir='../../Weights/2D/'
dir_recon = '/'
csv_path ='CT1_2D_CSV_PATH.csv'
R = 2

cmd = f"python3 Test_CT1_2D_tf2.py --R {R} --K 5 --csv_path csv {csv_path} --model_dir={model_dir} --dir_recon {dir_recon}"
sp.call(cmd,shell =True)
