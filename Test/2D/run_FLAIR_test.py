import os,time
import subprocess as sp
from datetime import datetime



import os,time
import subprocess as sp
from datetime import datetime

gpu_id = '0'
model_dir='../../Weights/2D/'

csv_path ='FLAIR_2D_CSV_PATH.csv'
R = 2

cmd = f"python3 test_FLAIR_2D_tf2.py --R {R} --K 10 --csv_path csv {csv_path} --model_dir={model_dir} "
sp.call(cmd,shell =True)
