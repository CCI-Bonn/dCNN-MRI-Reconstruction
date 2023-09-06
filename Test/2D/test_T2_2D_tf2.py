#%%
#%%
import os,time,io
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import sys
import numpy as np
import logging
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')
import nibabel as nib
import numpy as np
import csv
import matplotlib.pyplot as plt
from genPDF import give_mask
from datetime import datetime
from tqdm import tqdm
import supportingFunctions_T2_new_tf2 as sf
import model2D_tf2 as mm
import matplotlib.pyplot as plt
from preprocessData import cropdata, generateUndersampled
import tensorflow.compat.v1 as tfv
import argparse
from progiter import ProgIter
import random
import copy
from logging import DEBUG, INFO
from logger import log
import scipy.io as sio
from skimage.metrics import structural_similarity as skssim

tfv.disable_v2_behavior()

tfv.reset_default_graph()
config = tfv.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
#--------------------------------------------------------------
#% SET THESE PARAMETERS CAREFULLYd
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default = 1)
parser.add_argument('--n_gpu', type=int,  default = 0)
parser.add_argument('--R', type=int,  default = 4)
parser.add_argument('--epoch', type=int,  default = 100)
parser.add_argument('--load', type=str,default = '26Sep_1023am_5L_1K_200E_AG')
parser.add_argument('--csv_path', type=str,default = '/raid/Aditya/Recon/Tumor/Recon/modl2D/T2_EROTC_2D_TEST_ALL_TIME_NAS.csv')
parser.add_argument('--model_dir', type=str,default = '/raid/Aditya/Recon/Tumor/modl2D/')
parser.add_argument('--dir_recon', type=str,default = '/raid/Aditya/Recon/Tumor/Recon/modl2D/')

args = parser.parse_args()
gpu_id = str(args.n_gpu)
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

csv_path = args.csv_path

seq =  'T2_2D'
database = 'EROTC'
tfv.reset_default_graph()
config = tfv.ConfigProto()
config.gpu_options.allow_growth=True


tfv.reset_default_graph()
#%% choose a model from savedModels directory

nLayers=5
epochs=50
batchSize=1
gradientMethod='AG'
K=args.K
sigma=0.00
R_in = args.R
#subDirectory='14Mar_1105pm'
cwd=os.getcwd()
dir = args.model_dir
dir_recon = args.dir_recon

model_path = dir+'T2_2D_dcdw/R'+str(R_in)+'/'
if R_in == 2:
    subDirectory='09May_0123pm_5L_10K_50E_AG'
elif R_in == 4:
    subDirectory='09May_0602pm_5L_10K_100E_AG'
elif R_in == 6:
    subDirectory='10May_0119am_5L_10K_100E_AG'
elif R_in == 8:
    subDirectory='10May_0806am_5L_10K_100E_AG'
elif R_in == 10:
    subDirectory='10May_0315pm_5L_10K_100E_AG'
elif R_in == 15:
    subDirectory='12Jul_1131am_5L_10K_100E_AG'


wts=sf.getWeights(model_path+subDirectory)

filenames = sf.getFileNames(csv_path)
#tstMask = give_mask(patchSize,R_in)
# num_pats = 20
tstFilenames = filenames
# print(tstFilenames)

tstOrg = []
tstAtb = []
tstMask = []
tstMu = []
tstStd = []
tstShape = []
imgAffine = []
used_files = []

def load_nifti_data(filename,R):
    # print(filename[0])
    X = nib.load(filename[0])
    used_files = filename[0]
    scan = X.get_fdata()
    data_shape = np.shape(scan)
    # print(data_shape)
    nx,ny,nz = data_shape
    imgAffine = X.affine
    f_dir = 'Masks/T2/R'+str(R)
    if not os.path.exists(f_dir):
        os.makedirs(f_dir)
    mask_filename = f_dir + '/' + str(nx)+'_'+str(ny)+ '.mat'
    # print(np.shape(scan))
    if os.path.exists(mask_filename):
        tmp = sio.loadmat(mask_filename)['mask']
        mask = np.repeat(tmp[:,:,np.newaxis], data_shape[2], axis=2)
        # print("Loading old mask")
    else:
        # print("Creating New mask")
        mask = give_mask(data_shape,R_in)
        mdic = {"mask":mask[:,:,0]}
        sio.savemat(mask_filename,mdic)
    Atb = generateUndersampled(scan,mask,sigma=0.)
    mu = np.abs(np.expand_dims(np.mean(Atb),[0,1,2]))
    std = np.abs(np.expand_dims(np.std(Atb),[0,1,2]))
    
    scan = scan.transpose((2,0,1))
    Atb = Atb.transpose((2,0,1))
    mask = mask.transpose((2,0,1))
    used_files = filename[0]
    data_shape = np.shape(scan)
    return scan,Atb,mask,mu,std,used_files,data_shape,imgAffine


def save_nifti(recon,atb,i,aff,f):
    recon = np.abs(recon)
    recon = recon.transpose((1,2,0))
    recon.astype(np.int16)
    f = f.replace('/T2.nii.gz','')
    f_dir = f + '/recon/R' + str(R_in)
    if not os.path.exists(f_dir):
        os.makedirs(f_dir)
    f_save = f_dir + '/T2_recon.nii.gz'
    nft_img = nib.Nifti1Image(recon, aff)
    nib.save(nft_img,f_save)
    atb = np.abs(atb)
    atb = atb.transpose((1,2,0))
    atb.astype(np.int16)
    f_save = f_dir + '/T2_us.nii.gz'
    nft_img = nib.Nifti1Image(atb, aff)
    nib.save(nft_img,f_save)
    

def check_file_exits(f,R):
    if 'T2_2D.nii.gz' in f:
        f = f.replace('/T2_2D.nii.gz','')
        f_dir = f + '/recon/R' + str(R_in)   
        f_save = f_dir + '/T2_2D_recon.nii.gz'
        # print(f_save)
        if os.path.isfile(f_save):
            marker = True
        else:
            marker = False    
    elif 'T2.nii.gz' in f:
        f = f.replace('/T2.nii.gz','')
        f_dir = f + '/recon/R' + str(R_in)
        f_save = f_dir + '/T2_recon.nii.gz'
        # print(f_save)
        if os.path.isfile(f_save):
            marker = True
        else:
            marker = False
    return marker

    
for i in tstFilenames[0:1]:
    org,Atb,mask,mu,std,f_name,d_shape,affine = load_nifti_data(i,R_in)
    
print('Undersampling rates is %f'% (np.size(mask)/np.sum(mask)))

    



#%% Define Model

print ('Now loading the model ...')
#modelDir= cwd+'/savedModels_T1_R'+str(R_in)+'_new_par/'+subDirectory #complete path
# modelDir = cwd +'/'+ model_path+ subDirectory
modelDir = model_path+ subDirectory

#rec=np.empty(tstAtb.shape,dtype=np.complex64) #rec variable will have output
print(modelDir)
tfv.reset_default_graph()
loadChkPoint=tfv.train.latest_checkpoint(modelDir)
config = tfv.ConfigProto()
config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 0.6 sometimes works better for folks
#nx,ny,nz = tstMask.shape[0],tstMask.shape[1],tstMask.shape[2]
#csmT = tf.placeholder(tf.complex64,shape=(None,1,320,320),name='csmT')
atbT = tfv.placeholder(tf.complex64,shape=(None,None,None,None),name='atb')
muT = tfv.placeholder(tf.complex64,shape=(None,None,None,None),name='muT')
stdT = tfv.placeholder(tf.complex64,shape=(None,None,None,None),name='stdT')
maskT = tfv.placeholder(tf.complex64,shape=(None,None,None,None),name='stdT')

#orgT = tf.placeholder(tf.float32,shape=(None,None,None,2),name='org')
out=mm.makeModel(atbT,maskT,muT,stdT,False,nLayers,K,gradientMethod)
predT_dc=out['dc'+str(K)]
predT_dw=out['dw'+str(K)]
saver = tfv.train.Saver(tfv.global_variables(), max_to_keep=100)
rec1_dw = []
rec1_dc = []

tstFilenames_dummy = tstFilenames
nsamples = len(tstFilenames_dummy)



#%% Run reconstruction
with tfv.Session(config=config) as sess:
#    new_saver = tf.train.import_meta_graph(modelDir+'/modelTst.meta')
    sess.run(tfv.global_variables_initializer())
    sess=sf.assignWts(sess,nLayers,wts)
    #wts=sess.run(tf.global_variables())
    for i in tqdm(range(nsamples)):

        marker = check_file_exits(tstFilenames[i][0],R_in)

        if marker:
            print(f"Recon for {tstFilenames[i][0]} already exist")
        else:
            org,Atb,mask,mu,std,f_name,d_shape,affine = load_nifti_data(tstFilenames[i],R_in)
            nz,nx,ny = d_shape
            dataDict={atbT:Atb.reshape(-1,nz,nx,ny),maskT:mask.reshape(-1,nz,nx,ny),
                            muT:mu.reshape(-1,1,1,1),stdT:std.reshape(-1,1,1,1)}
            print(f_name,d_shape)
            data_dw,data_dc = sess.run([predT_dw,predT_dc],feed_dict=dataDict)
            data_dw = sf.r2c(data_dw)
            data_dc = sf.r2c(data_dc)
            
            save_nifti(data_dc,Atb,1.0,affine,f_name)
            # rec1_dw.append(sf.r2c(data_dw))
            # rec1_dc.append(sf.r2c(data_dc))
            #sys.stdout.write("-")
            #sys.stdout.flush()
   
print('Recon Complete')
#sys.stdout.write("]\n") # this ends the progress bar

#%% Save and plot 


