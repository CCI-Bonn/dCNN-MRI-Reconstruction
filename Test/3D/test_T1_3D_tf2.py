#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%%
import os,time,io
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import sys
import numpy as np
import logging
import tensorflow as tf
import nibabel as nib
import numpy as np
import csv
import matplotlib.pyplot as plt
from genPDF import give_mask
from datetime import datetime
from tqdm import tqdm
import supportingFunctions_T1_new_tf2 as sf
import model3D_tf2_32ch as mm
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
#% SET THESE PARAMETERS CAREFULLY
parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default = 1)
parser.add_argument('--n_gpu', type=int,  default = 0)
parser.add_argument('--R', type=int,  default = 4)
parser.add_argument('--epoch', type=int,  default = 100)
parser.add_argument('--load', type=str,default = '26Sep_1023am_5L_1K_200E_AG')
parser.add_argument('--csv_path', type=str,default = '/raid/Aditya/Recon/Tumor/Recon/modl3D_april23/T1_EROTC_3D_TEST_ALL_TIME_CPU_NAS.csv')
parser.add_argument('--model_dir', type=str,default = '/raid/Aditya/Recon/Tumor/modl3D/')
parser.add_argument('--dir_recon', type=str,default = '/raid/Aditya/Recon/Tumor/Recon/modl3D_april23/')
args = parser.parse_args()
gpu_id = str(args.n_gpu)
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id


csv_path = args.csv_path

seq =  'T1_3D'
database = 'EORTC'
tfv.reset_default_graph()
config = tfv.ConfigProto()
config.gpu_options.allow_growth=True

tfv.reset_default_graph()
#%% choose a model from savedModels directory



# In[2]:


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

model_path = dir+'T1_3D_dcdw/R'+str(R_in)+'/'
if R_in == 2:
    subDirectory='25Apr_0139pm_5L_5K_200E_AG'
elif R_in == 4:
    subDirectory='25Apr_0942pm_5L_5K_200E_AG'
elif R_in == 6:
    subDirectory='26Apr_0544am_5L_5K_200E_AG'
elif R_in == 8:
    subDirectory='26Apr_0141pm_5L_5K_200E_AG'
elif R_in == 10:
    subDirectory='26Apr_0950pm_5L_5K_200E_AG'
elif R_in == 15:
    subDirectory='19Jul_1202pm_5L_5K_200E_AG'



wts=sf.getWeights(model_path+subDirectory,chkPointNum='last')
lam_value = wts["Wts/lam1:0"]
print(f"Lambda value is {np.abs(lam_value) + 0.0001}")
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
    X = nib.load(filename[0])
    used_files = filename[0]
    scan = X.get_fdata()
    data_shape = np.shape(scan)
    pixdim = X.header['pixdim'][1:4]
    nx,ny,nz = data_shape
    imgAffine = X.affine
    print(f"{filename[0]} data size: {data_shape} pixdim: {pixdim}")
    f_dir = 'Masks/T1/R'+str(R)
    if not os.path.exists(f_dir):
        os.makedirs(f_dir)
    mask_filename = f_dir + '/' + str(nx)+'_'+str(ny)+'_'+str(nz) + '.mat'
    # print(np.shape(scan))
    if os.path.exists(mask_filename):
        mask = sio.loadmat(mask_filename)['mask']
        # print("Loading old mask")
    else:
        # print("Creating New mask")
        mask = give_mask(data_shape,R_in)
        mdic = {"mask":mask}
        sio.savemat(mask_filename,mdic)
    Atb = generateUndersampled(scan,mask,sigma=0.)
    mu = np.abs(np.expand_dims(np.mean(Atb),[0,1,2]))
    std = np.abs(np.expand_dims(np.std(Atb),[0,1,2]))
    return scan,Atb,mask,mu,std,used_files,data_shape,imgAffine


def save_nifti(recon,atb,i,aff,f):
    recon = np.abs(recon)
    # recon = recon.transpose((1,2,0))
    recon.astype(np.int16)
    if 'T1_3D.nii.gz' in f:
        f = f.replace('/T1_3D.nii.gz','')
        f_dir = f + '/recon/R' + str(R_in)
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)
        f_save = f_dir + '/T1_3D_recon.nii.gz'
        nft_img = nib.Nifti1Image(recon, aff)
        nib.save(nft_img,f_save)
        atb = np.abs(atb)
        # atb = atb.transpose((1,2,0))
        atb.astype(np.int16)
        f_save = f_dir + '/T1_3D_us.nii.gz'
        nft_img = nib.Nifti1Image(atb, aff)
    elif 'T1.nii.gz' in f:
        f = f.replace('/T1.nii.gz','')
        f_dir = f + '/recon/R' + str(R_in)
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)
        f_save = f_dir + '/T1_recon.nii.gz'
        nft_img = nib.Nifti1Image(recon, aff)
        nib.save(nft_img,f_save)
        atb = np.abs(atb)
        # atb = atb.transpose((1,2,0))
        atb.astype(np.int16)
        f_save = f_dir + '/T1_us.nii.gz'
        nft_img = nib.Nifti1Image(atb, aff)
    nib.save(nft_img,f_save)


def check_file_exits(f,R):
    if 'T1_3D.nii.gz' in f:
        f = f.replace('/T1_3D.nii.gz','')
        f_dir = f + '/recon/R' + str(R_in)   
        f_save = f_dir + '/T1_3D_recon.nii.gz'
        # print(f_save)
        if os.path.isfile(f_save):
            marker = True
        else:
            marker = False    
    elif 'T1.nii.gz' in f:
        f = f.replace('/T1.nii.gz','')
        f_dir = f + '/recon/R' + str(R_in)
        f_save = f_dir + '/T1_recon.nii.gz'
        # print(f_save)
        if os.path.isfile(f_save):
            marker = True
        else:
            marker = False
    return marker



def check_load_nifti_data(filename,R):
    print(filename[0])
    X = nib.load(filename[0])
    used_files = filename[0]
    scan = X.get_fdata()
    data_shape = np.shape(scan)
    print(data_shape)
    

def check_file_exits(f,R):
    if 'T1_3D.nii.gz' in f:
        f = f.replace('/T1_3D.nii.gz','')
        f_dir = f + '/recon/R' + str(R_in)   
        f_save = f_dir + '/T1_3D_recon.nii.gz'
        # print(f_save)
        if os.path.isfile(f_save):
            marker = True
        else:
            marker = False    
    elif 'T1.nii.gz' in f:
        f = f.replace('/T1.nii.gz','')
        f_dir = f + '/recon/R' + str(R_in)
        f_save = f_dir + '/T1_recon.nii.gz'
        # print(f_save)
        if os.path.isfile(f_save):
            marker = True
        else:
            marker = False
    return marker

# for i in tstFilenames[0:2]:
for i in tstFilenames[0:1]:
    # print(i)
    org,Atb,mask,mu,std,f_name,d_shape,affine = load_nifti_data(i,R_in)
    
print('Undersampling rates is %f'% (np.size(mask)/np.sum(mask)))


# In[5]:


print ('Now loading the model ...')

modelDir = model_path+ subDirectory


tfv.reset_default_graph()

config = tfv.ConfigProto()
config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 0.6 sometimes works better for folks
#nx,ny,nz = tstMask.shape[0],tstMask.shape[1],tstMask.shape[2]
#csmT = tf.placeholder(tf.complex64,shape=(None,1,320,320),name='csmT')
atbT = tfv.placeholder(tf.complex64,shape=(None,None,None,None),name='atbT')
maskT = tfv.placeholder(tf.complex64,shape=(None,None,None,None),name='maskT')
muT = tfv.placeholder(tf.complex64,shape=(None,None,None,None),name='muT')
stdT = tfv.placeholder(tf.complex64,shape=(None,None,None,None),name='stdT')
#orgT = tf.placeholder(tf.float32,shape=(None,None,None,2),name='org')
out=mm.makeModel(atbT,maskT,muT,stdT,False,nLayers,K,gradientMethod)
predT_dc=out['dc'+str(K)]
predT_dw=out['dw'+str(K)]
saver = tfv.train.Saver(tfv.global_variables(), max_to_keep=100)
rec1_dw = []
rec1_dc = []

nsamples = len(tstFilenames)



# In[9]:




# In[11]:


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
            pass
        else: 
            org,Atb,mask,mu,std,f_name,d_shape,affine = load_nifti_data(tstFilenames[i],R_in)
            # tstOrg.append(org)
            # tstAtb.append(Atb)
            # tstMask.append(mask)
            # tstMu.append(mu)
            # tstStd.append(std)
            # imgAffine.append(affine)
            # tstShape.append(d_shape)
            # used_files.append(f_name)
            nx,ny,nz = d_shape
            dataDict={atbT:Atb.reshape(-1,nx,ny,nz),maskT:mask.reshape(-1,nx,ny,nz),
                            muT:mu.reshape(-1,1,1,1),stdT:std.reshape(-1,1,1,1)}
            # print(used_files[i],d_shape)
            data_dw,data_dc = sess.run([predT_dw,predT_dc],feed_dict=dataDict)
            data_dw = sf.r2c(data_dw)
            data_dc = sf.r2c(data_dc)
            
            # rec1_dw.append(sf.r2c(data_dw))
            # rec1_dc.append(sf.r2c(data_dc))
            #sys.stdout.write("-")
            #sys.stdout.flush()
            
            save_nifti(data_dc[0,:],Atb,1.0,affine,f_name)
    
print('Recon Complete')

