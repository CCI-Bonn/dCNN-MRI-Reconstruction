import os,time,io
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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


# In[5]:


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
parser.add_argument('--csv_path', type=str,default = '/raid/Aditya/Recon/Tumor/Recon/modl2D/T1_EROTC_2D_TEST_ALL_TIME_NAS.csv')
parser.add_argument('--model_dir', type=str,default = '/raid/Aditya/Recon/Tumor/modl2D/')
parser.add_argument('--dir_recon', type=str,default = '/raid/Aditya/Recon/Tumor/Recon/modl2D/')

args = parser.parse_args()
gpu_id = str(args.n_gpu)
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

csv_path = args.csv_path

seq =  'T1_2D'
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

model_path = dir+'T1_2D_dcdw/R'+str(R_in)+'/'
if R_in == 2:
    subDirectory='09May_0119pm_5L_10K_100E_AG'
elif R_in == 4:
    subDirectory='10May_1014am_5L_10K_100E_AG'
elif R_in == 6:
    subDirectory='12May_0951am_5L_10K_100E_AG'
elif R_in == 8:
    subDirectory='12May_1040pm_5L_10K_100E_AG'
elif R_in == 10:
    subDirectory='12May_1205am_5L_10K_100E_AG'
elif R_in == 15:
    subDirectory='13Jul_0241pm_5L_10K_100E_AG'

logdir = dir_recon+'/'+seq+'/'+database+'/New/R'+str(R_in)+'/'
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
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
    f_dir = 'Masks/T1/R'+str(R)
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
    if 'T1_2D.nii.gz' in f:
        f = f.replace('/T1_2D.nii.gz','')
        f_dir = f + '/recon/R' + str(R_in)
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)
        f_save = f_dir + '/T1_2D_recon.nii.gz'
        nft_img = nib.Nifti1Image(recon, aff)
        nib.save(nft_img,f_save)
        atb = np.abs(atb)
        atb = atb.transpose((1,2,0))
        atb.astype(np.int16)
        f_save = f_dir + '/T1_2D_us.nii.gz'
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
        atb = atb.transpose((1,2,0))
        atb.astype(np.int16)
        f_save = f_dir + '/T1_us.nii.gz'
        nft_img = nib.Nifti1Image(atb, aff)
    nib.save(nft_img,f_save)
    

def check_load_nifti_data(filename,R):
    print(filename[0])
    X = nib.load(filename[0])
    used_files = filename[0]
    scan = X.get_fdata()
    data_shape = np.shape(scan)
    print(data_shape)

    
def check_file_exits(f,R):
    if 'T1_2D.nii.gz' in f:
        f = f.replace('/T1_2D.nii.gz','')
        f_dir = f + '/recon/R' + str(R_in)   
        f_save = f_dir + '/T1_2D_recon.nii.gz'
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


for i in tstFilenames[0:1]:
    org,Atb,mask,mu,std,f_name,d_shape,affine = load_nifti_data(i,R_in)
    
print('Undersampling rates is %f'% (np.size(mask)/np.sum(mask)))


def plot_figure_1(imgOrg,imgAtb,imgMask):
    idx = 0
    idx2 = 15
    mu = np.mean(imgOrg[idx2,:,:])
    sig = np.std(imgOrg[idx2,:,:])
    imgOrg1 = (imgOrg[idx2,:,:]-mu)/sig
    imgAtb1 = (np.abs(imgAtb[idx2,:,:])-mu)/sig
    imgMask1 = imgMask[idx2,:,:]
    #Display the output images
    print(np.shape(imgOrg1))
    fig, ax = plt.subplots(2,3,dpi=150)
    ax[0,0].imshow(np.fft.fftshift(imgMask1),cmap='gray')
    ax[0,0].set_title('Mask',fontsize=5)
    
    ax[0,1].imshow(imgOrg1,cmap='gray')
    ax[0,1].set_title('Original',fontsize=5)
    
    ax[0,2].imshow(imgAtb1,cmap='gray')
    ax[0,2].set_title('Input',fontsize=5)

    idx2 = 17

    mu = np.mean(imgOrg[idx2,:,:])
    sig = np.std(imgOrg[idx2,:,:])
    imgOrg1 = (imgOrg[idx2,:,:]-mu)/sig
    imgAtb1 = (np.abs(imgAtb[idx2,:,:])-mu)/sig
    imgMask1 = imgMask[idx2,:,:]

    ax[1,0].imshow(np.fft.fftshift(imgMask1),cmap='gray')
    ax[1,0].set_title('Mask',fontsize=5)
    
    ax[1,1].imshow(imgOrg1,cmap='gray')
    ax[1,1].set_title('Original',fontsize=5)
    
    ax[1,2].imshow(imgAtb1,cmap='gray')
    ax[1,2].set_title('Input',fontsize=5)

    for i in range(2):
        for j in range(3):
            ax[i,j].axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()
    # buf = io.BytesIO()
    # fig.savefig(buf, format='png')
    # buf.seek(0)
    # plt.close(fig)
    # return buf.getvalue()
    
def add_title(ax,img,ref,str_,fsize):
    psnr = sf.myPSNR(sf.normalize01(ref),sf.normalize01(img))
    ssim = skssim(sf.normalize01(ref),sf.normalize01(img),data_range = 1.0)
    title = "%s \nPSNR:%0.1f \n SSIM:%0.2f" % (str_,psnr,ssim.mean())
    ax.set_title(title,fontsize=fsize)
def plot_figure_2(imgOrg,imgAtb,imgRecon_dc,imgRecon_dw,imgMask,fname,fsize=10):
    nz = imgOrg.shape[0]
    idx2 = int(np.floor(nz/2) - np.floor(nz/10))
    fname = fname.replace('/raid/Aditya/Recon/Tumor/','')
    mu = np.mean(imgOrg[idx2,:,:])
    sig = np.std(imgOrg[idx2,:,:])
    imgOrg1 = (imgOrg[idx2,:,:]-mu)/sig
    imgAtb1 = (np.abs(imgAtb[idx2,:,:])-mu)/sig
    imgRecon1_dc = (np.abs(imgRecon_dc[idx2,:,:].squeeze()) - mu)/sig
    imgRecon1_dw = (np.abs(imgRecon_dw[idx2,:,:].squeeze()) - mu)/sig
    imgMask1 = imgMask[idx2,:,:]
    #Display the output images
    fig, ax = plt.subplots(2,5,dpi=150)
    fig.suptitle(fname,fontsize=4)
    ax[0,0].imshow(np.fft.fftshift(imgMask1),cmap='gray')
    mask_title = 'Mask R=%0.2f'% (np.size(imgMask1)/np.sum(imgMask1))
    ax[0,0].set_title(mask_title,fontsize=fsize)
    
    ax[0,1].imshow(imgOrg1,cmap='gray')
    ax[0,1].set_title('Original',fontsize=fsize)
    
    ax[0,2].imshow(imgAtb1,cmap='gray')
    # ax[0,2].set_title('Input')
    add_title(ax[0,2],imgOrg1,imgAtb1,'Input',fsize)
    ax[0,3].imshow(imgRecon1_dc,cmap='gray')
    # ax[0,3].set_title('Recon DC')
    add_title(ax[0,3],imgOrg1,imgRecon1_dc,'Recon DC',fsize)
    ax[0,4].imshow(imgRecon1_dw,cmap='gray')
    # ax[0,4].set_title('Recon DW')
    add_title(ax[0,4],imgOrg1,imgRecon1_dw,'Recon DW',fsize)
    
    idx2 = int(np.floor(nz/2) + np.floor(nz/10))

    mu = np.mean(imgOrg[idx2,:,:])
    sig = np.std(imgOrg[idx2,:,:])
    imgOrg1 = (imgOrg[idx2,:,:]-mu)/sig
    imgAtb1 = (np.abs(imgAtb[idx2,:,:])-mu)/sig
    imgRecon1_dc = (np.abs(imgRecon_dc[idx2,:,:].squeeze()) - mu)/sig
    imgRecon1_dw = (np.abs(imgRecon_dw[idx2,:,:].squeeze()) - mu)/sig
    imgMask1 = imgMask[idx2,:,:]
    #Display the output images
    ax[1,0].imshow(np.fft.fftshift(imgMask1),cmap='gray')
    mask_title = 'Mask R=%0.2f'% (np.size(imgMask1)/np.sum(imgMask1))
    ax[1,0].set_title(mask_title,fontsize=fsize)
    
    ax[1,1].imshow(imgOrg1,cmap='gray')
    ax[1,1].set_title('Original',fontsize=fsize)
    
    ax[1,2].imshow(imgAtb1,cmap='gray')
    # ax[1,2].set_title('Input')
    add_title(ax[1,2],imgOrg1,imgAtb1,'Input',fsize)
    ax[1,3].imshow(imgRecon1_dc,cmap='gray')
    ax[1,3].set_title('Recon DC')
    add_title(ax[1,3],imgOrg1,imgRecon1_dc,'Recon DC',fsize)
    ax[1,4].imshow(imgRecon1_dw,cmap='gray')
    ax[1,4].set_title('Recon DW')
    add_title(ax[1,4],imgOrg1,imgRecon1_dw,'Recon DW',fsize)
    for i in range(2):
        for j in range(5):
            ax[i,j].axis('off')


    plt.subplots_adjust(wspace=0, hspace=-0.65)
    # plt.subplots_adjust(bottom=0.3, top=0.7, hspace=-1.3)

    plt.tight_layout()
    # plt.show()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()
    


# In[6]:


plot_figure_1(org,Atb,mask)


# In[ ]:


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

strfigureT = tfv.placeholder(tf.string)
image = tf.image.decode_png(strfigureT, channels=4)
figureT = tf.expand_dims(image, 0)
figSumT = tfv.summary.image("Figure", figureT)

#%% Run reconstruction
writer = tfv.summary.FileWriter(logdir)
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
            plt_1 = plot_figure_2(org,Atb,data_dc,data_dw,mask,f_name,7)
            reconFigSum=sess.run(figSumT,feed_dict={strfigureT:plt_1})
            writer.add_summary(reconFigSum,i)
            save_nifti(data_dc,Atb,1.0,affine,f_name)
            # rec1_dw.append(sf.r2c(data_dw))
            # rec1_dc.append(sf.r2c(data_dc))
            #sys.stdout.write("-")
            #sys.stdout.flush()
   
writer.add_summary(reconFigSum,i)
print('Recon Complete')
#sys.stdout.write("]\n") # this ends the progress bar
