_**DISCLAIMER:** This repository is only intended for editors and reviewers of the manuscript "<ins>Deep-learning-based reconstruction of undersampled MRI for multi-fold reduction of scan time: a multicenter retrospective cohort study in neuro-oncology</ins>". Do not use this repository outside the review process or distribute it. The repository will be made publicly available upon acceptance of the manuscript via https://github.com/NeuroAI-HD/dCNN-MRI-Reconstruction_

# dCNN-MRI-Reconstruction
Code for MR Image reconstruction using physics-based neural network.


This code solves the following optimization problem:

    J(x) = argmin_x ||Ax-b||_2^2 + \alpha||x-Dw(x)||^2_2 

 `A` can be any measurement operator. Here we consider parallel imaging problem in MRI where
 the `A` operator consists of undersampling mask, FFT, and coil sensitivity maps.

`Dw(x)`: it represents the denoiser using a residual learning CNN.



## Architecture

The architecture of our network is shown below
<p align="center">
  <img src="img/Architecture_2.jpg" width="1000px" alt=""> 
</p>


1. Install the anaconda environment using environment.yml file in the repository.
2. Download the network weights using the link https://heibox.uni-heidelberg.de/f/d8f1dc4c3ae5412a97b9/?dl=1  and extract the weights.
3. The folder "Test/2D" contains all the test files for 2D reconstruction and folder "Test/3D" contains all the files for 3D reconstruction.
4. To reconstruct from undersampled MR data run the file run_SEQ_test.py  where SEQ = {CT1_2D,T1_2D,FLAIR,T2} for 2D sequences and SEQ = {CT1_3D,T1_3D} for 3D sequences.
5. In the file edit the following fields as per your configuration:
    ~~~
    gpu_id = '2'   # GPU ID on which to run Reconstruction
    model_dir='../../Weights/2D/'  # Path to weight file. For 2D sequences /PATH/Weights/2D and for 3D sequences /PATH/Weights/3D
    dir_recon = '/' # Path to tensorboard logdir for reconstruction visualization.
    csv_path ='CT1_2D_CSV_PATH.csv'  # Path to CSV file containing full path of the original file (e.g the entry to CSV file should be like "/home/mydata/patient1/01062023/CT1.nii.gz)
    R = 2 # Undersampling rate
    ~~~
