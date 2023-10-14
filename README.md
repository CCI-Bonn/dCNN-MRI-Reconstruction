**DISCLAIMER:** This repository is only intended for editors and reviewers of the manuscript "<ins>Deep-learning-based reconstruction of undersampled MRI for multi-fold reduction of scan time: a multicenter retrospective cohort study in neuro-oncology</ins>". Do not use this repository outside the review process or distribute it. The repository will be made publicly available upon acceptance of the manuscript via https://github.com/NeuroAI-HD/dCNN-MRI-Reconstruction

# dCNN-MRI-Reconstruction
Code for MR Image reconstruction using physics-based neural network.


This code solves the following optimization problem:

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B200%7D%20%5Csmall%20J%28x%29%20%3D%20%5Cunderset%7Bx%7D%7B%5Ctext%7Bargmin%7D%7D%20%5C%20%5C%20%7C%7CAx-b%7C%7C_2%5E2%20&plus;%20%5Calpha%7C%7Cx-D_w%28x%29%7C%7C%5E2_2" width="400px" alt=""> 
</p>
 
 `A` can be any measurement operator. Here we consider parallel imaging problem in MRI where
 the `A` operator consists of undersampling mask, FFT, and coil sensitivity maps.

`Dw(x)`: it represents the denoiser using a residual learning CNN.



## Architecture

The architecture of the network used in this study is shown below
<p align="center">
  <img src="img/Architecture_2.png" width="800px" alt=""> 
</p>

## Improvements over MoDL

1. Unlike existing MoDL, the network used in this study is capable of reconstructing 3D acquired sequences using a 3D undersampling mask. This also makes the prior more informative as the field of view for prior calculation increases.
2. The network used in this study is more stable than MoDL to changes while training. In classical MoDL the lagrangian parameter can fluctuate between positive and negative values which can blow up the loss values. This problem is addressed in the current networks. For solving any constraint optimization problem $\alpha$ has to be non-negative (strictly positive in our case). In the original implementation of MoDL there was no mechanism to restrict $\alpha$ to be positive and in our initial experiments, it was observed that $\alpha$ oscillates between positive and negative values during parameter updation. This leads to the divergence of neural network during training. The implementation was modified such that $\alpha$ remains positive.
3. The Data denoising block, which learns a denoising prior for MR reconstruction should only generate a noise prior from the input (output of data consistency layer at each iteration of dCNN). However, there is no guarantee for the Data denoising block to learn the noise prior instead of any other feature, which was observed during the experiments performed for this manuscript at higher undersampling rates.An explicit loss term was added between ground truth and the output of data denoising block to ensure that it learns a noise prior.
4. Additionally, the implementation of existing MoDL was modified to support multi-gpu training by splitting the minibatch across multiple GPUs. This was necessary for 3D volume reconstruction.

Therefore the modified model is more stable, easier to train and can offer better convergence - hence performance - which are substantial improvements over the original work.
## Execution

1. Install the anaconda environment using environment.yml file in the repository.
2. Download the network weights using the link https://heibox.uni-heidelberg.de/f/d8f1dc4c3ae5412a97b9/?dl=1  and extract the weights.
3. The folder "Test/2D" contains all the test files for 2D reconstruction and folder "Test/3D" contains all the files for 3D reconstruction from single coil simulated data from nifti files.
4. To reconstruct from undersampled MR data run the file run_SEQ_test.py  where SEQ = {CT1_2D,T1_2D,FLAIR,T2} for 2D sequences and SEQ = {CT1_3D,T1_3D} for 3D sequences.
5. In the file edit the following fields as per your configuration:
    ~~~
    gpu_id = '2'   # GPU ID on which to run Reconstruction
    model_dir='../../Weights/2D/'  # Path to weight file. For 2D sequences /PATH/Weights/2D and for 3D sequences /PATH/Weights/3D
    dir_recon = '/' # Path to tensorboard logdir for reconstruction visualization.
    csv_path ='CT1_2D_CSV_PATH.csv'  # Path to CSV file containing full path of the original file (e.g the entry to CSV file should be like "/home/mydata/patient1/01062023/CT1.nii.gz)
    R = 2 # Undersampling rate
    ~~~
## Demo
To test the reconstruction algorithm for different undersampling rates and sequences, we are attaching a sample case (same as shown in Figure 2 of main manuscript).
1. Download the sample case using the link [HERE](https://heibox.uni-heidelberg.de/f/b615c0390e0745988b36/?dl=1). This dataset corresponds to the representative case in Figure 2 of the main manuscript.
2. Unzip the file and copy the case in "Demo_Data" folder. The path to data should look like "Demo_Data/311_502/20140903".
3. Run tstDemo.ipynb file (instructions to run file are provided inside the jupyter notebook). The script calculated the kspace data using fourier transform and retrospectively undersamples the kspace data which is then used for reconstruction.
4. The nifti files estimated from undersampled data using zero filling and reconstructed nifti files using dCNN will be saved in a folder "Demo_Data/311_502/20140903/recon/" for different undersampling rates.
5. The image will be saved as Figure_Demo_300dpi.png in the main folder.
