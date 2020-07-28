# Implicit Neural Representations with Periodic Activation Functions
### [Project Page](https://vsitzmann.github.io/siren) | [Paper](https://arxiv.org/abs/2006.09661) | [Data](https://drive.google.com/drive/folders/1_iq__37-hw7FJOEUK1tX7mdp8SKB368K?usp=sharing)
[![Explore Siren in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb)<br>

[Vincent Sitzmann](https://vsitzmann.github.io/)\*,
[Julien N. P. Martel](http://www.jmartel.net)\*,
[Alexander W. Bergman](http://alexanderbergman7.github.io),
[David B. Lindell](http://www.davidlindell.com/),
[Gordon Wetzstein](https://stanford.edu/~gordonwz/)<br>
Stanford University, \*denotes equal contribution

This is the official implementation of the paper "Implicit Neural Representations with Periodic Activation Functions".

[![siren_video](https://img.youtube.com/vi/Q2fLWGBeaiI/0.jpg)](https://www.youtube.com/watch?v=Q2fLWGBeaiI)


## Google Colab
If you want to experiment with Siren, we have written a [Colab](https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb).
It's quite comprehensive and comes with a no-frills, drop-in implementation of SIREN. It doesn't require 
installing anything, and goes through the following experiments / SIREN properties:
* Fitting an image
* Fitting an audio signal
* Solving Poisson's equation
* Initialization scheme & distribution of activations
* Distribution of activations is shift-invariant
* Periodicity & behavior outside of the training range. 

## Tensorflow Playground
You can also play arond with a tiny SIREN interactively, directly in the browser, via the Tensorflow Playground [here](https://dcato98.github.io/playground/#activation=sine). Thanks to [David Cato](https://github.com/dcato98) for implementing this! 

## Get started
If you want to reproduce all the results (including the baselines) shown in the paper, the videos, point clouds, and 
audio files can be found [here](https://drive.google.com/drive/folders/1_iq__37-hw7FJOEUK1tX7mdp8SKB368K?usp=sharing).

You can then set up a conda environment with all dependencies like so:
```
conda env create -f environment.yml
conda activate siren
```

## High-Level structure
The code is organized as follows:
* dataio.py loads training and testing data.
* training.py contains a generic training routine.
* modules.py contains layers and full neural network modules.
* meta_modules.py contains hypernetwork code.
* utils.py contains utility functions, most promintently related to the writing of Tensorboard summaries.
* diff_operators.py contains implementations of differential operators.
* loss_functions.py contains loss functions for the different experiments.
* make_figures.py contains helper functions to create the convergence videos shown in the video.
* ./experiment_scripts/ contains scripts to reproduce experiments in the paper.

## Reproducing experiments
The directory `experiment_scripts` contains one script per experiment in the paper.

To monitor progress, the training code writes tensorboard summaries into a "summaries"" subdirectory in the logging_root.

### Image experiments
The image experiment can be reproduced with
```
python experiment_scripts/train_img.py --model_type=sine
```
The figures in the paper were made by extracting images from the tensorboard summaries. Example code how to do this can
be found in the make_figures.py script.

### Audio experiments
This github repository comes with both the "counting" and "bach" audio clips under ./data.

They can be trained with
```
python experiment_scipts/train_audio.py --model_type=sine --wav_path=<path_to_audio_file>
```

### Video experiments
The "bikes" video sequence comes with scikit-video and need not be downloaded. The cat video can be downloaded with the
link above.

To fit a model to a video, run
```
python experiment_scipts/train_video.py --model_type=sine --experiment_name bikes_video
```

### Poisson experiments
For the poisson experiments, there are three separate scripts: One for reconstructing an image from its gradients 
(train_poisson_grad_img.py), from its laplacian (train_poisson_lapl_image.py), and to combine two images 
(train_poisson_gradcomp_img.py).

Some of the experiments were run using the BSD500 datast, which you can download [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/).

### SDF Experiments
To fit a Signed Distance Function (SDF) with SIREN, you first need a pointcloud in .xyz format that includes surface normals.
If you only have a mesh / ply file, this can be accomplished with the open-source tool Meshlab.

To reproduce our results, we provide both models of the Thai Statue from the 3D Stanford model repository and the living room used in our paper
for download here.

To start training a SIREN, run:
```
python experiments_scripts/train_single_sdf.py --model_type=sine --point_cloud_path=<path_to_the_model_in_xyz_format> --batch_size=250000 --experiment_name=experiment_1
```
This will regularly save checkpoints in the directory specified by the rootpath in the script, in a subdirectory "experiment_1". 
The batch_size is typically adjusted to fit in the entire memory of your GPU. 
Our experiments show that with a 256, 3 hidden layer SIREN one can set the batch size between 230-250'000 for a NVidia GPU with 12GB memory.

To inspect a SDF fitted to a 3D point cloud, we now need to create a mesh from the zero-level set of the SDF. 
This is performed with another script that uses a marching cubes algorithm (adapted from the DeepSDF github repo) 
and creates the mesh saved in a .ply file format. It can be called with:
```
python experiments_scripts/test_single_sdf.py --checkpoint_path=<path_to_the_checkpoint_of_the_trained_model> --experiment_name=experiment_1_rec 
```
This will save the .ply file as "reconstruction.ply" in "experiment_1_rec" (be patient, the marching cube meshing step takes some time ;) )
In the event the machine you use for the reconstruction does not have enough RAM, running test_sdf script will likely freeze. If this is the case, 
please use the option --resolution=512 in the command line above (set to 1600 by default) that will reconstruct the mesh at a lower spatial resolution.

The .ply file can be visualized using a software such as [Meshlab](https://www.meshlab.net/#download) (a cross-platform visualizer and editor for 3D models).

### Helmholtz and wave equation experiments
The helmholtz and wave equation experiments can be reproduced with the train_wave_equation.py and train_helmholtz.py scripts.

## Torchmeta
We're using the excellent [torchmeta](https://github.com/tristandeleu/pytorch-meta) to implement hypernetworks. We 
realized that there is a technical report, which we forgot to cite - it'll make it into the camera-ready version!

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{sitzmann2019siren,
    author = {Sitzmann, Vincent
              and Martel, Julien N.P.
              and Bergman, Alexander W.
              and Lindell, David B.
              and Wetzstein, Gordon},
    title = {Implicit Neural Representations
              with Periodic Activation Functions},
    booktitle = {arXiv},
    year={2020}
}
```

## Contact
If you have any questions, please feel free to email the authors.
