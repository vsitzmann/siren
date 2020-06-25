# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
import configargparse

import imageio
from functools import partial
import random
from tqdm.autonotebook import tqdm
import time
import utils
from torch.utils.tensorboard import SummaryWriter

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--checkpoint_path', default=None, type=str, required=True,
               help='path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--dataset', type=str, default='celeba_32x32',
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model_type', type=str, default='sine',
               help='Nonlinearity in the neural implicit representation')
p.add_argument('--test_sparsity', type=float, default=200,
               help='Amount of subsampled pixels input into the set encoder')
p.add_argument('--partial_conv', action='store_true', default=False, help='Use a partial convolution encoder')
opt = p.parse_args()

if opt.experiment_name is None:
    opt.experiment_name = opt.checkpoint_path.split('/')[-3] + '_TEST'
else:
    opt.experiment_name = opt.checkpoint_path.split('/')[-3] + '_' + opt.experiment_name

assert opt.dataset == 'celeba_32x32'
img_dataset_test = dataio.CelebA(split='test', downsampled=True)
coord_dataset_test = dataio.Implicit2DWrapper(img_dataset_test, sidelength=(32, 32))
generalization_dataset_test = dataio.ImageGeneralizationWrapper(coord_dataset_test, test_sparsity=200,
                                                                generalization_mode='conv_cnp_test')
image_resolution = (32, 32)

img_dataset_train = dataio.CelebA(split='train', downsampled=True)
coord_dataset_train = dataio.Implicit2DWrapper(img_dataset_train, sidelength=(32, 32))
generalization_dataset_train = dataio.ImageGeneralizationWrapper(coord_dataset_train, test_sparsity=200,
                                                                 generalization_mode='conv_cnp_test')

# Define the model.
model = meta_modules.ConvolutionalNeuralProcessImplicit2DHypernet(in_features=img_dataset_test.img_channels,
                                                                  out_features=img_dataset_test.img_channels,
                                                                  image_resolution=image_resolution,
                                                                  partial_conv=opt.partial_conv)
model.cuda()
model.eval()

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)


# Load checkpoint
model.load_state_dict(torch.load(opt.checkpoint_path))

# First experiment: Upsample training image
model_input = {'coords':dataio.get_mgrid(image_resolution)[None,:].cuda(),
               'img_sparse':generalization_dataset_train[0][0]['img_sparse'].unsqueeze(0).cuda()}
model_output = model(model_input)

out_img = dataio.lin2img(model_output['model_out'], image_resolution).squeeze().permute(1,2,0).detach().cpu().numpy()
out_img += 1
out_img /= 2.
out_img = np.clip(out_img, 0., 1.)

imageio.imwrite(os.path.join(root_path, 'upsampled_train.png'), out_img)

# Second experiment: sample larger range
model_input = {'coords':dataio.get_mgrid(image_resolution)[None,:].cuda()*5,
               'img_sparse':generalization_dataset_train[0][0]['img_sparse'].unsqueeze(0).cuda()}
model_output = model(model_input)

out_img = dataio.lin2img(model_output['model_out'], image_resolution).squeeze().permute(1,2,0).detach().cpu().numpy()
out_img += 1
out_img /= 2.
out_img = np.clip(out_img, 0., 1.)

imageio.imwrite(os.path.join(root_path, 'outside_range.png'), out_img)

# Third experiment: interpolate between latent codes
idx1, idx2 = 57, 181
model_input_1 = {'coords': dataio.get_mgrid(image_resolution)[None, :].cuda(),
                 'img_sparse': generalization_dataset_train[idx1][0]['img_sparse'].unsqueeze(0).cuda()}
model_input_2 = {'coords': dataio.get_mgrid(image_resolution)[None, :].cuda(),
                 'img_sparse': generalization_dataset_train[idx2][0]['img_sparse'].unsqueeze(0).cuda()}

embedding_1 = model.get_hypo_net_weights(model_input_1)[1]
embedding_2 = model.get_hypo_net_weights(model_input_2)[1]
for i in np.linspace(0,1,8):
    embedding = i*embedding_1 + (1.-i)*embedding_2
    model_input = {'coords': dataio.get_mgrid(image_resolution)[None, :].cuda(), 'embedding': embedding}
    model_output = model(model_input)

    out_img = dataio.lin2img(model_output['model_out'], image_resolution).squeeze().permute(1,2,0).detach().cpu().numpy()
    out_img += 1
    out_img /= 2.
    out_img = np.clip(out_img, 0., 1.)

    if i == 0.:
        out_img_cat = out_img
    else:
        out_img_cat = np.concatenate((out_img_cat, out_img), axis=1)

imageio.imwrite(os.path.join(root_path, 'interpolated_image.png'), out_img_cat)

# Fourth experiment: Fit test images
def to_uint8(img):
    img = img * 255
    img = img.astype(np.uint8)
    return img

def getTestMSE(dataloader, subdir):
    MSEs = []
    total_steps = 0
    utils.cond_mkdir(os.path.join(root_path, subdir))
    utils.cond_mkdir(os.path.join(root_path, 'ground_truth'))

    with tqdm(total=len(dataloader)) as pbar:
        for step, (model_input, gt) in enumerate(dataloader):
            model_input['idx'] = torch.Tensor([model_input['idx']]).long()
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            with torch.no_grad():
                model_output = model(model_input)

            out_img = dataio.lin2img(model_output['model_out'], image_resolution).squeeze().permute(1, 2, 0).detach().cpu().numpy()
            out_img += 1
            out_img /= 2.
            out_img = np.clip(out_img, 0., 1.)
            gt_img = dataio.lin2img(gt['img'], image_resolution).squeeze().permute(1, 2, 0).detach().cpu().numpy()
            gt_img += 1
            gt_img /= 2.
            gt_img = np.clip(gt_img, 0., 1.)

            sparse_img = model_input['img_sparse'].squeeze().detach().cpu().permute(1,2,0).numpy()
            mask = np.sum((sparse_img == 0), axis=2) == 3
            sparse_img += 1
            sparse_img /= 2.
            sparse_img = np.clip(sparse_img, 0., 1.)
            sparse_img[mask, ...] = 1.

            imageio.imwrite(os.path.join(root_path, subdir, str(total_steps)+'_sparse.png'), to_uint8(sparse_img))
            imageio.imwrite(os.path.join(root_path, subdir, str(total_steps)+'.png'), to_uint8(out_img))
            imageio.imwrite(os.path.join(root_path, 'ground_truth', str(total_steps)+'.png'), to_uint8(gt_img))

            MSE = np.mean((out_img - gt_img) ** 2)
            MSEs.append(MSE)

            pbar.update(1)
            total_steps += 1

    return MSEs

sparsities = [10, 100, 1000, 'full', 'half']
for sparsity in sparsities:
    generalization_dataset_test.update_test_sparsity(sparsity)
    dataloader = DataLoader(generalization_dataset_test, shuffle=False, batch_size=1, pin_memory=True, num_workers=0)
    MSE = getTestMSE(dataloader, 'test_'+str(sparsity)+'_pixels')
    np.save(os.path.join(root_path, 'MSE_'+str(sparsity)+'_context.npy'), MSE)
    print(np.mean(MSE))
