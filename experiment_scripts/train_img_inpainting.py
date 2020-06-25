# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')
p.add_argument('--k1', type=float, default=1, help='weight on prior')
p.add_argument('--sparsity', type=float, default=0.1, help='percentage of pixels filled')
p.add_argument('--prior', type=str, default=None, help='prior')
p.add_argument('--downsample', action='store_true', default=False, help='use image downsampling kernel')

p.add_argument('--epochs_til_ckpt', type=int, default=25,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--dataset', type=str, default='camera',
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--mask_path', type=str, default=None, help='Path to mask image')
p.add_argument('--custom_image', type=str, default=None, help='Path to single training image')
opt = p.parse_args()


if opt.dataset == 'camera':
    img_dataset = dataio.Camera()
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=512, compute_diff='all')
    image_resolution = (512, 512)
if opt.dataset == 'camera_downsampled':
    img_dataset = dataio.Camera(downsample_factor=2)
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=256, compute_diff='all')
    image_resolution = (256, 256)
if opt.dataset == 'custom':
    img_dataset = dataio.ImageFile(opt.custom_image)
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=(img_dataset[0].size[1], img_dataset[0].size[0]),
                                             compute_diff='all')
    image_resolution = (img_dataset[0].size[1], img_dataset[0].size[0])

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh':
    model = modules.SingleBVPNet(type=opt.model_type, mode='mlp', out_features=img_dataset.img_channels, sidelength=image_resolution,
                                 downsample=opt.downsample)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode=opt.model_type, out_features=img_dataset.img_channels, sidelength=image_resolution,
                                 downsample=opt.downsample)
else:
    raise NotImplementedError
model.cuda()

root_path = os.path.join(opt.logging_root, opt.experiment_name)

if opt.mask_path:
    mask = Image.open(opt.mask_path)
    mask = ToTensor()(mask)
    mask = mask.float().cuda()
    percentage = torch.sum(mask).cpu().numpy() / np.prod(mask.shape)
    print("mask sparsity %f" % (percentage))
else:
    mask = torch.rand(image_resolution) < opt.sparsity
    mask = mask.float().cuda()

# Define the loss
if opt.prior is None:
    loss_fn = partial(loss_functions.image_mse, mask.view(-1,1))
elif opt.prior == 'TV':
    loss_fn = partial(loss_functions.image_mse_TV_prior, mask.view(-1,1), opt.k1, model)
elif opt.prior == 'FH':
    loss_fn = partial(loss_functions.image_mse_FH_prior, mask.view(-1,1), opt.k1, model)
summary_fn = partial(utils.write_image_summary, image_resolution)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)
