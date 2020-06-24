'''Reproduces Paper Sec. 4.1, Supplement Sec. 3, reconstruction from gradient.
'''

# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=16384)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=25,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--dataset', type=str, choices=['camera','bsd500'], default='camera',
               help='Dataset: choices=[camera,bsd500].')
p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

if opt.dataset == 'camera':
    img_dataset = dataio.Camera()
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=256, compute_diff='gradients')
elif opt.dataset == 'bsd500':
    # you can select the image your like in idx to sample
    img_dataset = dataio.BSD500ImageDataset(in_folder='../data/BSD500/train',
                                            idx_to_sample=[19])
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=256, compute_diff='gradients')

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh' or opt.model_type == 'softplus':
    model = modules.SingleBVPNet(type=opt.model_type, mode='mlp', sidelength=(256, 256))
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode=opt.model_type, sidelength=(256, 256))
else:
    raise NotImplementedError
model.cuda()

# Define the loss & summary functions
loss_fn = loss_functions.gradients_mse
summary_fn = utils.write_gradients_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False)
