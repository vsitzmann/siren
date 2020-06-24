'''Reproduces Paper Sec. 4.3 and Supplement Sec. 5'''

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
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
p.add_argument('--num_epochs', type=int, default=100000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--velocity', type=str, default='uniform', required=False, choices=['uniform', 'square', 'circle'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--pretrain', action='store_true', default=False, required=False, help='Pretrain dirichlet and neumann conditions')
p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

# if we have a velocity perturbation, offset the source
source_coords = [0., 0., 0.]

dataset = dataio.WaveSource(sidelength=340, velocity=opt.velocity,
                            source_coords=source_coords, pretrain=opt.pretrain)

dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

model = modules.SingleBVPNet(in_features=3, out_features=1, type=opt.model, mode=opt.mode,
                                final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
model.cuda()

# Define the loss
loss_fn = loss_functions.wave_pml
summary_fn = utils.write_wave_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=opt.clip_grad,
               use_lbfgs=opt.use_lbfgs)
