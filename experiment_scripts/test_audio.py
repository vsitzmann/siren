# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, modules

from torch.utils.data import DataLoader
import configargparse
import torch
import scipy.io.wavfile as wavfile

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default='audio',
               help='Name of subdirectory in logging_root where wav file will be saved.')
p.add_argument('--gt_wav_path', type=str, default='../data/gt_bach.wav', help='ground truth wav path')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')
p.add_argument('--checkpoint_path', required=True, help='Checkpoint to trained model.')

opt = p.parse_args()

audio_dataset = dataio.AudioFile(filename=opt.gt_wav_path)
coord_dataset = dataio.ImplicitAudioWrapper(audio_dataset)

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# Define the model and load in checkpoint path
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh':
    model = modules.SingleBVPNet(type=opt.model_type, mode='mlp', in_features=1)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode=opt.model_type, fn_samples=len(audio_dataset.data), in_features=1)
else:
    raise NotImplementedError
model.load_state_dict(torch.load(opt.checkpoint_path))
model.cuda()

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

# Get ground truth and input data
model_input, gt = next(iter(dataloader))
model_input = {key: value.cuda() for key, value in model_input.items()}
gt = {key: value.cuda() for key, value in gt.items()}

# Evaluate the trained model
with torch.no_grad():
    model_output = model(model_input)

waveform = torch.squeeze(model_output['model_out']).detach().cpu().numpy()
rate = torch.squeeze(gt['rate']).detach().cpu().numpy()
wavfile.write(os.path.join(opt.logging_root, opt.experiment_name, 'pred_waveform.wav'), rate, waveform)