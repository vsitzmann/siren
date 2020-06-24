import matplotlib.pyplot as plt
import numpy as np
import torch
import dataio
import os
import diff_operators
from torchvision.utils import make_grid, save_image
import skimage.measure
import cv2
import meta_modules
import scipy.io.wavfile as wavfile
import cmapy


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_result_img(experiment_name, filename, img):
    root_path = '/media/data1/sitzmann/generalization/results'
    trgt_dir = os.path.join(root_path, experiment_name)

    img = img.detach().cpu().numpy()
    np.save(os.path.join(trgt_dir, filename), img)


def densely_sample_activations(model, num_dim=1, num_steps=int(1e6)):
    input = torch.linspace(-1., 1., steps=num_steps).float()

    if num_dim == 1:
        input = input[...,None]
    else:
        input = torch.stack(torch.meshgrid(*(input for _ in num_dim)), dim=-1).view(-1, num_dim)

    input = {'coords':input[None,:].cuda()}
    with torch.no_grad():
        activations = model.forward_with_activations(input)['activations']
    return activations


def write_wave_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):

    sl = 256
    def scale_percentile(pred, min_perc=1, max_perc=99):
        min = np.percentile(pred.cpu().numpy(),1)
        max = np.percentile(pred.cpu().numpy(),99)
        pred = torch.clamp(pred, min, max)
        return (pred - min) / (max-min)

    with torch.no_grad():
        frames = [0.0, 0.05, 0.1, 0.15, 0.25]
        coords = [dataio.get_mgrid((1, sl, sl), dim=3)[None,...].cuda() for f in frames]
        for idx, f in enumerate(frames):
            coords[idx][..., 0] = f
        coords = torch.cat(coords, dim=0)

        Nslice = 10
        output = torch.zeros(coords.shape[0], coords.shape[1], 1)
        split = int(coords.shape[1] / Nslice)
        for i in range(Nslice):
            pred = model({'coords':coords[:, i*split:(i+1)*split, :]})['model_out']
            output[:, i*split:(i+1)*split, :] =  pred.cpu()

    min_max_summary(prefix + 'pred', pred, writer, total_steps)
    pred = output.view(len(frames), 1, sl, sl)

    plt.switch_backend('agg')
    fig = plt.figure()
    plt.subplot(2,2,1)
    data = pred[0, :, sl//2, :].numpy().squeeze()
    plt.plot(np.linspace(-1, 1, sl), data)
    plt.ylim([-0.01, 0.02])

    plt.subplot(2,2,2)
    data = pred[1, :, sl//2, :].numpy().squeeze()
    plt.plot(np.linspace(-1, 1, sl), data)
    plt.ylim([-0.01, 0.02])

    plt.subplot(2,2,3)
    data = pred[2, :, sl//2, :].numpy().squeeze()
    plt.plot(np.linspace(-1, 1, sl), data)
    plt.ylim([-0.01, 0.02])

    plt.subplot(2,2,4)
    data = pred[3, :, sl//2, :].numpy().squeeze()
    plt.plot(np.linspace(-1, 1, sl), data)
    plt.ylim([-0.01, 0.02])

    writer.add_figure(prefix + 'center_slice', fig, global_step=total_steps)

    pred = torch.clamp(pred, -0.002, 0.002)
    writer.add_image(prefix + 'pred_img', make_grid(pred, scale_each=False, normalize=True),
                     global_step=total_steps)


def write_helmholtz_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    sl = 256
    coords = dataio.get_mgrid(sl)[None,...].cuda()

    def scale_percentile(pred, min_perc=1, max_perc=99):
        min = np.percentile(pred.cpu().numpy(),1)
        max = np.percentile(pred.cpu().numpy(),99)
        pred = torch.clamp(pred, min, max)
        return (pred - min) / (max-min)

    with torch.no_grad():
        if 'coords_sub' in model_input:
            summary_model_input = {'coords':coords.repeat(min(2, model_input['coords_sub'].shape[0]),1,1)}
            summary_model_input['coords_sub'] = model_input['coords_sub'][:2,...]
            summary_model_input['img_sub'] = model_input['img_sub'][:2,...]
            pred = model(summary_model_input)['model_out']
        else:
            pred = model({'coords': coords})['model_out']

        if 'pretrain' in gt:
            gt['squared_slowness_grid'] = pred[...,-1, None].clone() + 1.
            if torch.all(gt['pretrain'] == -1):
                gt['squared_slowness_grid'] = torch.clamp(pred[...,-1, None].clone(), min=-0.999) + 1.
                gt['squared_slowness_grid'] = torch.where((torch.abs(coords[...,0,None]) > 0.75) | (torch.abs(coords[...,1,None]) > 0.75),
                                            torch.ones_like(gt['squared_slowness_grid']),
                                            gt['squared_slowness_grid'])
            pred = pred[...,:-1]

        pred = dataio.lin2img(pred)

        pred_cmpl = pred[...,0::2,:,:].cpu().numpy() + 1j * pred[...,1::2,:,:].cpu().numpy()
        pred_angle = torch.from_numpy(np.angle(pred_cmpl))
        pred_mag = torch.from_numpy(np.abs(pred_cmpl))

        min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
        min_max_summary(prefix + 'pred_real', pred[..., 0::2, :, :], writer, total_steps)
        min_max_summary(prefix + 'pred_abs', torch.sqrt(pred[..., 0::2, :, :]**2 + pred[..., 1::2, :, :]**2), writer, total_steps)
        min_max_summary(prefix + 'squared_slowness', gt['squared_slowness_grid'], writer, total_steps)

        pred = scale_percentile(pred)
        pred_angle = scale_percentile(pred_angle)
        pred_mag = scale_percentile(pred_mag)

        pred = pred.permute(1, 0, 2, 3)
        pred_mag = pred_mag.permute(1, 0, 2, 3)
        pred_angle = pred_angle.permute(1, 0, 2, 3)

    writer.add_image(prefix + 'pred_real', make_grid(pred[0::2, :, :, :], scale_each=False, normalize=True),
                     global_step=total_steps)
    writer.add_image(prefix + 'pred_imaginary', make_grid(pred[1::2, :, :, :], scale_each=False, normalize=True),
                     global_step=total_steps)
    writer.add_image(prefix + 'pred_angle', make_grid(pred_angle, scale_each=False, normalize=True),
                     global_step=total_steps)
    writer.add_image(prefix + 'pred_mag', make_grid(pred_mag, scale_each=False, normalize=True),
                     global_step=total_steps)

    if 'gt' in gt:
        gt_field = dataio.lin2img(gt['gt'])
        gt_field_cmpl = gt_field[...,0,:,:].cpu().numpy() + 1j * gt_field[...,1,:,:].cpu().numpy()
        gt_angle = torch.from_numpy(np.angle(gt_field_cmpl))
        gt_mag = torch.from_numpy(np.abs(gt_field_cmpl))

        gt_field = scale_percentile(gt_field)
        gt_angle = scale_percentile(gt_angle)
        gt_mag = scale_percentile(gt_mag)

        writer.add_image(prefix + 'gt_real', make_grid(gt_field[...,0,:,:], scale_each=False, normalize=True),
                         global_step=total_steps)
        writer.add_image(prefix + 'gt_imaginary', make_grid(gt_field[...,1,:,:], scale_each=False, normalize=True),
                         global_step=total_steps)
        writer.add_image(prefix + 'gt_angle', make_grid(gt_angle, scale_each=False, normalize=True),
                         global_step=total_steps)
        writer.add_image(prefix + 'gt_mag', make_grid(gt_mag, scale_each=False, normalize=True),
                         global_step=total_steps)
        min_max_summary(prefix + 'gt_real', gt_field[..., 0, :, :], writer, total_steps)

    velocity = torch.sqrt(1/dataio.lin2img(gt['squared_slowness_grid']))[:1]
    min_max_summary(prefix + 'velocity', velocity[..., 0, :, :], writer, total_steps)
    velocity = scale_percentile(velocity)
    writer.add_image(prefix + 'velocity', make_grid(velocity[...,0,:,:], scale_each=False, normalize=True),
                     global_step=total_steps)

    if 'squared_slowness_grid' in gt:
        writer.add_image(prefix + 'squared_slowness', make_grid(dataio.lin2img(gt['squared_slowness_grid'])[:2,:1],
                                                                scale_each=False, normalize=True),
                         global_step=total_steps)

    if 'img_sub' in model_input:
        writer.add_image(prefix + 'img', make_grid(dataio.lin2img(model_input['img_sub'])[:2,:1],
                                                                scale_each=False, normalize=True),
                         global_step=total_steps)

    if isinstance(model, meta_modules.NeuralProcessImplicit2DHypernetBVP):
        hypernet_activation_summary(model, model_input, gt, model_output, writer, total_steps, prefix)


def write_image_summary_small(image_resolution, mask, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    if mask is None:
        gt_img = dataio.lin2img(gt['img'], image_resolution)
        gt_dense = gt_img
    else:
        gt_img = dataio.lin2img(gt['img'], image_resolution) * mask
        gt_dense = gt_img

    pred_img = dataio.lin2img(model_output['model_out'], image_resolution)

    with torch.no_grad():
        img_gradient = torch.autograd.grad(model_output['model_out'], [model_output['model_in']],
                                           grad_outputs=torch.ones_like(model_output['model_out']), create_graph=True,
                                           retain_graph=True)[0]

        grad_norm = img_gradient.norm(dim=-1, keepdim=True)
        grad_norm = dataio.lin2img(grad_norm, image_resolution)
        writer.add_image(prefix + 'pred_grad_norm', make_grid(grad_norm, scale_each=False, normalize=True),
                         global_step=total_steps)

    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)
    write_psnr(pred_img, gt_dense, writer, total_steps, prefix + 'img_dense_')

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_img', pred_img, writer, total_steps)
    min_max_summary(prefix + 'gt_img', gt_img, writer, total_steps)

    hypernet_activation_summary(model, model_input, gt, model_output, writer, total_steps, prefix)


def make_contour_plot(array_2d,mode='log'):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    if(mode=='log'):
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels) # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif(mode=='lin'):
        num_levels = 10
        levels = np.linspace(-.5,.5,num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig


def write_sdf_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    slice_coords_2d = dataio.get_mgrid(512)

    with torch.no_grad():
        yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
        yz_slice_model_input = {'coords': yz_slice_coords.cuda()[None, ...]}

        yz_model_out = model(yz_slice_model_input)
        sdf_values = yz_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'yz_sdf_slice', fig, global_step=total_steps)

        xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                     torch.zeros_like(slice_coords_2d[:, :1]),
                                     slice_coords_2d[:,-1:]), dim=-1)
        xz_slice_model_input = {'coords': xz_slice_coords.cuda()[None, ...]}

        xz_model_out = model(xz_slice_model_input)
        sdf_values = xz_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'xz_sdf_slice', fig, global_step=total_steps)

        xy_slice_coords = torch.cat((slice_coords_2d[:,:2],
                                     -0.75*torch.ones_like(slice_coords_2d[:, :1])), dim=-1)
        xy_slice_model_input = {'coords': xy_slice_coords.cuda()[None, ...]}

        xy_model_out = model(xy_slice_model_input)
        sdf_values = xy_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)

        min_max_summary(prefix + 'model_out_min_max', model_output['model_out'], writer, total_steps)
        min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)


def hypernet_activation_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    with torch.no_grad():
        hypo_parameters, embedding = model.get_hypo_net_weights(model_input)

        for name, param in hypo_parameters.items():
            writer.add_histogram(prefix + name, param.cpu(), global_step=total_steps)

        writer.add_histogram(prefix + 'latent_code', embedding.cpu(), global_step=total_steps)


def write_video_summary(vid_dataset, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    resolution = vid_dataset.shape
    frames = [0, 60, 120, 200]
    Nslice = 10
    with torch.no_grad():
        coords = [dataio.get_mgrid((1, resolution[1], resolution[2]), dim=3)[None,...].cuda() for f in frames]
        for idx, f in enumerate(frames):
            coords[idx][..., 0] = (f / (resolution[0] - 1) - 0.5) * 2
        coords = torch.cat(coords, dim=0)

        output = torch.zeros(coords.shape)
        split = int(coords.shape[1] / Nslice)
        for i in range(Nslice):
            pred = model({'coords':coords[:, i*split:(i+1)*split, :]})['model_out']
            output[:, i*split:(i+1)*split, :] =  pred.cpu()

    pred_vid = output.view(len(frames), resolution[1], resolution[2], 3) / 2 + 0.5
    pred_vid = torch.clamp(pred_vid, 0, 1)
    gt_vid = torch.from_numpy(vid_dataset.vid[frames, :, :, :])
    psnr = 10*torch.log10(1 / torch.mean((gt_vid - pred_vid)**2))

    pred_vid = pred_vid.permute(0, 3, 1, 2)
    gt_vid = gt_vid.permute(0, 3, 1, 2)

    output_vs_gt = torch.cat((gt_vid, pred_vid), dim=-2)
    writer.add_image(prefix + 'output_vs_gt', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)
    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_vid', pred_vid, writer, total_steps)
    writer.add_scalar(prefix + "psnr", psnr, total_steps)


def write_image_summary(image_resolution, model, model_input, gt,
                        model_output, writer, total_steps, prefix='train_'):
    gt_img = dataio.lin2img(gt['img'], image_resolution)
    pred_img = dataio.lin2img(model_output['model_out'], image_resolution)

    img_gradient = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    img_laplace = diff_operators.laplace(model_output['model_out'], model_output['model_in'])

    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    pred_img = dataio.rescale_img((pred_img+1)/2, mode='clamp').permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
    pred_grad = dataio.grads2img(dataio.lin2img(img_gradient)).permute(1,2,0).squeeze().detach().cpu().numpy()
    pred_lapl = cv2.cvtColor(cv2.applyColorMap(dataio.to_uint8(dataio.rescale_img(
                             dataio.lin2img(img_laplace), perc=2).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()), cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)

    gt_img = dataio.rescale_img((gt_img+1) / 2, mode='clamp').permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
    gt_grad = dataio.grads2img(dataio.lin2img(gt['gradients'])).permute(1, 2, 0).squeeze().detach().cpu().numpy()
    gt_lapl = cv2.cvtColor(cv2.applyColorMap(dataio.to_uint8(dataio.rescale_img(
        dataio.lin2img(gt['laplace']), perc=2).permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()), cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)

    writer.add_image(prefix + 'pred_img', torch.from_numpy(pred_img).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'pred_grad', torch.from_numpy(pred_grad).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'pred_lapl', torch.from_numpy(pred_lapl).permute(2,0,1), global_step=total_steps)
    writer.add_image(prefix + 'gt_img', torch.from_numpy(gt_img).permute(2,0,1), global_step=total_steps)
    writer.add_image(prefix + 'gt_grad', torch.from_numpy(gt_grad).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'gt_lapl', torch.from_numpy(gt_lapl).permute(2, 0, 1), global_step=total_steps)

    write_psnr(dataio.lin2img(model_output['model_out'], image_resolution),
               dataio.lin2img(gt['img'], image_resolution), writer, total_steps, prefix+'img_')


def write_laplace_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    # Plot comparison images
    gt_img = dataio.lin2img(gt['img'])
    pred_img = dataio.lin2img(model_output['model_out'])

    output_vs_gt = torch.cat((dataio.rescale_img(gt_img), dataio.rescale_img(pred_img,perc=1e-2)), dim=-1)
    writer.add_image(prefix + 'comp_gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    # Plot comparisons laplacian (this is what has been fitted)
    gt_laplace = dataio.lin2img(gt['laplace'])
    pred_laplace = diff_operators.laplace(model_output['model_out'], model_output['model_in'])
    pred_laplace = dataio.lin2img(pred_laplace)

    output_vs_gt_laplace = torch.cat((gt_laplace, pred_laplace), dim=-1)
    writer.add_image(prefix + 'comp_gt_vs_pred_laplace', make_grid(output_vs_gt_laplace, scale_each=False, normalize=True),
                     global_step=total_steps)

    # Plot image gradient
    img_gradient = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    grads_img = dataio.grads2img(dataio.lin2img(img_gradient))
    writer.add_image(prefix + 'pred_grad', make_grid(grads_img, scale_each=False, normalize=True),
                     global_step=total_steps)

    # Plot gt image
    writer.add_image(prefix + 'gt_img', make_grid(gt_img, scale_each=False, normalize=True),
                     global_step=total_steps)

    # Plot gt laplacian
    # writer.add_image(prefix + 'gt_laplace', make_grid(gt_laplace, scale_each=False, normalize=True),
    #                  global_step=total_steps)
    gt_laplace_img = dataio.to_uint8(dataio.to_numpy(dataio.rescale_img(gt_laplace, 'scale', 1)))
    gt_laplace_img = cv2.applyColorMap(gt_laplace_img.squeeze(), cmapy.cmap('RdBu'))
    gt_laplace_img = cv2.cvtColor(gt_laplace_img, cv2.COLOR_BGR2RGB)
    writer.add_image(prefix + 'gt_lapl', torch.from_numpy(gt_laplace_img).permute(2, 0, 1), global_step=total_steps)

    # Plot pred image
    writer.add_image(prefix + 'pred_img', make_grid(pred_img, scale_each=False, normalize=True),
                     global_step=total_steps)

    # Plot pred gradient
    pred_gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    pred_grads_img = dataio.grads2img(dataio.lin2img(pred_gradients))
    writer.add_image(prefix + 'pred_grad', make_grid(pred_grads_img, scale_each=False, normalize=True),
                     global_step=total_steps)

    # Plot pred laplacian
    # writer.add_image(prefix + 'pred_lapl', make_grid(pred_laplace, scale_each=False, normalize=True),
    #                  global_step=total_steps)
    pred_laplace_img = dataio.to_uint8(dataio.to_numpy(dataio.rescale_img(pred_laplace,'scale',1)))
    pred_laplace_img = cv2.applyColorMap(pred_laplace_img.squeeze(),cmapy.cmap('RdBu'))
    pred_laplace_img = cv2.cvtColor(pred_laplace_img, cv2.COLOR_BGR2RGB)
    writer.add_image(prefix + 'pred_lapl', torch.from_numpy(pred_laplace_img).permute(2,0,1), global_step=total_steps)

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'gt_laplace', gt_laplace, writer, total_steps)
    min_max_summary(prefix + 'pred_laplace', pred_laplace, writer, total_steps)
    min_max_summary(prefix + 'pred_img', pred_img, writer, total_steps)
    min_max_summary(prefix + 'gt_img', gt_img, writer, total_steps)


def write_gradients_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    # Plot comparisons images
    gt_img = dataio.lin2img(gt['img'])
    pred_img = dataio.lin2img(model_output['model_out'])


    output_vs_gt = torch.cat((dataio.rescale_img(gt_img), dataio.rescale_img(pred_img,perc=1e-2)), dim=-1)
    writer.add_image(prefix + 'comp_gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)


    # Plot comparisons gradient (this is what has been fitted)
    gt_gradients = gt['gradients']
    gt_grads_img = dataio.grads2img(dataio.lin2img(gt_gradients))

    pred_gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    pred_grads_img = dataio.grads2img(dataio.lin2img(pred_gradients))

    output_vs_gt_gradients = torch.cat((gt_grads_img, pred_grads_img), dim=-1)
    writer.add_image(prefix + 'comp_gt_vs_pred_gradients', make_grid(output_vs_gt_gradients, scale_each=False, normalize=True),
                     global_step=total_steps)

    # Plot gt image
    writer.add_image(prefix + 'gt_img', make_grid(gt_img, scale_each=False, normalize=True),
                     global_step=total_steps)

    # Plot gt gradient
    writer.add_image(prefix + 'gt_grad', make_grid(gt_grads_img, scale_each=False, normalize=True),
                     global_step=total_steps)

    # Plot pred image
    writer.add_image(prefix + 'pred_img', make_grid(pred_img, scale_each=False, normalize=True),
                     global_step=total_steps)

    # Plot pred gradient
    writer.add_image(prefix + 'pred_grad', make_grid(pred_grads_img, scale_each=False, normalize=True),
                     global_step=total_steps)

    # Plot pred laplacian
    pred_laplace = diff_operators.laplace(model_output['model_out'], model_output['model_in'])
    pred_laplace = dataio.lin2img(pred_laplace)

    pred_laplace_img = dataio.to_uint8(dataio.to_numpy(dataio.rescale_img(pred_laplace,'scale',1)))
    pred_laplace_img = cv2.applyColorMap(pred_laplace_img.squeeze(),cmapy.cmap('RdBu'))
    pred_laplace_img = cv2.cvtColor(pred_laplace_img, cv2.COLOR_BGR2RGB)
    writer.add_image(prefix + 'pred_lapl', torch.from_numpy(pred_laplace_img).permute(2,0,1), global_step=total_steps)

    if 'laplace' in gt:
        # Plot gt laplacian
        gt_laplace = gt['laplace']
        gt_laplace_img = dataio.lin2img(gt_laplace)
        gt_laplace_img = dataio.to_uint8(dataio.to_numpy(dataio.rescale_img(gt_laplace_img, 'scale',1)))
        gt_laplace_img = cv2.applyColorMap(gt_laplace_img.squeeze(), cmapy.cmap('RdBu'))
        gt_laplace_img = cv2.cvtColor(gt_laplace_img, cv2.COLOR_BGR2RGB)
        writer.add_image(prefix + 'gt_lapl', torch.from_numpy(gt_laplace_img).permute(2,0,1), global_step=total_steps)

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'gt_grads', gt_gradients, writer, total_steps)
    min_max_summary(prefix + 'pred_laplace', pred_laplace, writer, total_steps)
    min_max_summary(prefix + 'pred_img', pred_img, writer, total_steps)
    min_max_summary(prefix + 'gt_img', gt_img, writer, total_steps)


def write_gradcomp_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    # Plot gt gradients (this is what has been fitted)
    gt_gradients = gt['gradients']
    gt_grads_img = dataio.grads2img(dataio.lin2img(gt_gradients))

    pred_gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    pred_grads_img = dataio.grads2img(dataio.lin2img(pred_gradients))

    output_vs_gt_gradients = torch.cat((gt_grads_img, pred_grads_img), dim=-1)
    writer.add_image(prefix + 'comp_gt_vs_pred_gradients', make_grid(output_vs_gt_gradients, scale_each=False, normalize=True),
                     global_step=total_steps)

    # Plot gt
    gt_grads1 = gt['grads1']
    gt_grads1_img = dataio.grads2img(dataio.lin2img(gt_grads1))

    gt_grads2 = gt['grads2']
    gt_grads2_img = dataio.grads2img(dataio.lin2img(gt_grads2))

    writer.add_image(prefix + 'gt_grads1', make_grid(gt_grads1_img, scale_each=False, normalize=True),
                     global_step=total_steps)
    writer.add_image(prefix + 'gt_grads2', make_grid(gt_grads2_img, scale_each=False, normalize=True),
                     global_step=total_steps)

    writer.add_image(prefix + 'gt_gradcomp', make_grid(gt_grads_img, scale_each=False, normalize=True),
                     global_step=total_steps)
    writer.add_image(prefix + 'pred_gradcomp', make_grid(pred_grads_img, scale_each=False, normalize=True),
                     global_step=total_steps)
    # Plot gt image
    gt_img1 = dataio.lin2img(gt['img1'])
    gt_img2 = dataio.lin2img(gt['img2'])
    writer.add_image(prefix + 'gt_img1', make_grid(gt_img1, scale_each=False, normalize=True),
                     global_step=total_steps)
    writer.add_image(prefix + 'gt_img2', make_grid(gt_img2, scale_each=False, normalize=True),
                     global_step=total_steps)

    # Plot pred compo image
    pred_img = dataio.rescale_img(dataio.lin2img(model_output['model_out']))
    writer.add_image(prefix + 'pred_comp_img', make_grid(pred_img, scale_each=False, normalize=True),
                     global_step=total_steps)

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'gt_laplace', gt_gradients, writer, total_steps)
    min_max_summary(prefix + 'pred_img', pred_img, writer, total_steps)


def write_audio_summary(logging_root_path, model, model_input, gt, model_output, writer, total_steps, prefix='train'):
    gt_func = torch.squeeze(gt['func'])
    gt_rate = torch.squeeze(gt['rate']).detach().cpu().numpy()
    gt_scale = torch.squeeze(gt['scale']).detach().cpu().numpy()
    pred_func = torch.squeeze(model_output['model_out'])
    coords = torch.squeeze(model_output['model_in'].clone()).detach().cpu().numpy()

    fig, axes = plt.subplots(3,1)

    strt_plot, fin_plot = int(0.05*len(coords)), int(0.95*len(coords))
    coords = coords[strt_plot:fin_plot]
    gt_func_plot = gt_func.detach().cpu().numpy()[strt_plot:fin_plot]
    pred_func_plot = pred_func.detach().cpu().numpy()[strt_plot:fin_plot]

    axes[1].plot(coords, pred_func_plot)
    axes[0].plot(coords, gt_func_plot)
    axes[2].plot(coords, gt_func_plot - pred_func_plot)

    axes[0].get_xaxis().set_visible(False)
    axes[1].axes.get_xaxis().set_visible(False)
    axes[2].axes.get_xaxis().set_visible(False)

    writer.add_figure(prefix + 'gt_vs_pred', fig, global_step=total_steps)

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_func', pred_func, writer, total_steps)
    min_max_summary(prefix + 'gt_func', gt_func, writer, total_steps)

    # write audio files:
    wavfile.write(os.path.join(logging_root_path, 'gt.wav'), gt_rate, gt_func.detach().cpu().numpy())
    wavfile.write(os.path.join(logging_root_path, 'pred.wav'), gt_rate, pred_func.detach().cpu().numpy())


def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)


def write_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5

        ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
        psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

        psnrs.append(psnr)
        ssims.append(ssim)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)
    writer.add_scalar(prefix + "ssim", np.mean(ssims), iter)
