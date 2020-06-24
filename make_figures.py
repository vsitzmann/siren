from collections import defaultdict
from tqdm import tqdm
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
import cv2
import cmapy
import skimage
import pathlib
from tensorboard.backend.event_processing import event_accumulator
from moviepy.editor import VideoFileClip, clips_array, vfx, ImageSequenceClip, CompositeVideoClip, concatenate_videoclips, VideoClip, TextClip
from matplotlib import animation


def extract_images_from_summary(events_path, tag_names_to_look_for, suffix='', img_outdir=None, colormap=None):
    print("Extracting data from tensorboard summary...")
    event_acc = event_accumulator.EventAccumulator(events_path, size_guidance={'images': 0})
    event_acc.Reload()

    # a suffix to append to the name if we save in outdir
    strsuffix = suffix

    if img_outdir is not None:
        outdir = pathlib.Path(img_outdir)
        outdir.mkdir(exist_ok=True, parents=True)

    # We are looking at all the images ...
    image_dict = defaultdict(list)
    for tag in event_acc.Tags()['images']:
        print("processing tag %s"%tag)
        events = event_acc.Images(tag)
        tag_name = tag.replace('/', '_')
        # ... that have the tag name: "tag_name_to_look_for"
        if tag_name in tag_names_to_look_for:
            tag_name = tag_name + strsuffix

            if img_outdir is not None:
                dirpath = outdir / tag_name
                dirpath.mkdir(exist_ok=True, parents=True)

            for index, event in enumerate(events):
                s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
                image = cv2.imdecode(s, cv2.IMREAD_COLOR)

                if colormap is not None:
                    image = cv2.applyColorMap(image[..., 0], cmapy.cmap(colormap))

                if img_outdir is not None:
                    outpath = dirpath / '{:04}.png'.format(index)
                    cv2.imwrite(outpath.as_posix(), image)

                image_dict[tag].append(image)
    return image_dict


def extract_from_summary(path, value_tag):
    ''' Extracts values and wall times from tensorboard summaries
    '''
    if os.path.isdir(path):
        path = glob.glob(os.path.join(path, "*"))[0]

    origin_wall_time = None
    wall_times = []
    values = []

    for event in tf.compat.v1.train.summary_iterator(path):
        if not origin_wall_time:
            origin_wall_time = event.wall_time
        for value in event.summary.value:
            if value.tag == value_tag:
                wall_times.append(event.wall_time - origin_wall_time)
                values.append(value.simple_value)
    return wall_times, values


def save_video(video_clip, filepath):
    video_clip.resize(width=1080)

    height, width = video_clip.h, video_clip.w

    if height % 2:
        height += 1

    video_clip.resize(width=1080, height=height).write_videofile(filepath, fps=25,
                                                                 audio_codec='libfdk_aac', audio=False)


def make_video_grid_from_filepaths(num_rows, num_cols, video_list, trgt_name,
                                   margin_color=(255,255,255), margin_width=0,
                                   column_wise=True):

    clip_array = [[] for _ in range(num_rows)]
    for col in range(num_cols):
        for row in range(num_rows):
            if column_wise:
                idx = col * num_rows + row
            else:
                idx = row * num_cols + col

            video_clip = VideoFileClip(video_list[idx]).margin(margin_width, color=margin_color)
            if margin_width > 0:
                video_clip = video_clip.margin(margin_width, color=margin_color)

            clip_array[row].append(video_clip)

    final_clip = clips_array(clip_array)
    save_video(final_clip, trgt_name)


def animated_line_plot(x_axis, data, trgt_path, legend_loc='lower right', plot_type=None):
    fig, ax = plt.subplots()
    fontdict = {'size': 16}
    ax.tick_params(axis='both', which='major', direction='in', labelsize=11)
    ax.set_ylabel("PSNR", fontdict=fontdict)
    ax.set_xlabel("Iterations", fontdict=fontdict)
    if plot_type == 'image':
        ax.set_xticks([5000, 10000, 15000])
        ax.set_xticklabels(['5,000', '10,000', '15,000'])
        ax.set_xlim(0, 15000)
        ax.set_yticks([10, 20, 30, 40, 50, 60])
        ax.set_ylim(0, 60)
    elif plot_type == 'poisson':
        ax.set_xticks([1000, 2000, 3000, 4000])
        ax.set_xticklabels(['1,000', '2,000', '3,000', '4,000'])
        ax.set_xlim(0, 4000)
        ax.set_yticks([5, 10, 15, 20, 25, 30, 35])
        ax.set_ylim(0, 35)
    ax.grid()

    lines = []
    for key, y in data.items():
        lobj = ax.plot(x_axis, y, label=key)[0]
        lines.append(lobj)

    ax.legend(loc=legend_loc, bbox_to_anchor=(0.95, 0.035))

    def update(num, x, data, lines):
        for idx, (_, y) in enumerate(data.items()):
            lines[idx].set_data(x[:num], y[:num])
        return lines

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=26, bitrate=1800)
    anim = animation.FuncAnimation(fig, update, fargs=[x_axis, data, lines],
                                   frames=len(x_axis), interval=1, blit=True)
    anim.save(trgt_path, writer=writer)


def make_video_from_tensorboard_summaries(summary_paths, trgt_path, image_extraction_dir, pred_tag_list,
                                          num_rows, num_cols, gt_tag_list=None, overwrite=False, colormap=None):
    for key in summary_paths.keys():
        if os.path.isdir(summary_paths[key]):
            summary_paths[key] = glob.glob(os.path.join(summary_paths[key], "*"))[0]

    video_filepaths = []

    def extract_images_make_videoclips(summary_path, root_dir, tag_list):
        if not os.path.exists(root_dir) or overwrite:
            extract_images_from_summary(summary_path,
                                        tag_names_to_look_for=tag_list,
                                        img_outdir=root_dir,
                                        colormap=colormap)

        for tag in tag_list:
            dir = os.path.join(root_dir, tag)
            video_path = os.path.join(dir, 'video.mp4')
            if not os.path.exists(video_path):
                print("Making video for %s" % dir)
                img_clip = ImageSequenceClip(dir, 26)
                save_video(img_clip, video_path)
            video_filepaths.append(video_path)

    # Extract ground truth
    gt_dir = os.path.join(image_extraction_dir, 'gt')
    summary_path = summary_paths[next(iter(summary_paths))]
    extract_images_make_videoclips(summary_path, gt_dir, gt_tag_list)

    # Extract all model predictions
    for key, summary_path in summary_paths.items():
        subdir = os.path.join(image_extraction_dir, key)
        summary_path = summary_path
        extract_images_make_videoclips(summary_path, subdir, pred_tag_list)

    # Now make joint video...
    if os.path.exists(trgt_path):
        val = input("The video %s exists. Overwrite? (y/n)" % trgt_path)
        if val == 'y':
            os.remove(trgt_path)

    make_video_grid_from_filepaths(num_rows, num_cols, video_list=video_filepaths,
                                   trgt_name=trgt_path, margin_width=0)


def glob_all_imgs(trgt_dir):
    '''Returns list of all images in trgt_dir
    '''
    all_imgs = []
    for ending in ['*.png', '*.tiff', '*.tif', '*.jpeg', '*.JPEG', '*.jpg', '*.bmp']:
        all_imgs.extend(glob.glob(os.path.join(trgt_dir, ending)))

    return all_imgs


def make_convergence_plot(gt_dir, img_dirs, trgt_path, animate=False, iters_info=None, plot_type=None):
    '''
    Args:
        img_dirs: dictionary with method name as key and path to the directory with the respective images as item
    '''
    if gt_dir is not None:
        gt_images = sorted(glob_all_imgs(gt_dir))

    psnrs = defaultdict(list)
    for key, path in tqdm(img_dirs.items()):
        psnrs_path = os.path.join(path, 'psnrs.npy')
        if os.path.exists(psnrs_path):
            psnrs[key] = np.load(psnrs_path).tolist()
            continue

        pred_images = sorted(glob_all_imgs(path))
        for gt_path, pred_path in tqdm(zip(gt_images, pred_images)):
            gt_img = imageio.imread(gt_path)
            pred_img = imageio.imread(pred_path)

            psnr = skimage.measure.compare_psnr(gt_img, pred_img)
            psnrs[key].append(psnr)

        np.save(psnrs_path, np.array(psnrs[key]))

    # Now make a line plot
    if gt_dir is not None:
        iterations = np.arange(len(gt_images))
    else:
        iterations = np.arange(0, iters_info['num_iters'], iters_info['step'])

    if animate:
        assert 'mp4' in trgt_path, "Filepath needs to be mp4"
        animated_line_plot(iterations, psnrs, trgt_path, plot_type=plot_type)
    else:
        fig, ax = plt.subplots()
        ax.tick_params(axis='both', which='major', direction='in', labelsize=8)
        ax.set_ylabel("PSNR")
        ax.set_xlabel("Iterations")
        ax.grid()

        ax.plot(iterations, psnrs[next(iter(psnrs))])
        fig.savefig(trgt_path, bbox='tight', bbox_inches='tight', pad_inches=0.)


def image_convergence_video():
    # Point each of the keys to the directory where you logged tensorboard summaries for this experiment.
    summary_paths = {"ReLU": "",
                     "Tanh": "",
                     "ReLU P.E.": "",
                     "RBF-ReLU": "",
                     "SIREN": ""}

    # This is the directory where all the images from the summaries will be extracted to.
    image_extraction_dir = './data/image_summaries'
    os.makedirs(image_extraction_dir, exist_ok=True)

    gt_tag_list = ['train_gt_img', 'train_gt_grad', 'train_gt_lapl']
    pred_tag_list = ['train_pred_img', 'train_pred_grad', 'train_pred_lapl']
    trgt_path = 'image_convergence.mp4'
    make_video_from_tensorboard_summaries(summary_paths, trgt_path, image_extraction_dir=image_extraction_dir,
                                          pred_tag_list=pred_tag_list, num_rows=3, num_cols=len(summary_paths)+1,
                                          gt_tag_list=gt_tag_list, overwrite=True)


def image_convergence_plot():
    animated = True

    # Point each of the keys to the directory where you logged tensorboard summaries for this experiment.
    summary_paths = {"ReLU": "",
                     "Tanh": "",
                     "ReLU P.E.": "",
                     "RBF-ReLU": "",
                     "SIREN": ""}

    # make convergence plot
    filename = 'image_psnr_convergence' + '.mp4' if animated else '.pdf'

    make_convergence_plot(None, img_dirs=summary_paths, animate=animated,
                          trgt_path=filename, iters_info={'num_iters':15001, 'step':5}, plot_type='image')


def extract_image_psnrs(summary_paths):
    for key, item in summary_paths.items():
        summary_file = os.listdir(item)[0]
        summary_file = os.path.join(item, summary_file)

        wall_times, values = extract_from_summary(summary_file, 'train_img_psnr')
        psnrs = [values for _, values in sorted(zip(wall_times, values))]

        np.save(os.path.join(item, 'psnrs.npy'), psnrs)


if __name__ == '__main__':
    image_convergence_video()
    image_convergence_plot()

