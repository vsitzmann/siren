import csv
import glob
import math
import os

import matplotlib.colors as colors
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.ndimage
import scipy.special
import skimage
import skimage.filters
import skvideo.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def grads2img(gradients):
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()

    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)


def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()


def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()


class InverseHelmholtz(Dataset):
    def __init__(self, source_coords, rec_coords, rec_val, sidelength, velocity='uniform', pretrain=False):

        super().__init__()
        torch.manual_seed(0)

        self.sidelength = sidelength
        self.mgrid = get_mgrid(self.sidelength).detach()
        self.velocity = velocity
        self.wavenumber = 20.
        self.pretrain = pretrain

        self.N_src_samples = 100  # how many times to sample around each small gaussian source
        self.sigma = 1e-4
        self.source = torch.Tensor([1.0, 1.0]).view(-1, 2)
        self.source_coords = torch.Tensor(source_coords).float()  # Nsrc, 2

        self.rec_coords = torch.Tensor(rec_coords).float()  # Nrec, 2
        self.rec = torch.zeros(self.rec_coords.shape[0], 2 * self.source_coords.shape[0])  # Nrec, 2*Nsrc
        for i in range(self.rec.shape[0]):
            self.rec[i, ::2] = torch.Tensor(rec_val.real)[i, :].float()  # * amplitude
            self.rec[i, 1::2] = torch.Tensor(rec_val.imag)[i, :].float()  # * amplitude

    def __len__(self):
        return 1

    def get_squared_slowness(self, coords):
        if self.velocity == 'square':
            squared_slowness = torch.zeros_like(coords)
            perturbation = 2.
            mask = (torch.abs(coords[..., 0]) < 0.3) & (torch.abs(coords[..., 1]) < 0.3)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))
        elif self.velocity == 'circle':
            squared_slowness = torch.zeros_like(coords)
            perturbation = 2.
            mask = (torch.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2) < 0.1)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))
        else:
            squared_slowness = torch.ones_like(coords)
            squared_slowness[..., 1] = 0.

        return squared_slowness

    def __getitem__(self, idx):

        N_src_coords = self.source_coords.shape[0]  # number of sources
        N_rec_coords = self.rec_coords.shape[0]
        coords = torch.zeros(self.sidelength ** 2, 2).uniform_(-1., 1.)

        samp_source_coords = torch.zeros(self.N_src_samples * N_src_coords, 2)
        for i in range(N_src_coords):
            samp_source_coords_r = 5e2 * self.sigma * torch.rand(self.N_src_samples, 1).sqrt()
            samp_source_coords_theta = 2 * np.pi * torch.rand(self.N_src_samples, 1)
            samp_source_coords_x = samp_source_coords_r * torch.cos(samp_source_coords_theta) \
                                   + self.source_coords[i, 0]
            samp_source_coords_y = samp_source_coords_r * torch.sin(samp_source_coords_theta) \
                                   + self.source_coords[i, 1]
            samp_source_coords[i * self.N_src_samples:(i + 1) * self.N_src_samples, :] = \
                torch.cat((samp_source_coords_x, samp_source_coords_y), dim=1)

        # Always include coordinates where source is nonzero
        coords[-self.N_src_samples * N_src_coords:, :] = samp_source_coords
        coords[:N_rec_coords, :] = self.rec_coords

        # sample each of the source gaussians separately 
        source_boundary_values = torch.zeros(coords.shape[0], 2 * N_src_coords)
        for i in range(N_src_coords):
            source_boundary_values[:, 2 * i:2 * i + 2] = self.source * \
                                                         gaussian(coords, mu=self.source_coords[i, :],
                                                                  sigma=self.sigma)[:, None]

        # truncate the source gaussians
        source_boundary_values[source_boundary_values < 1e-5] = 0.

        # add the receiver dirichlet conditions 
        rec_boundary_values = torch.zeros(coords.shape[0], self.rec.shape[1])
        rec_boundary_values[:N_rec_coords:, :] = self.rec

        # we don't know the squared slowness for the inverse problem
        squared_slowness = torch.Tensor([-1.])
        squared_slowness_grid = torch.Tensor([-1.])
        pretrain = torch.Tensor([-1.])

        if self.pretrain:
            squared_slowness = self.get_squared_slowness(coords)
            squared_slowness_grid = self.get_squared_slowness(self.mgrid)[:, 0, None]
            pretrain = torch.Tensor([1.])

        return {'coords': coords}, {'source_boundary_values': source_boundary_values,
                                    'rec_boundary_values': rec_boundary_values, 'squared_slowness': squared_slowness,
                                    'squared_slowness_grid': squared_slowness_grid, 'wavenumber': self.wavenumber,
                                    'pretrain': pretrain}


class SingleHelmholtzSource(Dataset):
    def __init__(self, sidelength, velocity='uniform', source_coords=[0., 0.]):
        super().__init__()
        torch.manual_seed(0)

        self.sidelength = sidelength
        self.mgrid = get_mgrid(self.sidelength).detach()
        self.velocity = velocity
        self.wavenumber = 20.

        self.N_src_samples = 100
        self.sigma = 1e-4
        self.source = torch.Tensor([1.0, 1.0]).view(-1, 2)
        self.source_coords = torch.tensor(source_coords).view(-1, 2)

        # For reference: this derives the closed-form solution for the inhomogenous Helmholtz equation.
        square_meshgrid = lin2img(self.mgrid[None, ...]).numpy()
        x = square_meshgrid[0, 0, ...]
        y = square_meshgrid[0, 1, ...]

        # Specify the source.
        source_np = self.source.numpy()
        hx = hy = 2 / self.sidelength
        field = np.zeros((sidelength, sidelength)).astype(np.complex64)
        for i in range(source_np.shape[0]):
            x0 = self.source_coords[i, 0].numpy()
            y0 = self.source_coords[i, 1].numpy()
            s = source_np[i, 0] + 1j * source_np[i, 1]

            hankel = scipy.special.hankel2(0, self.wavenumber * np.sqrt((x - x0) ** 2 + (y - y0) ** 2) + 1e-6)
            field += 0.25j * hankel * s * hx * hy

        field_r = torch.from_numpy(np.real(field).reshape(-1, 1))
        field_i = torch.from_numpy(np.imag(field).reshape(-1, 1))
        self.field = torch.cat((field_r, field_i), dim=1)

    def __len__(self):
        return 1

    def get_squared_slowness(self, coords):
        if self.velocity == 'square':
            squared_slowness = torch.zeros_like(coords)
            perturbation = 2.
            mask = (torch.abs(coords[..., 0]) < 0.3) & (torch.abs(coords[..., 1]) < 0.3)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))
        elif self.velocity == 'circle':
            squared_slowness = torch.zeros_like(coords)
            perturbation = 2.
            mask = (torch.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2) < 0.1)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))

        else:
            squared_slowness = torch.ones_like(coords)
            squared_slowness[..., 1] = 0.

        return squared_slowness

    def __getitem__(self, idx):
        # indicate where border values are
        coords = torch.zeros(self.sidelength ** 2, 2).uniform_(-1., 1.)
        source_coords_r = 5e2 * self.sigma * torch.rand(self.N_src_samples, 1).sqrt()
        source_coords_theta = 2 * np.pi * torch.rand(self.N_src_samples, 1)
        source_coords_x = source_coords_r * torch.cos(source_coords_theta) + self.source_coords[0, 0]
        source_coords_y = source_coords_r * torch.sin(source_coords_theta) + self.source_coords[0, 1]
        source_coords = torch.cat((source_coords_x, source_coords_y), dim=1)

        # Always include coordinates where source is nonzero
        coords[-self.N_src_samples:, :] = source_coords

        # We use the value "zero" to encode "no boundary constraint at this coordinate"
        boundary_values = self.source * gaussian(coords, mu=self.source_coords, sigma=self.sigma)[:, None]
        boundary_values[boundary_values < 1e-5] = 0.

        # specify squared slowness
        squared_slowness = self.get_squared_slowness(coords)
        squared_slowness_grid = self.get_squared_slowness(self.mgrid)[:, 0, None]

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'gt': self.field,
                                    'squared_slowness': squared_slowness,
                                    'squared_slowness_grid': squared_slowness_grid,
                                    'wavenumber': self.wavenumber}


class WaveSource(Dataset):
    def __init__(self, sidelength, velocity='uniform', source_coords=[0., 0., 0.],
                 pretrain=False):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.sidelength = sidelength
        self.mgrid = get_mgrid(self.sidelength).detach()
        self.velocity = velocity

        self.N_src_samples = 1000
        self.sigma = 5e-4
        self.source_coords = torch.tensor(source_coords).view(-1, 3)

        self.counter = 0
        self.full_count = 100e3

    def __len__(self):
        return 1

    def get_squared_slowness(self, coords):
        if self.velocity == 'square':
            squared_slowness = torch.zeros_like(coords[:, 0])
            perturbation = 2.
            mask = (torch.abs(coords[..., 0]) < 0.3) & (torch.abs(coords[..., 1]) < 0.3)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))
        elif self.velocity == 'circle':
            squared_slowness = torch.zeros_like(coords[:, 0])
            perturbation = 2.
            mask = (torch.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2) < 0.1)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))
        else:
            squared_slowness = torch.ones_like(coords[:, 0])
        return squared_slowness

    def __getitem__(self, idx):
        start_time = self.source_coords[0, 0]  # time to apply  initial conditions

        r = 5e2 * self.sigma * torch.rand(self.N_src_samples, 1).sqrt()
        phi = 2 * np.pi * torch.rand(self.N_src_samples, 1)

        # circular sampling
        source_coords_x = r * torch.cos(phi) + self.source_coords[0, 1]
        source_coords_y = r * torch.sin(phi) + self.source_coords[0, 2]
        source_coords = torch.cat((source_coords_x, source_coords_y), dim=1)

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.sidelength ** 2, 2).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.zeros(self.sidelength ** 2, 1).uniform_(start_time - 0.001, start_time + 0.001)
            coords = torch.cat((time, coords), dim=1)
            # make sure we spatially sample the source
            coords[-self.N_src_samples:, 1:] = source_coords
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is 0.75. 
            time = torch.zeros(self.sidelength ** 2, 1).uniform_(0, 0.4 * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial condition
            coords[-self.N_src_samples:, 1:] = source_coords
            coords[-2 * self.N_src_samples:, 0] = start_time

            # set up source
        normalize = 50 * gaussian(torch.zeros(1, 2), mu=torch.zeros(1, 2), sigma=self.sigma, d=2)
        boundary_values = gaussian(coords[:, 1:], mu=self.source_coords[:, 1:], sigma=self.sigma, d=2)[:, None]
        boundary_values /= normalize

        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            boundary_values = torch.where((coords[:, 0, None] == start_time), boundary_values, torch.Tensor([0]))
            dirichlet_mask = (coords[:, 0, None] == start_time)

        boundary_values[boundary_values < 1e-5] = 0.

        # specify squared slowness
        squared_slowness = self.get_squared_slowness(coords)[:, None]
        squared_slowness_grid = self.get_squared_slowness(self.mgrid)[:, None]

        self.counter += 1

        if self.pretrain and self.counter == 2000:
            self.pretrain = False
            self.counter = 0

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask,
                                    'squared_slowness': squared_slowness, 'squared_slowness_grid': squared_slowness_grid}


class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}


class Video(Dataset):
    def __init__(self, path_to_video):
        super().__init__()
        if 'npy' in path_to_video:
            self.vid = np.load(path_to_video)
        elif 'mp4' in path_to_video:
            self.vid = skvideo.io.vread(path_to_video).astype(np.single) / 255.

        self.shape = self.vid.shape[:-1]
        self.channels = self.vid.shape[-1]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.vid


class Camera(Dataset):
    def __init__(self, downsample_factor=1):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.img = Image.fromarray(skimage.data.camera())
        self.img_channels = 1

        if downsample_factor > 1:
            size = (int(512 / downsample_factor),) * 2
            self.img_downsampled = self.img.resize(size, Image.ANTIALIAS)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.downsample_factor > 1:
            return self.img_downsampled
        else:
            return self.img


class ImageFile(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.img = Image.open(filename)
        self.img_channels = len(self.img.mode)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


class CelebA(Dataset):
    def __init__(self, split, downsampled=False):
        # SIZE (178 x 218)
        super().__init__()
        assert split in ['train', 'test', 'val'], "Unknown split"

        self.root = '/media/data3/awb/CelebA/kaggle/img_align_celeba/img_align_celeba'
        self.img_channels = 3
        self.fnames = []

        with open('/media/data3/awb/CelebA/kaggle/list_eval_partition.csv', newline='') as csvfile:
            rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in rowreader:
                if split == 'train' and row[1] == '0':
                    self.fnames.append(row[0])
                elif split == 'val' and row[1] == '1':
                    self.fnames.append(row[0])
                elif split == 'test' and row[1] == '2':
                    self.fnames.append(row[0])

        self.downsampled = downsampled

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.fnames[idx])
        img = Image.open(path)
        if self.downsampled:
            width, height = img.size  # Get dimensions

            s = min(width, height)
            left = (width - s) / 2
            top = (height - s) / 2
            right = (width + s) / 2
            bottom = (height + s) / 2
            img = img.crop((left, top, right, bottom))
            img = img.resize((32, 32))

        return img


class ImplicitAudioWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.grid = np.linspace(start=-100, stop=100, num=dataset.file_length)
        self.grid = self.grid.astype(np.float32)
        self.grid = torch.Tensor(self.grid).view(-1, 1)

    def get_num_samples(self):
        return self.grid.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        rate, data = self.dataset[idx]
        scale = np.max(np.abs(data))
        data = (data / scale)
        data = torch.Tensor(data).view(-1, 1)
        return {'idx': idx, 'coords': self.grid}, {'func': data, 'rate': rate, 'scale': scale}


class AudioFile(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.rate, self.data = wavfile.read(filename)
        if len(self.data.shape) > 1 and self.data.shape[1] == 2:
            self.data = np.mean(self.data, axis=1)
        self.data = self.data.astype(np.float32)
        self.file_length = len(self.data)
        print("Rate: %d" % self.rate)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.rate, self.data


class Implicit2DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, compute_diff=None):

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength

        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.compute_diff = compute_diff
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.transform(self.dataset[idx])

        if self.compute_diff == 'gradients':
            img *= 1e1
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
        elif self.compute_diff == 'laplacian':
            img *= 1e4
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
        elif self.compute_diff == 'all':
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]

        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        in_dict = {'idx': idx, 'coords': self.mgrid}
        gt_dict = {'img': img}

        if self.compute_diff == 'gradients':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})

        elif self.compute_diff == 'laplacian':
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        elif self.compute_diff == 'all':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        return in_dict, gt_dict

    def get_item_small(self, idx):
        img = self.transform(self.dataset[idx])
        spatial_img = img.clone()
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        gt_dict = {'img': img}

        return spatial_img, img, gt_dict


class Implicit3DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, sample_fraction=1.):

        if isinstance(sidelength, int):
            sidelength = 3 * (sidelength,)

        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength, dim=3)
        data = (torch.from_numpy(self.dataset[0]) - 0.5) / 0.5
        self.data = data.view(-1, self.dataset.channels)
        self.sample_fraction = sample_fraction
        self.N_samples = int(self.sample_fraction * self.mgrid.shape[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.sample_fraction < 1.:
            coord_idx = torch.randint(0, self.data.shape[0], (self.N_samples,))
            data = self.data[coord_idx, :]
            coords = self.mgrid[coord_idx, :]
        else:
            coords = self.mgrid
            data = self.data

        in_dict = {'idx': idx, 'coords': coords}
        gt_dict = {'img': data}

        return in_dict, gt_dict


class ImageGeneralizationWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, test_sparsity=None, train_sparsity_range=(10, 200), generalization_mode=None):
        self.dataset = dataset
        self.sidelength = dataset.sidelength
        self.mgrid = dataset.mgrid
        self.test_sparsity = test_sparsity
        self.train_sparsity_range = train_sparsity_range
        self.generalization_mode = generalization_mode

    def __len__(self):
        return len(self.dataset)

    # update the sparsity of the images used in testing
    def update_test_sparsity(self, test_sparsity):
        self.test_sparsity = test_sparsity

    # generate the input dictionary based on the type of model used for generalization
    def get_generalization_in_dict(self, spatial_img, img, idx):
        # case where we use the convolutional encoder for generalization, either testing or training
        if self.generalization_mode == 'conv_cnp' or self.generalization_mode == 'conv_cnp_test':
            if self.test_sparsity == 'full':
                img_sparse = spatial_img
            elif self.test_sparsity == 'half':
                img_sparse = spatial_img
                img_sparse[:, 16:, :] = 0.
            else:
                if self.generalization_mode == 'conv_cnp_test':
                    num_context = int(self.test_sparsity)
                else:
                    num_context = int(
                        torch.empty(1).uniform_(self.train_sparsity_range[0], self.train_sparsity_range[1]).item())
                mask = spatial_img.new_empty(
                    1, spatial_img.size(1), spatial_img.size(2)).bernoulli_(p=num_context / np.prod(self.sidelength))
                img_sparse = mask * spatial_img
            in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sparse': img_sparse}
        # case where we use the set encoder for generalization, either testing or training
        elif self.generalization_mode == 'cnp' or self.generalization_mode == 'cnp_test':
            if self.test_sparsity == 'full':
                in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img, 'coords_sub': self.mgrid}
            elif self.test_sparsity == 'half':
                in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img[:512, :], 'coords_sub': self.mgrid[:512, :]}
            else:
                if self.generalization_mode == 'cnp_test':
                    subsamples = int(self.test_sparsity)
                    rand_idcs = np.random.choice(img.shape[0], size=subsamples, replace=False)
                    img_sparse = img[rand_idcs, :]
                    coords_sub = self.mgrid[rand_idcs, :]
                    in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img_sparse, 'coords_sub': coords_sub}
                else:
                    subsamples = np.random.randint(self.train_sparsity_range[0], self.train_sparsity_range[1])
                    rand_idcs = np.random.choice(img.shape[0], size=self.train_sparsity_range[1], replace=False)
                    img_sparse = img[rand_idcs, :]
                    coords_sub = self.mgrid[rand_idcs, :]

                    rand_idcs_2 = np.random.choice(img_sparse.shape[0], size=subsamples, replace=False)
                    ctxt_mask = torch.zeros(img_sparse.shape[0], 1)
                    ctxt_mask[rand_idcs_2, 0] = 1.

                    in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img_sparse, 'coords_sub': coords_sub,
                               'ctxt_mask': ctxt_mask}
        else:
            in_dict = {'idx': idx, 'coords': self.mgrid}

        return in_dict

    def __getitem__(self, idx):
        spatial_img, img, gt_dict = self.dataset.get_item_small(idx)
        in_dict = self.get_generalization_in_dict(spatial_img, img, idx)
        return in_dict, gt_dict



# in_folder: where to find the data (train, val, test)
# color: whether to load in color
# idx_to_sample: which index to sample (usefull if wanting to fit a single image)
# preload: whether or not to preload in memory
class BSD500ImageDataset(Dataset):
    def __init__(self,
                 in_folder='data/BSD500/train',
                 is_color=False,
                 size=[321, 321],  # BSD is 481x321
                 preload=True,
                 idx_to_sample=[]):
        self.in_folder = in_folder
        self.size = size
        self.idx_to_sample = idx_to_sample
        self.is_color = is_color
        self.preload = preload
        if (self.is_color):
            self.img_channels = 3
        else:
            self.img_channels = 1

        self.img_filenames = []
        self.img_preloaded = []
        for idx, filename in enumerate(sorted(glob.glob(self.in_folder + '/*.jpg'))):
            # print(f'Gathering img #{idx}')
            self.img_filenames.append(filename)

            if (self.preload):
                # print(f'... preloaded')
                img = self.load_image(filename)
                self.img_preloaded.append(img)

        if (self.preload):
            assert (len(self.img_preloaded) == len(self.img_filenames))

    def load_image(self, filename):
        img = Image.open(filename, 'r')
        if not self.is_color:
            img = img.convert("L")
        img = img.crop((0, 0, self.size[0], self.size[1]))

        return img

    def __len__(self):
        # If we have specified specific idx to sample from, we only
        # return from those, otherwise, we want to return from the whole
        # dataset
        if (len(self.idx_to_sample) != 0):
            return len(self.idx_to_sample)
        else:
            return len(self.img_filenames)

    def __getitem__(self, item):
        # if we have specified specific idx to sample from, convert
        # back the item number to the actual item we can sample from,
        # otherwise you can directly use the item since the length
        # corresponds to all the files in the directory.
        if (len(self.idx_to_sample) != 0):
            idx = self.idx_to_sample[item]
        else:
            idx = item

        if (self.preload):
            img = self.img_preloaded[idx]
        else:
            img = self.load_image(self.img_filenames[idx])

        return img


class CompositeGradients(Dataset):
    def __init__(self, img_filepath1, img_filepath2,
                 sidelength=None,
                 is_color=False):
        super().__init__()

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)

        self.is_color = is_color
        if (self.is_color):
            self.channels = 3
        else:
            self.channels = 1

        self.img1 = Image.open(img_filepath1)
        self.img2 = Image.open(img_filepath2)

        if not self.is_color:
            self.img1 = self.img1.convert("L")
            self.img2 = self.img2.convert("L")
        else:
            self.img1 = self.img1.convert("RGB")
            self.img2 = self.img2.convert("RGB")

        self.transform = Compose([
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.mgrid = get_mgrid(sidelength)

        self.img1 = self.transform(self.img1)
        self.img2 = self.transform(self.img2)

        paddedImg = .85 * torch.ones_like(self.img1)
        paddedImg[:, 512 - 340:512, :] = self.img2
        self.img2 = paddedImg

        self.grads1 = self.compute_gradients(self.img1)
        self.grads2 = self.compute_gradients(self.img2)

        self.comp_grads = (.5 * self.grads1 + .5 * self.grads2)

        self.img1 = self.img1.permute(1, 2, 0).view(-1, self.channels)
        self.img2 = self.img2.permute(1, 2, 0).view(-1, self.channels)

    def compute_gradients(self, img):
        if not self.is_color:
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
        else:
            gradx = np.moveaxis(scipy.ndimage.sobel(img.numpy(), axis=1), 0, -1)
            grady = np.moveaxis(scipy.ndimage.sobel(img.numpy(), axis=2), 0, -1)

        grads = torch.cat((torch.from_numpy(gradx).reshape(-1, self.channels),
                           torch.from_numpy(grady).reshape(-1, self.channels)),
                          dim=-1)
        return grads

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        in_dict = {'idx': idx, 'coords': self.mgrid}
        gt_dict = {'img1': self.img1,
                   'img2': self.img2,
                   'grads1': self.grads1,
                   'grads2': self.grads2,
                   'gradients': self.comp_grads}

        return in_dict, gt_dict
