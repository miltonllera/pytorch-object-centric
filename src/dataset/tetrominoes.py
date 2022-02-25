import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class Tetrominoes(Dataset):
    """
    Dataset of Tetris-like objects, several of which can
    appear on screen at a time.

    TODO: Add description of dimensions here
    """
    files = {"full": "data/raw/tetrominoes/all_tetrominoes.npz"}

    factors = {"shape", "color", "x", "y", "visibility"}

    categorical = np.array([1, 1, 0, 0, 1])

    img_size = (3, 35, 35)

    unique_values = {'shape': np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.,
                                        9., 10., 11., 12., 13., 14., 15.,
                                        16., 17., 18., 19.]),
                     'color': np.array([0., 1., 2., 3., 4., 5., 6.]),
                     'x'    : np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.,
                                        9., 10., 11., 12., 13., 14., 15., 16.,
                                        17., 18., 19., 20, 21, 22, 23, 24, 25,
                                        26, 27, 28, 29]),
                     'y'    : np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.,
                                        9., 10., 11., 12., 13., 14., 15., 16.,
                                        17., 18., 19., 20, 21, 22, 23, 24, 25,
                                        26, 27, 28, 29]),
                     'visibility': np.array([1.])}

    def __init__(self, images, masks, factor_values, target_transform=None):
        self.images = images
        self.masks = masks
        self.factor_values = factor_values

        img_transform = [transforms.ToTensor(),
                         transforms.ConvertImageDtype(torch.float32),
                         transforms.Lambda(lambda x: (x - 0.5) * 2)]

        self.transform = transforms.Compose(img_transform)
        self.target_transform = target_transform

    def __getitem__(self, idx):
        return (self.transform(self.images[idx]),
                self.factor_values[idx],
                self.masks[idx])

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return 'Tetrominoes'


def load_raw(path):
    data_zip = np.load(path, allow_pickle=True)

    # Extract colors
    images = data_zip['image']
    masks = data_zip['mask']
    x = data_zip['x']
    y = data_zip['y']
    shape = data_zip['shape']
    color = data_zip['color']
    visibility = data_zip['visibility']

    # color values from binary to integer
    integer_color = np.zeros_like(color)
    for i in range(3):
        integer_color[:, :, i] = color[:, :, i] * 2 ** (2 - i)
    color = np.add.reduce(integer_color, axis=2)

    factor_values = np.stack([shape, color, x, y, visibility], axis=-1)

    return images, factor_values, masks


def load(data_filters=(None, None), train=True, variant='full', path=None):
    if path is None:
        try:
            path = Tetrominoes.files[variant]
        except KeyError:
            raise ValueError('Variant {} does not exist'.format(variant))

    images, factor_values, masks = load_raw(path)

    return Tetrominoes(images, masks, factor_values)
