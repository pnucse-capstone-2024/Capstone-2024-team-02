import numpy as np
from scipy.ndimage import binary_fill_holes
from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize
import torchvision
import torch
import os


def get_extension(filename, extensions):
    for extension in extensions:
        if filename.endswith(extension):
            return extension


def crop_image(image):
    nonzero_mask = binary_fill_holes(image != 0)
    mask_voxel_coords = np.stack(np.where(nonzero_mask))
    minidx = np.min(mask_voxel_coords, axis=1)
    maxidx = np.max(mask_voxel_coords, axis=1) + 1
    resizer = tuple([slice(*i) for i in zip(minidx, maxidx)])
    return resizer


def preprocess_image(image, crop, spacing, spacing_target):
    image = image[crop].transpose(2, 1, 0)
    spacing_target[0] = spacing[0]
    new_shape = np.round(spacing / spacing_target * image.shape).astype(int)
    image = np.stack(
        [
            resize(slice, new_shape[1:], 3, cval=0, mode="edge", anti_aliasing=False)
            for slice in image
        ]
    )
    image -= image.mean()
    image /= image.std() + 1e-8
    return image


def postprocess_image(image, info, phase, current_spacing):
    postprocessed = np.zeros(info["shape_{}".format(phase)])
    crop = info["crop_{}".format(phase)]
    original_shape = postprocessed[crop].shape
    original_spacing = info["spacing"]
    tmp_shape = tuple(
        np.round(
            original_spacing[1:] / current_spacing[1:] * original_shape[:2]
        ).astype(int)[::-1]
    )
    image = np.argmax(image, axis=1)
    image = np.array(
        [
            torchvision.transforms.Compose(
                [AddPadding(tmp_shape), CenterCrop(tmp_shape), OneHot()]
            )({"gt": slice})["gt"]
            for slice in image
        ]
    )
    image = resize_segmentation(
        image.transpose(1, 3, 2, 0), image.shape[1:2] + original_shape, order=1
    )
    image = np.argmax(image, axis=0)
    postprocessed[crop] = image
    return postprocessed


class AddPadding:
    def __init__(self, output_size):
        self.output_size = output_size

    def resize_image_by_padding(self, image, new_shape, pad_value=0):
        shape = tuple(list(image.shape))
        new_shape = tuple(
            np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0)
        )
        res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
        start = np.array(new_shape) / 2.0 - np.array(shape) / 2.0
        res[
            int(start[0]) : int(start[0]) + int(shape[0]),
            int(start[1]) : int(start[1]) + int(shape[1]),
        ] = image
        return res

    def __call__(self, sample):

        sample = {
            k: (
                self.resize_image_by_padding(v, new_shape=self.output_size)
                if isinstance(v, np.ndarray)
                else v
            )
            for k, v in sample.items()
        }

        return sample


class OneHot:
    def one_hot(self, seg, num_classes=4):
        return np.eye(num_classes)[seg.astype(int)].transpose(2, 0, 1)

    def __call__(self, sample):
        if "gt" in sample.keys():
            if isinstance(sample["gt"], list):
                sample["gt"] = [self.one_hot(y) for y in sample["gt"]]
            else:
                sample["gt"] = self.one_hot(sample["gt"])
        return sample


class CenterCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def center_crop_2D_image(self, img, center_crop):
        if all(np.array(img.shape) <= center_crop):
            return img
        center = np.array(img.shape) / 2.0
        return img[
            int(center[0] - center_crop[0] / 2.0) : int(
                center[0] + center_crop[0] / 2.0
            ),
            int(center[1] - center_crop[1] / 2.0) : int(
                center[1] + center_crop[1] / 2.0
            ),
        ]

    def __call__(self, sample):
        sample = {
            k: (
                self.center_crop_2D_image(v, center_crop=self.output_size)
                if isinstance(v, np.ndarray)
                else v
            )
            for k, v in sample.items()
        }
        return sample


class ToTensor:
    def __call__(self, sample):
        sample["data"] = torch.from_numpy(sample["data"][None, :, :]).float()
        return sample


transform = torchvision.transforms.Compose(
    [AddPadding((256, 256)), CenterCrop((256, 256)), ToTensor()]
)


class Patient(torch.utils.data.Dataset):
    def __init__(self, patient_info, patient_image, patient_id, transform=None):
        self.data = patient_image
        self.transform = transform
        self.info = patient_info
        self.id = patient_id

    def __len__(self):
        return self.info["shape_ED"][2] + self.info["shape_ES"][2]

    def __getitem__(self, slice_id):
        is_es = slice_id >= len(self) // 2
        slice_id = slice_id - len(self) // 2 if is_es else slice_id
        sample = {
            "data": (
                self.data["ED"][slice_id] if not is_es else self.data["ES"][slice_id]
            )
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
