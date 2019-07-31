# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transforms used in the PBA Augmentation Policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import collections
import inspect
import random

import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image  # pylint:disable=g-multiple-import

from pba.augmentation_transforms import random_flip, zero_pad_and_crop, cutout_numpy  # pylint: disable=unused-import
from pba.augmentation_transforms import TransformFunction
from pba.augmentation_transforms import ALL_TRANSFORMS, NAME_TO_TRANSFORM, TRANSFORM_NAMES  # pylint: disable=unused-import
from pba.augmentation_transforms import pil_wrap, pil_unwrap  # pylint: disable=unused-import
from pba.augmentation_transforms import MEANS, STDS, PARAMETER_MAX  # pylint: disable=unused-import
from pba.augmentation_transforms import _rotate_impl, _posterize_impl, _shear_x_impl, _shear_y_impl, _translate_x_impl, _translate_y_impl, _crop_impl, _solarize_impl, _cutout_pil_impl, _enhancer_impl


def apply_policy(policy, img, aug_policy, dset, image_size, verbose=False):
    """Apply the `policy` to the numpy `img`.

  Args:
    policy: A list of tuples with the form (name, probability, level) where
      `name` is the name of the augmentation operation to apply, `probability`
      is the probability of applying the operation and `level` is what strength
      the operation to apply.
    img: Numpy image that will have `policy` applied to it.
    aug_policy: Augmentation policy to use.
    dset: Dataset, one of the keys of MEANS or STDS.
    image_size: Width and height of image.
    verbose: Whether to print applied augmentations.

  Returns:
    The result of applying `policy` to `img`.
  """
    if aug_policy == 'cifar10':
        count = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.5, 0.0])
    else:
        raise ValueError('Unknown aug policy.')
    if count != 0:
        pil_img = pil_wrap(img, dset)
        policy = copy.copy(policy)
        random.shuffle(policy)
        for xform in policy:
            assert len(xform) == 3
            name, probability, level = xform
            assert 0. <= probability <= 1.
            assert 0 <= level <= PARAMETER_MAX
            xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(
                probability, level, image_size)
            pil_img, res = xform_fn(pil_img)
            if verbose and res:
                print("Op: {}, Magnitude: {}, Prob: {}".format(name, level, probability))
            count -= res
            assert count >= 0
            if count == 0:
                break
        return pil_unwrap(pil_img, dset, image_size)
    else:
        return img


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def pil_transformer(self, probability, level, image_size):
        """Builds augmentation function which returns resulting image and whether augmentation was applied."""

        def return_function(im):
            res = False
            if random.random() < probability:
                if 'image_size' in inspect.getargspec(self.xform).args:
                    im = self.xform(im, level, image_size)
                else:
                    im = self.xform(im, level)
                res = True
            return im, res

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)

    def str(self):
        return self.name


################## Transform Functions ##################
identity = TransformT('identity', lambda pil_img, level: pil_img)
flip_lr = TransformT(
    'FlipLR', lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
flip_ud = TransformT(
    'FlipUD', lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
# pylint:disable=g-long-lambda
auto_contrast = TransformT(
    'AutoContrast',
    lambda pil_img, level: ImageOps.autocontrast(pil_img.convert('RGB')).convert('RGBA')
)
equalize = TransformT(
    'Equalize',
    lambda pil_img, level: ImageOps.equalize(pil_img.convert('RGB')).convert('RGBA')
)
invert = TransformT(
    'Invert',
    lambda pil_img, level: ImageOps.invert(pil_img.convert('RGB')).convert('RGBA')
)
# pylint:enable=g-long-lambda
blur = TransformT('Blur',
                  lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))
smooth = TransformT('Smooth',
                    lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))
rotate = TransformT('Rotate', _rotate_impl)
posterize = TransformT('Posterize', _posterize_impl)
shear_x = TransformT('ShearX', _shear_x_impl)
shear_y = TransformT('ShearY', _shear_y_impl)
translate_x = TransformT('TranslateX', _translate_x_impl)
translate_y = TransformT('TranslateY', _translate_y_impl)
crop_bilinear = TransformT('CropBilinear', _crop_impl)
solarize = TransformT('Solarize', _solarize_impl)
cutout = TransformT('Cutout', _cutout_pil_impl)
color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))

HP_TRANSFORMS = [
    rotate,
    translate_x,
    translate_y,
    brightness,
    color,
    invert,
    sharpness,
    posterize,
    shear_x,
    solarize,
    shear_y,
    equalize,
    auto_contrast,
    cutout,
    contrast
]

NAME_TO_TRANSFORM = collections.OrderedDict((t.name, t) for t in HP_TRANSFORMS)
HP_TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()
NUM_HP_TRANSFORM = len(HP_TRANSFORM_NAMES)
