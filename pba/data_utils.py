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
"""Data utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
try:
    import cPickle as pickle
except:
    import pickle
import os
import random
import numpy as np
import tensorflow as tf
import torchvision

from autoaugment.data_utils import unpickle
import pba.policies as found_policies
from pba.utils import parse_log_schedule
import pba.augmentation_transforms_hp as augmentation_transforms_pba
import pba.augmentation_transforms as augmentation_transforms_autoaug


# pylint:disable=logging-format-interpolation


def parse_policy(policy_emb, augmentation_transforms):
    policy = []
    num_xform = augmentation_transforms.NUM_HP_TRANSFORM
    xform_names = augmentation_transforms.HP_TRANSFORM_NAMES
    assert len(policy_emb
               ) == 2 * num_xform, 'policy was: {}, supposed to be: {}'.format(
                   len(policy_emb), 2 * num_xform)
    for i, xform in enumerate(xform_names):
        policy.append((xform, policy_emb[2 * i] / 10., policy_emb[2 * i + 1]))
    return policy


def shuffle_data(data, labels):
    """Shuffle data using numpy."""
    np.random.seed(0)
    perm = np.arange(len(data))
    np.random.shuffle(perm)
    data = data[perm]
    labels = labels[perm]
    return data, labels


class DataSet(object):
    """Dataset object that produces augmented training and eval data."""

    def __init__(self, hparams):
        self.hparams = hparams
        self.epochs = 0
        self.curr_train_index = 0

        self.parse_policy(hparams)
        self.load_data(hparams)

        # Apply normalization
        self.train_images = self.train_images.transpose(0, 2, 3, 1) / 255.0
        self.val_images = self.val_images.transpose(0, 2, 3, 1) / 255.0
        self.test_images = self.test_images.transpose(0, 2, 3, 1) / 255.0
        if not hparams.recompute_dset_stats:
            mean = self.augmentation_transforms.MEANS[hparams.dataset + '_' +
                                                      str(hparams.train_size)]
            std = self.augmentation_transforms.STDS[hparams.dataset + '_' +
                                                    str(hparams.train_size)]
        else:
            mean = self.train_images.mean(axis=(0, 1, 2))
            std = self.train_images.std(axis=(0, 1, 2))
            self.augmentation_transforms.MEANS[hparams.dataset + '_' + str(hparams.train_size)] = mean
            self.augmentation_transforms.STDS[hparams.dataset + '_' + str(hparams.train_size)] = std
        tf.logging.info('mean:{}    std: {}'.format(mean, std))

        self.train_images = (self.train_images - mean) / std
        self.val_images = (self.val_images - mean) / std
        self.test_images = (self.test_images - mean) / std

        assert len(self.test_images) == len(self.test_labels)
        assert len(self.train_images) == len(self.train_labels)
        assert len(self.val_images) == len(self.val_labels)
        tf.logging.info('train dataset size: {}, test: {}, val: {}'.format(
            len(self.train_images), len(self.test_images), len(self.val_images)))

    def parse_policy(self, hparams):
        """Parses policy schedule from input, which can be a list, list of lists, text file, or pickled list.

        If list is not nested, then uses the same policy for all epochs.

        Args:
        hparams: tf.hparams object.
        """
        # Parse policy
        if hparams.use_hp_policy:
            self.augmentation_transforms = augmentation_transforms_pba

            if isinstance(hparams.hp_policy,
                          str) and hparams.hp_policy.endswith('.txt'):
                if hparams.num_epochs % hparams.hp_policy_epochs != 0:
                    tf.logging.warning(
                        "Schedule length (%s) doesn't divide evenly into epochs (%s), interpolating.",
                        hparams.num_epochs, hparams.hp_policy_epochs)
                tf.logging.info(
                    'schedule policy trained on {} epochs, parsing from: {}, multiplier: {}'
                    .format(
                        hparams.hp_policy_epochs, hparams.hp_policy,
                        float(hparams.num_epochs) / hparams.hp_policy_epochs))
                raw_policy = parse_log_schedule(
                    hparams.hp_policy,
                    epochs=hparams.hp_policy_epochs,
                    multiplier=float(hparams.num_epochs) /
                    hparams.hp_policy_epochs)
            elif isinstance(hparams.hp_policy,
                            str) and hparams.hp_policy.endswith('.p'):
                assert hparams.num_epochs % hparams.hp_policy_epochs == 0
                tf.logging.info('custom .p file, policy number: {}'.format(
                    hparams.schedule_num))
                with open(hparams.hp_policy, 'rb') as f:
                    policy = pickle.load(f)[hparams.schedule_num]
                raw_policy = []
                for num_iters, pol in policy:
                    for _ in range(num_iters * hparams.num_epochs //
                                   hparams.hp_policy_epochs):
                        raw_policy.append(pol)
            else:
                raw_policy = hparams.hp_policy

            if isinstance(raw_policy[0], list):
                self.policy = []
                split = len(raw_policy[0]) // 2
                for pol in raw_policy:
                    cur_pol = parse_policy(pol[:split],
                                           self.augmentation_transforms)
                    cur_pol.extend(
                        parse_policy(pol[split:],
                                     self.augmentation_transforms))
                    self.policy.append(cur_pol)
                tf.logging.info('using HP policy schedule, last: {}'.format(
                    self.policy[-1]))
            elif isinstance(raw_policy, list):
                split = len(raw_policy) // 2
                self.policy = parse_policy(raw_policy[:split],
                                           self.augmentation_transforms)
                self.policy.extend(
                    parse_policy(raw_policy[split:],
                                 self.augmentation_transforms))
                tf.logging.info('using HP Policy, policy: {}'.format(
                    self.policy))

        else:
            self.augmentation_transforms = augmentation_transforms_autoaug
            tf.logging.info('using ENAS Policy or no augmentaton policy')
            if 'svhn' in hparams.dataset:
                self.good_policies = found_policies.good_policies_svhn()
            else:
                assert 'cifar' in hparams.dataset
                self.good_policies = found_policies.good_policies()

    def reset_policy(self, new_hparams):
        self.hparams = new_hparams
        self.parse_policy(new_hparams)
        tf.logging.info('reset aug policy')
        return

    def load_cifar(self, hparams):
        train_labels = []
        test_labels = []
        num_data_batches_to_load = 5
        total_batches_to_load = num_data_batches_to_load
        train_batches_to_load = total_batches_to_load
        assert hparams.train_size + hparams.validation_size <= 50000
        # Determine how many images we have loaded
        train_dataset_size = 10000 * num_data_batches_to_load

        if hparams.dataset == 'cifar10':
            train_data = np.empty(
                (total_batches_to_load, 10000, 3072), dtype=np.uint8)
            datafiles = [
                'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
                'data_batch_5'
            ]
            datafiles = datafiles[:train_batches_to_load]
            test_data = np.empty((1, 10000, 3072), dtype=np.uint8)
            datafiles.append('test_batch')
            num_classes = 10
        elif hparams.dataset == 'cifar100':
            assert num_data_batches_to_load == 5
            train_data = np.empty((1, 50000, 3072), dtype=np.uint8)
            datafiles = ['train']
            test_data = np.empty((1, 10000, 3072), dtype=np.uint8)
            datafiles.append('test')
            num_classes = 100

        for file_num, f in enumerate(datafiles):
            d = unpickle(os.path.join(hparams.data_path, f))
            if f == 'test' or f == 'test_batch':
                test_data[0] = copy.deepcopy(d['data'])
            else:
                train_data[file_num] = copy.deepcopy(d['data'])
            if hparams.dataset == 'cifar10':
                labels = np.array(d['labels'])
            else:
                labels = np.array(d['fine_labels'])
            nsamples = len(labels)
            for idx in range(nsamples):
                if f == 'test' or f == 'test_batch':
                    test_labels.append(labels[idx])
                else:
                    train_labels.append(labels[idx])
        train_data = train_data.reshape(train_dataset_size, 3072)
        test_data = test_data.reshape(10000, 3072)
        train_data = train_data.reshape(-1, 3, 32, 32)
        test_data = test_data.reshape(-1, 3, 32, 32)
        train_labels = np.array(train_labels, dtype=np.int32)
        test_labels = np.array(test_labels, dtype=np.int32)

        self.test_images, self.test_labels = test_data, test_labels
        train_data, train_labels = shuffle_data(train_data, train_labels)
        train_size, val_size = hparams.train_size, hparams.validation_size
        assert 50000 >= train_size + val_size
        self.train_images = train_data[:train_size]
        self.train_labels = train_labels[:train_size]
        self.val_images = train_data[train_size:train_size + val_size]
        self.val_labels = train_labels[train_size:train_size + val_size]
        self.num_classes = num_classes

    def load_svhn(self, hparams):
        train_labels = []
        test_labels = []
        if hparams.dataset == 'svhn':
            assert hparams.train_size == 1000
            assert hparams.train_size + hparams.validation_size <= 73257
            train_loader = torchvision.datasets.SVHN(
                root=hparams.data_path, split='train', download=True)
            test_loader = torchvision.datasets.SVHN(
                root=hparams.data_path, split='test', download=True)
            num_classes = 10
            train_data = train_loader.data
            test_data = test_loader.data
            train_labels = train_loader.labels
            test_labels = test_loader.labels
        elif hparams.dataset == 'svhn-full':
            assert hparams.train_size == 73257 + 531131
            assert hparams.validation_size == 0
            train_loader = torchvision.datasets.SVHN(
                root=hparams.data_path, split='train', download=True)
            test_loader = torchvision.datasets.SVHN(
                root=hparams.data_path, split='test', download=True)
            extra_loader = torchvision.datasets.SVHN(
                root=hparams.data_path, split='extra', download=True)
            num_classes = 10
            train_data = np.concatenate(
                [train_loader.data, extra_loader.data], axis=0)
            test_data = test_loader.data
            train_labels = np.concatenate(
                [train_loader.labels, extra_loader.labels], axis=0)
            test_labels = test_loader.labels
        else:
            raise ValueError(hparams.dataset)

        self.test_images, self.test_labels = test_data, test_labels
        train_data, train_labels = shuffle_data(train_data, train_labels)
        train_size, val_size = hparams.train_size, hparams.validation_size
        if hparams.dataset == 'svhn-full':
            assert train_size + val_size <= 604388
        else:
            assert train_size + val_size <= 73257
        self.train_images = train_data[:train_size]
        self.train_labels = train_labels[:train_size]
        self.val_images = train_data[-val_size:]
        self.val_labels = train_labels[-val_size:]
        self.num_classes = num_classes

    def load_test(self, hparams):
        """Load random data and labels."""
        test_size = 200
        self.num_classes = 200
        self.train_images = np.random.random((hparams.train_size, 3, 224, 224)) * 255
        self.val_images = np.random.random((hparams.validation_size, 3, 224, 224)) * 255
        self.test_images = np.random.random((test_size, 3, 224, 224)) * 255
        self.train_labels = np.random.randint(0, self.num_classes, (hparams.train_size))
        self.val_labels = np.random.randint(0, self.num_classes, (hparams.validation_size))
        self.test_labels = np.random.randint(0, self.num_classes, (test_size))

    def load_data(self, hparams):
        """Load raw data from specified dataset.

        Assumes data is in NCHW format.

        Populates:
            self.train_images: Training image data.
            self.train_labels: Training ground truth labels.
            self.val_images: Validation/holdout image data.
            self.val_labels: Validation/holdout ground truth labels.
            self.test_images: Testing image data.
            self.test_labels: Testing ground truth labels.
            self.num_classes: Number of classes.
            self.num_train: Number of training examples.
            self.image_size: Width/height of image.

        Args:
            hparams: tf.hparams object.
        """
        if hparams.dataset == 'cifar10' or hparams.dataset == 'cifar100':
            self.load_cifar(hparams)
        elif hparams.dataset == 'svhn' or hparams.dataset == 'svhn-full':
            self.load_svhn(hparams)
        elif hparams.dataset == 'test':
            self.load_test(hparams)
        else:
            raise ValueError('unimplemented')

        self.num_train = self.train_images.shape[0]
        self.image_size = self.train_images.shape[2]
        self.train_labels = np.eye(self.num_classes)[np.array(
            self.train_labels, dtype=np.int32)]
        self.val_labels = np.eye(self.num_classes)[np.array(
            self.val_labels, dtype=np.int32)]
        self.test_labels = np.eye(self.num_classes)[np.array(
            self.test_labels, dtype=np.int32)]
        assert len(self.train_images) == len(self.train_labels)
        assert len(self.val_images) == len(self.val_labels)
        assert len(self.test_images) == len(self.test_labels)
        assert self.train_images.shape[2] == self.train_images.shape[3]

    def next_batch(self, iteration=None):
        """Return the next minibatch of augmented data."""
        next_train_index = self.curr_train_index + self.hparams.batch_size
        if next_train_index > self.num_train:
            # Increase epoch number
            epoch = self.epochs + 1
            self.reset()
            self.epochs = epoch
        batched_data = (
            self.train_images[self.curr_train_index:self.curr_train_index +
                              self.hparams.batch_size],
            self.train_labels[self.curr_train_index:self.curr_train_index +
                              self.hparams.batch_size])
        final_imgs = []

        dset = self.hparams.dataset + '_' + str(self.hparams.train_size)
        images, labels = batched_data
        for data in images:
            if not self.hparams.no_aug:
                if not self.hparams.use_hp_policy:
                    # apply autoaugment policy
                    epoch_policy = self.good_policies[np.random.choice(
                        len(self.good_policies))]
                    final_img = self.augmentation_transforms.apply_policy(
                        epoch_policy,
                        data,
                        dset=dset,
                        image_size=self.image_size)
                else:
                    # apply PBA policy)
                    if isinstance(self.policy[0], list):
                        # single policy
                        if self.hparams.flatten:
                            final_img = self.augmentation_transforms.apply_policy(
                                self.policy[random.randint(
                                    0,
                                    len(self.policy) - 1)],
                                data,
                                self.hparams.aug_policy,
                                dset,
                                image_size=self.image_size)
                        else:
                            final_img = self.augmentation_transforms.apply_policy(
                                self.policy[iteration],
                                data,
                                self.hparams.aug_policy,
                                dset,
                                image_size=self.image_size)
                    elif isinstance(self.policy, list):
                        # policy schedule
                        final_img = self.augmentation_transforms.apply_policy(
                            self.policy,
                            data,
                            self.hparams.aug_policy,
                            dset,
                            image_size=self.image_size)
                    else:
                        raise ValueError('Unknown policy.')
            else:
                final_img = data
            if self.hparams.dataset == 'cifar10' or self.hparams.dataset == 'cifar100':
                final_img = self.augmentation_transforms.random_flip(
                    self.augmentation_transforms.zero_pad_and_crop(
                        final_img, 4))
            elif 'svhn' in self.hparams.dataset:
                pass
            else:
                tf.logging.log_first_n(tf.logging.WARN, 'Using default random flip and crop.', 1)
                final_img = self.augmentation_transforms.random_flip(
                    self.augmentation_transforms.zero_pad_and_crop(
                        final_img, 4))
            # Apply cutout
            if not self.hparams.no_cutout:
                if 'cifar10' == self.hparams.dataset:
                    final_img = self.augmentation_transforms.cutout_numpy(
                        final_img, size=16)
                elif 'cifar100' == self.hparams.dataset:
                    final_img = self.augmentation_transforms.cutout_numpy(
                        final_img, size=16)
                elif 'svhn' in self.hparams.dataset:
                    final_img = self.augmentation_transforms.cutout_numpy(
                        final_img, size=20)
                else:
                    tf.logging.log_first_n(tf.logging.WARN, 'Using default cutout size (16x16).', 1)
                    final_img = self.augmentation_transforms.cutout_numpy(
                        final_img)
            final_imgs.append(final_img)
        batched_data = (np.array(final_imgs, np.float32), labels)
        self.curr_train_index += self.hparams.batch_size
        return batched_data

    def reset(self):
        """Reset training data and index into the training data."""
        self.epochs = 0
        # Shuffle the training data
        perm = np.arange(self.num_train)
        np.random.shuffle(perm)
        assert self.num_train == self.train_images.shape[
            0], 'Error incorrect shuffling mask'
        self.train_images = self.train_images[perm]
        self.train_labels = self.train_labels[perm]
        self.curr_train_index = 0
