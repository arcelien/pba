"""Builds the Wide-ResNet Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import autoaugment.custom_ops as ops
from autoaugment.wrn import residual_block, _res_add


def build_wrn_model(images, num_classes, wrn_size, depth=28):
    """Builds the WRN model.

  Build the Wide ResNet model from https://arxiv.org/abs/1605.07146.

  Args:
    images: Tensor of images that will be fed into the Wide ResNet Model.
    num_classes: Number of classed that the model needs to predict.
    wrn_size: Parameter that scales the number of filters in the Wide ResNet
      model.
    depth: Number of layers of model.

  Returns:
    The logits of the Wide ResNet model.

  28-10 is wrn_size of 160 -> k = 10, and depth=28 -> blocks = 4
  40-2 is wrn_size of 32 -> k = 2, and depth=40 -> blocks = 6
  """
    assert (depth - 4) % 6 == 0
    kernel_size = wrn_size
    filter_size = 3
    num_blocks_per_resnet = (depth - 4) // 6  # 4
    filters = [
        min(kernel_size, 16), kernel_size, kernel_size * 2, kernel_size * 4
    ]
    strides = [1, 2, 2]  # stride for each resblock

    # Run the first conv
    with tf.variable_scope('init'):
        x = images
        output_filters = filters[0]
        x = ops.conv2d(x, output_filters, filter_size, scope='init_conv')

    first_x = x  # Res from the beginning
    orig_x = x  # Res from previous block

    for block_num in range(1, 4):
        with tf.variable_scope('unit_{}_0'.format(block_num)):
            activate_before_residual = True if block_num == 1 else False
            x = residual_block(
                x,
                filters[block_num - 1],
                filters[block_num],
                strides[block_num - 1],
                activate_before_residual=activate_before_residual)
        for i in range(1, num_blocks_per_resnet):
            with tf.variable_scope('unit_{}_{}'.format(block_num, i)):
                x = residual_block(
                    x,
                    filters[block_num],
                    filters[block_num],
                    1,
                    activate_before_residual=False)
        x, orig_x = _res_add(filters[block_num - 1], filters[block_num],
                             strides[block_num - 1], x, orig_x)
    final_stride_val = np.prod(strides)
    x, _ = _res_add(filters[0], filters[3], final_stride_val, x, first_x)
    with tf.variable_scope('unit_last'):
        x = ops.batch_norm(x, scope='final_bn')
        x = tf.nn.relu(x)
        x = ops.global_avg_pool(x)
        logits = ops.fc(x, num_classes)
    return logits
