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
"""AutoAugment augmentation policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from autoaugment.policies import good_policies  # pylint: disable=unused-import


def good_policies_svhn():
    return [[('ShearX', 0.9, 4), ('Invert', 0.2, 3)], [('ShearY', 0.9, 8),
                                                       ('Invert', 0.7, 5)],
            [('Equalize', 0.6, 5),
             ('Solarize', 0.6, 6)], [('Invert', 0.9, 3), ('Equalize', 0.6, 3)],
            [('Equalize', 0.6, 1), ('Rotate', 0.9, 3)], [
                ('ShearX', 0.9, 4), ('AutoContrast', 0.8, 3)
            ], [('ShearY', 0.9, 8), ('Invert', 0.4, 5)], [
                ('ShearY', 0.9, 5), ('Solarize', 0.2, 6)
            ], [('Invert', 0.9, 6), ('AutoContrast', 0.8, 1)],
            [('Equalize', 0.6, 3),
             ('Rotate', 0.9, 3)], [('ShearX', 0.9, 4), ('Solarize', 0.3, 3)], [
                 ('ShearY', 0.8, 8), ('Invert', 0.7, 4)
             ], [('Equalize', 0.9, 5), ('TranslateY', 0.6, 6)], [
                 ('Invert', 0.9, 4), ('Equalize', 0.6, 7)
             ], [('Contrast', 0.3, 3), ('Rotate', 0.8, 4)], [
                 ('Invert', 0.8, 5), ('TranslateY', 0.0, 2)
             ], [('ShearY', 0.7, 6), ('Solarize', 0.4, 8)], [
                 ('Invert', 0.6, 4), ('Rotate', 0.8, 4)
             ], [('ShearY', 0.3, 7), ('TranslateX', 0.9, 3)], [
                 ('ShearX', 0.1, 6), ('Invert', 0.6, 5)
             ], [('Solarize', 0.7, 2), ('TranslateY', 0.6, 7)], [
                 ('ShearY', 0.8, 4), ('Invert', 0.8, 8)
             ], [('ShearX', 0.7, 9), ('TranslateY', 0.8, 3)], [
                 ('ShearY', 0.8, 5), ('AutoContrast', 0.7, 3)
             ], [('ShearX', 0.7, 2), ('Invert', 0.1, 5)]]
