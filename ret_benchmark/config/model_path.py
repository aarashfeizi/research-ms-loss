# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

# -----------------------------------------------------------------------------
# Config definition of imagenet pretrained model path
# -----------------------------------------------------------------------------


from yacs.config import CfgNode as CN

MODEL_PATH = {
    'bninception': "~/pretrained_models/bn_inception-52deb4733.pth", # still empty
    'resnet50': "~/pretrained_models/resnet50-0676ba61.pth",
}

MODEL_PATH = CN(MODEL_PATH)
