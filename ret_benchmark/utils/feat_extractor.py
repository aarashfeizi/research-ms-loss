# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from tqdm import tqdm


def feat_extractor(model, data_loader, logger=None):
    model.eval()
    feats = list()

    with tqdm(total=len(data_loader), desc='Embeddings for validation...') as t:
        for i, batch in enumerate(data_loader):
            imgs = batch[0].cuda()

            with torch.no_grad():
                out = model(imgs).data.cpu().numpy()
                feats.append(out)

            if logger is not None and (i + 1) % 100 == 0:
                logger.debug(f'Extract Features: [{i + 1}/{len(data_loader)}]')
            del out
            t.update()
    feats = np.vstack(feats)
    return feats
