# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
import copy
import datetime
import time

import numpy as np
import torch
import h5py
import os

from sklearn.metrics.pairwise import cosine_similarity

from ret_benchmark.data.evaluations import RetMetric
from ret_benchmark.utils.feat_extractor import feat_extractor
from ret_benchmark.utils.freeze_bn import set_bn_eval
from ret_benchmark.utils.metric_logger import MetricLogger


def make_batch_bce_labels(labels):
    """
    :param labels: e.g. tensor of size (N,1)
    :return: binary matrix of labels of size (N, N)
    """

    l_ = labels.repeat(len(labels)).reshape(-1, len(labels))
    l__ = labels.repeat_interleave(len(labels)).reshape(-1, len(labels))

    final_bce_labels = (l_ == l__).type(torch.float32)

    # final_bce_labels.fill_diagonal_(0)

    return final_bce_labels
def get_samples(l, k):
    if len(l) < k:
        to_ret = np.random.choice(l, k, replace=True)
    else:
        to_ret = np.random.choice(l, k, replace=False)

    return to_ret
def get_xs_ys(bce_labels, k=1):
    """

    :param bce_labels: tensor of (N, N) with 0s and 1s
    :param k: number of pos and neg samples per anch
    :return:

    """
    xs = []
    ys = []
    bce_labels_copy = copy.deepcopy(bce_labels)
    bce_labels_copy.fill_diagonal_(-1)
    for i, row in enumerate(bce_labels_copy):
        neg_idx = torch.where(row == 0)[0]
        pos_idx = torch.where(row == 1)[0]

        ys.extend(get_samples(neg_idx, k))
        ys.extend(get_samples(pos_idx, k))
        xs.extend(get_samples([i], 2 * k))

    return xs, ys

def calc_auroc(embeddings, labels):
    from sklearn.metrics import roc_auc_score
    bce_labels = make_batch_bce_labels(labels)
    similarities = cosine_similarity(embeddings)

    xs, ys = get_xs_ys(bce_labels)

    true_labels = bce_labels[xs, ys]
    predicted_labels = similarities[xs, ys]

    return roc_auc_score(true_labels, predicted_labels)


def __save_h5(data_description, data, data_type, path):
    h5_feats = h5py.File(path, 'w')
    h5_feats.create_dataset(data_description, data=data, dtype=data_type)
    h5_feats.close()


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        logger
):
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_loader)

    start_iter = arguments["iteration"]
    best_iteration = -1
    best_recall = 0
    best_auc = 0

    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets) in enumerate(train_loader, start_iter):

        if iteration % cfg.VALIDATION.VERBOSE == 0 or iteration == max_iter:
            model.eval()
            logger.info('Validation')
            labels = val_loader.dataset.label_list
            labels = np.array([int(k) for k in labels])
            feats = feat_extractor(model, val_loader, logger=logger)

            ret_metric = RetMetric(feats=feats, labels=labels)

            recall_curr = ret_metric.recall_k(1)
            auc_curr = calc_auroc(embeddings=feats, labels=torch.tensor(labels))

            if auc_curr > best_auc:
                best_recall = recall_curr
                best_auc = auc_curr
                best_iteration = iteration
                logger.info(f'Best iteration {iteration}: recall@1: {best_recall:.3f}')
                logger.info(f'Best iteration {iteration}: auroc: {auc_curr:.3f}')

                checkpointer.save(f"best_model")
                checkpointer.tag_last_checkpoint("best_model.pth")
            else:
                logger.info(f'Recall@1 at iteration {iteration:06d}: {recall_curr:.3f}')
                logger.info(f'Auroc at iteration {iteration:06d}: {auc_curr:.3f}')


        model.train()
        model.apply(set_bn_eval)

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = torch.stack([target.to(device) for target in targets])

        feats = model(images)
        loss = criterion(feats, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time, loss=loss.item())

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.1f} GB",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                )
            )

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:06d}".format(iteration))

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    logger.info(f"Best iteration: {best_iteration :06d} | best recall {best_recall} | best auroc {best_auc}")

def save_embs(
        cfg,
        model,
        val_loader,
        logger
):
    logger.info(f"Saving Embeddings for {cfg.DATA.TEST_IMG_SOURCE}")
    print(f"Saving Embeddings for {cfg.DATA.TEST_IMG_SOURCE}")

    val_file_name = os.path.split(cfg.DATA.TEST_IMG_SOURCE)[1]
    val_type = val_file_name[:val_file_name.find('.')]

    model.eval()
    logger.info('Validation (Saving Embeddings)')
    labels = val_loader.dataset.label_list
    labels = np.array([int(k) for k in labels])
    feats = feat_extractor(model, val_loader, logger=logger)

    ret_metric = RetMetric(feats=feats, labels=labels)
    recall_curr = ret_metric.recall_k(1)

    best_recall = recall_curr

    logger.info(f'{val_type} Recall@1: {best_recall:.3f}')

    save_path = cfg.SAVE_DIR
    __save_h5('data', labels, 'i8',
            os.path.join(save_path, f'{cfg.DATA.DATASET_NAME}_{val_type}_Classes.h5'))
    __save_h5('data', feats, 'f',
            os.path.join(save_path, f'{cfg.DATA.DATASET_NAME}_{val_type}_Feats.h5'))

    return True