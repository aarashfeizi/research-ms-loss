# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import numpy as np
import faiss
import torch

class RetMetric(object):
    def __init__(self, feats, labels, ks=[1, 2, 4, 8]):

        if len(feats) == 2 and type(feats) == list:
            """
            feats = [gallery_feats, query_feats]
            labels = [gallery_labels, query_labels]
            """
            self.is_equal_query = False

            self.gallery_feats, self.query_feats = feats
            self.gallery_labels, self.query_labels = labels

        else:
            self.is_equal_query = True
            self.gallery_feats = self.query_feats = feats
            self.gallery_labels = self.query_labels = labels

        # print('computing sim_mat')
        # self.sim_mat = np.array(torch.matmul(torch.tensor(self.query_feats), torch.tensor(self.gallery_feats).t()))
        # print('done computing sim_mat!')
        # print('computing r@1')
        # self.recall_at_k_dict = self.get_recall_at_k(ks)
        # print('done computing r@1')

        print('computing sim_mat')
        self.sim_mat = np.array(torch.matmul(torch.tensor(self.query_feats), torch.tensor(self.gallery_feats).t()))
        print('done computing sim_mat!')

    def recall_k(self, k=1):
        m = len(self.sim_mat)

        match_counter = 0

        for i in range(m):
            pos_sim = self.sim_mat[i][self.gallery_labels == self.query_labels[i]]
            neg_sim = self.sim_mat[i][self.gallery_labels != self.query_labels[i]]

            thresh = np.sort(pos_sim)[-2] if self.is_equal_query else np.max(pos_sim)

            if np.sum(neg_sim > thresh) < k:
                match_counter += 1
        return float(match_counter) / m

    # def recall_k(self, k=1):
    #     return self.recall_at_k_dict[k]

    def get_faiss_knn(self, k=1500, gpu=False):  # method "cosine" or "euclidean"
        assert self.gallery_feats.dtype == np.float32
        assert self.query_feats.dtype == np.float32

        valid = False

        D, I, self_D = None, None, None

        dim = self.gallery_feats.shape[1]

        if gpu:
            try:
                index_flat = faiss.IndexFlatIP(dim)
                res = faiss.StandardGpuResources()
                index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
                index_flat.add(self.gallery_feats)  # add vectors to the index
                print('Using GPU for KNN!!'
                      ' Thanks FAISS! :)')
            except:
                print('Didn\'t fit it GPU, No gpus for faiss! :( ')
                index_flat = faiss.IndexFlatIP(dim)
                index_flat.add(self.gallery_feats)  # add vectors to the index
        else:
            print('No gpus for faiss! :( ')
            index_flat = faiss.IndexFlatIP(dim)
            index_flat.add(self.gallery_feats)  # add vectors to the index

        assert (index_flat.ntotal == self.gallery_feats.shape[0])

        while not valid:
            print(f'get_faiss_knn metric is cosine: for top {k}')

            D, I = index_flat.search(self.query_feats, k)

            D_notself = []
            I_notself = []

            self_distance = []
            max_dist = np.array(D.max(), dtype=np.float32)
            for i, (i_row, d_row) in enumerate(zip(I, D)):
                if len(np.where(i_row == i)[0]) > 0:  # own index in returned indices
                    self_distance.append(d_row[np.where(i_row == i)])
                    I_notself.append(np.delete(i_row, np.where(i_row == i)))
                    D_notself.append(np.delete(d_row, np.where(i_row == i)))
                else:
                    self_distance.append(max_dist)
                    I_notself.append(np.delete(i_row, len(i_row) - 1))
                    D_notself.append(np.delete(d_row, len(i_row) - 1))

            self_D = np.array(self_distance, dtype=np.float32)
            D = np.array(D_notself, dtype=np.int32)
            I = np.array(I_notself, dtype=np.int32)
            if len(self_D) == D.shape[0]:
                valid = True
            else:  # self was not found for all examples
                print(f'self was not found for all examples, going from k={k} to k={k * 2}')
                k *= 2

        return D, I, self_D

    def get_recall_at_k(self, ks):
        all_lbls = np.unique(self.query_labels)

        num = self.gallery_labels.shape[0]

        k_max = min(1500, num)

        _, I, self_D = self.get_faiss_knn(k=k_max, gpu=True)

        recall_at_k = Accuracy_At_K(classes=np.array(all_lbls), ks=ks)

        for idx, lbl in enumerate(self.query_labels):
            ret_lbls = self.gallery_labels[I[idx]]
            recall_at_k.update(lbl, ret_lbls)

        total = recall_at_k.get_all_metrics()

        return total

class Accuracy_At_K():

    def __init__(self, ks=[1, 2, 4, 8], classes=np.array([])):
        self.ks = ks
        self.k_valus = {i: 0 for i in self.ks}

        self.r_values = {i: 0 for i in self.ks}

        self.n = 0

        self.classes = classes
        self.class_tot = len(self.classes)
        self.lbl2idx = {c: i for i, c in enumerate(self.classes)}


    def update(self, lbl, ret_lbls):
        # all_lbl = sum(ret_lbls == lbl)

        for k in self.ks:
            if lbl in ret_lbls[:k]:
                self.k_valus[k] += 1

        self.n += 1

    def __str__(self):
        output_str = ''
        metrics = self.get_all_metrics()

        for k, v in metrics.items():
            output_str += f'{k} = {v}\n'

        return output_str

    def get_all_metrics(self):
        output_dict = {}
        for k in self.ks:
            final_k = self.k_valus[k] / max(self.n, 1)
            output_dict[k] = final_k

        return output_dict
