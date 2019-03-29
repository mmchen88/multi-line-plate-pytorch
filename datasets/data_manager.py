from __future__ import print_function, absolute_import

import glob
import re
from os import path as osp
import os
import numpy as np
"""Dataset classes"""


class plateDataset(object):
    """
    plateDataset
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    def __init__(self, dataset_dir, mode, root='D:/pythonProjects/multi-line-plate-recognition-master'):
        self.dataset_dir = dataset_dir
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train_data')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()
        train_relabel = (mode == 'retrieval')
        train, num_train_imgs = self._process_dir(self.train_dir, relabel=train_relabel)
        query, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> plateDataset loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} ".format(num_train_imgs))
        print("  query    | {:5d} ".format(num_query_imgs))
        print("  gallery  | {:5d} ".format(num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:8d}".format(num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_names = os.listdir(dir_path)
        img_paths = [os.path.join(dir_path, img_name) for img_name in img_names \
            if img_name.endswith('jpg') or img_name.endswith('png')]
        pattern = re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)_([-\d]+)_([-\d]+)_([-\d]+)_([-\d]+)')

        pid_container = []
        dataset = []
        for img_path in img_paths:
            pid = np.array(list(map(int, pattern.search(img_path).groups())))
            if len(pid) < 7: continue  # junk images are just ignored
            pid_container.append(pid)
            dataset.append((img_path, pid))

        num_imgs = len(dataset)
        return dataset, num_imgs

def init_dataset(name, mode):
    return plateDataset(name, mode)
