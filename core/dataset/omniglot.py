# -*- coding: utf-8 -*-
# @File : omniglot.py
# @Author: Runist
# @Time : 2022/7/6 10:41
# @Software: PyCharm
# @Brief:

import random
import numpy as np
import glob # 用来查找文件目录和文件，并将搜索的到的结果返回到一个列表中
from PIL import Image
import torch.nn.functional as F
import torch

from core.dataset import MAMLDataset


class OmniglotDataset(MAMLDataset):
    def get_file_list(self, data_path):
        """
        Get all fonts list.
        Args:
            data_path: Omniglot Data path

        Returns: fonts list

        """
        return [f for f in glob.glob(data_path + "**/character*", recursive=True)]

    def get_one_task_data(self):
        """
        Get ones task maml data, include one batch support images and labels, one batch query images and labels.
        Returns: support_data, query_data

        """
        img_dirs = random.sample(self.file_list, self.n_way) # 随机采样
        support_data = []
        query_data = []

        support_image = []
        support_label = []
        query_image = []
        query_label = []

        for label, img_dir in enumerate(img_dirs): # 组合为一个索引序列
            img_list = [f for f in glob.glob(img_dir + "**/*.png", recursive=True)]
            images = random.sample(img_list, self.k_shot + self.q_query)

            # Read support set
            for img_path in images[:self.k_shot]:
                image = Image.open(img_path)
                image = np.array(image)
                image = np.expand_dims(image / 255., axis=0)
                support_data.append((image, label))

            # Read query set
            for img_path in images[self.k_shot:]:
                image = Image.open(img_path)
                image = np.array(image)
                image = np.expand_dims(image / 255., axis=0)
                query_data.append((image, label))

        # shuffle support set
        random.shuffle(support_data) # 用于将一个列表中的元素打乱顺序
        for data in support_data:
            support_image.append(data[0])
            support_label.append(data[1])

        # shuffle query set
        random.shuffle(query_data)
        for data in query_data:
            query_image.append(data[0])
            query_label.append(data[1])

        return np.array(support_image), np.array(support_label), np.array(query_image), np.array(query_label)
