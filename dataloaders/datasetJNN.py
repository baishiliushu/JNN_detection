import os
import random
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

import torch
from torch.utils.data import Dataset

from config import Config
from utils.utils import augment_img


class DatasetJNN(Dataset):

    def __init__(self, imageFolderDataset, is_training=True):
        self.imageFolderDataset = imageFolderDataset
        self.is_training = is_training

    def __getitem__(self, index):
        # get the index for the current batch
        img0_tuple = self.imageFolderDataset.imgs[index]

        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)

        # Search for class by looping random indexes
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        # Query preproc
        query_id = img0_tuple[0].split("/")
        query_id = query_id[len(query_id) - 1]
        query_annot = ET.parse(Config.annotations_dir + query_id + ".xml")

        qboxes = []
        qclasses = []
        for obj in query_annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            label = obj.find('name').text.lower().strip()
            if label == img0_tuple[1]:
                qboxes.append([xmin, ymin, xmax, ymax])
                qclasses.append(label)

        # crop
        query_random_index = random.randrange(0, len(qboxes))
        qbox = qboxes[query_random_index]
        qlabel = qclasses[query_random_index]
        img0 = img0.crop(qbox[0], qbox[1], qbox[2], qbox[3])

        # Target preproc
        target_id = img1_tuple[0].split("/")
        target_id = target_id[len(target_id) - 1]
        target_annot = ET.parse(Config.annotations_dir + target_id + ".xml")

        tboxes = []
        for obj in target_annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            if label == qlabel:
                tboxes.append([xmin, ymin, xmax, ymax])

        boxes = np.asarray(tboxes, dtype=np.int32)

        if self.is_training:
            img1, boxes = augment_img(img1, boxes)

            w, h = img1.size[0], img1.size[1]
            boxes[:, 0::2] = np.clip(boxes[:, 0::2] / w, 0.001, 0.999)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2] / h, 0.001, 0.999)

            # resize images
            img0 = img0.resize((Config.imq_w, Config.imq_h))
            img1 = img1.resize((Config.im_w, Config.im_h))

            # To float tensors
            img0 = torch.from_numpy(np.array(img0)).float() / 255
            img1 = torch.from_numpy(np.array(img1)).float() / 255
            img0 = img0.permute(2, 0, 1)
            img1 = img1.permute(2, 0, 1)

            boxes = torch.from_numpy(boxes)

            return img0, img1, boxes

        else:
            w, h = img1.size[0], img1.size[1]
            boxes[:, 0::2] = np.clip(boxes[:, 0::2] / w, 0.001, 0.999)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2] / h, 0.001, 0.999)

            # resize images
            img0 = img0.resize((Config.imq_w, Config.imq_h))
            img1 = img1.resize((Config.im_w, Config.im_h))

            # To float tensors
            img0 = torch.from_numpy(np.array(img0)).float() / 255
            img1 = torch.from_numpy(np.array(img1)).float() / 255
            img0 = img0.permute(2, 0, 1)
            img1 = img1.permute(2, 0, 1)

            boxes = torch.from_numpy(boxes)

            return img0, img1, boxes, img0_tuple[1], (w, h, target_id)

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
