import os.path
import random
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

import torch
from torch.utils.data import Dataset

from config import Config
from utils.utils import augment_img_cls, letterbox_image, judge_pillow_image_is_wrong, rgb_open_check_pil


class DatasetJNN_COCO_CLS(Dataset):

    def __init__(self, VOC_path, mode="train", annn_middel_path="annotations/train_voc", is_training=True):

        self.VOC_path = VOC_path
        self.is_training = is_training
        self.image_paths = []
        self.ann_path = os.path.join(self.VOC_path, annn_middel_path)

        self.unseen_classes = ['cow', 'sheep', 'aeroplane', 'bottle', 'bus',  'train', 'person']
        # self.unseen_classes = ['bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sofa', 'train', 'tvmonitor']
        # self.unseen_classes = ['cow', 'sheep', 'cat', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sofa', 'train', 'tvmonitor']
        f = open(self.VOC_path + "ImageSets/Main/" + mode + ".txt", "r")
        [self.image_paths.append(line.replace("\n", "")) for line in f.readlines()]
        f.close()
        random.seed(123)

    def _get_annos_by_fname(self, fname):
        annotation = ET.parse(os.path.join(self.ann_path, fname + ".xml"))
        return annotation

    def _get_boxes_and_classes(self, annotation):
        boxes = []
        classes = []
        for obj in annotation.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            tlabel = obj.find('name').text.lower().strip()

            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(tlabel)
        return boxes, classes

    def _get_boxes_and_gen_binary_label(self, selected_class, annotation):
        boxes = []
        classes = []
        unseen_count = 0
        found_current_class = False
        for obj in annotation.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            tlabel = obj.find('name').text.lower().strip()
            if tlabel in self.unseen_classes:
                unseen_count += 1
                # dog dog dog unseen_1 : should use for training
                continue
            if selected_class == tlabel:
                found_current_class = True
                tlabel = 1
            else:
                tlabel = 0
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(tlabel)
        return boxes, classes, unseen_count, found_current_class

    def _get_target_with_binary_all_index(self, selected_class, must_fount_selected_classs=False):
        while True:
            tindex = random.randint(0, len(self.image_paths) - 1)
            t_name = self.image_paths[tindex]
            t_annot = self._get_annos_by_fname(t_name)
            tboxes, tclasses, unseen_count, found = self._get_boxes_and_gen_binary_label(selected_class, t_annot)
            if (len(tboxes) == 0) or (unseen_count == len(tboxes)):
                tindex = random.randint(0, len(self.image_paths) - 1)
                continue
            if (must_fount_selected_classs is True) and (found is False):
                tindex = random.randint(0, len(self.image_paths) - 1)
                continue
            break

        return t_name, tboxes, tclasses

    def __getitem__(self, index):
        # print("org index {}".format(index))
        # get query data multi-progress?
        query_random_index = None
        qboxes = None

        while True:
            q_name = self.image_paths[index]
            q_annot = self._get_annos_by_fname(q_name)
            qboxes, qclasses = self._get_boxes_and_classes(q_annot)
            qboxes, qclasses, unseen_flag = self.filter_boxes(qboxes, qclasses)

            if len(qboxes) == 0 or unseen_flag:
                index = random.randint(0, len(self.image_paths) - 1)
                continue

            # Select a random box in the image as a query and crop
            query_random_index = random.randrange(0, len(qboxes))
            qbox = qboxes[query_random_index]
            qclass = qclasses[query_random_index]
            break
        # get target data

        t_name, tboxes, tclasses = \
            self._get_target_with_binary_all_index(qclass, must_fount_selected_classs=random.choices([True, False], [0.6, 0.4])[0])
        q_jpg_path = self.VOC_path + "train2017/" + q_name + ".jpg"
        q_im = rgb_open_check_pil(q_jpg_path)
        if judge_pillow_image_is_wrong(q_im):
            print("[Err]query image is empty:{}".format(q_jpg_path))
        t_jpg_path = self.VOC_path + "train2017/" + t_name + ".jpg"
        t_im = rgb_open_check_pil(t_jpg_path)
        if judge_pillow_image_is_wrong(t_im):
            print("[Err]target image is empty:{}".format(t_jpg_path))
        q_im = q_im.crop((qbox[0], qbox[1], qbox[2], qbox[3]))
        if judge_pillow_image_is_wrong(q_im):
            print("[Err]query crop {} image is empty:{}(classes{}, boxes{}, index:{})"
                  "".format((qbox[0], qbox[1], qbox[2], qbox[3]), q_jpg_path, qclass, qboxes, query_random_index))
        boxes = np.asarray(tboxes, dtype=np.float32)
        class_values = np.asarray(tclasses, dtype=np.float32)
        if self.is_training:
            t_im, boxes, class_values = augment_img_cls(t_im, boxes, class_values)

            w, h = t_im.size[0], t_im.size[1]
            boxes[:, 0::2] = np.clip(boxes[:, 0::2] / w, 0.001, 0.999)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2] / h, 0.001, 0.999)

            # resize images
            if Config.letter_box_for_query_img:
                q_im = letterbox_image(q_im, (Config.imq_w, Config.imq_h))
                t_im = letterbox_image(t_im, (Config.im_w, Config.im_h))
            else:
                q_im = q_im.resize((Config.imq_w, Config.imq_h))
                t_im = t_im.resize((Config.im_w, Config.im_h))

            # To float tensors
            q_im = torch.from_numpy(np.array(q_im)).float() / 255
            t_im = torch.from_numpy(np.array(t_im)).float() / 255
            q_im = q_im.permute(2, 0, 1)
            t_im = t_im.permute(2, 0, 1)

            boxes = torch.from_numpy(boxes)
            num_obj = torch.Tensor([boxes.size(0)]).long()
            class_values = torch.from_numpy(class_values)
            return q_im, t_im, boxes, class_values, num_obj

        else:
            w, h = t_im.size[0], t_im.size[1]

            # resize images
            if Config.letter_box_for_query_img:
                q_im = letterbox_image(q_im, (Config.imq_w, Config.imq_h))
            else:
                q_im = q_im.resize((Config.imq_w, Config.imq_h))
            t_im = t_im.resize((Config.im_w, Config.im_h))

            # To float tensors
            q_im = torch.from_numpy(np.array(q_im)).float() / 255
            t_im = torch.from_numpy(np.array(t_im)).float() / 255
            q_im = q_im.permute(2, 0, 1)
            t_im = t_im.permute(2, 0, 1)

            boxes = torch.from_numpy(boxes)

            return q_im, t_im, boxes, qclass, (w, h, q_name, t_name)


    def filter_boxes(self, boxes, classes):
        # filters unseen classes boxes when training all the others while testing
        reset = False
        out_boxes = []
        out_classes = []
        object_have_seen = []
        for i in range(len(classes)):
            if (self.is_training and classes[i] not in self.unseen_classes) \
                    or (not self.is_training):
                # not self.is_training and classes[i] in self.unseen_classes
                out_classes.append(classes[i])
                out_boxes.append(boxes[i])
            else:
                # reset = True # one seen will give up all objects in this image
                object_have_seen.append(True)
        if len(object_have_seen) == len(classes):
            reset = True
        return out_boxes, out_classes, reset

    def __len__(self):
        return len(self.image_paths)
