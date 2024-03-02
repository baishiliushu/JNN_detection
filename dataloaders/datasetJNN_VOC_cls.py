import random
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

import torch
from torch.utils.data import Dataset

from config import Config
from utils.utils import augment_img_cls, letterbox_image, judge_pillow_image_is_wrong


class DatasetJNN_VOC_CLS(Dataset):

    def __init__(self, VOC_path, mode="trainval", year="20072012", is_training=True):

        self.VOC_path = VOC_path
        self.is_training = is_training
        self.image_paths = []


        self.unseen_classes = ['cow', 'sheep', 'cat', 'aeroplane', 'person']
        #self.unseen_classes = ['bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sofa', 'train', 'tvmonitor']
        # self.unseen_classes = ['cow', 'sheep', 'cat', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sofa', 'train', 'tvmonitor']

        if "2007" in year:
            f = open(self.VOC_path + "VOC2007/ImageSets/Main/" + mode + ".txt", "r")
            [self.image_paths.append(line.replace("\n", "")) for line in f.readlines()]
            f.close()
        if "2012" in year:
            f = open(self.VOC_path + "VOC2012/ImageSets/Main/" + mode + ".txt", "r")
            [self.image_paths.append(line.replace("\n", "")) for line in f.readlines()]
            f.close()
        if not ("2007" in year) and not ("2012" in year):
            raise Exception('Ill defined dataset')

        random.seed(123)

    def _get_name_by_index(self, index):
        fname = self.image_paths[index]
        path_helper = self.get_year_path(fname)
        return fname, path_helper

    def _get_annos_by_fname(self, path_helper, fname):
        annotation = ET.parse(self.VOC_path + path_helper + "Annotations/" + fname + ".xml")
        return annotation

    def _get_boxes_and_classes_by_fname(self, annotation):
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
            tpath_helper = self.get_year_path(t_name)
            t_annot = self._get_annos_by_fname(tpath_helper, t_name)
            tboxes, tclasses, unseen_count, found = self._get_boxes_and_gen_binary_label(selected_class, t_annot)
            if (len(tboxes) == 0) or (unseen_count == len(tboxes)) :
                tindex = random.randint(0, len(self.image_paths) - 1)
                continue
            if (must_fount_selected_classs is True) and (found is False):
                tindex = random.randint(0, len(self.image_paths) - 1)
                continue
            break

        return tpath_helper, t_name, tboxes, tclasses

    def __getitem__(self, index):
        # print("org index {}".format(index))
        # get query data
        while True:
            q_name = self.image_paths[index]
            qpath_helper = self.get_year_path(q_name)
            q_annot = self._get_annos_by_fname(qpath_helper, q_name)

            qboxes, qclasses = self.get_boxes(q_annot)
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
        # tpath_helper, t_name, tboxes, tclasses  = self.get_targets(qclass)
        tpath_helper, t_name, tboxes, tclasses  = \
            self._get_target_with_binary_all_index(qclass, must_fount_selected_classs=random.choices([True, False], [0.6, 0.4])[0])

        q_im = Image.open(self.VOC_path + qpath_helper + "JPEGImages/" + q_name + ".jpg")
        if judge_pillow_image_is_wrong(q_im):
            print("[Err]query image is empty:{}".format(q_name))
        t_im = Image.open(self.VOC_path + tpath_helper + "JPEGImages/" + t_name + ".jpg")
        if judge_pillow_image_is_wrong(t_im):
            print("[Err]target image is empty:{}".format(t_name))
        q_im = q_im.crop((qbox[0], qbox[1], qbox[2], qbox[3]))
        if judge_pillow_image_is_wrong(q_im):
            print("[Err]query crop {} image is empty:{}".format((qbox[0], qbox[1], qbox[2], qbox[3]), q_name))
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

    def get_targets(self, qclass):
        while True:
            tindex = random.randint(0, len(self.image_paths) - 1)
            t_name = self.image_paths[tindex]
            tpath_helper = self.get_year_path(t_name)
            t_annot = self._get_annos_by_fname(tpath_helper, t_name)

            tboxes, tclasses = self.get_boxes(t_annot, qclass=random.choices(['', qclass], [0.4, 0.6])[0])
            tboxes, tclasses, unseen_flag = self.filter_boxes(tboxes, tclasses)

            if len(tboxes) == 0 or unseen_flag:
                tindex = random.randint(0, len(self.image_paths) - 1)
                continue
            break
        # print("[000]index {} , q_name {}, tindex {},  t_name {} ".format(index, q_name, tindex, t_name))
        for i in range(0, len(tclasses)):
            if tclasses[i] == qclass:
                tclasses[i] = 1
            else:
                tclasses[i] = 0
        # print("[111]qclass(type{}) :{} v.s. tclasses(type{}):{} ".format(type(qclass), qclass, type(tclasses), tclasses))
        return tpath_helper, t_name, tboxes, tclasses


    def get_boxes(self, annotation, qclass=''):
        boxes = []
        classes = []
        for obj in annotation.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            tlabel = obj.find('name').text.lower().strip()
            
            if qclass != '':  # check query class
                if tlabel == qclass:
                    boxes.append([xmin, ymin, xmax, ymax])
                    classes.append(tlabel)
            else:
                # Otherwise append boxes for all classes in the image
                boxes.append([xmin, ymin, xmax, ymax])
                classes.append(tlabel)
        
        return boxes, classes

    def filter_boxes(self, boxes, classes):
        # filters unseen classes boxes when training all the others while testing
        reset = False
        out_boxes = []
        out_classes = []
        object_have_seen = []
        for i in range(len(classes)):
            if (self.is_training and classes[i] not in self.unseen_classes)\
                    or (not self.is_training ):
                #not self.is_training and classes[i] in self.unseen_classes
                out_classes.append(classes[i])
                out_boxes.append(boxes[i])
            else:
                # reset = True # one seen will give up all objects in this image
                object_have_seen.append(True)
        if len(object_have_seen) == len(classes):
            reset = True
        return out_boxes, out_classes, reset

    def get_year_path(self, img_name):
        if img_name.startswith('20'):  # Quick workaround: all VOC2012 imgs start with 20XX_
            return "VOC2012/"
        else:
            return "VOC2007/"

    def __len__(self):
        return len(self.image_paths)
