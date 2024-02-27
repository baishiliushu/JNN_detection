import random
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

import torch
from torch.utils.data import Dataset

from config import Config
from utils.utils import augment_img, letterbox_image


class DatasetJNN_VOC_ABBY(Dataset):

    def __init__(self, VOC_path, mode="train_only_jpg", is_training=True):

        self.VOC_path = VOC_path
        self.is_training = is_training
        self.image_paths = []


        self.unseen_classes = ['jlw_person,jlw_cat,jlw_wire']
        #self.unseen_classes = ['bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sofa', 'train', 'tvmonitor']
        # self.unseen_classes = ['cow', 'sheep', 'cat', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sofa', 'train', 'tvmonitor']
        f = open(self.VOC_path + "Annos/TRAIN/" + "/Main/" + mode + ".txt", "r")
        [self.image_paths.append(line.replace("\n", "")) for line in f.readlines()]
        f.close()
        random.seed(123)
        random.shuffle(self.image_paths)

    def __getitem__(self, index):

        # get query data
        while True:
            q_name = self.image_paths[index]
            q_annot = ET.parse(self.VOC_path + "Annos/TRAIN/" + q_name + ".xml")

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
        while True:
            tindex = random.randint(0, len(self.image_paths) - 1)
            t_name = self.image_paths[tindex]
            t_annot = ET.parse(self.VOC_path + "Annos/TRAIN/" + t_name + ".xml")

            tboxes, tclasses = self.get_boxes(t_annot, qclass=qclass)
            tboxes, tclasses, unseen_flag = self.filter_boxes(tboxes, tclasses)

            if len(tboxes) == 0:
                tindex = random.randint(0, len(self.image_paths) - 1)
                continue
            break
        q_file = self.VOC_path + "JPEGImages/TRAIN/" + q_name + ".jpg"
        t_file = self.VOC_path + "JPEGImages/TRAIN/" + t_name + ".jpg"

        q_im = Image.open(q_file)
        t_im = Image.open(t_file)

        q_im = q_im.crop((qbox[0], qbox[1], qbox[2], qbox[3]))


        if str(q_im.mode) != "RGB":
            q_im = q_im.convert("RGB")
            print("[DEBUG]querry convert RBG {}".format(q_file))
        if str(t_im.mode) != "RGB":
            t_im = t_im.convert("RGB")
            print("[DEBUG]sence convert RBG {}".format(t_file))
        boxes = np.asarray(tboxes, dtype=np.float32)
        if self.is_training:
            t_im, boxes = augment_img(t_im, boxes)

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

            return q_im, t_im, boxes, num_obj

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
        
        for i in range(len(classes)):
            if (self.is_training and classes[i] not in self.unseen_classes)\
                    or (not self.is_training and classes[i] in self.unseen_classes):
                out_classes.append(classes[i])
                out_boxes.append(boxes[i])
            else:
                reset = True
        return out_boxes, out_classes, reset

    def __len__(self):
        return len(self.image_paths)
