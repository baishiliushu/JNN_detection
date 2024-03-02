import math
import os.path
import time
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as dset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim

from dataloaders.datasetJNN import DatasetJNN
from dataloaders.datasetJNN_VOC_cls import DatasetJNN_VOC_CLS
from dataloaders.datasetJNN_COCO_cls import DatasetJNN_COCO_CLS
from dataloaders.datasetJNN_COCOsplit import DatasetJNN_COCOsplit
from dataloaders.datasetJNN_VOC_abby import DatasetJNN_VOC_ABBY

from utils.utils import Utils
from utils.utils import logg_init_obj
from utils.utils import network_choice
from utils.utils import judge_tensor_is_zero

from config import Config
from model.darkJNNBinaryCls import DarkJNNCls


class Trainer:
    def __init__(self):
        self.log_name = None
        if not os.path.exists(Config.model_father_path):
            try:
                os.makedirs(Config.model_father_path)
            except OSError:
                print("[ERR]mkdir model_father_path {} failed.".format(Config.model_father_path))
                exit(-1)
        if Config.log_of_train:
            self.log_name = "{}/console_train_{}.log".format(Config.model_father_path, time.time())
            logg_init_obj(self.log_name)
        print("lr:     ", Config.lr)
        print("batch: {}, num_workers:{} ".format(Config.batch_size, Config.num_workers))
        print("epochs: ", Config.epochs)
        print("dataset:", Config.dataset)
        print("network_type:", Config.network_type)
        print("load_pretrianed_weight:", Config.load_pretrianed_weight)
        print("anchors:{}, giou:{}".format(Config.anchors, Config.use_giou))
        print("size:{},{} ".format(Config.im_w, Config.im_h))
        print("object_scale:{}, noobject_scale: {},class_scale: {}, coord_scale: {}"
              "".format(Config.object_scale, Config.noobject_scale, Config.class_scale, Config.coord_scale))

    @staticmethod
    def train():

        torch.cuda.manual_seed(123)
        print("Training process initialized...")

        if Config.dataset == "VOC":
            print("dataset: ", Config.voc_dataset_dir)
            dataset = DatasetJNN_VOC_CLS(Config.voc_dataset_dir)
        if Config.dataset == "coco":
            dataset = DatasetJNN_COCO_CLS(Config.coco_dataset_dir)
        # elif Config.dataset == "coco":
        #     print("dataset: ", Config.coco_dataset_dir)
        #     dataset = DatasetJNN_COCO(Config.coco_dataset_dir)
        # elif Config.dataset == "coco_split":
        #     print("dataset: ", Config.coco_dataset_dir, "--Split: ", Config.coco_split)
        #     dataset = DatasetJNN_COCOsplit(Config.coco_dataset_dir, Config.coco_split)
        # elif Config.dataset == "VOC_ABBY":
        #     dataset = DatasetJNN_VOC_ABBY(Config.voc_rubby_dataset_dir)
        # else:
        #     print("dataset: ", Config.training_dir)
        #     folder_dataset = dset.ImageFolder(root=Config.training_dir)
        #     dataset = DatasetJNN(imageFolderDataset=folder_dataset)

        train_dataloader = DataLoader(dataset,
                                      shuffle=True,
                                      num_workers=Config.num_workers,
                                      batch_size=Config.batch_size,
                                      drop_last=True,
                                      collate_fn=Utils.custom_collate_fn)


        model = DarkJNNCls()
        print("trainCls loaded net :\n{}".format(model))
        time.sleep(3)
        lr = Config.lr

        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=Config.momentum, weight_decay=Config.weight_decay)

        starting_ep = 0
        if Config.continue_training:
            last_model = Config.model_path + Config.model_endless
            if os.path.isfile(last_model):
                checkpoint = torch.load(last_model)
                model.load_state_dict(checkpoint['model'])
                starting_ep = checkpoint['epoch'] + 1
                lr = checkpoint['lr']
                Trainer.adjust_learning_rate(optimizer, lr)
                print("[INFO]Continue training start epoch is : ", starting_ep)
            else:
                print("[WARN]Cannot Continue-training, NOT found file {}".format(last_model))

        model.cuda()
        print("model convert into cuda.")
        model.train()
        print("model recall train().")
        counter = []
        loss_history = []

        best_loss = 10 ** 15
        best_epoch = 0
        break_counter = 0  # break after 20 epochs without loss improvement

        for epoch in range(starting_ep, Config.epochs):
            print("##################NO.{} EPOCH [{},{})####################".format(epoch, starting_ep, Config.epochs))
            start_time = time.time()

            average_model_time = 0
            average_optim_time = 0

            average_epoch_loss = 0
            average_loc_loss = 0
            average_conf_loss = 0
            average_cls_loss = 0
            print('current learning rate is {}'.format(lr))
            if epoch in Config.decay_lrs:
                lr = Config.decay_lrs[epoch]
                Trainer.adjust_learning_rate(optimizer, lr)
                print('adjust learning rate to {}'.format(lr))

            for i, data in enumerate(train_dataloader, 0):

                if (i % 3000 == 0):
                   print(str(i) + "/" + str(len(train_dataloader)))  # progress

                img0, img1, targets, classes_gt, num_obj = data
                # print("classes_gt:{}".format(classes_gt))
                img0, img1, targets, classes_gt, num_obj = Variable(img0).cuda(), Variable(img1).cuda(), targets.cuda(), classes_gt.cuda(), num_obj.cuda()
                if judge_tensor_is_zero(targets):
                    print("[WARN]tensor data targets is all zero.")
                    continue
                if judge_tensor_is_zero(img0):
                    print("[WARN]tensor data img0 is all zero.")
                    continue
                if judge_tensor_is_zero(img1):
                    print("[WARN]tensor data img1 is all zero.")
                    continue
                model_timer = time.time()
                loc_l, conf_l, cls_l = model(img0, img1, targets, classes_gt, num_obj, training=True)
                # loss = loc_l.mean() + conf_l.mean()
                loc_l_mean_handle = (loc_l / loc_l.numel()).sum()
                conf_l_mean_handle = (conf_l / conf_l.numel()).sum()
                cls_l_mean_handle = (cls_l / cls_l.numel()).sum()
                loss = loc_l_mean_handle + conf_l_mean_handle + cls_l_mean_handle
                if math.isnan(loc_l):
                    print("[ERR] loc_l_mean_handle is nan:{}".format(loc_l))
                if math.isnan(conf_l):
                    print("[ERR] conf_l_mean_handle is nan:{}".format(conf_l))
                if math.isnan(cls_l):
                    print("[ERR] cls_l_mean_handel is nan:{}".format(cls_l))
                model_timer = time.time() - model_timer
                average_model_time += model_timer

                optim_timer = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optim_timer = time.time() - optim_timer
                average_optim_time += optim_timer

                average_epoch_loss += loss
                average_loc_loss += loc_l
                average_conf_loss += conf_l
                average_cls_loss += cls_l
                if (i % 100 == 0):
                   print("[INFO] data batch {} : loss-sum {}, loss is {} = loc{} + conf{} + "
                         "cls{}".format(i, average_epoch_loss, loss, loc_l_mean_handle,  conf_l_mean_handle, cls_l_mean_handle))  # progress

            end_time = time.time() - start_time
            print("time: ", end_time)

            others_timer = end_time - average_model_time - average_optim_time
            print("data+ time: ", others_timer / 3600)
            print("model time: ", average_model_time / 3600)
            print("optim time: ", average_optim_time / 3600)
            average_epoch_loss = average_epoch_loss / i
            if i == 0:
                average_epoch_loss = average_epoch_loss / i + 1

            print("Epoch number {}\n Current loss {}\n".format(epoch, average_epoch_loss))
            counter.append(epoch)
            loss_history.append(average_epoch_loss.item())

            if average_epoch_loss < best_loss:
                print("------Best:")
                break_counter = 0
                best_loss = average_epoch_loss
                best_epoch = epoch
                save_name = Config.best_model_path
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': average_epoch_loss,
                    'lr': lr
                }, "{}_total{}".format(save_name, Config.model_endless))
                torch.save({
                    'model': model.state_dict(),
                }, "{}{}".format(save_name, Config.model_endless))

            save_name = Config.model_path
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': average_epoch_loss,
                'lr': lr
            }, "{}{}".format(save_name, Config.model_endless))
            torch.save({
                'model': model.state_dict(),
            }, "{}_only_weight{}".format(save_name, Config.model_endless))

            print("")
            if break_counter >= 20:
                print("Training break...")
                #break

            break_counter += 1

        print("best: ", best_epoch)
        plt.plot(counter, loss_history)
        plt.show()

    @staticmethod
    def adjust_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr