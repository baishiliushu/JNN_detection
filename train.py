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
from dataloaders.datasetJNN_VOC import DatasetJNN_VOC
from dataloaders.datasetJNN_COCO import DatasetJNN_COCO
from dataloaders.datasetJNN_COCOsplit import DatasetJNN_COCOsplit
from dataloaders.datasetJNN_VOC_abby import DatasetJNN_VOC_ABBY

from utils.utils import Utils
from utils.utils import logg_init_obj
from utils.utils import network_choice
from utils.utils import judge_tensor_is_zero

from config import Config


class Trainer:

    @staticmethod
    def train():
        if Config.log_of_train:
            logg_init_obj("log/console_train_{}.log".format(time.time()))
        torch.cuda.manual_seed(123)
        print("Training process initialized...")

        if Config.dataset == "VOC":
            print("dataset: ", Config.voc_dataset_dir)
            dataset = DatasetJNN_VOC(Config.voc_dataset_dir)
        elif Config.dataset == "coco":
            print("dataset: ", Config.coco_dataset_dir)
            dataset = DatasetJNN_COCO(Config.coco_dataset_dir)
        elif Config.dataset == "coco_split":
            print("dataset: ", Config.coco_dataset_dir, "--Split: ", Config.coco_split)
            dataset = DatasetJNN_COCOsplit(Config.coco_dataset_dir, Config.coco_split)
        elif Config.dataset == "VOC_ABBY":
            dataset = DatasetJNN_VOC_ABBY(Config.voc_rubby_dataset_dir)
        else:
            print("dataset: ", Config.training_dir)
            folder_dataset = dset.ImageFolder(root=Config.training_dir)
            dataset = DatasetJNN(imageFolderDataset=folder_dataset)

        train_dataloader = DataLoader(dataset,
                                      shuffle=True,
                                      num_workers=Config.num_workers,
                                      batch_size=Config.batch_size,
                                      drop_last=True,
                                      collate_fn=Utils.custom_collate_fn)

        print("lr:     ", Config.lr)
        print("batch:  ", Config.batch_size)
        print("epochs: ", Config.epochs)

        model = network_choice()
        print("DEV-BRANCH MARK loaded net :\n{}".format(model))
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

            print('current learning rate is {}'.format(lr))
            if epoch in Config.decay_lrs:
                lr = Config.decay_lrs[epoch]
                Trainer.adjust_learning_rate(optimizer, lr)
                print('adjust learning rate to {}'.format(lr))

            for i, data in enumerate(train_dataloader, 0):

                if (i % 3000 == 0):
                   print(str(i) + "/" + str(len(train_dataloader)))  # progress

                img0, img1, targets, num_obj = data
                img0, img1, targets, num_obj = Variable(img0).cuda(), Variable(img1).cuda(), targets.cuda(), num_obj.cuda()
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
                loc_l, conf_l = model(img0, img1, targets, num_obj, training=True)
                # loss = loc_l.mean() + conf_l.mean()
                loc_l_mean_handle = (loc_l / loc_l.numel()).sum()
                conf_l_mean_handle = (conf_l / conf_l.numel()).sum()
                loss = loc_l_mean_handle + conf_l_mean_handle
                if math.isnan(loc_l_mean_handle):
                    print("[ERR] loc_l_mean_handle is nan:{}".format(loc_l_mean_handle))
                if math.isnan(conf_l_mean_handle):
                    print("[ERR] conf_l_mean_handle is nan:{}".format(conf_l_mean_handle))
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
                if (i % 100 == 0):
                   print("[INFO] data batch {} : loss is {}".format(i, loss))  # progress


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
