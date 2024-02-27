import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset

import os
import time
import cv2
import numpy as np
from PIL import Image
from PIL import ImageOps
from config import Config

from model.decoder import decode
from dataloaders.datasetJNN import DatasetJNN
from dataloaders.datasetJNN_VOC import DatasetJNN_VOC
from dataloaders.datasetJNN_COCO import DatasetJNN_COCO
from dataloaders.datasetJNN_COCOsplit import DatasetJNN_COCOsplit
from utils.utils import letterbox_image
from utils.utils import network_choice
from utils.utils import logg_init_obj

def preprocess_byPIL_base(file_name, resize_w, resize_h, hist=False, letterbox=True, gray=False):
    q_im = Image.open(file_name)
    if q_im.mode != 'RGB':
        q_im = q_im.convert("RGB")
    if hist:
        q_im = ImageOps.equalize(q_im, mask=None)
    # q_im = q_im.resize((resize_w, resize_h))
    if not letterbox:
        q_im = q_im.resize((resize_w, resize_h))
    else:
        q_im = letterbox_image(q_im, (resize_w, resize_h))
        #q_im = q_im.resize((resize_w, resize_h))

    cv_q_img = np.array(q_im)
    if gray is True:
        # gray
        image_gray_color = cv2.cvtColor(cv_q_img, cv2.COLOR_BGR2GRAY)
        # color   RGB
        image_gray_color = cv2.cvtColor(image_gray_color, cv2.COLOR_GRAY2RGB)
        q_im = Image.fromarray(image_gray_color)
        cv_q_img = cv2.cvtColor(image_gray_color, cv2.COLOR_RGB2BGR)
    cv_q_img = cv_q_img[:, :, ::-1].copy()
    # To float tensors
    q_im = torch.from_numpy(np.array(q_im)).float() / 255
    img0 = q_im.permute(2, 0, 1)
    img0 = torch.unsqueeze(img0, 0)
    return img0, cv_q_img

def preprocess_byPIL(test_img_top_path, file_name, resize_w, resize_h, file_endless=".jpg", hist=False, letterbox=True, gray=False):
    img0, cv_q_img = preprocess_byPIL_base(os.path.join(test_img_top_path, file_name) + file_endless,
                                           resize_w, resize_h, hist , letterbox , gray)
    return img0, cv_q_img


def get_milliseconds_timestamp():
    return int(time.time() * 1000)


def copy_roi_and_save(cv_s_img, box_xywh):
    image_roi = np.zeros((box_xywh[3], box_xywh[2], 3), dtype=np.uint8)
    return image_roi


def sorted_with_conf(t):
    _, indices = torch.sort(t, descending=True, dim=0)  # sort along with cols-value-decrease
    print(indices)
    print(indices[:, -2])
    idx = indices[:, -2]
    t1 = t[idx[0]]  # top1
    print(t1)
    ls = []
    for ix in idx:
        ls.append(t[ix].unsqueeze(0))
    out = torch.cat(ls)
    return out


def feed_forward_process(model, q_name, t_name, img_q, img_s, cv_sence, conf, nums, is_save_rect=True, rst_path=""):
    detections = None
    fount_sample_count = 0
    loss_sample_count = 0
    visiable_image = cv_sence.copy()
    im_infos = (visiable_image.shape[1], visiable_image.shape[0], q_name, t_name)
    with torch.no_grad():
        test_t = get_milliseconds_timestamp()
        img_q, img_s = Variable(img_q).cuda(), Variable(img_s).cuda()
        model_output = model(img_q, img_s, [])
        print("[INFO]model_output: {}".format(len(model_output)))
        im_info = {'width': im_infos[0], 'height': im_infos[1]}
        output = [item[0].data for item in model_output]
        detections = decode(output, im_info, conf_threshold=conf, nms_threshold=nums)
        test_t = get_milliseconds_timestamp() - test_t
        print("[INFO]detections: {}, using time:{}".format(detections, test_t))
        highest_score = -1.0
        if len(detections) > 0:
            fount_sample_count =  1
            for detection in detections:
                start_pt = (int(detection[0].item()), int(detection[1].item()))
                end_pt = (int(detection[2].item()), int(detection[3].item()))
                visiable_image = cv2.rectangle(visiable_image, start_pt, end_pt, (0, 255, 0), 2)
                visiable_image = cv2.putText(visiable_image, "{:.4f}".format(detection[4].item()),
                                    (int(detection[2].item() - 25), int(detection[3].item() + 3)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 122, 222), 1, cv2.LINE_AA)
                print(start_pt, end_pt)

                if highest_score < detection[4].item():
                    highest_score = detection[4].item()
            if is_save_rect:
                cv2.imwrite("{}.jpg".format(os.path.join(rst_path, t_name)), visiable_image)

        else:
            loss_sample_count = 1
    return detections, visiable_image, fount_sample_count, loss_sample_count


class Tester:

    @staticmethod
    def test():

        print("testing...")

        Config.batch_size = 1

        #Config.model_path = "testmodel_last.pt"
        #Config.model_path = "/home/leon/opt-exprements/expments/JNN_detection/check_points/model_best.pt"
        print("mAP files output path: " + Config.mAP_path)

        model_path = Config.best_model_path + Config.model_endless

        print("model: ", model_path)
        print("conf: ", Config.conf_thresh)
        print("iou thresh:  ", Config.conf_thresh)

        if Config.dataset == "VOC":
            print("dataset: ", Config.voc_dataset_dir)
            dataset = DatasetJNN_VOC(Config.voc_dataset_dir, mode="test", year="2007", is_training=False)
        elif Config.dataset == "coco":
            print("dataset: ", Config.coco_dataset_dir)
            dataset = DatasetJNN_COCO(Config.coco_dataset_dir, is_training=False)
        elif Config.dataset == "coco_split":
            print("dataset: ", Config.coco_dataset_dir, "--Split: ", Config.coco_split)
            dataset = DatasetJNN_COCOsplit(Config.coco_dataset_dir, Config.coco_split, is_training=False)
        else:
            print("dataset: ", Config.testing_dir)
            folder_dataset = dset.ImageFolder(root=Config.testing_dir)
            dataset = DatasetJNN(imageFolderDataset=folder_dataset, is_training=False)

        dataloader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)

        model = network_choice()

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        print("epoch: ", str(checkpoint['epoch'] + 1))

        model.cuda()
        model.eval()

        with torch.no_grad():

            for i, data in enumerate(dataloader, 0):

                if (i % 1000 == 0):
                    print(str(i) + "/" + str(len(dataset)))  # progress

                img0, img1, targets, label, im_infos = data
                img0, img1, targets = Variable(img0).cuda(), Variable(img1).cuda(), targets.cuda()

                model_output = model(img0, img1, targets)

                im_info = {'width': im_infos[0].item(), 'height': im_infos[1].item()}
                output = [item[0].data for item in model_output]

                detections = decode(output, im_info, conf_threshold=Config.conf_thresh,nms_threshold=Config.nms_thresh)

                if len(detections) > 0:

                    # mAP files
                    pair_id = im_infos[2][0].split('.')[0] + "_" +im_infos[3][0].split('.')[0]

                    detection_str = ""
                    gt_str = ""

                    f = open(Config.mAP_path + "groundtruths/" + pair_id + ".txt", "a+")
                    for box_idx in range(len(targets)):

                        gt_str += label[0].replace(" ", "_") + " " \
                                  + str(targets[0][box_idx][0].item()) + " " \
                                  + str(targets[0][box_idx][1].item()) + " " \
                                  + str(targets[0][box_idx][2].item()) + " " \
                                  + str(targets[0][box_idx][3].item()) + "\n"

                    f.seek(0)
                    if not (gt_str in f.readlines()):
                        f.write(gt_str)
                    f.close()

                    f = open(Config.mAP_path + "detections/" + pair_id + ".txt", "a+")
                    for detection in detections:
                        detection_str += label[0].replace(" ", "_") + " " \
                                      + str(detection[4].item()) + " "\
                                      + str(detection[0].item()) + " "\
                                      + str(detection[1].item()) + " "\
                                      + str(detection[2].item()) + " "\
                                      + str(detection[3].item()) + "\n"

                    f.seek(0)
                    if not (detection_str in f.readlines()):
                        f.write(detection_str)
                    f.close()


    @staticmethod
    def test_one_OL(model_path, test_img_top_path, q_name, search_path_top, hist_option, rst_path, conf, nums):
        """ Tests a a pair of images """

        print("testing one image...")

        #Config.model_path = "/home/mmv/Documents/2.projects/JNN_detection/trained_models/dJNN_COCOsplit2/testmodel_last_split2.pt"
        model_type = "darknet19" # "no_config"
        middle_name = ""
        if model_type == "darknet19":
            middle_name = "coco_voc199epoch/"
        #
        #model_path = "/home/leon/opt-exprements/expments/JNN_detection/check_points/{}model_best.pt".format(middle_name)
        # model_path = Config.best_model_path + Config.model_endless

        model = network_choice(model_type)  # DarkJNN()

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])

        model.cuda()
        model.eval()

        # (3m1, 3m6), (rbc1, rbc43), hp(33971473, 70609284), blizzard(1, 6), gen_electric(7, 31), warner(10, 18)
        # goodyear(13, 20), airhawk(12, 1), gap(34, 36), levis(14, 30)

        # test_img_top_path = "/home/leon/opt-exprements/coco/val2017/"
        # q_name = "000000008629" #
        # t_name = "000000209530" #
        # test_img_top_path = '/home/leon/opt-exprements/expments/data_test/template_match/template_data/my_test_data/'
        # q_name = "xiaofangxiang"  # ['bin', 'miehuoqi', 'yaoshi', 'desk', 'yizi',  'shuiping']
        img0, cv_q_img = preprocess_byPIL(test_img_top_path, "{}".format(q_name), Config.imq_w, Config.imq_h,
                                          letterbox=False)

        #test_img_top_path = "/home/leon/opt-exprements/expments/data_test/template_match/match_data/{}/".format("dir-mosaic")  # q_name
        search_path = os.path.join(search_path_top, q_name)
        if not os.path.exists(search_path):
            search_path = search_path_top
        sence_imgs = os.listdir(search_path)
        if len(sence_imgs) < 0:
            return
        s_checks = []
        r_checks = []
        for i, s in enumerate(sence_imgs, 0):
            if ".jpg" not in s:
                r_checks.append(s)
                filenames = os.listdir(os.path.join(search_path,s))
                for f in filenames:
                    s_checks.append(os.path.join(os.path.join(search_path,s), f))
            else:
                sence_imgs[i] = os.path.join(search_path, s)
        if len(s_checks) > 0:
            sence_imgs = s_checks
        for r in r_checks:
            if r in sence_imgs:
                sence_imgs.remove(r)
        # hist_option = False
        #conf = 0.3  # Config.conf_thresh
        #nums = 0.4  # Config.nms_th17_1703669057750530.jpgresh
        #rst_path = '/home/leon/opt-exprements/expments/data_test/template_match/own_rst_model_coco_e98'
        if not os.path.exists(rst_path):
            os.mkdir(rst_path)
        rst_path = os.path.join(rst_path, q_name)
        if hist_option:
            rst_path = rst_path + "_histed"
        if not os.path.exists(rst_path):
            os.mkdir(rst_path)
        fount_sample_count = 0
        loss_sample_count = 0
        for t_name in sence_imgs:
            img1, cv_im = preprocess_byPIL_base(t_name, Config.im_w, Config.im_h, hist=hist_option, letterbox=False)
            #t_name = "53_1703664893160227.jpg"  #
            t_name = t_name.split("/")[-1]
            t_name = t_name.split(".jpg")[0]

            im_infos = (cv_im.shape[1], cv_im.shape[0], q_name, t_name)
            with torch.no_grad():
                test_t = get_milliseconds_timestamp()
                img0, img1 = Variable(img0).cuda(), Variable(img1).cuda()
                model_output = model(img0, img1, [])
                print("model_output: {}".format(model_output))
                im_info = {'width': im_infos[0], 'height': im_infos[1]}
                output = [item[0].data for item in model_output]
                detections = decode(output, im_info, conf_threshold=conf, nms_threshold=nums)
                test_t = get_milliseconds_timestamp() - test_t
                print("detections: {}, using time:{}".format(detections, test_t))
                highest_score = -1.0
                if len(detections) > 0:
                    fount_sample_count = fount_sample_count + 1
                    for detection in detections:
                        start_pt = (int(detection[0].item()), int(detection[1].item()))
                        end_pt = (int(detection[2].item()), int(detection[3].item()))
                        image = cv2.rectangle(cv_im, start_pt, end_pt, (0, 255, 0), 2)
                        image = cv2.putText(image, "{:.4f}".format(detection[4].item()),
                                            (min(int(detection[2].item() - 25), im_infos[0] - 1), min(int(detection[3].item() + 15), im_infos[1] - 1)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 122, 222), 1, cv2.LINE_AA)
                        if highest_score < detection[4].item():
                            highest_score = detection[4].item()
                        print(start_pt, end_pt)

                    cv2.imwrite("{}.jpg".format(os.path.join(rst_path, t_name)), image)

                else:
                    loss_sample_count = loss_sample_count + 1
                    print("under the conf {}, nums {}, sample data {} is no result output.".format(conf, nums, t_name))
                # cv2.imshow("{} with S{:.4f} C{} N{} T:{}".format(t_name, float(highest_score), conf, nums, ""),
                #            cv_im)
                # cv_q_show = cv2.resize(cv_q_img, (320, 320))
                # name_of_support_img = "{}-conf {}-nms {}".format(q_name, conf, nums)
                # cv2.namedWindow(name_of_support_img, 0)
                # cv2.moveWindow(name_of_support_img, 710, 100)
                # cv2.imshow(name_of_support_img, cv_q_show)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        pet_found = float(fount_sample_count/len(sence_imgs))
        pet_found = pet_found * 100
        print("Finished!(found count {} + loss count {} ?= {} total. fount / total = {}%)".format(fount_sample_count,
                                                                                                  loss_sample_count,
                                                                                                  len(sence_imgs),
                                                                                                  pet_found))

    @staticmethod
    def test_on_cross_cats(model_file, query_cat_name:str, query_base_path, search_base_path, rst_base_path, hist_flag=False, conf=0.3, nms=0.1):
        if Config.log_of_train:
            logg_init_obj("./log/console_test_{}.log".format(time.time()))
        model_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_file)
        if not os.path.isfile(model_file):
            print("[ERR]model tested NOT exist {}".format(model_file))
            exit(-1)
        if not (os.path.exists(search_base_path)):
            print("[ERR]search top path not exist:{}".format(search_base_path))
            exit(-2)
        model = network_choice()  # DarkJNN()
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        model.eval()

        if not (os.path.exists(rst_base_path)):
            print("[INFO]rst top path not exist:{}".format(rst_base_path))
            os.mkdir(rst_base_path)

        img_q, cv_q_img = preprocess_byPIL(query_base_path, "{}".format(query_cat_name), Config.imq_w, Config.imq_h,
                                           letterbox=True, gray=True)
        for s in os.listdir(search_base_path):
            search_path = os.path.join(search_base_path, s)
            if not (os.path.exists(search_path)):
                print("[WARN]search sub-path not exist:{},jump scene:{}".format(search_path, s))
                continue
            sence_imgs = os.listdir(search_path)
            if len(sence_imgs) < 1:
                print("[WARN]search images do not exist in {}, jump.".format(search_path))
                continue
            path_name_endless = "mix"
            if query_cat_name == s:
                path_name_endless = "canter"
            rst_current_dir = os.path.join(rst_base_path, "{}-{}_{}".format(query_cat_name, s, path_name_endless))
            print("[INFO]--------{} testing on {}, rst locate {}------------".format(query_cat_name, search_path, rst_current_dir))

            f_count = 0
            l_count = 0
            need_create_save_path = True
            for file_s in sence_imgs:
                s_name = file_s.split(".jpg")[0]
                img_s, cv_s_img = preprocess_byPIL(search_path, "{}".format(s_name), Config.im_w, Config.im_h, letterbox=False)
                detections, visiable_image, fount_sample_count, loss_sample_count = \
                    feed_forward_process(model, query_cat_name, s, img_q, img_s, cv_s_img, conf, nms, False)
                if detections is None or len(detections) < 1:
                    continue
                # todo: sort tensors [detections]
                highest_score = 0
                roi_imgs = list()
                if len(detections) > 0 and need_create_save_path:
                    if not (os.path.exists(rst_current_dir)):
                        os.mkdir(rst_current_dir)
                        need_create_save_path = False
                for i, d in enumerate(detections):
                    start_pt = (int(d[0].item()), int(d[1].item()))
                    end_pt = (int(d[2].item()), int(d[3].item()))
                    image_crop_roi = cv_s_img[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]]
                    score_roi = {}
                    save_path_score_key = "{}.jpg".format(os.path.join(rst_current_dir, "{:3f}-{}-{}".format(d[4].item(), s_name, i)))
                    score_roi[save_path_score_key] = image_crop_roi
                    roi_imgs.append(score_roi)
                    if highest_score < d[4].item():
                        highest_score = d[4].item()
                for r_pair in roi_imgs:
                    for c in r_pair:
                        if (r_pair[c].shape[0] > 0) and (r_pair[c].shape[1] > 0):
                            print("[INFO]save roi {}".format(c))
                            cv2.imwrite(c, r_pair[c])

                f_count += fount_sample_count
                l_count += loss_sample_count
                if loss_sample_count > 0:
                    print("[WARN]under conf {}, nums {}, {} in {} NO output.".format(conf, nms, query_cat_name, s_name))
                print("[SUM]--------{} tested {};rst found :{}, miss:{}, locate {}------------".format(query_cat_name, search_path, rst_current_dir, f_count, l_count))

                cv_q_show = cv2.resize(cv_q_img, (320, 320))
                name_of_support_img = "{}-conf {}-nms {}".format(query_cat_name, conf, nms)
                cv2.namedWindow(name_of_support_img, 0)
                cv2.moveWindow(name_of_support_img, 710, 100)
                # cv2.imshow("{} with S{:.4f} C{} N{} T:{}".format(s, float(highest_score), conf, nms, ""),
                #            visiable_image)
                # cv2.imshow(name_of_support_img, cv_q_show)
                # cv2.waitKey(0)
                cv2.destroyAllWindows()


    @staticmethod
    def test_one_COCO():
        """ Tests a a pair of images """

        print("testing one image...")

        #Config.model_path = "/home/mmv/Documents/2.projects/JNN_detection/trained_models/dJNN_COCOsplit2/testmodel_last_split2.pt"
        model_path = Config.best_model_path + Config.model_endless

        model = network_choice()  # DarkJNN()

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        model.eval()

        coco_dataset = dset.CocoDetection(Config.coco_dataset_dir,
                                          Config.coco_dataset_dir + "annotations/instances_val2017.json")

        # (3m1, 3m6), (rbc1, rbc43), hp(33971473, 70609284), blizzard(1, 6), gen_electric(7, 31), warner(10, 18)
        # goodyear(13, 20), airhawk(12, 1), gap(34, 36), levis(14, 30)
        q_name = "000000024144"
        t_name = "000000306700"
        # /home/mmv/Documents/3.datasets / coco /
        q_im = Image.open("{}val2017/".format(Config.coco_dataset_dir) + q_name + ".jpg")
        t_im = Image.open("{}val2017/".format(Config.coco_dataset_dir) + t_name + ".jpg")

        # find image id and (first) annotation
        for id in coco_dataset.coco.imgs:
            if coco_dataset.coco.imgs[id]['file_name'] == q_name + ".jpg":
                break
        for ann_id in coco_dataset.coco.anns:
            if coco_dataset.coco.anns[ann_id]['image_id'] == id:
                print(coco_dataset.coco.anns[ann_id])
                break
        qbox = coco_dataset.coco.anns[ann_id]['bbox']
        qbox = [qbox[0], qbox[1], qbox[0] + qbox[2], qbox[1] + qbox[3]]
        q_im = q_im.crop((qbox[0], qbox[1], qbox[2], qbox[3]))

        w, h = t_im.size[0], t_im.size[1]
        im_infos = (w, h, q_name, t_name)

        qcv_im = np.array(q_im)
        qcv_im = qcv_im[:, :, ::-1].copy()
        cv_im = np.array(t_im)
        cv_im = cv_im[:, :, ::-1].copy()

        q_im = q_im.resize((Config.imq_w, Config.imq_h))
        t_im = t_im.resize((Config.im_w, Config.im_h))

        # To float tensors
        q_im = torch.from_numpy(np.array(q_im)).float() / 255
        t_im = torch.from_numpy(np.array(t_im)).float() / 255
        img0 = q_im.permute(2, 0, 1)
        img1 = t_im.permute(2, 0, 1)
        img0 = torch.unsqueeze(img0, 0)
        img1 = torch.unsqueeze(img1, 0)

        with torch.no_grad():
            #
            img0, img1 = Variable(img0).cuda(), Variable(img1).cuda()

            model_output = model(img0, img1, [])

            im_info = {'width': im_infos[0], 'height': im_infos[1]}
            output = [item[0].data for item in model_output]

            detections = decode(output, im_info, conf_threshold=Config.conf_thresh, nms_threshold=Config.nms_thresh)

            if len(detections) > 0:

                for detection in detections:
                    start_pt = (int(detection[0].item()), int(detection[1].item()))
                    end_pt = (int(detection[2].item()), int(detection[3].item()))
                    image = cv2.rectangle(cv_im, start_pt, end_pt, (0, 255, 0), 3)
                    print(start_pt, end_pt)

                cv2.imshow("q", qcv_im)
                cv2.imshow("res", image)
                cv2.waitKey()
            else:
                print("No detctions found")

