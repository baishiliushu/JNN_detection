import argparse
import os
import sys
import shutil

import cv2
from lxml import etree, objectify

# 将标签信息写入xml
from tqdm import tqdm
import random
import time
images_nums = 0
category_nums = 0
bbox_nums = 0


def save_anno_to_xml(filename, size, objs, save_path, folder_name="VOC_CONVERT", source_database='VOC_CONVERT'):
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder("{}".format(folder_name)),
        E.filename(filename),
        E.source(
            E.database("The {} Database".format(source_database)),
            E.annotation("PASCAL VOC"),
            E.image("flickr")
        ),
        E.size(
            E.width(size[1]),
            E.height(size[0]),
            E.depth(size[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose("Unspecified"),
            E.truncated(0),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[1][0]),
                E.ymin(obj[1][1]),
                E.xmax(obj[1][2]),
                E.ymax(obj[1][3])
            )
        )
        anno_tree.append(anno_tree2)
    anno_path = os.path.join(save_path, filename[:-3] + "xml")
    etree.ElementTree(anno_tree).write(anno_path, pretty_print=True)


def xywhn2xyxy(bbox, size):
    bbox = list(map(float, bbox))
    size = list(map(float, size))
    xmin = max((bbox[0] - bbox[2] / 2.) * size[1], 1)
    ymin = max((bbox[1] - bbox[3] / 2.) * size[0], 1)
    xmax = min((bbox[0] + bbox[2] / 2.) * size[1], size[1])
    ymax = min((bbox[1] + bbox[3] / 2.) * size[0], size[0])
    box = [xmin, ymin, xmax, ymax]
    return list(map(int, box))


def gen_xml_files(current_save_path, current_image_path, current_anno_path, category_maps, unsave_categorys):
    global images_nums,  bbox_nums
    if not os.path.exists(current_anno_path):
        print("ERROR {} dose not exists".format(current_anno_path))
        return
    if not os.path.exists(current_image_path):
        print("ERROR {} dose not exists".format(current_image_path))
        return
    if os.path.exists(current_save_path):
        shutil.rmtree(current_save_path)
        print("[WARNING]:save path {} exist, delete.".format(save_path))
    # os.makedirs(current_save_path)
    os.system("mkdir -p {}".format(current_save_path))
    if not os.path.exists(current_save_path):
        exit(-2)
    images = [os.path.join(current_image_path, i) for i in os.listdir(current_image_path)]
    files = [os.path.join(current_anno_path, i) for i in os.listdir(current_anno_path)]
    images_index = dict((v.split(os.sep)[-1][:-4], k) for k, v in enumerate(images))
    images_nums += len(images)
    for file in tqdm(files):
        if os.path.splitext(file)[-1] != '.txt' or 'classes' in file.split(os.sep)[-1]:
            continue
        if file.split(os.sep)[-1][:-4] in images_index:
            index = images_index[file.split(os.sep)[-1][:-4]]
            img = cv2.imread(images[index])
            shape = img.shape
            filename = images[index].split(os.sep)[-1]
        else:
            continue
        objects = []
        with open(file, 'r') as fid:
            for i in fid.readlines():
                i = i.strip().split()
                current_obj_category_id = int(i[0])
                category_name = category_maps[current_obj_category_id]
                if category_name in unsave_categorys:
                    print("[INFO]:don't use object <{}>".format(category_name))
                    continue
                bbox = xywhn2xyxy((i[1], i[2], i[3], i[4]), shape)
                obj = [category_name, bbox]
                objects.append(obj)
        bbox_nums += len(objects)
        save_anno_to_xml(filename, shape, objects, current_save_path)
    return


def dir_struct(base_dir):
    middle_dirs = []
    for root, dirs, files in os.walk(base_dir):
        # for file in files:
        #     all_files.append(os.path.join(root, file))
        if 'coco' in root:
            continue
        if 'baidu' in root:
            continue

        if len(dirs) == 0:
            m = str(root).replace(base_dir, "")
            middle_dirs.append(m)
        # for dir in dirs:
        #     all_dirs.append(os.path.join(root, dir))
    return middle_dirs


def parseXmlFilse(image_path, anno_path, save_path, category_txt="labels-classes-map.txt", unsave_categorys=['jlw_person', ]):
    global category_nums
    assert os.path.exists(image_path), "ERROR {} dose not exists".format(image_path)
    assert os.path.exists(anno_path), "ERROR {} dose not exists".format(anno_path)
    if os.path.exists("done_dirs_yolo2voc.txt"):
        shutil.rmtree("done_dirs_yolo2voc.txt")
    category_set = []
    with open(anno_path + category_txt, 'r') as f:
        for i in f.readlines():
            category_set.append(i.strip())
    category_nums = len(category_set)
    category_maps = dict((k, v) for k, v in enumerate(category_set))
    list_annos = dir_struct(anno_path)

    # for i in os.listdir(anno_path):
    #     if os.path.isdir(anno_path + i):
    #         list_annos.append(i)
    if len(list_annos) < 1:
        exit(-1)
    with open("all_dirs_yolo2voc.txt", 'w') as run_record_file:
        for l in list_annos:
            run_record_file.write(l + "\n")
    has_record_category = []
    for current_category in list_annos:
        current_anno_path =  os.path.join(anno_path, current_category)  # anno_path + '20220720_dog/data-4/gray/cam0'
        current_image_path = os.path.join(image_path, current_category)
        current_save_path = os.path.join(save_path, current_category)
        gen_xml_files(current_save_path, current_image_path, current_anno_path, category_maps, unsave_categorys)
        if current_category not in has_record_category:
            with open("done_dirs_yolo2voc.txt", 'a') as run_record_file:
                run_record_file.write(current_category + "\n")
        has_record_category.append(current_category)

def write_txt_from_xml_sub_path(xml_dir="/home/leon/opt-exprements/coco/annotations/train_voc"):
    xmls = []
    for root, dirs, files in os.walk(xml_dir):
        xml_names = [*map(lambda f: os.path.join("", str(f).replace(".xml", "\n")), files)]
        xmls.extend(xml_names)

    with open("/home/leon/opt-exprements/coco/ImageSets/Main/train_order.txt", "x") as t:
        t.writelines(xmls)
    random.shuffle(xmls)
    with open("/home/leon/opt-exprements/coco/ImageSets/Main/train.txt", "x") as t:
        t.writelines(xmls)



def write_imagesets_txt(xml_base_dir, txt_file_path="Main", train_val_interval=8):
    middle_paths = dir_struct(xml_base_dir)
    all_txt_ordered_file_path = os.path.join(xml_base_dir, txt_file_path, "train_val.txt")
    txt_file_train_path = os.path.join(xml_base_dir, txt_file_path, "train_data.txt")
    txt_file_val_path = os.path.join(xml_base_dir, txt_file_path, "val_data.txt")
    if os.path.exists(txt_file_path):
        shutil.rmtree(txt_file_path)
    xmls = []
    xmls_train = []
    xmls_val = []
    for m in middle_paths:
        if 'Main' in m:
            continue
        for root, dirs, files in os.walk(os.path.join(xml_base_dir, m)):
            xml_names = [*map(lambda f:os.path.join(m, str(f).replace(".xml", "\n")), files)]
            print("[INFO]dir count:{}".format(xml_names.__len__()))
            xmls.extend(xml_names)
            interval_tmp = train_val_interval
            if len(xml_names) > 1:
                if len(xml_names) < interval_tmp + 1:
                    interval_tmp = len(xml_names)
                val_names = xml_names[1::interval_tmp]
                xmls_val.extend(val_names)
            for val in xml_names:
                if val in xmls_val:
                    xml_names.remove(val)
            xmls_train.extend(xml_names)
            print("[INFO]train count :{}, val count:{}".format(xml_names.__len__(), val_names.__len__()))

    with open(all_txt_ordered_file_path, "w") as t:
        t.writelines(xmls)
    # if xmls.__len__() > 0:
    #     print(xmls.__len__(), xmls[0])
    #     random.shuffle(xmls)
    with open(txt_file_train_path, "w") as t:
        t.writelines(xmls_train)
    with open(txt_file_val_path, "w") as t:
        t.writelines(xmls_val)
    print("Finish txt.(contain {} xml/image pair path)".format(xmls_train.__len__()))


def check_endless(endless='.jpg'):
    top_path = "/home/leon/mount_point_c/datas_jinglianwen/ABBY/JPEGImages/TRAIN/"
    txt_file = top_path.replace("JPEGImages/TRAIN/", "Annos/TRAIN/Main/train_data.txt")
    if not os.path.isfile(txt_file):
        print("NO such file {}".format(txt_file))
        exit(-1)
    context_set = []
    with open(txt_file, "r") as org:
        for i in org.readlines():
            context_set.append(i.strip())
    checked_set_jpg = []
    checked_set_png = []
    for t in context_set:
        image_file = top_path + t + ".jpg"
        if os.path.exists(image_file):
            line = t+"\n"
            checked_set_jpg.append(line)
        else:
            image_file = top_path + t + ".png"
            if os.path.exists(image_file):
                line = t + "\n"
                checked_set_png.append(line)
    print("lenth org :{} ({}), lenth jpg :{} ({})".format(len(context_set), context_set[0], len(checked_set_jpg), checked_set_jpg[0]))
    print("if {} ?= {} <- jpg{} + png{}".format(len(context_set), (len(checked_set_jpg)+len(checked_set_png)), len(checked_set_jpg), len(checked_set_png)))
    with open(txt_file.replace("train_data", "train_only_jpg"), "w") as t:
        print("writting...")
        t.writelines(checked_set_jpg)

    with open(txt_file.replace("train_data", "train_only_png"), "w") as t:
        print("writting...")
        t.writelines(checked_set_png)


def clean_image_format_RBG():
    from PIL import Image
    image_path = "/home/leon/opt-exprements/coco/train2017/" #"/home/leon/mount_point_c/datas_jinglianwen/ABBY/JPEGImages/TRAIN/"
    txt_file = "/home/leon/opt-exprements/coco/ImageSets/Main/train.txt" #"ABBY/Annos/TRAIN/Main/trainval_only_jpg.txt"
    context_set = []
    with open(txt_file, "r") as org:
        for i in org.readlines():
            context_set.append(i.strip())
    option_count = 0
    for i in context_set:
        image_file = os.path.join(image_path, i + ".jpg")
        im = Image.open(image_file)
        if str(im.mode) == "RGB":
            continue
        else:
            im = im.convert("RGB")
            im.save(image_file)
            option_count += 1
            print("RGB Covered {} ".format(image_file))
    print("Optioned {} of {}".format(option_count, len(context_set)))


if __name__ == '__main__':
    """
    脚本说明：
        本脚本用于将yolo格式的标注文件.txt转换为voc格式的标注文件.xml
    参数说明：
        anno_path:标注文件txt存储路径
        save_path:json文件输出的文件夹
        image_path:图片路径
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--anno-path', type=str, default='./data/labels/yolo', help='yolo txt path')
    parser.add_argument('-s', '--save-path', type=str, default='./data/convert/voc', help='xml save path')
    parser.add_argument('--image-path', default='./data/images')

    opt = parser.parse_args()

    if len(sys.argv) > 1:
        print(opt)
        parseXmlFilse(**vars(opt))
        print("image nums: {}".format(images_nums))
        print("category nums: {}".format(category_nums))
        print("bbox nums: {}".format(bbox_nums))
    else:
        data_path = "/home/leon/mount_point_c/datas_jinglianwen/ABBY/"
        data_path = "/home/leon/mount_point_c/datas_jinglianwen/ABBY/"
        anno_path = data_path + 'labels/TRAIN/'
        image_path = data_path + 'JPEGImages/TRAIN/' #20220720_dog/data-4/gray/cam0
        save_path = data_path + 'Annos/ForTRAIN/'
        # write_imagesets_txt(data_path + 'Annos/TRAIN/')
        # check_endless()
        clean_image_format_RBG()
        exit(0)
        parseXmlFilse(image_path, anno_path, save_path)
        print("image nums: {}".format(images_nums))
        print("category nums: {}".format(category_nums))
        print("bbox nums: {}".format(bbox_nums))
