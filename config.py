class Config:

    ####### Datasets
    dataset = "coco"  # {VOC, coco, coco_split, other}
    # paths to [other] datasets
    training_dir = "/home/leon/opt-exprements/coco/train2017/"
    testing_dir = "/home/leon/opt-exprements/coco/test2017/"
    annotations_dir = "/home/mmv/Documents/3.datasets/openlogo/Annotations/"
    # path to VOC
    voc_dataset_dir = "/home/leon/opt-exprements/expments/vocdevkit/"
    # path to COCO
    coco_dataset_dir = "/home/leon/opt-exprements/coco/"
    coco_split = 4  # Defines the split to test for the VOC split experiment
    network_type = "darknet19"  # darknet19 mobile_net_v2 resnet18
    ####### Model params
    batch_size = 8
    epochs = 2
    lr = 0.0001
    decay_lrs = {60: 0.00001, 90: 0.000001}
    weight_decay = 0.0005
    momentum = 0.9

    num_workers = 8

    im_w = 416 #416 414
    im_h = 416 #416
    imq_w = 208 #208 206
    imq_h = 208 #208

    continue_training = False
    log_of_train = False
	letter_box_for_query_img = False  # if enabled, g-box need changed
    thresh = .6

    jitter = 0.3

    saturation = 1.5
    exposure = 1.5
    hue = .1

    strides = 32

    ####### Model save/load path
    best_model_path = "check_points/model_best"
    model_path = "check_points/model_last"
    model_endless = ".pt"
    anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]

    ####### testing
    conf_thresh = 0.5
    nms_thresh = 0.45

    # path to mAP repository
    mAP_path = "/home/leon/opt-exprements/expments/JNN_detection/mAP/"

    ####### Loss
    object_scale = 5
    noobject_scale = 1
    class_scale = 1
    coord_scale = 1

