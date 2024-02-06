import random
import time
import torch
from config import Config
from onnxsim import simplify
import onnx as OX
import onnxruntime as OXRT
import numpy as np
from utils.utils import logg_init_obj
from utils.utils import network_choice

MODEL_TOP_PATH = "check_points"
# "/home/leon/mount_point_two/data-od"  #
PT_FORMAT_MODEL = '{}/model_last.pt'.format(MODEL_TOP_PATH)
# '{}/mobilenet_v2-b0353104.pth'.format(MODEL_TOP_PATH)  #
ONNX_ORG_FORMAT_MODEL = '{}/model_last.onnx'.format(MODEL_TOP_PATH)
ONNX_SIM_FORMAT_MODEL = "{}/simplified_last.onnx".format(MODEL_TOP_PATH)
NET_STRUCT = network_choice()


def convert_RGB_mode(image):
    rgb_image = image
    if image.mode != "RGB":
        rgb_image = image.convert("RGB")
        image_mode = rgb_image.mode
        num_channels = len(image_mode)
        print("convert to RGB,{} ,{}".format(rgb_image.mode, num_channels))
    return rgb_image


def gen_onnx_by_define(net, input_shapes, total_file_path=None):
    if total_file_path is not None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # pytorch模型加载
        net = torch.load(total_file_path, map_location=device)
    batch_size = 1  # 批处理大小
    # set the model to inference mode
    net.eval()
    ins = []
    input_names = []
    index_of_name = 1
    for x in input_shapes:
        ins.append(torch.randn(batch_size, *x))
        input_names.append("x_{}".format(index_of_name))
        index_of_name += index_of_name
    dummy_input = tuple(input_names)
    torch.onnx.export(net, dummy_input, ONNX_ORG_FORMAT_MODEL, input_names=input_names, verbose=True)
    OX.save(OX.shape_inference.infer_shapes(OX.load(ONNX_ORG_FORMAT_MODEL)), ONNX_ORG_FORMAT_MODEL)


def export_from_best_pt(net, xq, xt, use_cpu=True, file_path=PT_FORMAT_MODEL):

    if use_cpu:
        checkpoint = torch.load(file_path, map_location="cpu")
    else:
        checkpoint = torch.load(file_path)
    # if 'total' in file_path or 'last' in file_path:
    #     checkpoint = checkpoint['model']
    checkpoint = checkpoint['model']
    net.load_state_dict(checkpoint)
    if not use_cpu:
       net.cuda()
    net.eval()
    dummy_input = (xq, xt)
    input_names = ['x2', 'x1']
    output_names = ['delta_pred', 'conf_pred']
    torch.onnx.export(net, dummy_input, ONNX_ORG_FORMAT_MODEL, input_names=input_names, output_names=output_names, verbose='True')
    OX.save(OX.shape_inference.infer_shapes(OX.load(ONNX_ORG_FORMAT_MODEL)), ONNX_ORG_FORMAT_MODEL)


def export_by_net_define(use_cpu=True):

    # net = MobileNetV2()
    from torchvision.models import MobileNetV2
    net = MobileNetV2()

    # net = NET_STRUCT
    net.eval()
    dummy_input = torch.randn(1, 3, Config.imq_h, Config.imq_w)
    input_names = ['x2']
    output_names = ['output']
    torch.onnx.export(net, dummy_input, ONNX_ORG_FORMAT_MODEL, input_names=input_names, output_names=output_names, verbose='True')
    OX.save(OX.shape_inference.infer_shapes(OX.load(ONNX_ORG_FORMAT_MODEL)), ONNX_ORG_FORMAT_MODEL)
    dummy_input = torch.randn(1, 3, Config.im_h, Config.im_w)
    input_names = ['x1']
    new_file_name = ONNX_ORG_FORMAT_MODEL.split(".onnx")[0] + "_448" + ".onnx"
    torch.onnx.export(net, dummy_input,  new_file_name, input_names=input_names, output_names=output_names,
                      verbose='True')
    OX.save(OX.shape_inference.infer_shapes(OX.load(new_file_name)), new_file_name)


def simplified_identify_node(onnx_path=ONNX_ORG_FORMAT_MODEL):
    onnx_model = OX.load(onnx_path)
    # 简化模型
    simplified_model, check = simplify(onnx_model)
    # 保存简化后的模型
    OX.save_model(simplified_model, ONNX_SIM_FORMAT_MODEL)
    OX.save(OX.shape_inference.infer_shapes(OX.load(ONNX_SIM_FORMAT_MODEL)), ONNX_SIM_FORMAT_MODEL)
    print("simplified_model.")


def convert_onnx():
    logg_init_obj("convert_onnx.txt")
    # 设置示例输入
    input_query = torch.randn(1, 3, Config.imq_h, Config.imq_w)
    # (3, Config.imq_h, Config.imq_w) # torch.randn(1, 3, Config.imq_h, Config.imq_w)
    input_scnce = torch.randn(1, 3, Config.im_h, Config.im_w)
    # (3, Config.im_h, Config.im_w) # torch.randn(1, 3, Config.im_h, Config.im_w)
    # 将模型导出为 ONNX 格式
    # torch.onnx.export(model, input_query, input_sence, 'checkpoint/onnx_model.onnx')  #
    net = NET_STRUCT
    export_from_best_pt(net, input_query, input_scnce)
    print("convert to onnx")


def inference_test(onnx_path=ONNX_SIM_FORMAT_MODEL, pt_path=PT_FORMAT_MODEL):

    # OXRT
    xq = torch.randn(1, 3, Config.imq_h, Config.imq_w)
    xt = torch.randn(1, 3, Config.im_h, Config.im_w)
    inputs = (xq, xt)
    infer_ = OXRT.InferenceSession(onnx_path)
    inputs_name = infer_.get_inputs()
    output_name = infer_.get_outputs()
    print("infer inputs -> {}, output -> {}".format(inputs_name, output_name))
    onnx_rst = infer_.run([output_name[0].name, output_name[1].name], {inputs_name[0].name: xq.numpy(), inputs_name[1].name: xt.numpy()})
    n0 = onnx_rst[0]
    n1 = onnx_rst[1]
    # pytorch
    # net = DarkJNN()
    net = NET_STRUCT
    checkpoint = torch.load(pt_path, map_location="cpu")
    net.load_state_dict(checkpoint['model'])
    # net.cuda()
    net.eval()
    torch_rst = net(xq, xt, [])
    t0 = torch_rst[0].detach().numpy()
    t1 = torch_rst[1].detach().numpy()
    print(np.allclose(n0, t0, rtol=1e-03, atol=1e-05))
    print(np.allclose(n1, t1, rtol=1e-03, atol=1e-05))
    print("test finished.")


def export_model_struct(net, save_file="model_struct.txt"):
    with open(save_file, 'w') as f:
        # 打印模型到model.txt
        print(net, file=f)
        # 打印模型参数
        for params in net.state_dict():
            f.write("{}\t{}\n".format(params, net.state_dict()[params]))
        f.write("****************{}****************\n".format(time.time()))


def estimate_flops_params():
    logg_init_obj("net_flops_params.txt")
    from thop import profile
    from thop import clever_format
    from torchstat import stat
    net = NET_STRUCT  # 定义好的网络模型
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of params {:3f}M".format(total / 1e6))
    export_model_struct(net)
    xq = torch.randn(1, 3, Config.imq_h, Config.imq_w)
    xt = torch.randn(1, 3, Config.im_h, Config.im_w)
    flops, params = profile(net, (xq, xt, [],))
    print('[thop]flops(M): {}(25548.25728), params(M): {}(52.102073)'.format(flops / 1e6, params / 1e6))
    flops, params = clever_format([flops, params], "% .3f")
    print('[thop]flops: ', flops, 'params: ', params)
    from torchsummary import summary
    summary(net.cuda(), input_size=[(3, Config.imq_h, Config.imq_w), (3, Config.im_h, Config.im_w)]) #, device='cpu'

    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(net, ((3, Config.imq_h, Config.imq_w), (3, Config.im_h, Config.im_w)),
    #                                          as_strings=True, print_per_layer_stat=True)
    # print('{:<30}  {:<8}'.format('Computational complexity(macs): ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters(params): ', params))

"""
    input: "/Sigmoid_output_0"
input: "/Exp_output_0"
output: "delta_pred"
name: "/Concat_4"
op_type: "Concat"
attribute {
  name: "axis"
  type: INT
  i: -1
}
    """


def edit_onnx(file="check_points/edit-onnx/simplified_best.onnx"):
    onnx_model = OX.load(file)
    graph = onnx_model.graph
    node = graph.node
    index = 0
    for n in node:
        if "Concat_4" in n.name:
            print(n)
            break
        index += 1

    old_scale_node = node[index]
    node_name = old_scale_node.name
    print("if No.{} node {} == /Concat_4".format(index, node_name))
    attr = OX.helper.make_attribute('axis', 2)  # 添加属性
    new_scale_node = OX.helper.make_node(
        name=node_name,
        op_type="Concat",
        inputs=["/Sigmoid_output_0", "/Exp_output_0"],
        outputs=['delta_pred']
    )
    graph.node.remove(old_scale_node)
    graph.node.insert(index, new_scale_node)
    node[index].attribute.insert(0, attr)
    node[index].name = node_name
    new_scale_node = node[index]
    print("new one:{}".format(new_scale_node))
    OX.checker.check_model(onnx_model, full_check=True)
    OX.save(onnx_model, 'check_points/edit-onnx/e190_d19_cat_edit.onnx')


def gen_qual_input_txt(every_picked_num=10,
                       templete_img_top_path='/home/leon/opt-exprements/expments/data_test/template_match/template_data/my_test_data/',
                       scene_path="/home/leon/opt-exprements/expments/data_test/template_match/match_data/",
                       cp_top_path="/home/leon/mount_point_c/acuity-toolkit-binary-6.24.7/JNN/"):
    import os
    q_names = ['bin', 'ac_remote', 'cellphone', 'desk', 'earphones', 'glasses', 'handchain', 'miehuoqi', 'ring',
               'shuiping', 'sliper', 'udisk', 'xiaofangxiang', 'yaoshi', 'yizi']
    cp_org_pair_scene_datas = list()
    cp_org_pair_tempfile_datas = list()
    cp_dst_pair_scene_datas = list()
    cp_dst_pair_tempfile_datas = list()
    txt_context_dst_scene_paths = list()
    txt_context_dst_tempfiles = list()
    for q in q_names:
        temp_file = os.path.join(templete_img_top_path, q)
        temp_file = "{}.jpg".format(temp_file)
        if not os.path.isfile(temp_file):
            q_names.remove(q)
            continue
        test_img_top_path = os.path.join(scene_path, q)
        sence_imgs = os.listdir(test_img_top_path)

        if len(sence_imgs) < 1:
            q_names.remove(q)
            continue
        random.shuffle(sence_imgs)
        current_cp_scene_files = random.sample(sence_imgs, min(every_picked_num, len(sence_imgs)))
        sence_file_names = current_cp_scene_files[:]
        for i in range(0, len(current_cp_scene_files)):
            current_cp_scene_files[i] = os.path.join(test_img_top_path, current_cp_scene_files[i])
        cp_org_pair_scene_datas.append(current_cp_scene_files)
        cp_org_pair_tempfile_datas.append(temp_file)

        current_top_path = os.path.join(cp_top_path, "data/")
        if not os.path.exists(current_top_path):
            os.mkdir(current_top_path)
        current_top_path = os.path.join(current_top_path, q)
        if not os.path.exists(current_top_path):
            os.mkdir(current_top_path)
        current_top_path = "{}/".format(current_top_path)
        for i in range(0, len(sence_file_names)):
            sence_file_names[i] = os.path.join(current_top_path, sence_file_names[i])
        cp_dst_pair_scene_datas.append(sence_file_names)
        dst_temp_file = temp_file.replace(templete_img_top_path, "{}/".format(os.path.join(cp_top_path, "data")))
        cp_dst_pair_tempfile_datas.append(dst_temp_file)
        for i in range(0, len(sence_file_names)):
            s_file_in_txt = sence_file_names[i]
            txt_context_dst_scene_paths.append(s_file_in_txt.replace(current_top_path, "./data/{}/".format(q)))
            temp_file_in_txt = dst_temp_file.replace(cp_top_path, "./")
            txt_context_dst_tempfiles.append(temp_file_in_txt)

    if len(cp_org_pair_scene_datas) != len(cp_dst_pair_scene_datas):
        exit(-1)
    if len(txt_context_dst_scene_paths) != len(txt_context_dst_tempfiles):
        exit(-2)
    for i in range(0, len(cp_dst_pair_scene_datas)):
        if len(cp_dst_pair_scene_datas[i]) != len(cp_dst_pair_scene_datas[i]):
            exit(-3)
        cmd_cp_templte_file = "cp {} {}".format(cp_org_pair_tempfile_datas[i], cp_dst_pair_tempfile_datas[i])
        print("-----{}---- will apply operation:{}".format(i, cmd_cp_templte_file))
        os.system(cmd_cp_templte_file)
        for j in range(0, len(cp_dst_pair_scene_datas[i])):
            cmd_cp = "cp {} {}".format(cp_org_pair_scene_datas[i][j], cp_dst_pair_scene_datas[i][j])
            print("-----{}----{}--- will apply operation:{}".format(i, j, cmd_cp))
            os.system(cmd_cp)

    with open(os.path.join(cp_top_path, "dataset_scene.txt"), "w") as f:
        for line in txt_context_dst_scene_paths:
            f.writelines("{}\n".format(line))
    with open(os.path.join(cp_top_path, "dataset_query.txt"), "w") as f:
        for line in txt_context_dst_tempfiles:
            f.writelines("{}\n".format(line))

    print("cp and txt finished. data files total count:{}".format(len(txt_context_dst_tempfiles)))

        #


def crop_and_mosaic(base_image, other_image_anns=[]):
    # other_image_anns : ['/0/1/2/a.xml', '']
    from PIL import Image
    import xml.etree.ElementTree as ET
    im = Image.open(base_image)
    for i, a in enumerate(other_image_anns, 0):
        shif_i = i + 1
        annotation = ET.parse(a)
        query_path = a.replace('/xml', '/image')
        query_path = query_path.replace(".xml", ".jpg")
        cropped_img = Image.open(query_path)
        boxes = []

        for obj in annotation.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            tlabel = obj.find('name').text.lower().strip()
            if tlabel == 'glasses':
                continue
            boxes.append([xmin, ymin, xmax, ymax])
        if len(boxes) < 1:
            continue
        cropped_img = cropped_img.crop(random.choice(boxes))
        im.paste(cropped_img, (max(shif_i * 50, (cropped_img.size[0] - cropped_img.size[0])),
                               max(shif_i* 50, (cropped_img.size[1] - cropped_img.size[1]))))
        print("debug")
    return im

def test_mosaic():
    import os
    background_path = "/home/leon/opt-exprements/expments/data_test/template_match/match_data/ac_remote/"
    background = os.listdir(background_path)[4]
    background = os.path.join(background_path, background)
    base_xml_path = '/home/leon/opt-exprements/expments/data_test/data_0118/xml'
    xml_paths = os.listdir(base_xml_path)
    xml_files = []
    for p in xml_paths:
        xmls = os.listdir(os.path.join(base_xml_path, p))
        xml_files.append(os.path.join(base_xml_path, p, random.choice(xmls)))

    mosaic_img = crop_and_mosaic(background, xml_files)
    mosaic_img.save("mosaic_image_no_glasses.jpg")
    mosaic_img.show()

def main():
    # edit_onnx()
    # convert_onnx()
    # simplified_identify_node()
    # inference_test(onnx_path="check_points/edit-onnx/e190_d19_cat_edit.onnx",
    #                pt_path="check_points/coco_voc199epoch/model_best_total.pt")
    # estimate_flops_params()
    # export_by_net_define()
    # gen_qual_input_txt()
    test_mosaic()
    pass

if __name__ == "__main__":
    main()
