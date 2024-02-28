from test import Tester
from config import ConfigInit
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("--model_path", type=str, default="check_points/e70-voc-60-448/model_best.pt",
                        help="path to model file coco_voc199epoch-416 e60-rubby-16wjpg-448 ")
    parser.add_argument("--test_img_top_path", type=str,
                        default="/home/leon/opt-exprements/expments/data_test/template_match/template_data/my_test_data/",
                        help="path to template image path  rubby_views  my_test_data")
    parser.add_argument("--q_name", type=str, default="h-roi", help="name of object")
    parser.add_argument("--search_path", type=str,
                        default="/home/leon/opt-exprements/expments/data_test/template_match/rubby-test/0223/",
                        help="father path to match_data path, will add the q_name to gen final total dirname match_data rubby-test")
    parser.add_argument("--hist_option", type=bool, default=False, help="using hist preprocess when True")
    parser.add_argument("--rst_path", type=str,
                        default='/home/leon/opt-exprements/expments/data_test/template_match/backs/cc_448_0228',
                        help="father path of result, will  add the q_name and hist_option to gen final total dirname")
    parser.add_argument("--conf", type=str, default="0.5", help="conf")
    parser.add_argument("--nms", type=str, default="0.1", help="nms")
    args = parser.parse_args()
    return args


def main():
    # test()
    args = parse_args()
    Tester.test_one_OL(args.model_path, args.test_img_top_path, args.q_name, args.search_path, args.hist_option, args.rst_path, float(args.conf), float(args.nms))

    Tester.test_on_cross_cats(args.model_path, args.q_name, args.test_img_top_path, args.search_path, args.rst_path, conf=float(args.conf), nms=float(args.nms))
    # Tester.test_one_COCO()


if __name__ == "__main__":
    config_obj = ConfigInit()
    main()
