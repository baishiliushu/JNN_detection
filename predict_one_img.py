from test import Tester
from config import ConfigInit
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="check_points/model_last.pt",
                        help="path to model file")
    parser.add_argument("--test_img_top_path", type=str,
                        default="/home/leon/opt-exprements/expments/data_test/template_match/template_data/my_test_data/",
                        help="path to template image path")
    parser.add_argument("--q_name", type=str, default="bin", help="name of object")
    parser.add_argument("--search_path", type=str,
                        default="/home/leon/opt-exprements/expments/data_test/template_match/match_data/",
                        help="father path to match image path, will add the q_name to gen final total dirname")
    parser.add_argument("--hist_option", type=bool, default=False, help="using hist preprocess when True")
    parser.add_argument("--rst_path", type=str,
                        default='/home/leon/opt-exprements/expments/data_test/template_match/tt',
                        help="father path of result, will  add the q_name and hist_option to gen final total dirname")
    parser.add_argument("--conf", type=str, default="0.3", help="conf")
    parser.add_argument("--nms", type=str, default="0.1", help="nms")
    args = parser.parse_args()
    return args


def main():
    # test()
    args = parse_args()
    Tester.test_one_OL(args.model_path, args.test_img_top_path, args.q_name, args.search_path, args.hist_option,
                       args.rst_path, float(args.conf), float(args.nms))
    # Tester.test_one_COCO()


if __name__ == "__main__":
    config_obj = ConfigInit()
    main()
