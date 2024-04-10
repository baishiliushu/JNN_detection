import os
import shutil
import argparse

def add_prefix_to_files(directory, prefix):
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否以'.jpg'结尾
        if filename.endswith('.jpg'):
            # 构建原始文件路径
            src_path = os.path.join(directory, filename)
            # 构建目标文件路径
            dest_path = os.path.join(directory, prefix + filename)
            # 重命名文件
            shutil.move(src_path, dest_path)
            print(f"Renamed '{filename}' to '{prefix + filename}'")


def cp_files_to_scene_dir(rubby_data_path, scene_dir):
    prefix_based_time = rubby_data_path.split("/")[-1]
    cam_path = os.path.join(rubby_data_path, "cam0")
    if not os.path.exists(cam_path):
        print(f"jpg path not exists {cam_path}")
        return
    for filename in os.listdir(cam_path):
        if filename.endswith('.jpg'):
            src_path = os.path.join(cam_path, filename)
            dest_path = os.path.join(scene_dir, prefix_based_time + filename)
            shutil.copy(src_path, dest_path)
            print(f"Copied '{filename}' to '{prefix_based_time  + filename}'")


def main():
    key_prefixes_in_line = ["RUBBY", "DATASET_SAVE", "DATE_ENDLESS"]
    key_values_conter = ["", "", "d"]

    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Add a prefix to all JPG files in a specified directory.")
    # 添加命令行参数
    parser.add_argument("maps_txt", type=str, default="/home/leon/mount_point_two/rubby-data-track/dl-need-scp-0409/maps.txt", nargs="?", help="map txt for Replacing JPG files.")
    # 解析命令行参数
    args = parser.parse_args()
    if args.maps_txt == "":
        print("empty maps_txt")
        exit(-1)
    if not os.path.exists(args.maps_txt):
        print(f"don't exists maps_txt {args.maps_txt}")
        exit(-1)
    with open(args.maps_txt, encoding="utf-8") as mf:
        lines = mf.readlines()
        for l, line in enumerate(lines, 0):
            line = line.strip()
            if len(line) == 0:
                print(f"No.{l} is empty line.")
                continue
            contexts = line.split(":")
            if len(contexts) < 2:
                continue
            if contexts[0] in key_prefixes_in_line:
                for i, kp in enumerate(key_prefixes_in_line, 0):
                    if contexts[0] == kp:
                        key_values_conter[i] += contexts[1]
            else:
                last_endless_by_line_count = max(l - 2, 0)
                dst_dir_name = key_values_conter[2] + f"n{last_endless_by_line_count}=" + contexts[0]
                dst_dir_name = os.path.join(key_values_conter[1], dst_dir_name)
                if not os.path.exists(dst_dir_name):
                    os.mkdir(dst_dir_name)
                dst_dir_name = os.path.join(dst_dir_name, "img")
                if not os.path.exists(dst_dir_name):
                    os.mkdir(dst_dir_name)

                src_dir_names = contexts[1].split(",")
                for src_n in src_dir_names:
                    src_sequences = src_n.split("|")
                    mid_base = src_sequences[0].split("/")
                    mid_base = src_sequences[0].replace(mid_base[-1], "")
                    for s in src_sequences:
                        _src = os.path.join(key_values_conter[0], mid_base, s.split("/")[-1])
                        cp_files_to_scene_dir(_src, dst_dir_name)


if __name__ == "__main__":
    main()

