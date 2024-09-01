import os
import sys
import argparse
import shutil
import math
from collections import OrderedDict

import json
import cv2

from labelme import utils


class Labelme2YOLO(object):
    def __init__(self, json_dir, output_dir, images_dir, to_seg=False):
        self._json_dir = json_dir
        self._output_dir = output_dir
        self._images_dir = images_dir
        self._label_id_map = self._get_label_id_map(self._json_dir)
        self._to_seg = to_seg

        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

    def _get_label_id_map(self, json_dir):
        label_set = set()
        for file_name in os.listdir(json_dir):
            if file_name.endswith('json'):
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                if 'shapes' not in data:
                    print(f"Warning: 'shapes' key not found in {json_path}. Skipping this file.")
                    continue
                if isinstance(data['shapes'], list):
                    shapes = data['shapes']
                else:
                    shapes = [data['shapes']]
                for shape in shapes:
                    label_set.add(shape['label'])
        return OrderedDict([(label, label_id) for label_id, label in enumerate(label_set)])

    def convert(self):
        json_names = [file_name for file_name in os.listdir(self._json_dir)
                      if os.path.isfile(os.path.join(self._json_dir, file_name)) and
                      file_name.endswith('.json')]

        for json_name in json_names:
            json_path = os.path.join(self._json_dir, json_name)
            json_data = json.load(open(json_path))

            if 'shapes' not in json_data:
                print(f"Warning: 'shapes' key not found in {json_path}. Skipping this file.")
                continue

            print(f'Converting {json_name} ...')

            img_path = self._generate_image_path(json_data, json_name)

            yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
            self._save_yolo_label(json_name, yolo_obj_list)

    def convert_one(self, json_name):
        json_path = os.path.join(self._json_dir, json_name)
        json_data = json.load(open(json_path))

        if 'shapes' not in json_data:
            print(f"Warning: 'shapes' key not found in {json_path}. Skipping this file.")
            return

        print(f'Converting {json_name} ...')

        img_path = self._generate_image_path(json_data, json_name)

        yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
        self._save_yolo_label(json_name, yolo_obj_list)

    def _get_yolo_object_list(self, json_data, img_path):
        yolo_obj_list = []
        img_h, img_w, _ = cv2.imread(img_path).shape
        if isinstance(json_data['shapes'], list):
            shapes = json_data['shapes']
        else:
            shapes = [json_data['shapes']]
        for shape in shapes:
            if shape['shape_type'] == 'circle':
                yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w)
            else:
                yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w)
            yolo_obj_list.append(yolo_obj)
        return yolo_obj_list

    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape['label']]
        obj_center_x, obj_center_y = shape['points'][0]
        radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 +
                           (obj_center_y - shape['points'][1][1]) ** 2)
        if self._to_seg:
            retval = [label_id]
            n_part = radius / 10
            n_part = int(n_part) if n_part > 4 else 4
            n_part2 = n_part << 1
            pt_quad = [None for i in range(0, 4)]
            pt_quad[0] = [[obj_center_x + math.cos(i * math.pi / n_part2) * radius,
                           obj_center_y - math.sin(i * math.pi / n_part2) * radius]
                          for i in range(1, n_part)]
            pt_quad[1] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[0]]
            pt_quad[1].reverse()
            pt_quad[3] = [[x1, obj_center_y * 2 - y1] for x1, y1 in pt_quad[0]]
            pt_quad[3].reverse()
            pt_quad[2] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[3]]
            pt_quad[2].reverse()
            pt_quad[0].append([obj_center_x, obj_center_y - radius])
            pt_quad[1].append([obj_center_x - radius, obj_center_y])
            pt_quad[2].append([obj_center_x, obj_center_y + radius])
            pt_quad[3].append([obj_center_x + radius, obj_center_y])
            for i in pt_quad:
                for j in i:
                    j[0] = round(float(j[0]) / img_w, 6)
                    j[1] = round(float(j[1]) / img_h, 6)
                    retval.extend(j)
            return retval
        obj_w = 2 * radius
        obj_h = 2 * radius
        yolo_center_x = round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)
        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _get_other_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape['label']]
        if self._to_seg:
            retval = [label_id]
            for i in shape['points']:
                i[0] = round(float(i[0]) / img_w, 6)
                i[1] = round(float(i[1]) / img_h, 6)
                retval.extend(i)
            return retval

        def __get_object_desc(obj_port_list):
            __get_dist = lambda int_list: max(int_list) - min(int_list)
            x_lists = [port[0] for port in obj_port_list]
            y_lists = [port[1] for port in obj_port_list]
            return min(x_lists), __get_dist(x_lists), min(y_lists), __get_dist(y_lists)

        obj_x_min, obj_w, obj_y_min, obj_h = __get_object_desc(shape['points'])
        yolo_center_x = round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
        yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)
        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _save_yolo_label(self, json_name, yolo_obj_list):
        txt_path = os.path.join(self._output_dir, json_name.replace('.json', '.txt'))
        with open(txt_path, 'w+') as f:
            for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
                yolo_obj_line = ""
                for i in yolo_obj:
                    yolo_obj_line += f'{i} '
                yolo_obj_line = yolo_obj_line[:-1]
                if yolo_obj_idx != len(yolo_obj_list) - 1:
                    yolo_obj_line += '\n'
                f.write(yolo_obj_line)

    def _generate_image_path(self, json_data, json_name):
        img_path_original = os.path.join(self._images_dir, json_data['imagePath'])
        if not os.path.exists(img_path_original):
            raise FileNotFoundError(f"Image not found: {img_path_original}")
        return img_path_original

    def _save_dataset_yaml(self):
        yaml_path = os.path.join(self._output_dir, 'dataset.yaml')
        with open(yaml_path, 'w+') as yaml_file:
            yaml_file.write('labels: %s\n' % self._output_dir)
            yaml_file.write('nc: %i\n\n' % len(self._label_id_map))
            names_str = ''
            for label, _ in self._label_id_map.items():
                names_str += "'%s', " % label
            names_str = names_str.rstrip(', ')
            yaml_file.write('names: [%s]' % names_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str,
                        help='Please input the path of the labelme json files.')
    parser.add_argument('--output_dir', type=str,
                        help='Please input the path where YOLO txt files will be saved.')
    parser.add_argument('--images_dir', type=str,
                        help='Please input the path of the images directory.')
    parser.add_argument('--val_size', type=float, nargs='?', default=0.1,
                        help='Please input the validation dataset size, for example 0.1 ')
    parser.add_argument('--json_name', type=str, nargs='?', default=None,
                        help='If you put json name, it would convert only one json file to YOLO.')
    parser.add_argument('--seg', action='store_true',
                        help='Convert to YOLOv5 v7.0 segmentation dataset')
    args = parser.parse_args(sys.argv[1:])
    convertor = Labelme2YOLO(args.json_dir, args.output_dir, args.images_dir, to_seg=args.seg)
    if args.json_name is None:
        convertor.convert()
    else:
        convertor.convert_one(args.json_name)
