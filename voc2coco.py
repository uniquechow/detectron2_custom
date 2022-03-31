# -*- coding=utf-8 -*-
#!/usr/bin/python
import sys
import os
import shutil
import numpy as np
import json
import xml.etree.ElementTree as ET
import mmcv
# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
PRE_DEFINE_CATEGORIES = {}

def get(root, name):
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def convert(xml_list, xml_dir, json_file):
    '''
    :param xml_list: 需要转换的XML文件列表    ['bc3.xml']
    :param xml_dir: XML的存储文件夹    str : './demo/voc/Annotations'
    :param json_file: 导出json文件的路径  str: './demo/coco/annotations/val2014.json'
    :return: None
    '''
    list_fp = xml_list
    image_id = 1
    # 标注基本结构
    json_dict = {"images":[],
                 "type": "detection",
                 "annotations": [],
                 "categories": []}

    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip() # 去除头尾空格， return str: 'bc55.xml'
        print(" Processing {}".format(line))
        # 解析XML
        xml_f = os.path.join(xml_dir, line)  # return str: './demo/voc/Annotations/bc55.xml'
        # 开始解析xml文件
        tree = ET.parse(xml_f)
        root = tree.getroot()
        filename = line.split('.')[0] + '.jpg'
        # 取出图片名字
        image_id+=1
        size = get_and_check(root, 'size', 1)
        # 图片的基本信息
        width, height = int(get_and_check(size, 'width', 1).text), int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename,
                 'height': height,
                 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)
        # 处理每个标注的检测框
        for obj in get(root, 'object'):
            # 取出检测框类别名称
            category = get_and_check(obj, 'name', 1).text
            # 更新类别ID字典
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin, ymin= int(get_and_check(bndbox, 'xmin', 1).text) - 1, int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax,ymax = int(get_and_check(bndbox, 'xmax', 1).text), int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width, o_height = abs(xmax - xmin),abs(ymax - ymin)
            annotation = dict()
            annotation['area'] = o_width*o_height
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, o_width, o_height]
            annotation['category_id'] = category_id
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            # 设置分割数据，点的顺序为逆时针方向
            annotation['segmentation'] = [[xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin]]
            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1

    # 写入类别ID字典
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    # 导出到json
    #mmcv.dump(json_dict, json_file)
    print(type(json_dict))
    json_data = json.dumps(json_dict)
    with open(json_file, 'w') as w:
        w.write(json_data)

def parsetxtfile(txt_path):
    txt_lists = open(txt_path).readlines()
    new_txtlist = []
    for i in range(0, len(txt_lists)):
        txt_list = txt_lists[i].strip()
        new_txtlist.append(txt_list + '.xml')
    return new_txtlist  # return:  list:  [bc3.xml, bc4.xml....]

def check_cocopath(rootpath):
    if not os.path.exists(os.path.join(root_path, 'coco/annotations')):
        os.makedirs(os.path.join(root_path, 'coco/annotations'))
    if not os.path.exists(os.path.join(root_path, 'coco/train')):
        os.makedirs(os.path.join(root_path, 'coco/train'))
    if not os.path.exists(os.path.join(root_path, 'coco/val')):
        os.makedirs(os.path.join(root_path, 'coco/val'))

if __name__ == '__main__':
    #===============仅需改该4个路径即可,建议绝对路径==============#
    # 原始voc格式jpg和xml
    root_path = ''  # 当前文件所在位置的目录，留空即可
    voc_jpg_path = '/home/chow/0_project/1_transporter_ai0721/datasets/3class_train_AB/images'
    voc_xml_path ='/home/chow/0_project/1_transporter_ai0721/datasets/3class_train_AB/Annotations'
    # 原始voc中split后的数据，在ImageSets中
    traintxt_path = '/home/chow/0_project/1_transporter_ai0721/datasets/3class_train_AB/ImageSets/Main/train.txt'  # 训练txt
    valtxt_path = '/home/chow/0_project/1_transporter_ai0721/datasets/3class_train_AB/ImageSets/Main/val.txt'      # 验证txt
    #=============================================#

    check_cocopath(root_path)

    xml_dir = os.path.join(voc_xml_path) #已知的voc的标注, str: ./demo/voc/Annotations
    xml_labels = os.listdir(xml_dir)  # 所有xml的list: ['bc3.xml', 'bc57.xml', 'bc62.xml',....]
    # 解析训练、验证的txt文件
    new_traintxtlist, new_valtxtlist= parsetxtfile(traintxt_path), parsetxtfile(valtxt_path)

    # validation data
    json_file = os.path.join(root_path, 'coco/annotations/val2014.json') # str: './demo/coco/annotations/instances_val2014.json'
    convert(new_valtxtlist, xml_dir, json_file)

    for xml_file in new_valtxtlist: # val data
        img_name = xml_file[:-4] + '.jpg'
        shutil.copy(os.path.join(voc_jpg_path, img_name),
                    os.path.join(root_path, 'coco/val', img_name))

    # train data
    json_file = os.path.join(root_path, 'coco/annotations/train.json')
    convert(new_traintxtlist, xml_dir, json_file)
    for xml_file in new_traintxtlist:  # train data
        img_name = xml_file[:-4] + '.jpg'
        shutil.copy(os.path.join(voc_jpg_path, img_name),
                    os.path.join(root_path, 'coco/train', img_name))