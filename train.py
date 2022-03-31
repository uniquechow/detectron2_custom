#https://www.youtube.com/watch?v=GoItxr16ae8&list=PLUE9cBml08ygkbShjPzCQQy_TkSDC-d1T

from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
import os
import pickle
from utils import *

# 在detectron2 modelzoo中选择模型，
config_file_path = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
checkpoint_url = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'

# config_file_path = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'
# checkpoint_url = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'


output_dir = './output/object_detection'
num_classes = 3  #define model num_classes
input_size = [480, 480]   # [train, val]

device = 'cuda'

#====def datasets path====#
train_dataset_name = '3class_train'
train_images_path = './coco/train2014'  #训练集图片路径
train_json_annot_path = './coco/annotations/instances_train2014.json' # 训练集json文件路径

test_dataset_name = '3class_test'
test_images_path = './coco/val2014'
test_json_annot_path = './coco/annotations/instances_val2014.json'
#=========================#

cfg_save_path = 'OD_cfg.pickle'  # 用于保存训练时的配置，在test可以直接导入

# 在detectron2注册datasets
register_coco_instances(name=train_dataset_name, metadata={}, json_file=train_json_annot_path, image_root=train_images_path)
register_coco_instances(name=test_dataset_name, metadata={}, json_file=test_json_annot_path, image_root=test_images_path)

plot_samples(dataset_name=train_dataset_name, n=2) # 检查datasets是否有误

##
def main():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name,
                        test_dataset_name, num_classes, device, output_dir, input_size)
    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)  # 保存cfg文件

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()

if __name__ == '__main__':
    main()
