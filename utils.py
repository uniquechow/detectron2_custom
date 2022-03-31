
from detectron2.data import DatasetCatalog, MetadataCatalog,build_detection_train_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg # 用于初始化参数
from detectron2 import model_zoo  # 用于加载模型
from detectron2.engine import HookBase
import detectron2.utils.comm as comm
import torch

import random
import cv2
import matplotlib.pyplot as plt
from libs.plots import plot_one_box, Colors

import json

def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
    # verfiy whether or not our annnotations are correctly picked up by detectron2
    for s in random.sample(dataset_custom, n): #读取n个来自cutsomdataset的数据
        img = cv2.imread(s['file_name'])
        # cv2 input: bgr; detectron2 input rgb
        v = Visualizer(img[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15, 20))
        plt.imshow(v.get_image())
        plt.show()

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name,
                  test_dataset_name, num_classes, device, output_dir, input_size):
    cfg = get_cfg()  # here is the model init config

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path)) # 模型结构文件
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url) # 模型权重文件
    cfg.DATASETS.TRAIN = (train_dataset_name,) # 训练集,前提：已经在detectron上register
    cfg.DATASETS.TEST = (test_dataset_name,)   # 测试集，前提：已经在detectron上register

    cfg.DATALOADER.NUM_WORKERS = 4  # 加载数据线程数

    cfg.SOLVER.IMS_PER_BATCH = 8   # BatchSize
    cfg.SOLVER.BASR_LR = 0.00025   # learning rate
    cfg.SOLVER.MAX_ITER = 1000     # iterations
    cfg.SOLVER.STEPS = []          # 学习率衰减iterations

    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = input_size[0]  # 训练图片输入的最大尺寸
    cfg.INPUT.MAX_SIZE_TEST = input_size[1]  # 测试数据输入的最大尺寸
    cfg.INPUT.MIN_SIZE_TRAIN = input_size[0]  # 训练图片输入的最小尺寸，可以设定为多尺度训练
    cfg.INPUT.MIN_SIZE_TEST = input_size[1]

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  #定义检测类别数
    cfg.MODEL.DEVICE = device      # cuda or cpu
    cfg.OUTPUT_DIR = output_dir

    return cfg



# detect on video , 需先丢该
def on_video(videoPath, predictor):
    cap = cv2.VideoCapture(videoPath)
    if (cap.isOpened()== False):
        print('error opening file...')
        return
    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:, :, ::-1], metadata={}, instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions['instances'].to('cpu'))

        cv2.imshow('Result', output.get_image()[:,:,::-1])
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break

# 用于计算训练中validation loss
class ValidationLoss(HookBase):
    def __init__(self, cfg, DATASETS_VAL_NAME):  # 多加一个DATASETS_VAL_NAME参数（小改动）
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = DATASETS_VAL_NAME  ##
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,
                                                 **loss_dict_reduced)











