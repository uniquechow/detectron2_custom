# just for predictions
from detectron2.engine import DefaultPredictor
import os
import pickle
from libs.plots import on_image
from utils import *

cfg_save_path = 'OD_cfg.pickle'  #导入配置参数

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

# 加载权重
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

# image_path = './coco/val2014/c1.jpg'   # 选择其中一张test图片作为demo
image_path = './coco/ccc1.png'
line_thickness = 3
names = ['basket', 'orange', 'fullbasket']  # 类别名字，可在annotations的json最后找

on_image(image_path, predictor, names=names, line_thickness=line_thickness)

