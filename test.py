# just for predictions
from detectron2.engine import DefaultPredictor
import os
import pickle

from utils import *

cfg_save_path = 'OD_cfg.pickle'

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

# 加载权重
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

image_path = './coco/val2014/c1.jpg'   # 选择其中一张test图片作为demo
on_image(image_path, predictor)

