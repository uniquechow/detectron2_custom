from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

register_coco_instances("3class_test", {}, "./coco/annotations/instances_val2014.json", "./coco/val2014")

config_file_path = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'


cfg = get_cfg()
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
cfg.OUTPUT_DIR = './output/object_detection'

# cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
# cfg.merge_from_file("configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml")

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("3class_test", cfg, False, output_dir="./output/eval")
val_loader = build_detection_test_loader(cfg, "3class_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))