from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import os
import pickle

from utils import *


config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" #Link to downloading coco pretrained weights
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

output_dir = "output_weights/image_segmentation" #dir to save our custom object detection model

num_classes = 1 # define number of classes

device = "cuda" # "cpu"

train_dataset_name = "document_train"
train_images_path = "card_data/card_images"
train_json_annot_path = "card_data/train.json"

test_dataset_name = "document_test"
test_images_path = "card_data/card_images"
test_json_annot_path = "card_data/test.json"

cfg_save_path = "IS_cfg.pickle"

#################################

register_coco_instances(name = train_dataset_name, metadata={}, json_file = train_json_annot_path, image_root = train_images_path) #to register our coco dataset

register_coco_instances(name = test_dataset_name, metadata={}, json_file = test_json_annot_path, image_root = test_images_path)

plot_samples(dataset_name=train_dataset_name, n =5 ) #to verify the output.

#################################

def main():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device,output_dir)
        
    with open(cfg_save_path, 'wb') as f: #saving the newly created config file for test
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)


    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True) #making output dir where model weights will be saved

    trainer = DefaultTrainer(cfg) #load default trainer
    trainer.resume_or_load(resume=False) #if you want to resume training from previous checkpoint change it into True
    
    trainer.train()
        
    evaluator = COCOEvaluator(test_dataset_name, cfg, False, output_dir)
    val_loader = build_detection_test_loader(cfg, test_dataset_name)
    print(inference_on_dataset(trainer.model, val_loader, evaluator))   
        
if __name__ == '__main__':
    main()
        
        

