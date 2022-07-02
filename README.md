# Training Framework

A generalised training framework consisting of all kinds of models like Yolo, RCNN etc. with default model configurations.

## Description

This repo will walk you from installation, to training, evaluating and testing your model with a custom dataset on various state-of-the-art object-detection models like Faster R-CNN, YOLO, SSD and mask RCNN for segmentation. Exporting model weights for inference on differnt frameworks like tensorflow, pytorch and darknet


## Darknet
Download the yolov3 weights in Darknet dir:
`wget https://pjreddie.com/media/files/yolov3.weights`

Make Sure in `Makefile` 'gpu == 1'.
From darknet dir run `make`

For Testing Darknet:
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg

###  Custom Training,
Copy "yolov3.cfg" file from cfg to custom_data/cfg dir, and rename to `yolov3-custom.cfg`

## Making changes in yolov3-custom.cfg file:

The maximum number of iterations for which our network should be trained is set with the param `max_batches=4000`. Also update `steps=3200,3600` which is 80%, 90% of max_batches, you can set value based on your training requirements.

classes param in the yolo layers to based on number of classes you are workning with like for one or `2 class` at `line numbers: 610, 696, 783`.

Similarly we will need to update the filters param based on the classes count `filters=(classes + 5) * 3`. 
For a single class we should set `filters=18` at `line numbers: 603, 689, 776`.

## Updating custom_data dir,
### Updating "custom.names" file : Mention all class name,
### Updating "detector.names" file : 
`classes=1`
train=custom_data/train.txt //Path to text file of images path for training.
valid=custom_data/test.txt // Path to text file of images path for testing.
names=custom_data/custom.names //Path to the class names
backup=backup/ //path to save weights

`Test.txt` need to store the path of each image used for testing
`Train.txt`  need to store the path of each image used for training

## Command to initialise training, 
```bash
./darknet detector detector train custom_data/detector.data custom_data/cfg/yolov3-custom.cfg yolov3.weights
```
## Evaluating your taining,
```bash 
./darknet detector test custom_data/detector.data custom_data/cfg/yolov3-custom.cfg backup/yolov3_final.weights -ext_output -out eval.json < eval.txt
./darknet detector map data/obj.data custom_data/cfg/yolov3-custom.cfg backup/yolov3_final.weights
./darknet detector recall data/obj.data custom_data/cfg/yolov3-custom.cfg backup/yolov3_final.weights
```
#note eval.json will store all the output bounding box for each input image path stored in the eval.txt, //eval.txt will be prepared exactly like `test.txt/train.txt`


## Tensorflow object detection api

## Requirements

-Python 3.8.1, Tensorflow-gpu 2.4.1 and cuda 11.5
-Tensorflow repositories: [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [Tensorflow object detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

## Installation
- You can check the compatible versions of any tensorflow version with cuda and cudnn versions from [here](https://www.tensorflow.org/install/source#tested_build_configurations), install tensorflow-gpu using following command.
```bash 
pip install tensorflow-gpu==2.2.0
```
- After that, you should install the Object Detection API.
```bash 
pip install object-detection
```
- Then proceed to the python package installation as follows:
cd models/research
compile protos:
```bash
protoc object_detection/protos/*.proto --python_out=.
```
Install TensorFlow Object Detection API as a python package:
```bash
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```
To run the examples in this repo, you will need some additional dependencies:
```bash
# install OpenCV python package
pip install opencv-python
pip install opencv-contrib-python
```
## Pretrained models
You can download model weights of your choice from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Unzip it to the models dir.

- [faster_rcnn_inception_v2](https://drive.google.com/open?id=1LRCSWIkX_i6ijScMfaxSte_5a_x9tjWF)
- [faster_rcnn_resnet_101](https://drive.google.com/open?id=15OxyPlqyOOlUdsbUmdrexKLpHy1l5tP9)
- [ssd_mobilenet_v1](https://drive.google.com/open?id=1U31RhUvE1Urr5Q92AJynMvl-oFBVRxxg)

## Preparing data for initializing training.

-if annotation in XML format.
* Converting XML to CSV format using 'xml_to_csv.py' in data_gen dir. 
train : 
```bash 
python3 xml_to_csv.py --annot_dir data_images/train --out_csv_path train_labels.csv
```
test : 
```bash 
python3 xml_to_csv.py --annot_dir data_images/test --out_csv_path test_labels.csv
```
-if annotation in json format.
* Converting json to xml format using 'labelme2voc.py' in data_gen dir.
```bash 
labelme data_annotated --labels labels.txt --nodata --autosave
```
* Genrating TFrecords using 'generate_tfrecord.py' in data_gen dir.
train : 
```bash 
python3 generate_tfrecord.py --path_to_images data_images/train --path_to_annot train_labels.csv --path_to_label_map fruit.pbtxt --path_to_save_tfrecords train.records
```
test : 
```bash 
python3 generate_tfrecord.py --path_to_images data_gen/data_images/test --path_to_annot data_gen/test_labels.csv --train  path_to_label_map data_gen/fruit.pbtxt --path_to_save_tfrecords data_gen/test.records
```
* Download the corresponding original config file from [here](https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2). eg [ssd_mobilenet_v2_320x320_coco17_tpu-8.config](https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/ssd_mobilenet_v2_320x320_coco17_tpu-8.config) and make the following changes based on your use case:
* Used `num_classes: 1` based on number of classes, instead of 90 classes in coco dataset.
* Changed `fine_tune_checkpoint_type: "classification"` to `fine_tune_checkpoint_type: "detection"` as we are using the pre-trained detection model as initialization.
* Added the path of the pretrained model in the field `fine_tune_checkpoint:`, for example using the mobilenet v2 model `fine_tune_checkpoint: "../models/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"`  
* Changed `batch_size: 512` to a reasonable number based on GPU memory like `batch_size: 16`
* Added the maximum number of training iterations in `num_steps:`, and also used the same number in `total_steps:`
* Adapted the learning rate to our model and batch size (originally they used higher learning rates because they had bigger batch sizes). This values needs some testing and tuning:
    ``` 
    cosine_decay_learning_rate {
        learning_rate_base: 0.025
        total_steps: 3000
        warmup_learning_rate: 0.005
        warmup_steps: 100 }
    ```
* The `label_map_path:` should point labelmap file `label_map_path: "labelmap.pbtxt"`
* You need to set the `tf_record_input_reader` under both `train_input_reader` and `eval_input_reader`. This should point to the tfrecords we generated (one for training and one for validation).
    ```
    train_input_reader: {
        label_map_path: "labelmap.pbtxt"
        tf_record_input_reader {
            input_path: "train.record"
        }
    }
    ``` 
* Prepare the labelmap according to your data, the [labelmap file](models/raccoon_labelmap.pbtxt) contains:

```
item {
  id: 1
  name: 'class name'
}
```
## Initialize training
* Allocating gpu memory in `model_main_tf2.py` based on gpu memory you want to utilise for training the model. 
`config.gpu_options.per_process_gpu_memory_fraction=0.2`
* Once configuration file is prepared, initialize training using following commands:
```bash 
python3 model_main_tf2.py --pipeline_config_path ssd_mobilenet_v2.config --model_dir model/train/ --alsologtostderr
```
* Evaluating Model performance : 
```bash
python3 model_main_tf2.py --pipeline_config_path ssd_mobilenet_v2.config --model_dir model/ssd_mobilenet_v2/train --checkpoint_dir model/ssd_mobilenet_v2/train
```
Note that running the evaluation script along with the training requires another GPU dedicated for the evaluation. So, if you don't have enough resources, you can ignore running the validation script, and run it only once when the training is done. However, you can run the evaluation on the CPU, while the training is running on the GPU. Simply by adding this flag before running the evaluation script `export CUDA_VISIBLE_DEVICES="-1"`, which makes all the GPUs not visible for tensoflow, so it will use the CPU instead.
* Visualising model on Tensorboard :
```bash
tensorboard --port 6004 --logdir=.
```
## Exporting your trained model for inference
When the training is done, Tensorflow saves the trained model as a checkpoint. Now we will see how to export the models to a format that can be used for inference, this final format usually called saved model or frozen model.
```bash
python3 exporter_main_v2.py --input_type="image_tensor" --pipeline_config_path=ssd_mobilenet_v2.config --trained_checkpoint_dir=model/ssd_mobilenet_v2/train --output_directory=model/ssd_mobilenet_v2/weights
```
The weights directory will contain 2 folders; saved_model and checkpoint. The saved_model directory contains the frozen model, and that is what we will use for inference. The checkpoint contains the last checkpoint in training, which is useful if you will use this model as pretrained model for another training in the future, similar to the checkpoint we used from the pretrained model with coco dataset.

## Setup for inferencing different models

- Clone this git repo, go to `Tensorflow -> Models -> Reaseach -> Object_detection dir`, and Download the object_detection.
- Make folder by name `Model` and download the weight files given above in this dir.
- Make folder by name `images`, where you can place all the test images for inference.
- Run `Object_detection` notebook for inferencing the images.
- Inference without tensorflow visualization utils dependency for drawing bounding boxes we can use `object_detection_pil.py` python script which uses PIL draw bounding boxes.  

## PyTorch

## Requirements

- Python == 3.8.10
- Torch == 1.8.1
- Torchvision == 0.9.1
- Detectron2 == 0.6
- OpenCV

## Installation
- You can check the compatible versions of any pytorch version with cuda and cudnn versions from [here](https://pytorch.org/get-started/previous-versions/)
- Clone detectron2 repo from [here](https://github.com/facebookresearch/detectron2)

## Preparing data for initializing training.

- Test trian split use `test_train_split.ipynb` you can split dataset in train, test, val in different ratios based on your requirments.
- For converting csv to coco you can use `csv_to_coco.ipynb` in data_preprocessing dir.
- For converting json to coco you can refer [here](https://github.com/fcakyon/labelme2coco)

- Visualizing data feeding to detectron2. using `plot_samples` function in `train.py`.
- `config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"` for segmentation. [dectron2 weights](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
- `output_dir = "output_weights/image_segmentation"` dir to save our custom object detection model

- `num_classes = 1` define number of classes

- `device = "cuda" # "cpu"`

- Path to trian dir.
```
train_dataset_name = "document_train"
train_images_path = "card_data/card_images"
train_json_annot_path = "card_data/train.json"
```

- Path to test dir.
```
test_dataset_name = "document_test"`
test_images_path = "card_data/card_images"
test_json_annot_path = "card_data/test.json"
```

- `cfg_save_path = "IS_cfg.pickle"`
- get_train_cfg function you can define max number iterations.

- ` python3 train.py`

## Inferencing images
- define test image path in `test.py` and run `python3 test.py`
