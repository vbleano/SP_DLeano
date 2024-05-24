try:
    import os
    import sys
    import json
    import datetime
    import numpy as np
    import cv2
    import tensorflow as tf
    import skimage.draw
    from mrcnn.visualize import display_instances, display_top_masks
    from mrcnn.utils import extract_bboxes
    import matplotlib.pyplot as plt
    import imgaug
    from mrcnn.config import Config
    from mrcnn import utils
    from mrcnn.model import MaskRCNN
    from PIL import Image, ImageDraw


except Exception as e:
    print(e)

class CocoLikeDataset(utils.Dataset):
    def load_data(self,annotation_json,images_dir):

        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()

        source_name="coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print("error")
                return
            self.add_class(source_name,class_id,class_name)

        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("warning")
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print(key)

                image_path = os.path.abspath(os.path.join(images_dir,image_file_name))
                image_annotations = annotations[image_id]

                self.add_image(
                    source = source_name,
                    image_id = image_id,
                    path = image_path,
                    width = image_width,
                    height = image_height,
                    annotations = image_annotations
                )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks=[]
        class_ids=[]


        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))

            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids


dataset_train = CocoLikeDataset()
dataset_train.load_data("Dataset/train/labels/All_Annotations.json", "Dataset/train")
dataset_train.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data("Dataset/train/labels/All_Annotations.json", "Dataset/train")
dataset_val.prepare()

dataset = dataset_train
image_ids = dataset.image_ids

#for testing
# for image_id in image_ids:
#     image = dataset.load_image(image_id)
#     mask, class_ids = dataset.load_mask(image_id)
#     # limit to total number of classes (ROOT, SHOOT, NODE, BACKGROUND)
#     display_top_masks(image, mask, class_ids, dataset.class_names, limit=4)
#     break

# image_id = 0
# image = dataset_train.load_image(image_id)
# mask,class_ids = dataset_train.load_mask(image_id)
# bbox = extract_bboxes(mask)
# display_instances(image,bbox,mask,class_ids,dataset_train.class_names)


class MetricTrackingConfig(Config):
    NAME = "MetricTracking_cfg_coco"
    NUM_CLASSES = 1 + 3
    STEPS_PER_EPOCH = 100

config = MetricTrackingConfig()
# config.display()

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR,"logs")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
tf.compat.v1.disable_eager_execution()

model = MaskRCNN(mode="training", config=config,model_dir="mask_rcnn_coco.h5")
# model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# model.train(dataset_train, dataset_train, learning_rate=config.LEARNING_RATE, epochs=25, layers='heads')