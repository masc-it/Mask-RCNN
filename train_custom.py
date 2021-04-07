import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import glob
import argparse

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
class CustomConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 30

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = CustomConfig()
config.display()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
    
class CustomDataset(utils.Dataset):
    
    w = 128
    h = 128
    
    def load_imgs(self, base_dir):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "food")
        
        img_path = os.path.join(base_dir, "imgs")
        
        mask_path = os.path.join(base_dir, "masks")

        for i, filename in enumerate(glob.glob(img_path + '/*.jpg')):
            
            name = os.path.splitext(os.path.basename(filename))[0]
            
            # img = imread(filename)
            # h,w = img.shape[:2]
            self.add_image("shapes", image_id=i, path=filename, class_="food",
                           mask=os.path.join(mask_path, name + ".jpg"))


    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        orig_img = cv2.imread(info["path"])
        
        # resized_img = cv2.resize(orig_img, (self.w, self.h))
        
        return orig_img

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["class_"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask_path = info['mask']
        
        masks = []
        
        orig_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # resized_mask = cv2.resize(orig_mask, (self.w, self.h))
        
        masks.append(orig_mask)
        
        class_id = 1 # self.map_source_class_id(info['class_'])
                
        class_ids = np.array([class_id])
        
        mask = np.stack(masks, axis=2).astype(np.bool)
        return mask, class_ids.astype(np.int32)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Mask-RCNN custom train")
    parser.add_argument('-t', '--train', type=str, required=True, help='train folder')
    parser.add_argument('-v', '--val', type=str, required=True, help='test folder')
    
    args = parser.parse_args()

    # Training dataset
    dataset_train = CustomDataset()
    dataset_train.load_imgs(args.train)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_imgs(args.val) # "F:\\datasets\\food\\test")
    dataset_val.prepare()

    # Load and display random samples
    #image_ids = np.random.choice(dataset_train.image_ids, 4)
    #for image_id in image_ids:
    #    image = dataset_train.load_image(image_id)
    #    mask, class_ids = dataset_train.load_mask(image_id)
    #    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
                              
    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
        
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=3, 
                layers='heads')