import os
import numpy as np
import torch
import cv2

import importlib.resources as pkg_resources
from omegaconf import OmegaConf
from hydra import compose, initialize
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
import torch.nn.functional as F

from . import input_dictionary as dict

from clip_dinoiser.models.builder import build_model
from clip_dinoiser.segmentation.datasets.pascal_context import PascalContextDataset
from clip_dinoiser.helpers.visualization import mask2rgb
from clip_dinoiser import checkpoints
import clip_dinoiser.resources as resources

class clip_dinoiser_config:
    # init
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMG_FOLDER_PATH = '../ExampleImgs'
    
    # model setup
    CONFIG_NAME = 'COMPASS.yaml'
    CHECKPOINT_PATH = 'clip_dinoiser/checkpoints/last.pt'

    # tuning dials
    IMG_RESIZE = 1280

class clip_dinoiser_pipeline:
    # initialize variables
    def __init__(self, config, prompts):
        # copy over config
        self.config = config

        # initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # set up checkpoint
        self.checkpoint = resources.load_checkpoint('last.pt')

        # initialize prompts
        if isinstance(prompts, str):
            prompts = dict.word_bank(prompts)
        if len(prompts) == 1:
            prompts = ['background'] + prompts
        self.palette = dict.get_label_colors(prompts)
        self.prompts = prompts

        # initialize model
        cfg = resources.load_config(self.config.CONFIG_NAME)
        self.model = build_model(cfg.model, class_names=self.prompts)
        self.model.load_state_dict(self.checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.model.apply_found = 'background' in self.prompts

    # read in images given a folder path
    def load_images(self, folder_path):
        abs_folder_path = os.path.abspath(os.path.join(self.config.BASE_DIR, folder_path))
        image_paths = [os.path.join(abs_folder_path, f) for f in os.listdir(abs_folder_path) if f.endswith('.png')]
        return [(cv2.imread(p), os.path.basename(p)) for p in image_paths]

    # process images and produce mask segments
    def produce_masks(self, image, display_soft_edges=True):
        # image loading and preprocessing
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        w, h = img_pil.size
        img_resize = self.config.IMG_RESIZE
        img_pil = img_pil.resize((img_resize, img_resize))

        # create tensor based on image
        img_tens = T.ToTensor()(img_pil).unsqueeze(0).to(self.device)

        # extract mask segments
        with torch.inference_mode():
            output, dinoised_feats = self.model(img_tens, apply_softmax=False, get_features=True)
            output = output.cpu()
            dinoised_feats = dinoised_feats.cpu()

        # resize tensor to fit original image dimensions
        if display_soft_edges:
            output = F.interpolate(output, size=(h, w), mode="bilinear", align_corners=False)
        else:
            output = F.interpolate(output, size=(h,w), mode="nearest")

        # select the class with the highest score for each pixel
        output = output[0].argmax(dim=0)
        return output
        
    # display mask segments over original image
    def display_results(self, output, filename):
        mask = mask2rgb(output, self.palette)
        name = os.path.splitext(filename)[0]  # strip ".png" extension

        fig = plt.figure()
        plt.imshow(mask)
        plt.axis('off')
        save_path = os.path.join('./OutputImgs', f'{name}_output.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def display_labels(self):
        n = len(self.prompts)
        palette_array = np.array(self.palette).reshape(1, n, 3) / 255.0  # normalize colors for matplotlib

        fig, ax = plt.subplots(figsize=(max(8, n), 2))
        ax.imshow(palette_array, aspect='auto')
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(self.prompts, rotation=45, ha='right')
        ax.set_yticks([])
        ax.set_title("Labels")
        plt.tight_layout()

        save_path = os.path.join('./OutputImgs', 'labels.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def run(self, image_arr=None, folder_path=None, save_images=False):
        # parameter format evaluation        
        if image_arr is not None and folder_path is not None:
            raise ValueError("either a single image or folder of images must be provided, not both")

        # format images
        out_masks = [] # masks to be outputted
        if folder_path is not None:
            images = self.load_images(folder_path)
        else:
            images = [(image_arr, "singleshot")]
        
        # run primary processing loop
        for image, filename in images:
            # run CLIP DINOiser model
            masks = self.produce_masks(image, display_soft_edges=True)

            pred_masks = []
            for class_id in range(len(self.prompts)):
                pred_mask = (masks == class_id).numpy().astype(bool)
                pred_masks.append(pred_mask)
            out_masks.append(pred_masks)

            # save images
            if save_images:
                self.display_results(masks, filename)
        
        # save labels (only once)
        if save_images:
            self.display_labels()

        return out_masks

def load_model(prompts):
    config = clip_dinoiser_config()
    pipeline = clip_dinoiser_pipeline(config, prompts)
    return pipeline

if __name__ == '__main__':
    config = clip_dinoiser_config()
    pipeline = clip_dinoiser_pipeline(config)
    pipeline.run()