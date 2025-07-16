import os
import numpy as np
import torch
import cv2
import time

import importlib.resources as pkg_resources
from omegaconf import OmegaConf
from hydra import compose, initialize
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
import torch.nn.functional as F

import clipdino_dictionary as dict

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
    CFG = 'COMPASS.yaml'
    CHECKPOINT_PATH = 'clip_dinoiser/checkpoints/last.pt'
    WORD_BANK_NAME = 'extended'

    # tuning dials
    IMG_RESIZE = 1280

class clip_dinoiser_pipeline:
    # initialize variables
    def __init__(self, config):
        # copy over config
        self.config = config

        # initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # set up checkpoint
        self.checkpoint = resources.load_checkpoint('last.pt')

        # initialize prompts
        self.prompts = dict.word_bank(self.config.WORD_BANK_NAME)
        if len(self.prompts) == 1:
            self.prompts = ['background'] + self.prompts
        self.palette = dict.get_label_colors(self.prompts)

        # initialize model
        cfg = resources.load_config(self.config.CFG)
        self.model = build_model(cfg.model, class_names=self.prompts)
        self.model.load_state_dict(self.checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.model.apply_found = 'background' in self.prompts

        # set up image folder path
        self.img_path = os.path.abspath(os.path.join(self.config.BASE_DIR, self.config.IMG_FOLDER_PATH))

    # read in images given a folder path
    def load_images(self):
        image_paths = [os.path.join(self.img_path, f) for f in os.listdir(self.img_path) if f.endswith('.png')]
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
        start = time.perf_counter()
        with torch.inference_mode():
            output, dinoised_feats = self.model(img_tens, apply_softmax=False, get_features=True)
            output = output.cpu()
            dinoised_feats = dinoised_feats.cpu()
        elapsed = time.perf_counter() - start
        print(f'extract masks: {elapsed:.2f}s')

        # resize tensor to fit original image dimensions
        if display_soft_edges:
            output = F.interpolate(output, scale_factor=self.model.vit_patch_size, mode="bilinear",
                                   align_corners=False)[..., :h, :w]
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

    def run(self, img_resize):
        images = self.load_images()
        for image, filename in images:
            masks = self.produce_masks(image, display_soft_edges=False)
            self.display_results(masks, filename)
        self.display_labels()

if __name__ == '__main__':
    config = clip_dinoiser_config()
    pipeline = clip_dinoiser_pipeline(config)
    pipeline.run(1280)