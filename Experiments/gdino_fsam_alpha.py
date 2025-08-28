import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import time

from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
import importlib.resources as pkg_resources
import json
import gc

from . import input_dictionary as dict

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

from ultralytics import FastSAM

class GroundingDINOConfig:
    # model setup
    CONFIG_NAME = "COMPASS_SwinT.py"
    CHECKPOINT_NAME = "groundingdino_swint_ogc.pth"
    
    # tuning dials
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.3
    IMG_RESIZE = 2000

    CPU_ONLY = False

class GroundingDINOPipeline:
    def __init__(self, config):
        self.config = config
        self.device = "cpu" if config.CPU_ONLY else "cuda"
        self.model = self.load_model()

    def load_model(self):
        # Load config file as a path
        with pkg_resources.path('groundingdino.config', self.config.CONFIG_NAME) as config_path:
            args = SLConfig.fromfile(str(config_path))
            model = build_model(args)

        # Load checkpoint as a path
        with pkg_resources.path('groundingdino.weights', self.config.CHECKPOINT_NAME) as ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

        model.eval()
        return model.to(self.device)

    def produce_boxes(self, image, prompts):
        # set parameters
        box_threshold = self.config.BOX_THRESHOLD
        text_threshold = self.config.TEXT_THRESHOLD
 
        # image preparation
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        transform = T.Compose([
            T.RandomResize([self.config.IMG_RESIZE], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)
        
        # prompts preparation
        prompts = " . ".join(prompts)
        prompts = prompts.strip().lower()
        if not prompts.endswith("."):
            prompts += "."

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor[None], captions=[prompts])

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        filt_mask = logits.max(dim=1)[0] > box_threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]
        scores = logits_filt.max(dim=1)[0]  # Confidence scores

        tokenizer = self.model.tokenizer
        tokenized = tokenizer(prompts)
        pred_phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            for logit in logits_filt
        ]

        H, W = image.shape[:2]
        boxes_filt = box_ops.box_cxcywh_to_xyxy(boxes_filt)
        boxes_filt = boxes_filt * torch.tensor([W, H, W, H], device=boxes_filt.device)

        del image_tensor
        del outputs
        del logits
        del boxes
        del logits_filt
        torch.cuda.empty_cache()

        return boxes_filt, pred_phrases, scores
    
class FastSAMConfig:
    # model setup
    MODEL_PATH = "../FastSAM/weights/FastSAM-x.pt"  # Update with actual model file path

    IMG_SIZE = 1024
    CONF_THRESHOLD = 0.1
    IOU_THRESHOLD = 0.6
    
class FastSAMPipeline:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FastSAM(config.MODEL_PATH)

    def produce_masks(self, image):
        with torch.inference_mode():
            everything_results = self.model(
                image,
                device=self.device,
                retina_masks=True,
                imgsz=self.config.IMG_SIZE,
                conf=self.config.CONF_THRESHOLD,
                iou=self.config.IOU_THRESHOLD,
                verbose=False
            )
        # deallocate everything_results
        all_masks = everything_results[0].masks.data.detach().clone().bool()
        everything_results[0].masks.data = None
        everything_results = None
        del everything_results
        torch.cuda.empty_cache()

        return all_masks

class MainConfig:
    # input/output
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = "../../OutputImgs"

class MainPipeline:
    def __init__(self, config, gdino_config, fsam_config, prompts):
        self.config = config
        self.gdino_pipeline = GroundingDINOPipeline(gdino_config)
        self.fsam_pipeline = FastSAMPipeline(fsam_config)

        if isinstance(prompts, str):
            prompts = dict.word_bank(prompts)
        self.prompts = prompts

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.out_path = os.path.abspath(os.path.join(self.config.BASE_DIR, self.config.OUTPUT_DIR))

    def load_images(self, folder_path):
        abs_folder_path = os.path.abspath(os.path.join(self.config.BASE_DIR, folder_path))
        image_paths = [os.path.join(abs_folder_path, f) for f in os.listdir(abs_folder_path) if f.endswith('.png')]
        return [(cv2.imread(p), os.path.basename(p)) for p in image_paths]
    
    def match_boxes_and_masks(self, boxes, masks):
        # keep a copy of the full-res masks for later
        orig_masks = masks.clone()

        # --- Step 0: convert FastSAM masks to bounding boxes ---
        mask_bboxes = []
        for mask in masks:
            ys, xs = torch.nonzero(mask, as_tuple=True)
            if len(xs) == 0 or len(ys) == 0:
                mask_bboxes.append(None)
            else:
                x1, y1 = xs.min().item(), ys.min().item()
                x2, y2 = xs.max().item(), ys.max().item()
                mask_bboxes.append((x1, y1, x2, y2))
        
        # --- Step 1: compute IoU between boxes and mask_bboxes ---
        bboxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=self.device)  # (B, 4)
        mask_bboxes_tensor = torch.tensor(
            [mb if mb is not None else (0, 0, 0, 0) for mb in mask_bboxes],
            dtype=torch.float32, device=self.device
        )  # (M, 4)
        
        B = bboxes_tensor.shape[0]
        M = mask_bboxes_tensor.shape[0]

        # Compute intersection
        x1 = torch.max(bboxes_tensor[:, None, 0], mask_bboxes_tensor[None, :, 0])
        y1 = torch.max(bboxes_tensor[:, None, 1], mask_bboxes_tensor[None, :, 1])
        x2 = torch.min(bboxes_tensor[:, None, 2], mask_bboxes_tensor[None, :, 2])
        y2 = torch.min(bboxes_tensor[:, None, 3], mask_bboxes_tensor[None, :, 3])

        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        intersection = inter_w * inter_h  # (B, M)

        # Compute union
        area_box = (bboxes_tensor[:, 2] - bboxes_tensor[:, 0]) * (bboxes_tensor[:, 3] - bboxes_tensor[:, 1])
        area_mask = (mask_bboxes_tensor[:, 2] - mask_bboxes_tensor[:, 0]) * (mask_bboxes_tensor[:, 3] - mask_bboxes_tensor[:, 1])
        union = area_box[:, None] + area_mask[None, :] - intersection
        ious = intersection / union.clamp(min=1e-6)

        # --- Step 2: assign each bbox its best mask ---
        best_ious, best_mask_idxs = ious.max(dim=1)  # (B,)

        # --- Step 3: resolve conflicts (only highest IoU per mask) ---
        final_assignments = [None] * B
        mask_to_best = {}

        for bbox_idx, (mask_idx, score) in enumerate(zip(best_mask_idxs.tolist(), best_ious.tolist())):
            if score == 0:  # no overlap
                continue
            if (mask_idx not in mask_to_best) or (score > mask_to_best[mask_idx][1]):
                if mask_idx in mask_to_best:
                    old_bbox = mask_to_best[mask_idx][0]
                    final_assignments[old_bbox] = None
                mask_to_best[mask_idx] = (bbox_idx, score)
                final_assignments[bbox_idx] = mask_idx

        # --- Step 4: build final mask list ---
        matched_masks = []
        for mask_idx in final_assignments:
            if mask_idx is None:
                matched_masks.append(None)
            else:
                matched_masks.append(orig_masks[mask_idx].cpu().numpy())

        # --- Step 5: clear GPU memory ---
        del masks, orig_masks
        del bboxes_tensor, mask_bboxes_tensor
        torch.cuda.empty_cache()

        return matched_masks

    def display_results(self, image, masks, labels, scores=None):
        # Display and save results
        vis_image = image.copy()

        # option of displaying scores
        if scores is None:
            scores = [None] * len(labels)

        # loop through each mask
        for mask, label, score in zip(masks, labels, scores):
            if mask is None:
                continue

            if label in dict.LABEL_COLOR_MAP:
                color = dict.LABEL_COLOR_MAP[label]
                color = tuple(color[::-1]) # convert from RGB to BGR
            else:
                color = tuple(np.random.randint(0, 255, 3).tolist())

            vis_image[mask.astype(bool)] = (
                0.5 * vis_image[mask.astype(bool)] + 0.5 * np.array(color)
            ).astype(np.uint8)

            # Draw label at the mask centroid
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0:
                # Compute centroid
                y_center = int(np.mean(y_indices))
                x_center = int(np.mean(x_indices))
                
                if score is not None:
                    text = f"{label} ({score:.2f})"
                else:
                    text = f"{label}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                
                # Get text size to center properly
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = x_center - text_width // 2
                text_y = y_center + text_height // 2  # baseline shift
                
                cv2.putText(
                    vis_image,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    lineType=cv2.LINE_AA,
                )

        return vis_image
    
    # type is a string
    # image is a NumPy array
    # prompts is an array of strings or single string in reference to a word bank
    # folder_name is a string
    def run(self, image_arr=None, folder_path=None, save_images=False, type='debugging'):
        # parameter format evaluation
        if image_arr is not None and folder_path is not None:
            raise ValueError("either a single image or folder of images must be provided, not both")

        # simplify variable names
        fsam = self.fsam_pipeline
        gdino = self.gdino_pipeline
        prompts = self.prompts

        # format images
        if folder_path is not None:
            images = self.load_images(folder_path)
        else:
            images = [(image_arr, "singleshot")]

        # run primary processing loop
        out_masks = [] # masks to be outputted
        for image, filename in images:
            # run Grounding DINO and FastSAM models
            boxes, labels, scores = gdino.produce_boxes(image, prompts)
            masks = fsam.produce_masks(image)

            # match GDINO boxes to FSAM masks
            matched_masks = self.match_boxes_and_masks(boxes, masks)

            # match masks to labels
            H, W = image.shape[:2]
            pred_masks = [np.zeros((H, W), dtype=bool) for _ in prompts]
            for label, mask in zip(labels, matched_masks):
                if mask is None:
                    continue
                label = label.strip().lower()
                if label in prompts:
                    idx = prompts.index(label)
                    pred_masks[idx] = np.logical_or(pred_masks[idx], mask)
            out_masks.append(pred_masks)

            # save images
            if save_images:
                vis_image = None
                if type == 'debugging':
                    # create image
                    vis_image = self.display_results(image, matched_masks, labels, scores=scores)

                if type == 'evaluation':
                    # create image
                    vis_image = self.display_results(image, pred_masks, prompts)

                if vis_image is None:
                    raise ValueError("unrecognized value for 'type'")

                # Save image
                name = os.path.splitext(filename)[0]  # strip ".png" extension
                out_file = os.path.join(self.out_path, f"{name}_output.png")
                cv2.imwrite(out_file, vis_image)
            
            # deallocate big data
            del boxes
            del labels
            del scores
            del masks
            torch.cuda.empty_cache()
            gc.collect()
        
        return out_masks

def load_model(prompts):
    gdino_config = GroundingDINOConfig()
    fsam_config = FastSAMConfig()
    main_config = MainConfig()
    main_pipeline = MainPipeline(main_config, gdino_config, fsam_config, prompts)
    return main_pipeline