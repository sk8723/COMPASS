import os
import numpy as np
import torch
import cv2
import time

from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
import importlib.resources as pkg_resources
import ast

import input_dictionary as dict

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

class GroundingDINOConfig:
    # input/output
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_PATH = "../ExampleImgs"
    WORD_BANK_NAME = "extended"

    OUTPUT_DIR = "../../OutputImgs"

    # model setup
    CONFIG_NAME = "testing_SwinT.py"
    CHECKPOINT_NAME = "groundingdino_swint_ogc.pth"
    
    # tuning dials
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.25
    TOKEN_SPANS = None  # e.g., [[[2, 5]]] or None
    IMG_RESIZE = 2000

    CPU_ONLY = False


class GroundingDINOPipeline:
    def __init__(self, config):
        self.config = config
        self.device = "cpu" if config.CPU_ONLY else "cuda"
        self.model = self.load_model()
        self.img_path = os.path.abspath(os.path.join(self.config.BASE_DIR, self.config.IMAGE_PATH))
        self.prompts = " . ".join(dict.word_bank(self.config.WORD_BANK_NAME))

    def load_model(self):
        # Load config file as a path
        with pkg_resources.path('groundingdino.config', self.config.CONFIG_NAME) as config_path:
            args = SLConfig.fromfile(str(config_path))
            args.device = self.device
            model = build_model(args)

        # Load checkpoint as a path
        with pkg_resources.path('groundingdino.weights', self.config.CHECKPOINT_NAME) as ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

        model.eval()
        return model.to(self.device)

    def load_images(self):

        image_paths = [os.path.join(self.img_path, f) for f in os.listdir(self.img_path) if f.endswith('.png')]
        return [(cv2.imread(p), os.path.basename(p)) for p in image_paths]

    def produce_boxes(self, image):
        # set parameters
        prompts = self.prompts
        box_threshold = self.config.BOX_THRESHOLD
        text_threshold = None if self.config.TOKEN_SPANS else self.config.TEXT_THRESHOLD
        token_spans = ast.literal_eval(self.config.TOKEN_SPANS) if self.config.TOKEN_SPANS else None

        # image preparation
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        transform = T.Compose([
            T.RandomResize([self.config.IMG_RESIZE], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)
        
        prompts = prompts.strip().lower()
        if not prompts.endswith("."):
            prompts += "."

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor[None], captions=[prompts])

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        if token_spans is None:
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
        else:
            positive_maps = create_positive_map_from_span(
                self.model.tokenizer(prompts), token_span=token_spans
            ).to(image_tensor.device)

            logits_for_phrases = positive_maps @ logits.T
            all_phrases, all_boxes, all_scores = [], [], []

            for span, logit_vec in zip(token_spans, logits_for_phrases):
                phrase = ' '.join([prompts[s:e] for s, e in span])
                mask = logit_vec > box_threshold
                all_boxes.append(boxes[mask])
                all_scores.append(logit_vec[mask])
                all_phrases.extend([phrase] * mask.sum().item())

            boxes_filt = torch.cat(all_boxes, dim=0).cpu()
            scores = torch.cat(all_scores, dim=0).cpu()
            pred_phrases = all_phrases

        return boxes_filt, pred_phrases, scores



    def display_results(self, image, filename, boxes, labels, scores, rand_color=True, compress_img_size=False):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        W, H = image_pil.size
        for box, label, score in zip(boxes, labels, scores):
            box = box * torch.tensor([W, H, W, H], device=box.device)
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = [int(v) for v in box]

            if rand_color:
                color = tuple(np.random.randint(0, 255, size=3).tolist())
            else:
                color = dict.LABEL_COLOR_MAP.get(label, (128, 128, 128))
            
            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

            display_text = f"{label} ({score:.2f})"
            font_path = os.path.abspath(os.path.join(self.config.BASE_DIR, "../fonts/arial.ttf"))
            font = ImageFont.truetype(font_path, size=50)
            bbox = draw.textbbox((x0, y0), display_text, font)
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), display_text, fill="white", font=font)

            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

        name = os.path.splitext(filename)[0]  # strip ".png" extension
        save_path = os.path.abspath(os.path.join(self.config.BASE_DIR, self.config.OUTPUT_DIR))
        
        if compress_img_size:
            image_resized = image_pil.resize((490, 410), Image.Resampling.LANCZOS)
            image_saved = image_resized
        else:
            image_saved = image_pil

        image_saved.save(os.path.join(save_path, f'{name}_boxes.png'))

    def run(self):
        images = self.load_images()
        total_time = 0
        for image, filename in images:
            start = time.perf_counter()
            # # # processing
            boxes, phrases, scores = self.produce_boxes(image)
            self.display_results(image, filename, boxes, phrases, scores, compress_img_size=False)
            # # #
            elapsed = time.perf_counter() - start
            print(f'{filename} processing time: {elapsed:.2f}s')
            total_time += elapsed
        if images:
            avg_time = total_time / len(images)
            print(f'Average processing time: {avg_time:.2f}s')

if __name__ == "__main__":
    config = GroundingDINOConfig()
    pipeline = GroundingDINOPipeline(config)
    pipeline.run()