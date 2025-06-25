import numpy as np
import os

import matplotlib.pyplot as plt
import cv2
import torch
from ultralytics import FastSAM

import torch.nn.functional as F
import clip
from PIL import Image

# ----------------------------- Segmentation -----------------------------
# size of the original images
IMG_W = 2448
IMG_H = 2048

# size the input image
# images that are too large will cause the algorithm to fail
# must be a multiple of 32
IN_RESIZE = 512

# scale of the output image
# images that are too large will display larger than the screen
OUT_SCALE = 0.25

# FastSAM model parameters
CONF = 0.2
IOU = 0.9

# read in an image given the name
def input_image(image_name):
    path = os.path.join('/home/sk8723/Shark/Big Picture/', image_name)
    original_image = cv2.imread(path)
    global IMG_W 
    IMG_W = original_image.shape[1]
    global IMG_H
    IMG_H = original_image.shape[0]
    return original_image


# dispaly a segmented image given the mask image
def display_segimg(everything_results, clip_labels=None, display_masks=False):
    base_img = everything_results[0].orig_img.copy()
    base_img = cv2.resize(base_img, (int(IMG_W * OUT_SCALE), int(IMG_H * OUT_SCALE)))
    
    masks = everything_results[0].masks.data

    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(mask.cpu().numpy(), (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask_resized > 0).astype(np.uint8)

        color = np.random.randint(100, 255, size=3).tolist()
        colored_mask = cv2.merge([mask_bin * color[0], mask_bin * color[1], mask_bin * color[2]])
        base_img = cv2.addWeighted(base_img, 1.0, colored_mask.astype(np.uint8), 0.4, 0)

        # Add CLIP label if available and valid
        if clip_labels and i < len(clip_labels):
            label, score = clip_labels[i]
            if label is not None:
                M = cv2.moments(mask_bin)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(base_img, f"{label} ({score:.2f})", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Segmented Image with CLIP Labels", base_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if display_masks:
        if len(masks) == 0:
            print(f"Masks: {masks}")
            print("NO MASKS FOUND")
            exit()
        cpu_masks = masks.cpu().numpy().astype(bool)

        for mask_idx in range(cpu_masks.shape[0]):
            mask = (cpu_masks[mask_idx,:,:] * 255).astype(np.uint8)
            cv2.imshow("Mask",mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# the full process of segmentation given an image name
def seg_process(fastsam_model, image_name, conf=CONF, iou=IOU):
    # read in image and resize it
    bgr_img = input_image(image_name)
    bgr_img = cv2.resize(bgr_img, (IN_RESIZE, IN_RESIZE)) 

    # segment image
    everything_results = fastsam_model(
        bgr_img,
        device=device,
        retina_masks=True,
        imgsz=bgr_img.shape[:2], # 256
        conf=conf,
        iou=iou,
    )

    return everything_results

# ----------------------------- Text Association -----------------------------
# CLIP model parameters
CLIP_CONF = 0.5

# computes the similarity between two embeddings
# logit_scale: learned parameter in CLIP
def cosine_similarity(emb1, emb2, logit_scale):
    cos_sim = F.cosine_similarity(emb1, emb2, dim=1)
    scaled_sim = logit_scale * cos_sim
    return scaled_sim.softmax(dim=-1).detach().cpu().numpy()

def clip_process(clip_model, image_name, raw_texts, everything_results):
    # initialize logit scale
    clip_logit_scale = clip_model.logit_scale.exp()
   
    # load image
    bgr_img = input_image(image_name)
    
    text_tokens = clip.tokenize(raw_texts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()

    # preprocess batched image inputs
    image_tensors = []
    valid_masks = []

    masks = everything_results[0].masks.data.cpu().numpy()
    for i, mask in enumerate(masks):
        # resize mask back to original image size
        mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        cropped = cv2.bitwise_and(bgr_img, bgr_img, mask=mask.astype(np.uint8))

        # skip empty or very small crops
        if cropped.shape[0] < 50 or cropped.shape[1] < 50 or np.count_nonzero(cropped) == 0:
            continue

        # Convert to PIL image for CLIP
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        clip_img = clip_preprocess(pil_img).unsqueeze(0) # not moved to device yet
        image_tensors.append(clip_img)
        valid_masks.append(i)

    # if too many masks are too small, they will all be skipped
    if not image_tensors:
        return [(None, None)] * len(masks)  # no usable masks

    image_batch = torch.cat(image_tensors).to(device)

    # Encode with CLIP
    with torch.no_grad():
        image_features = clip_model.encode_image(image_batch).float()

    # Compute similarity
    scores = cosine_similarity(image_features, text_features, clip_logit_scale)

    # Map back to full mask list
    labels_scores = [(None, None)] * len(masks)
    for idx, mask_idx in enumerate(valid_masks):
        best_idx = scores[idx].argmax()
        best_score = scores[idx][best_idx]
        best_label = raw_texts[best_idx]
        if best_score >= CLIP_CONF:
            labels_scores[mask_idx] = (best_label, best_score)

    return labels_scores

if __name__ == '__main__':

    # run on the GPU instead of the CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize models
    fastsam_model = FastSAM("FastSAM-s.pt")
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

    class_names = ['water', 'boat', 'dock', 'tree']

    image_name = 'boats.png'
    results = seg_process(fastsam_model, image_name, conf=0.1, iou=0.1)
    clip_labels = clip_process(clip_model, image_name, class_names, results)

    display_segimg(results, clip_labels)