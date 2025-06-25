import numpy as np
import os

import cv2

import torch

from ultralytics import FastSAM

import clip
from PIL import Image

# ----------------------------- Dictionary -----------------------------

# Define categories
categories_dict = {
    'Water Surface': ['Water Surface', 'River Surface', 'Lake Surface', 
            'Reflection Water', 'Lagoon Surface', 'Estuary Surface', 
            'Harbor Surface', 'Bay Surface'],
    'Vegetation': ['Vegetation', 'Tree', 'Palm Tree', 'Oak Tree', 'Cypress'],
    'Boat': ['Boat', 'Ship', 'Dinghy'],
    'Boat parts': ['Boat parts', 'Motor', 'Mast', 'Hull', 'Sail'],
    'Rock': ['Rock', 'Boulder', 'Stones', 'Pebbles', 'Gravel'],
    'Sand': ['Sand', 'Beach'],
    'Grass': ['Grass', 'Field'],
    'Dock': ['Dock', 'Pier', 'Jettie', 'Wharf'],
    'Street Light Pole': ['Street Light Pole', 'Piling'],
    'Tire': ['Tire', 'Wheel'],
    'Street': ['Street', 'Sidewalk', 'Trail'],
    'Boat Ramp': ['Boat Ramp', 'Railings'],
    'Shoreline Barrier': ['Shoreline Barrier', 'Sea Wall', 'Wall', 'Barrier', 'Fence', 'Retaining Wall', 'Mossy Wall'],
    'Bridge': ['Bridge', 'Bridge Support'],
    'Buoy': ['Buoy', 'Water Buoy'],
    'Building': ['Building', 'House', 'Shed', 'Cabin'],
    'Lighthouse': ['Lighthouse'],
    'Person': ['Person'],
    'Car': ['Car', 'Truck', 'Van', 'Motorcycle']
}

# Define categories for water, land, and objects
categories = [item for sublist in categories_dict.values() for item in sublist]

# Predefine colors for each main category
category_colors = {
    'Water Surface': (255, 0, 0),       # Blue
    'Vegetation': (0, 128, 0),          # Green
    'Boat': (0, 0, 255),                # Red
    'Boat parts': (0, 255, 255),        # Yellow
    'Rock': (128, 128, 128),            # Gray
    'Sand': (0, 204, 255),              # Light Orange
    'Grass': (0, 255, 0),               # Bright Green
    'Dock': (128, 0, 128),              # Purple
    'Pole': (255, 255, 0),              # Cyan
    'Tire': (147, 20, 255),             # Pink
    'Street': (19, 69, 139),            # Dark Brown
    'Boat Ramp': (180, 130, 70),        # Steel Blue
    'Shoreline Barrier': (105, 105, 105), # Dark Gray
    'Bridge': (192, 192, 192),          # Silver
    'Buoy': (180, 105, 255),            # Hot Pink
    'Building': (255, 0, 255),          # Magenta
    'Lighthouse': (230, 216, 173),      # Light Blue
    'Person': (0, 69, 255),             # Orange Red
    'Car': (128, 0, 0)                  # Dark Navy
}

# ----------------------------- Display Support -----------------------------
# helper function to display one mask with text
# info is the unit from mask_info_list
def display_mask(info, display_img):
    mask_np = info['mask']
    color = info['color']
    label = info['label']
    score = info['score']

    # display contours or masks
    if DISPLAY_CONTOURS:
        cv2.drawContours(display_img, info['contours'], -1, color, 2)  # Thickness = 2
    else:
        display_img[mask_np > 0] = (display_img[mask_np > 0] * (1 - MASK_ALPHA) + np.array(color, dtype=np.float32) * MASK_ALPHA).astype(np.uint8)
        
    # Draw label at center of mask bounding box
    if DISPLAY_LABELS:
        x, y, w, h = info['bbox']
        text_x = x + w // 2
        text_y = y + h // 2
        cv2.putText(display_img, f'{label} ({score:.2f})', (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(display_img, f'{label} ({score:.2f})', (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return display_img

# mouse interaction
cursor_pos = (-1, -1)
def on_mouse(event, x, y, flags, param):
    global cursor_pos
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_pos = (x, y)

DISPLAY_CONTOURS = True    # display contours or masks
MASK_ALPHA = 0.5            # opacity of masks
DISPLAY_LABELS = False      # display labels or only masks
MOUSE_HOVER_MODE = True     # mouse hover mode or static mode

# ----------------------------- Tuning Dials -----------------------------
IMG_RESIZE = 512            # input image resize value

SAM_CONF = 0.1              # conf value used in the FastSAM model
SAM_IOU = 0.1               # IoU value used in the FastSAM model

CLIP_CONF = 0               # threshold for which masks CLIP removes

# ----------------------------- Main -----------------------------
if __name__ == '__main__':
    # ------------------------- Initialization -------------------------
    # run on GPU instead of CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # import images
    folder_path = './ExampleImgs'  # Replace with your actual path
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    bgr_imgs = [cv2.imread(p) for p in image_paths]

    # initialize models
    fastsam_model = FastSAM("FastSAM-s.pt")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # ------------------------- Image Processing -------------------------
    for img_idx, bgr_img in enumerate(bgr_imgs):
        bgr_img = cv2.resize(bgr_img, (IMG_RESIZE, IMG_RESIZE))
        img_combined = bgr_img.copy()  # This will be used for combined view with CLIP
        
        # ------------------------- Segmentation -------------------------
        fastsam_results = fastsam_model(
            bgr_img,
            device=device,
            retina_masks=True,
            imgsz=bgr_img.shape[:2],
            conf=SAM_CONF,
            iou=SAM_IOU,
        )

        # ------------------------- Text Association -------------------------
        # Prepare CLIP text embeddings for "water", "land", and "object"
        text_inputs = torch.cat([clip.tokenize(f"a photo of {category}") for category in categories]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        mask_info_list = []

        # loop through each mask segment
        for mask in fastsam_results[0].masks.data:
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            # convert segments into bounding boxes
            # CLIP processes bounding boxes better than segments
            x, y, w, h = cv2.boundingRect(mask_np)
            cropped = bgr_img[y:y+h, x:x+w]

            # skip empty or very small crops
            if cropped.size == 0 or w < 10 or h < 10:
                continue

            # Convert to PIL image for CLIP
            pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            image_input = clip_preprocess(pil_crop).unsqueeze(0).to(device)
            
            # calculate best score
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                best_idx = similarity[0].argmax().item()
                best_score = similarity[0][best_idx].item()
            
            if best_score >= CLIP_CONF:
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best_label = categories[best_idx]
                main_category = next((key for key, values in categories_dict.items() if best_label in values or best_label == key), None)
                color = category_colors.get(main_category, (255, 255, 255))

                mask_info_list.append({
                    'mask': mask_np,
                    'contours': contours,
                    'bbox': (x, y, w, h),
                    'label': best_label,
                    'score': best_score,
                    'color': color
                })

        # ------------------------- Display -------------------------
        display_img = bgr_img.copy()

        if MOUSE_HOVER_MODE:
            cv2.namedWindow("Interactive View")
            cv2.setMouseCallback("Interactive View", on_mouse)

            while True:
                # reset display_img
                display_img = bgr_img.copy()

                # check if cursor is within image bounds and over non-zero mask pixel
                hovered_info = None
                for info in mask_info_list:
                    mask_np = info['mask']
                    if 0 <= cursor_pos[0] < mask_np.shape[1] and 0 <= cursor_pos[1] < mask_np.shape[0]:
                        if mask_np[cursor_pos[1], cursor_pos[0]] > 0:
                            hovered_info = info
                            break

                # if cursor is over a mask, only display that mask
                # if cursor is over no mask, display all masks
                if hovered_info is not None:
                    display_img = display_mask(hovered_info, display_img) 
                else:
                    for info in mask_info_list:
                        display_img = display_mask(info, display_img) 
                        
                cv2.imshow("Interactive View", display_img)
                key = cv2.waitKey(1)
                if key != -1:
                    break

            cv2.destroyAllWindows()
            # hit escape to end the program
            if key == 27:
                break
        else:
            for info in mask_info_list:
                display_img = display_mask(info, display_img)  
                      
            cv2.imshow("Static View", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # hit escape to end the program
            if key == 27:
                break