import os
import re
from argparse import ArgumentParser
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from PIL import Image
import cv2
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import pickle as pkl

import torch
import torch.nn.functional as F
from ultralytics import YOLO, FastSAM
import clip


def load_transformation(file_path):
    transform_1d = np.load(file_path)
    trans_mat = np.eye(4)
    rot = R.from_quat(transform_1d[3:])
    trans_mat[:3,:3] = rot.as_matrix()
    trans_mat[:3,3] = transform_1d[:3]
    return trans_mat

def get_yolo_class_masks(img, yolo_results):
    IMG_H, IMG_W = img.shape[0], img.shape[1]
    
    # Dictionary to store binary masks for each YOLO class
    class_masks = {}

    # Iterate over all YOLO masks and classes
    for mask, cls in zip(yolo_results[0].masks.data, yolo_results[0].boxes.cls):
        class_id = int(cls.item())  # Get class id
        
        # Convert mask to binary and resize it to match the image dimensions
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)  # Convert mask to 0-255
        mask_resized = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

        # Create a binary mask for this specific class
        if class_id not in class_masks:
            class_masks[class_id] = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        
        # Combine the current mask for this class
        class_masks[class_id] = np.maximum(class_masks[class_id], mask_resized)

        # # Optionally: Apply dilation and blur to make the mask more smooth
        # kernel = np.ones((5, 5), np.uint8)  # Larger kernel for stronger dilation
        # dilated_mask = cv2.dilate(class_masks[class_id], kernel, iterations=2)
        # blurred_mask = cv2.GaussianBlur(dilated_mask, (15, 15), 0)
        # # Update the class mask with the final blurred version
        # class_masks[class_id] = blurred_mask

    return class_masks

def remove_yolo_masks(img, yolo_masks):
    IMG_H, IMG_W = img.shape[0], img.shape[1]
    
    # Create a single binary mask covering all terrain segments
    combined_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    for cls_id, mask in yolo_masks.items():
        combined_mask = np.maximum(combined_mask, mask)

    # Expand the mask to cover a larger area
    kernel = np.ones((5, 5), np.uint8)  # Larger kernel for stronger dilation
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)

    # Smooth edges with Gaussian blur
    blurred_mask = cv2.GaussianBlur(dilated_mask, (15, 15), 0)

    # Apply inverted blurred mask to remove terrain
    non_terrain_image = cv2.bitwise_and(img, img, mask=255 - blurred_mask)
    
    return non_terrain_image

def compute_aabb(points):
    """
    Compute the axis-aligned bounding box (AABB) for a given set of 3D points.
    
    Args:
        points (numpy.ndarray): Nx3 array of 3D points.
    
    Returns:
        tuple: (min_bound, max_bound) where each is a (3,) array.
    """
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    return min_bound, max_bound

def tensor_cosine_similarity(emb1, emb2):
    unscaled_logit_cos_sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=2)
    return unscaled_logit_cos_sim

def random_color():
    return np.random.rand(3).tolist()  # Generates a random RGB color

def arg_parser():
    parser = ArgumentParser()
    # Point Cloud Data
    parser.add_argument('--data_folder', 
                        type=str, 
                        default="/docker_ros2_ws/src/oasis2/data/south_campus_4_21_2025", 
                        help='Directory to where folders of local and global scans were saved')
    parser.add_argument('--continue_processing', 
                        action='store_true', 
                        help='Picks up where you last left off')
    
    # # Folder to Save Results
    # parser.add_argument('--output_file', type=str, default='results/ouster/labeling_clouds/', help='Path to save the labeled point cloud (PLY file)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    data_folder = args.data_folder
    
    ## Define parameters
    DEBUG_MODE = True #False
    K = np.array([
        [457.978759765625, 0.0, 387.35205078125], 
        [0.0, 457.719482421875, 240.11021423339844], 
        [0.0, 0.0, 1.0]
    ]) # 768 x 480
    scan_display_step_sz = 500
    theta_cos_sim = 0.9   # cos_sim threshold to determine a match
    do_DBSCAN = True
    dbscan_global = DBSCAN(eps=2.0, min_samples=5) # change depending on voxel sizes of global PC
    
    ##########################
    ## Load Models and Data ##
    ##########################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = '/FastSAM.pt'
    fastsam_model = FastSAM(model_path)
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    logit_scale = clip_model.logit_scale.exp()
    yolo_model = YOLO("/custom_yolo_model_3cls.pt") # orig_and_sunnyfeb_11nano_640sz/weights/best.pt
    
    # Load global point cloud
    global_pc_folder = os.path.join(data_folder, "global_pc")
    global_pc_files = sorted(Path(global_pc_folder).glob("*.npy"))        
    latest_global_pc_file = global_pc_files[-1] # global_map_idx 33 for sunny midday data
    global_pc = np.load(latest_global_pc_file)
    # Dictionary to hold matching global points
    global_kdtree = KDTree(global_pc[:, :3])  # Using the x, y, z columns
    
    # Load time-synced sensor data and transformation files
    local_pc_folder = os.path.join(data_folder, "local_pc")
    local_images_folder = os.path.join(data_folder, "local_images")
    transforms_cam2local_folder = os.path.join(data_folder, "transformations_cam2local_pc")
    transforms_local2global_folder = os.path.join(data_folder, "transformations_local2global_pc")
    camera_timestamps = []
    for local_image_file in Path(local_images_folder).glob("*.jpg"):
        # Assuming filename format: local_cam_img_{timestamp}.jpg
        timestamp = float(local_image_file.stem.split("_")[-1])  # Extract timestamp as float
        camera_timestamps.append(timestamp)
    camera_timestamps.sort()  # Sort timestamps for easier matching

    t_begin = time.time()
    if args.continue_processing:
        print("Picking up where you last left off")
        ## Find last saved data
        start_string = "ptxpt_gidx2imgidx_dict_itr"
        matching_files = [f for f in os.listdir(data_folder) if f.startswith(start_string)]
        numbers = []
        for f in matching_files:
            match = re.search(rf"{re.escape(start_string)}(\d+)", f)
            if match:
                numbers.append(int(match.group(1)))
        last_scan_idx = max(numbers) if numbers else None
        
        ## Load saved data
        with open(data_folder+f"/ptxpt_pc_dict_itr{last_scan_idx}.pkl", "rb") as f:
            pc_dict = pkl.load(f)
        clip_tensor = torch.load(data_folder+f"/ptxpt_clip_tensor_itr{last_scan_idx}.pt")
        with open(data_folder+f"/ptxpt_gidx2imgidx_dict_itr{last_scan_idx}.pkl", "rb") as f:
            map_globalidx2imgidx = pkl.load(f)
        img_clip_tensor = torch.load(data_folder+f"/img_clip_tensor_itr{last_scan_idx}.pt")
        img_clips = list(img_clip_tensor.unbind(dim=0))
        with open(data_folder+f"/saved_img_names_itr{last_scan_idx}.pkl","rb") as f:
            saved_img_names = pkl.load(f)
        num_scans = last_scan_idx+1
    else:
        print("Starting at the beginning")
        # Embed yolo class names with CLIP
        yolo_clip_embs = []
        yolo_classes = yolo_model.names
        for cls_idx, cls_str in yolo_classes.items():
            clip_emb = clip_model.encode_text(clip.tokenize([cls_str]).to(device)).float()
            yolo_clip_embs.append(clip_emb)
        clip_tensor = torch.vstack(yolo_clip_embs) # (num_classes, 512)
        pc_dict = {}#defaultdict(lambda: defaultdict(int))  # {point_index: {clip_id: count}}
        num_scans = 0
        img_clips = []
        saved_img_names = []
        map_globalidx2imgidx = {} # map from global_index to img_index {g_idx: set(0,34,2), ...}
    
    
    for scan_idx, local_pc_file in enumerate(sorted(Path(local_pc_folder).glob("*.npy"))):
        if args.continue_processing and scan_idx <= last_scan_idx:
            continue
        itr_t0 = time.time()
        print(f"\nScan Index {scan_idx}\n")
        timestamp = local_pc_file.stem.split("_")[-1]  # Assuming format local_pc_{timestamp}.npy
        transform_cam_to_local_file = os.path.join(transforms_cam2local_folder, f"transform_cam_to_lidar_{timestamp}.npy")
        transform_local_to_global_file = os.path.join(transforms_local2global_folder, f"transform_lidar_to_map_{timestamp}.npy")

        # Check if corresponding image and transformation files exist
        if not os.path.exists(transform_cam_to_local_file) or not os.path.exists(transform_local_to_global_file):
            continue
        # Choose time-sync(-ish) camera image
        closest_cam_timestamp = min(camera_timestamps, key=lambda x: abs(x - float(timestamp)))
        local_image_file = os.path.join(local_images_folder, f"local_cam_img_{closest_cam_timestamp}.jpg")
        
        #####################
        ## Load Local Data ##
        #####################
        local_pc = np.load(local_pc_file)
        local_image = cv2.imread(local_image_file)
        IMG_H, IMG_W = local_image.shape[:2]
        transform_cam_to_local = load_transformation(transform_cam_to_local_file)
        transform_local_to_global = load_transformation(transform_local_to_global_file)
        # transform_cam_to_global = transform_cam_to_local @ transform_local_to_global
        
        ##########################
        ## CLIP vector of image ##
        ##########################
        preprocessed_image = clip_preprocess(Image.fromarray(local_image)).unsqueeze(0).to(device)
        with torch.no_grad():
            img_encoded = clip_model.encode_image(preprocessed_image)
        img_clips.append(img_encoded)
        saved_img_names.append(local_image_file)

        ##################################
        ## Keep only 3D Points in Image ##
        ##################################
        in_lidar_frame = True
        keep_pc_idxs = []
        map_yx2idx = {} # map from 2D image coord to 3D PC index
        map_local2globalidx = {} # map from 3D local PC index to 3D global PC index
        lidar_img = np.zeros((IMG_H,IMG_W))
        if in_lidar_frame:
            R_L2S = np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0,  0, 1]
            ])
        for pt_idx, p_L in enumerate(local_pc):
            if in_lidar_frame:
                p_S = np.dot(R_L2S, p_L[:3]) # Transform 3d point into sensor frame
            else:
                p_S = p_L
            vec = np.array([p_S[0],p_S[1],p_S[2],1.0]) # Homogenous Coordinates in Sensor frame
            cam_point = np.dot(transform_cam_to_local, vec)
            cam_vec = cam_point[:3]
            cam_vec2 = np.dot(K, cam_vec)
            x = int(round(cam_vec2[0] / cam_vec2[2]))
            y = int(round(cam_vec2[1] / cam_vec2[2]))
            if (x >= 0) and (x < IMG_W) and (y >= 0) and (y < IMG_H) and (p_S[1] > 0):
                keep_pc_idxs.append(pt_idx)
                map_yx2idx[(y,x)] = pt_idx
                lidar_img[y,x] = p_L[3] # intensity
                p_G = transform_local_to_global @ np.append(p_L[:3],1.0)
                l2g_dist, g_idx = global_kdtree.query(p_G[:3]) # Find the closest point in the global point cloud
                map_local2globalidx[pt_idx] = g_idx
                if g_idx in map_globalidx2imgidx.keys():
                    map_globalidx2imgidx[g_idx].add(num_scans)
                else:
                    map_globalidx2imgidx[g_idx] = set([num_scans])
        # lidar_img = (lidar_img - np.min(lidar_img)) / np.ptp(lidar_img)
        lidar_img = lidar_img / np.max(lidar_img)
        lidar_img_bool = lidar_img.astype(bool) # True for non-zero intensities
        filtered_local_pc = local_pc[keep_pc_idxs] # 4D pointcloud in FoV
        if DEBUG_MODE and (scan_idx % scan_display_step_sz == 0):
            cv2.imshow("LiDAR Image",lidar_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #######################
        ## Yolo Segmentation ##
        #######################
        yolo_results = yolo_model(local_image, conf=0.1)
        if DEBUG_MODE and (scan_idx % scan_display_step_sz == 0):
            img_with_masks = yolo_results[0].plot(
                labels=True,
                boxes=True,
                masks=True,
                probs=False,
                conf=False,
                show=False,
                save=False,
            )
            cv2.imshow("YOLO Segmentation", img_with_masks)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        yolo_masks = get_yolo_class_masks(local_image, yolo_results)
        
        ## Associate Global Points to CLIP embs ##
        for cls_id, mask in yolo_masks.items():
            filtered_lidar_img = np.bitwise_and(lidar_img_bool, mask.astype(bool))
            y_indices, x_indices = np.where(filtered_lidar_img > 0)
            for y, x in zip(y_indices, x_indices):
                g_idx = map_local2globalidx[map_yx2idx[(y,x)]]
                if g_idx in pc_dict:
                    if cls_id in pc_dict[g_idx]:
                        pc_dict[g_idx][cls_id] += 1
                    else:
                        pc_dict[g_idx][cls_id] = 1
                else:
                    pc_dict[g_idx] = {cls_id: 1}
        
        ##################
        ## FastSAM+CLIP ##
        ##################
        filtered_image = remove_yolo_masks(local_image, yolo_masks)
        if DEBUG_MODE and (scan_idx % scan_display_step_sz == 0):
            cv2.imshow("Removed YOLO Masks", filtered_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        fastsam_results = fastsam_model(
            filtered_image,
            device=device,
            retina_masks=True,
            imgsz=filtered_image.shape[:2], # 256
            conf=0.003,#0.003,
            iou=0.25, # Lower values remove more overlapping masks
            max_det=100, # maximum number of detections
        )
        if DEBUG_MODE and (scan_idx % scan_display_step_sz == 0):
            mask_img = fastsam_results[0].plot(
                conf=False, 
                labels=False,
                boxes=False,
                probs=False,
                color_mode="instance",
            )
            cv2.imshow(f"FastSAM Masks", mask_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        fastsam_masks = fastsam_results[0].masks.data.bool()  # Using PyTorch tensors instead of NumPy
        if fastsam_masks.numel() == 0:  # Check if no masks are found
            print("NO MASKS FOUND")
            exit()
        
        buffer = 10
        fastsam_clip_embs = []
        fastsam_global_idxs = []
        for mask_idx, mask in enumerate(fastsam_masks):
            # Convert mask to 8-bit
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)

            masked_img = np.zeros_like(filtered_image)
            masked_img_filt = np.zeros_like(filtered_image)
            masked_img[mask_np > 0] = local_image[mask_np > 0]
            masked_img_filt[mask_np > 0] = filtered_image[mask_np > 0]
            if np.count_nonzero(masked_img) == 0:
                continue
            if np.count_nonzero(masked_img_filt)/np.count_nonzero(masked_img) < 0.5:
                # print("Skipping mask that matches terrain") 
                continue
            # print("Non-zero Orig:",np.count_nonzero(masked_img),"Filtered:",np.count_nonzero(masked_img_filt))
            # print("Filtered/Orig =",np.count_nonzero(masked_img_filt)/np.count_nonzero(masked_img))
            # if DEBUG_MODE and (scan_idx % scan_display_step_sz == 0):
            #     cv2.imshow(f"FastSAM mask {mask_idx}", masked_img)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            
            # Faster bounding box extraction
            x, y, w, h = cv2.boundingRect(mask_np)
            mask_x1, mask_y1 = max(0, x - buffer), max(0, y - buffer)
            mask_x2, mask_y2 = min(filtered_image.shape[1], x + w + buffer), min(filtered_image.shape[0], y + h + buffer)

            # Dilate mask using OpenCV or try Gaussian blur
            kernel_sz = 5
            kernel = np.ones((kernel_sz, kernel_sz), np.uint8)
            mask_np = cv2.dilate(mask_np, kernel, iterations=3)

            # Apply mask to image
            masked_img = np.zeros_like(filtered_image)
            masked_img[mask_np > 0] = filtered_image[mask_np > 0]
                
            # # Store numpy boolean fastsam mask
            # sam_masks[mask_idx,:,:] = mask_np

            # Extract region of interest
            masked_bb_img = masked_img[mask_y1:mask_y2, mask_x1:mask_x2]
            # if DEBUG_MODE and (scan_idx % scan_display_step_sz == 0) and (mask_idx % 25 == 0):
            #     cv2.imshow(f"FastSAM mask {mask_idx}", masked_bb_img)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            
            # CLIP Preprocessing
            masked_image = clip_preprocess(Image.fromarray(masked_bb_img)).unsqueeze(0).to(device)

            # Encode with CLIP
            with torch.no_grad():
                masked_encoded = clip_model.encode_image(masked_image)
            
            ## Add to list of observations
            masked_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
            filtered_lidar_img = np.bitwise_and(lidar_img_bool, masked_img_gray.astype(bool))
            y_indices, x_indices = np.where(filtered_lidar_img > 0)
            corresponding_global_idxs = []
            corresponding_global_pc = []
            for y, x in zip(y_indices, x_indices):
                g_idx = map_local2globalidx[map_yx2idx[(y,x)]]
                corresponding_global_idxs.append(g_idx)
                corresponding_global_pc.append(global_pc[g_idx,:3])
            if len(corresponding_global_idxs) == 0:
                continue
            
            if do_DBSCAN:
                ## Apply DBSCAN to global PC
                corresponding_global_idxs_arr = np.array(corresponding_global_idxs)
                corresponding_global_pc_arr = np.array(corresponding_global_pc)
                labels = dbscan_global.fit_predict(corresponding_global_pc_arr)
                unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
                if counts.size == 0 or counts.max() == 0:
                    continue
                largest_pts_cluster_sz = 0
                largest_global_pt_indices = None
                for l, count in zip(unique_labels,counts):
                    if count > largest_pts_cluster_sz:
                        largest_pts_cluster_sz = count
                        largest_global_pt_indices = corresponding_global_idxs_arr[labels==l]
                largest_global_pt_indices = largest_global_pt_indices.tolist()
            else:
                largest_global_pt_indices = corresponding_global_idxs
            if len(largest_global_pt_indices) == 0:
                continue
            fastsam_clip_embs.append(masked_encoded)            
            fastsam_global_idxs.append(largest_global_pt_indices)
        fastsam_clip_embs_tensor = torch.vstack(fastsam_clip_embs)
        scores = tensor_cosine_similarity(fastsam_clip_embs_tensor, clip_tensor) # (num_masks, num_clip_ids)
        
        ## Add in any new CLIP embedding vectors and associate global indices to CLIP vector index
        for mask_idx in range(scores.shape[0]): # num masks
            if scores[mask_idx,:].max().item() > theta_cos_sim:
                max_clip_id = scores[mask_idx,:].argmax().item()
            else:
                clip_tensor = torch.cat([clip_tensor, fastsam_clip_embs_tensor[mask_idx,:].unsqueeze(0)], dim=0)
                max_clip_id = clip_tensor.shape[0] - 1
            for g_idx in fastsam_global_idxs[mask_idx]:
                if g_idx in pc_dict:
                    if max_clip_id in pc_dict[g_idx]:
                        pc_dict[g_idx][max_clip_id] += 1
                    else:
                        pc_dict[g_idx][max_clip_id] = 1
                else:
                    pc_dict[g_idx] = {max_clip_id: 1}
        
        ## Save Semantic Point Cloud
        if (scan_idx % scan_display_step_sz == 0):
            with open(data_folder+f"/ptxpt_pc_dict_itr{scan_idx}.pkl", "wb") as f:
                pkl.dump(pc_dict, f)
            torch.save(clip_tensor,data_folder+f"/ptxpt_clip_tensor_itr{scan_idx}.pt")
            with open(data_folder+f"/ptxpt_gidx2imgidx_dict_itr{scan_idx}.pkl", "wb") as f:
                pkl.dump(map_globalidx2imgidx, f)
            img_clip_tensor = torch.vstack(img_clips)
            torch.save(img_clip_tensor,data_folder+f"/img_clip_tensor_itr{scan_idx}.pt")
            with open(data_folder+f"/saved_img_names_itr{scan_idx}.pkl","wb") as f:
                pkl.dump(saved_img_names, f)
            
            print(f"\nSaved pc_dict and clip_tensor at iteration {scan_idx}\n")
            
        
        ## Debugging/Displaying Progress
        rgb_colors = [[1,0,0],[0,1,0],[0,0,1]]
        count_threshold = 2
        # Display Global Point Cloud colored by CLIP IDs
        if DEBUG_MODE and (scan_idx % scan_display_step_sz == 0):
            global_pts = {}    
            for global_idx in range(global_pc.shape[0]):
                if global_idx in pc_dict.keys():
                    max_class, max_count = max(pc_dict[global_idx].items(), key=lambda x: x[1])
                    # Make sure max_count is more than some threshold
                    if max_count < count_threshold:
                        if -1 in global_pts.keys():
                            global_pts[-1].append(global_idx)
                        else:
                            global_pts[-1] = [global_idx]    
                    elif max_class in global_pts.keys():
                        global_pts[max_class].append(global_idx)
                    else:
                        global_pts[max_class] = [global_idx]
                else:
                    if -1 in global_pts.keys():
                        global_pts[-1].append(global_idx)
                    else:
                        global_pts[-1] = [global_idx]
            pcds = []
            for class_id in global_pts.keys():
                pcd = o3d.geometry.PointCloud()
                if class_id == -1:
                    pcd.points = o3d.utility.Vector3dVector(global_pc[global_pts[class_id],:3])
                    pcd.paint_uniform_color([0.5,0.5,0.5])
                else:
                    if class_id < len(rgb_colors):
                        pcd.points = o3d.utility.Vector3dVector(global_pc[global_pts[class_id], :3])
                        pcd.paint_uniform_color(rgb_colors[class_id])
                    else:
                        # Otherwise, generate a random color
                        random_col = random_color()
                        pcd.points = o3d.utility.Vector3dVector(global_pc[global_pts[class_id], :3])
                        pcd.paint_uniform_color(random_col)
                pcds.append(pcd)
            o3d.visualization.draw_geometries(pcds)

        itr_t1 = time.time()
        num_scans += 1
        print(f"Iteration {scan_idx} runtime:",itr_t1 - itr_t0,"sec")
    t_end = time.time()
    print("Finished iterating through all lidar scans in",t_end-t_begin,"seconds for",num_scans,"scans")
    
    ## Save Semantic Point Cloud
    with open(data_folder+f"/ptxpt_pc_dict_itr{num_scans}.pkl", "wb") as f:
        pkl.dump(pc_dict, f)
    torch.save(clip_tensor,data_folder+f"/ptxpt_clip_tensor_itr{num_scans}.pt")
    with open(data_folder+f"/ptxpt_gidx2imgidx_dict_itr{num_scans}.pkl", "wb") as f:
        pkl.dump(map_globalidx2imgidx, f)
    img_clip_tensor = torch.vstack(img_clips)
    torch.save(img_clip_tensor,data_folder+f"/img_clip_tensor_itr{num_scans}.pt")
    with open(data_folder+f"/saved_img_names_itr{num_scans}.pkl","wb") as f:
        pkl.dump(saved_img_names, f)
    
    print(f"\nSaved pc_dict and clip_tensor at iteration {num_scans}\n")
    
    ## Visualize the final global map colored by class
    global_pts = {}    
    for global_idx in range(global_pc.shape[0]):
        if global_idx in pc_dict.keys():
            max_class, max_count = max(pc_dict[global_idx].items(), key=lambda x: x[1])
            # Make sure max_count is more than some threshold
            if max_count < count_threshold:
                if -1 in global_pts.keys():
                    global_pts[-1].append(global_idx)
                else:
                    global_pts[-1] = [global_idx]    
            elif max_class in global_pts.keys():
                global_pts[max_class].append(global_idx)
            else:
                global_pts[max_class] = [global_idx]
        else:
            if -1 in global_pts.keys():
                global_pts[-1].append(global_idx)
            else:
                global_pts[-1] = [global_idx]
    pcds = []
    for class_id in global_pts.keys():
        pcd = o3d.geometry.PointCloud()
        if class_id == -1:
            pcd.points = o3d.utility.Vector3dVector(global_pc[global_pts[class_id],:3])
            pcd.paint_uniform_color([0.5,0.5,0.5])
        else:
            if class_id < len(rgb_colors):
                pcd.points = o3d.utility.Vector3dVector(global_pc[global_pts[class_id], :3])
                pcd.paint_uniform_color(rgb_colors[class_id])
            else:
                # Otherwise, generate a random color
                random_col = random_color()
                pcd.points = o3d.utility.Vector3dVector(global_pc[global_pts[class_id], :3])
                pcd.paint_uniform_color(random_col)
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)