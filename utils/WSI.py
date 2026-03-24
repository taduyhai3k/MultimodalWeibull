import openslide
import numpy as np
import cv2
import pickle
from ContourCheck import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, Contour_Checking_fn
import os
import torch
from PIL import Image, ImageDraw





class WholeSlideImage:
    def __init__(self, path, patch_size=512, final_size=512, level=0):
        self.slide = openslide.OpenSlide(path)
        self.level_count = self.slide.level_count
        self.dimensions = self.slide.dimensions
        self.level_dimensions = self.slide.level_dimensions
        self.level_downsamples = self._assertLevelDownsamples()
        self.patch_size = patch_size
        self.final_size = final_size
        self.contours_tissue = None
        self.contours_tumor = None
    def _assertLevelDownsamples(self):
        r"""The level_downsamples provided by OpenSlide may not be accurate. This function estimates the downsample factors based on the level dimensions and compares them to the provided downsample factors. If they differ, it uses the estimated values instead.
        """
        level_downsamples = []
        dim_0 = self.level_dimensions[0]
        
        for downsample, dim in zip(self.level_downsamples,   self.level_dimensions):
            estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
        
        return level_downsamples
    def initSegmentation(self, mask_file):
        r"""Initializes the segmentation by loading the mask data from a pickle file. The mask data is expected to contain 'holes' and 'tissues' which are stored in the instance variables holes_tissue and contours_tissue respectively.
        """
        with open(mask_file, 'rb') as f:
            mask_data = pickle.load(f)
        self.holes_tissue = mask_data['holes']
        self.contours_tissue = mask_data['tissues']
    def _filter_contours(self, contours, hierarchy, filter_params):
        filtered = []
        hierarchy_parent = np.flatnonzero(hierarchy[:,1]==-1)
        all_holes = []
        for idx in hierarchy_parent:
            contour = contours[idx]
            holes_indice_in_contour = np.flatnonzero(hierarchy[:,1]==idx)
            contour_area = cv2.contourArea(contour)
            holes_area = [cv2.contourArea(contours[i]) for i in holes_indice_in_contour]
            real_area = contour_area - sum(holes_area)
            if real_area == 0:
                continue
            elif tuple(filter_params['real_area_patch'],) < tuple(real_area,):
                filtered.append(idx)
                all_holes.append(holes_indice_in_contour)
        foreground_contours = [contours[i] for i in filtered]
        hole_contours = []
        for index in all_holes:
            unfiltered_hole = [contours[i] for i in index]
            unfiltered_hole = sorted(unfiltered_hole, key = lambda x: cv2.contourArea(x), reverse=True)
            unfiltered_hole = unfiltered_hole[:filter_params['max_n_holes']]
            filtered_hole = [hole for hole in unfiltered_hole if cv2.contourArea(hole) > filter_params['min_hole_area_patch']]
            hole_contours.append(filtered_hole)
        return foreground_contours, hole_contours
    def segmentTissue(self, seg_level = 0, sthresh = 20,sthresh_up = 255, mthresh = 7, close = 0, use_otsu = False, ref_patch_size = 512, exclude_id = [], keep_id = [] ,filter_params = {'real_area_patch': 1000, 'min_hole_area_patch': 500, 'max_n_holes': 5}):
        r"""Segments the tissue in the whole slide image at a specified level using a combination of color thresholding, median filtering, and morphological operations. The function allows for various parameters to control the segmentation process, including thresholds for binary segmentation, median filter size, and morphological closing. It also includes options for filtering contours based on area and the number of holes, as well as excluding or keeping specific contour IDs."""
        img = np.array(self.slide.read_region((0,0), seg_level, self.level_dimensions[seg_level]))
        img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_median = cv2.medianBlur(img_HSV[:,:,1], mthresh)
        if use_otsu:
            _, img_binary = cv2.threshold(img_median, 0, sthresh_up, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, img_binary = cv2.threshold(img_median, sthresh, sthresh_up, cv2.THRESH_BINARY)
        if close > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close, close))
            img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
        scale_factor = self.level_downsamples[seg_level]
        scale_factor_ref= int(ref_patch_size**2/ (scale_factor[0] * scale_factor[1]))   
        filter_params = filter_params.copy()
        filter_params['real_area_patch'] = filter_params['real_area_patch'] * scale_factor_ref
        filter_params['min_hole_area_patch'] = filter_params['min_hole_area_patch'] * scale_factor_ref
        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:,2:]
        if filter_params is not None:
            foreground_contours, hole_contours = self._filter_contours(contours, hierarchy, filter_params)
        self.contours_tissue = self.scaleContour(foreground_contours, scale_factor)
        self.holes_tissue = self.scaleHole(hole_contours, scale_factor)
        if len(keep_id) > 0:
            contours_id = set(keep_id) - set(exclude_id)
        else:
            contours_id = set(range(len(self.contours_tissue))) - set(exclude_id)
        self.contours_tissue = [self.contours_tissue[i] for i in contours_id]
        self.holes_tissue = [self.holes_tissue[i] for i in contours_id]
        
    def scaleContour(self, contours, scale_factor):
        return [np.array(cont * scale_factor, dtype=np.int32) for cont in contours]
    def scaleHole(self, holes, scale_factor):
        return [[np.array(hole * scale_factor, dtype=np.int32) for hole in hole_group] for hole_group in holes] 
    def _getpatchGenerator(self, contour, holes,level, patch_size, final_size, white_black = True, white_thresh = 15, black_thresh = 50, contour_fn = "four_pt"):
        start_x, start_y, width, height = cv2.boundingRect(contour) if contour is not None else (0, 0, self.dimensions[level][0], self.dimensions[level][1])
        step = patch_size
        if isinstance(contour_fn, str):
            if contour_fn == "four_pt":
                contour_check_fn = isInContourV3_Easy(contour= contour, patch_size= patch_size)
            elif contour_fn == "four_pt_hard":
                contour_check_fn = isInContourV3_Hard(contour= contour, patch_size= patch_size)
            elif contour_fn == "center_pt":
                contour_check_fn = isInContourV2(contour= contour, patch_size= patch_size)
            elif contour_fn == "basic":
                contour_check_fn = isInContourV1(contour= contour)
            else:
                raise ValueError("Invalid contour_fn string. Expected one of 'four_pt', 'four_pt_hard', 'center_pt', or 'basic'.")
        else:
            assert isinstance(contour_fn, Contour_Checking_fn), "contour_fn must be a string or an instance of Contour_Checking_fn"
            contour_check_fn = contour_fn
        img_w, img_h = self.dimensions[level]
        stop_x = start_x + width
        stop_y = start_y + height
        count = 0
        for y in range(start_y, stop_y, step):
            for x in range(start_x, stop_x, step):
                if not self.isInContours(contour_check_fn, (x, y), holes, patch_size):
                    continue
                count += 1
                patch_image = self.slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
                patch_image = patch_image.resize((final_size, final_size))
                if white_black:
                    if self.isWhitePatch(np.array(patch_image), white_thresh) or self.isBlackPatch(np.array(patch_image), black_thresh):
                        continue
                yield patch_image, (x, y)


    def isInHoles(self, holes, pt, patch_size):
        r"""Checks if a given point is within any of the specified holes. The function iterates through each hole and uses the cv2.pointPolygonTest to determine if the point is inside the hole. If the point is found to be inside any hole, the function returns 1; otherwise, it returns 0 after checking all holes."""
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                return 1
        
        return 0
    def isInContours(self, cont_check_fn, pt, holes=None, patch_size=256):
        r"""Determines if a given point is within the contours defined by the cont_check_fn and not within any holes. The function first checks if the point is within the contours using the provided contour checking function. If it is, and if there are holes defined, it then checks if the point is within any of the holes. If the point is within the contours and not within any holes, it returns 1; otherwise, it returns 0."""
        if cont_check_fn(pt):
            if holes is not None:
                return not self.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0
    def isWhitePatch(self, patch, satThresh=5):
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        return True if np.mean(patch_hsv[:,:,1]) < satThresh else False

    def isBlackPatch(self,  patch, rgbThresh=40):
        return True if np.all(np.mean(patch, axis = (0,1)) < rgbThresh) else False

    def isBlackPatch_S(self, patch, rgbThresh=20, percentage=0.05):
        num_pixels = patch.size[0] * patch.size[1]
        return True if np.all(np.array(patch) < rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

    def isWhitePatch_S(self, patch, rgbThresh=220, percentage=0.2):
        num_pixels = patch.size[0] * patch.size[1]
        return True if np.all(np.array(patch) > rgbThresh, axis=(2)).sum() > num_pixels * percentage else False
    

class EnhancedWSI(WholeSlideImage):
    def generate_visualization(self, coords, patch_size, vis_level=4):
        """Tạo ảnh thumbnail nhỏ và tô màu các vùng đã lấy patch"""
        # 1. Lấy ảnh thumbnail của toàn bộ slide
        total_dim = self.slide.dimensions
        thumb_img = self.slide.get_thumbnail(self.slide.level_dimensions[vis_level])
        thumb_img = thumb_img.convert('RGB')
        draw = ImageDraw.Draw(thumb_img)
        
        # 2. Tính tỉ lệ scale giữa Level 0 và Level hiển thị
        scale_x, scale_y = self.level_downsamples[vis_level]
        
        # 3. Vẽ các ô vuông
        for x, y in coords:
            # Quy đổi tọa độ về level hiển thị
            x_vis, y_vis = x / scale_x, y / scale_y
            p_vis_x = patch_size / scale_x
            p_vis_y = patch_size / scale_y
            
            # Vẽ hình chữ nhật (màu xanh lá nhạt, viền đậm)
            draw.rectangle([x_vis, y_vis, x_vis + p_vis_x, y_vis + p_vis_y], outline="lime", width=1)
            
        return thumb_img

    def process_and_extract(self, extractor, batch_size, save_feature_dir):
        """Hàm tổng hợp: Cắt patch và trích xuất đặc trưng ngay lập tức"""
        os.makedirs(save_feature_dir, exist_ok=True)
        slide_name = os.path.basename(self.slide._filename).split('.')[0]
        all_features = []
        all_coords = []

        # Tự động segment nếu chưa làm
        if self.contours_tissue is None:
            self.segmentTissue()

        print(f"Extracting features for {slide_name}...")
        list_of_patches = []    
        count = 0
        for i, cont in enumerate(self.contours_tissue):
            # Sử dụng generator em đã viết
            patch_gen = self._getpatchGenerator(
                cont, self.holes_tissue[i], level=0, 
                patch_size=self.patch_size, final_size=self.final_size
            )
            if count < batch_size:
                list_of_patches.append(patch_gen)
                count += 1
            else:
                patchs = []
                coords = []
                for patch_and_coord in list_of_patches:
                    patchs.append(patch_and_coord[0])
                    coords.append(patch_and_coord[1])
                features = extractor.extract_batch(patchs)
                all_features.append(features.unsqueeze(0))
                all_coords.append(torch.tensor(coords).unsqueeze(0))
                count = 0
        self.generate_visualization(coords, self.patch_size).save(os.path.join(save_feature_dir, f"{slide_name}_visualization.png"))
        if len(all_features) > 0:
            # Gom tất cả đặc trưng của 1 slide vào 1 file duy nhất để tối ưu
            
            combined_data = {
                'features': torch.cat(all_features, dim=0), # [N, 768]
                'coords': torch.stack(all_coords)           # [N, 2]
            }
            save_path = os.path.join(save_feature_dir, f"{slide_name}_features.pt")
            torch.save(combined_data, save_path)
            print(f"Saved {len(all_features)} patches to {save_path}")
            return save_path            

