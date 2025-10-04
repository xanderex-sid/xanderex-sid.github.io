import argparse
import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
import gc
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path
from abc import ABC, abstractmethod

import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

def paths_from_json(json_file_path: str) -> List[str]:
    """Load image paths from JSON file."""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return data['image_paths']
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found at: {json_file_path}")
    except KeyError:
        raise KeyError("JSON file must contain 'image_paths' key")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Invalid JSON file: {json_file_path}")

class ImageUtils:
    @staticmethod
    def get_jpg_files(root_dir: str) -> List[str]:
        """Get all JPG files in directory recursively."""
        jpg_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    jpg_files.append(os.path.join(root, file))
        return jpg_files

    @staticmethod
    def smallest_circumscribing_rectangle(
        x1: float, y1: float, x2: float, y2: float,
        x3: float, y3: float, x4: float, y4: float,
        buffer: float = 0.05
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """Calculate smallest rectangle containing the points with buffer."""
        points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)

        return (
            min_x*(1 - buffer), min_y*(1 - buffer),
            min_x*(1 - buffer), max_y*(1 + buffer),
            max_x*(1 + buffer), max_y*(1 + buffer),
            max_x*(1 + buffer), min_y*(1 - buffer)
        )

class TilingUtils:
    def safe_polygon(coords):
        poly = Polygon(coords)
        if not poly.is_valid or poly.area == 0:
            return None
        return poly
        
    def parse_points(x_str, y_str):
        x = list(map(float, x_str.split(',')))
        y = list(map(float, y_str.split(',')))
        return list(zip(x, y))
    
    def to_counter_clockwise(coords):
        poly = Polygon(coords)
        if not poly.exterior.is_ccw:
            coords = list(poly.exterior.coords)[::-1][:-1]
        return coords
    
    def polygon_to_str_coords(polygon: Polygon):
        coords = list(polygon.exterior.coords)[:-1]
        xpoints = ",".join([f"{pt[0]}" for pt in coords])
        ypoints = ",".join([f"{pt[1]}" for pt in coords])
        return xpoints, ypoints
    
    def minimum_rotated_rectangle(polygon: Polygon):
        return polygon.minimum_rotated_rectangle
    
    def split_rotated_rectangle(rect: Polygon, parts: int):
        rect_coords = np.array(rect.exterior.coords[:-1])  # 4 corners
        vec1 = rect_coords[1] - rect_coords[0]
        vec2 = rect_coords[3] - rect_coords[0]
    
        # Ensure width is the longer side
        if np.linalg.norm(vec1) > np.linalg.norm(vec2):
            width_vec_raw = vec1
            height_vec_raw = vec2
        else:
            width_vec_raw = vec2
            height_vec_raw = vec1
    
        width_len = np.linalg.norm(width_vec_raw)
        height_len = np.linalg.norm(height_vec_raw)
    
        width_vec = width_vec_raw / width_len
        height_vec = height_vec_raw / height_len
    
        origin = rect_coords[0]
        tile_polys = []
    
        tile_width = width_len / parts
        tile_height = height_len
    
        for i in range(parts):
            start = origin + width_vec * tile_width * i
            corners = [
                start,
                start + width_vec * tile_width,
                start + width_vec * tile_width + height_vec * tile_height,
                start + height_vec * tile_height
            ]
            tile_polys.append(Polygon(corners))
    
        return tile_polys

    
    def calculate_iou_over_defect(defect_poly, parent_poly):
        intersection = defect_poly.intersection(parent_poly).area
        return intersection / defect_poly.area if defect_poly.area > 0 else 0

# ------------

class ComponentProcessor:
    """
    This Processor does all the processes for component detection.
    """
    def __init__(self, strategy: ComponentStrategy = StandardComponentDetectionStrategy(), strategy_args: dict) -> None:
        """
        Here we initialise our processor.
        """

        self._strategy = strategy
        self.strategy_args = strategy_args

    @property
    def strategy(self) -> ComponentStrategy:
        """
        Maintaining reference to one of the ComponentStrategy objects.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: ComponentStrategy) -> None:
        """
        To change ComponentStrategy object at runtime.
        """
        self._strategy = strategy

    @property
    def strategy_args(self) -> dict:
        """
        Maintaining reference to one of the strategy arguments.
        """
        return self._strategy_args

    @strategy.setter
    def strategy_args(self, strategy_args: dict) -> None:
        """
        To change strategy arguments at runtime.
        """
        self._strategy_args = strategy_args

    def detect(self, images: List[str]) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Detect components in images using batched processing."""

        self._strategy_args["images"] = images
        self._strategy_args["model"] = YOLO(self._strategy_args["model_path"])

        if "batch_size" not in self._strategy_args:
            self._strategy_args["batch_size"] = 4
        
        components_by_type, data = self._strategy.detect(**self._strategy_args)
        return components_by_type, data
        

# -------

class ComponentStrategy(ABC):
    """
    The ComponentStrategy interface declares operations common to all supported versions
    of detection algorithm.

    The ComponentProcessor uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    # def _process_batch(self, model, image_batch: List[str]) -> List:
    #     """Process a batch of images with memory cleanup."""
    #     try:
    #         with torch.cuda.amp.autocast():
    #             results = model(image_batch)
    #         torch.cuda.synchronize()
    #         gc.collect()
    #         torch.cuda.empty_cache()
    #         return results
    #     except RuntimeError as e:
    #         if "out of memory" in str(e):
    #             print("CUDA OOM detected, clearing cache and reducing batch...")
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #             results = []
    #             for img in image_batch:
    #                 try:
    #                     result = model([img])
    #                     results.extend(result)
    #                 except Exception as e:
    #                     print(f"Error processing single image: {str(e)}")
    #             return results
    #         raise e

    def _process_batch(self, model, image_batch: List[str]) -> List:
        """Process a batch of images on MPS with memory cleanup (no CUDA-specific code)."""
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        try:
            # Inference on MPS (no autocast support)
            results = model(image_batch)
            gc.collect()
            return results
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("MPS OOM detected, reducing batch size...")
                gc.collect()
                results = []
                for img in image_batch:
                    try:
                        result = model([img])
                        results.extend(result)
                    except Exception as e:
                        print(f"Error processing single image: {str(e)}")
                return results
            raise e

    @abstractmethod
    def detect(self, images: List[str], model, label_dict: dict, batch_size: int, **kargs) -> Tuple[Dict[str, List], Dict[str, List]]:
        pass


class StandardComponentDetectionStrategy(ComponentStrategy):
    def detect(self, images: List[str], model, label_dict: dict, batch_size: int, **kargs) -> Tuple[Dict[str, List], Dict[str, List]]:
        components_by_type = defaultdict(list)
        data = {}

        for i in tqdm(range(0, len(images), batch_size), desc="Processing component batches"):
            batch = images[i:i + batch_size]
            
            try:
                results = self._process_batch(model, batch)
                
                for r in results:
                    orients = r.obb
                    obboxes = orients.xyxyxyxy
                    classes = orients.cls
                    path = r.path

                    preds = []
                    for idx in range(len(classes)):
                        component_name = label_dict[str(int(classes[idx]))]
                        component_bbox = obboxes[idx].cpu().numpy().tolist()

                        components_by_type[component_name].append([
                            path, component_name, component_bbox
                        ])
                        preds.append((component_name, component_bbox))

                    data[path] = preds
                    
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue

            torch.cuda.empty_cache()
            gc.collect()

        return components_by_type, data

# -------

class CroppingProcessor:
    """
    This Processor does cropping of detected components.
    """
    def __init__(self, strategy: CroppingStrategy = StandardCroppingStrategy(), strategy_args: dict = {}, output_dir: str) -> None:
        """
        Here we initialise our processor.
        """

        self._strategy = strategy
        self._strategy_args = strategy_args

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @property
    def strategy(self) -> ComponentStrategy:
        """
        Maintaining reference to one of the ComponentStrategy objects.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: ComponentStrategy) -> None:
        """
        To change ComponentStrategy object at runtime.
        """
        self._strategy = strategy

    @property
    def strategy_args(self) -> dict:
        """
        Maintaining reference to one of the strategy arguments.
        """
        return self._strategy_args

    @strategy.setter
    def strategy_args(self, strategy_args: dict) -> None:
        """
        To change strategy arguments at runtime.
        """
        self._strategy_args = strategy_args

    def crop(self, components_by_type: Dict[str, List])  -> Dict[str, Dict[str, List]]:
        """This method crops the component"""

        self._strategy_args["components_by_type"] = components_by_type
        self._strategy_args["output_dir"] = self.output_dir

        if "batch_size" not in self._strategy_args:
            self._strategy_args["batch_size"] = 16

        mapping = self._strategy.crop(**self._strategy_args)
        return mapping

class CroppingStrategy(ABC):
    """
    The CroppingStrategy interface declares operations common to all supported versions
    of cropping.

    The CroppingStrategy uses this interface to call the algorithm defined by Concrete
    Strategies.
    """
    @abstractmethod
    def crop(self, components_by_type: Dict[str, List], component_type: str, buffer: float, output_dir: str, batch_size: int = 16, **kargs)  -> Dict[str, Dict[str, List]]:
        pass

class StandardCroppingStrategy(CroppingStrategy):
    def crop(self, components_by_type: Dict[str, List], component_type: str, buffer: float, output_dir: str, batch_size: int = 16, **kargs)  -> Dict[str, Dict[str, List]]:
        """Crop detected components using batched processing."""
        mapping = {component_type: {'original_image': [], 'cropped_image': [], 'bbox': []}}

        if component_type not in components_by_type:
            print(f"Skipping {component_type} - not found")
            return mapping

        print(f"Processing {component_type}...")
        df = pd.DataFrame(
            components_by_type[component_type],
            columns=['names', 'labels', 'bboxes']
        )

        component_dir = os.path.join(output_dir, component_type)
        os.makedirs(component_dir, exist_ok=True)

        unique_files = df['names'].unique()
        for i in tqdm(range(0, len(unique_files), batch_size), 
                        desc=f"Processing {component_type} batches"):
            batch_files = unique_files[i:i + batch_size]
            
            for file_path in batch_files:
                df_temp = df[df['names'] == file_path]

                try:
                    image = cv2.imread(file_path)
                    if image is None:
                        return mapping

                    subdir_name = os.path.basename(file_path).split('.')[0]
                    subdir_path = os.path.join(component_dir, subdir_name)
                    os.makedirs(subdir_path, exist_ok=True)

                    for idx, (_, row) in enumerate(df_temp.iterrows()):
                        try:
                            points = row['bboxes']

                            x1, y1, x2, y2, x3, y3, x4, y4 = ImageUtils.smallest_circumscribing_rectangle(
                                points[0][0], points[0][1],
                                points[1][0], points[1][1],
                                points[2][0], points[2][1],
                                points[3][0], points[3][1],
                                buffer
                            )

                            width = int(x3 - x1)
                            height = int(y3 - y1)

                            if width <= 0 or height <= 0:
                                return mapping

                            y1, y3 = max(0, int(y1)), min(image.shape[0], int(y3))
                            x1, x3 = max(0, int(x1)), min(image.shape[1], int(x3))

                            cropped = image[int(y1):int(y3), int(x1):int(x3)]
                            output_path = os.path.join(subdir_path, f'{idx}.jpg')

                            cv2.imwrite(output_path, cropped)

                            mapping[component_type]['original_image'].append(file_path)
                            mapping[component_type]['cropped_image'].append(output_path)
                            mapping[component_type]['bbox'].append({
                                'x1': x1, 'y1': y1,
                                'x2': x2, 'y2': y2,
                                'x3': x3, 'y3': y3,
                                'x4': x4, 'y4': y4,
                                'width': width,
                                'height': height
                            })
                        
                        except Exception as e:
                            print(f"Error processing detection: {str(e)}")

                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")

            gc.collect()
        
        return mapping

class RotatedCroppingStrategy(CroppingStrategy):
    def crop(self, components_by_type: Dict[str, List], component_type: str, buffer: float, output_dir: str, batch_size: int = 16, **kargs)  -> Dict[str, Dict[str, List]]:
        """Crop detected components using batched processing."""
        mapping = {component_type: {'original_image': [], 'cropped_image': [], 'bbox': []}}

        if component_type not in components_by_type:
            print(f"Skipping {component_type} - not found")
            return mapping

        print(f"Processing {component_type}...")
        df = pd.DataFrame(
            components_by_type[component_type],
            columns=['names', 'labels', 'bboxes']
        )

        component_dir = os.path.join(output_dir, component_type)
        os.makedirs(component_dir, exist_ok=True)

        unique_files = df['names'].unique()
        for i in tqdm(range(0, len(unique_files), batch_size), 
                        desc=f"Processing {component_type} batches"):
            batch_files = unique_files[i:i + batch_size]
            
            for file_path in batch_files:
                df_temp = df[df['names'] == file_path]

                try:
                    image = cv2.imread(file_path)
                    if image is None:
                        return mapping

                    subdir_name = os.path.basename(file_path).split('.')[0]
                    subdir_path = os.path.join(component_dir, subdir_name)
                    os.makedirs(subdir_path, exist_ok=True)

                    for idx, (_, row) in enumerate(df_temp.iterrows()):
                        try:
                            points = row['bboxes']

                            output_path = os.path.join(subdir_path, f'{idx}.jpg')
                                width, height, ordered_box_list = self.save_rotated_parents(
                                    [points[0][0], points[0][1],
                                    points[1][0], points[1][1],
                                    points[2][0], points[2][1],
                                    points[3][0], points[3][1]], image, output_path, (buffer * 100)
                                )

                                x1, y1, x2, y2, x3, y3, x4, y4 = (
                                    points[0][0], points[0][1],
                                    points[1][0], points[1][1],
                                    points[2][0], points[2][1],
                                    points[3][0], points[3][1]
                                )

                                mapping[component_type]['original_image'].append(file_path)
                                mapping[component_type]['cropped_image'].append(output_path)
                                mapping[component_type]['bbox'].append({
                                    'x1': x1, 'y1': y1,
                                    'x2': x2, 'y2': y2,
                                    'x3': x3, 'y3': y3,
                                    'x4': x4, 'y4': y4,
                                    'width': width,
                                    'height': height,
                                    'ordered_box': ordered_box_list
                                })
                        
                        except Exception as e:
                            print(f"Error processing detection: {str(e)}")

                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")

            gc.collect()
        
        return mapping 

    def save_rotated_parents(self, obb_coords, image, output_path, buffer=50):
        import cv2
        import numpy as np
        from shapely.geometry import Polygon
    
        pts = np.array([(obb_coords[i], obb_coords[i + 1]) for i in range(0, 8, 2)], dtype=np.float32)
    
        polygon = TilingUtils.safe_polygon(pts)
        if polygon is None:
            raise ValueError("Invalid polygon from OBB coords.")
    
        buffered_polygon = polygon.buffer(buffer)
    
        mrr = TilingUtils.minimum_rotated_rectangle(buffered_polygon)
        box = np.array(mrr.exterior.coords)[:-1].astype(np.float32)
    
        # Order the rectangle points: top-left, top-right, bottom-right, bottom-left
        def order_points(pts):
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            return np.array([
                pts[np.argmin(s)],
                pts[np.argmin(diff)],
                pts[np.argmax(s)],
                pts[np.argmax(diff)]
            ], dtype=np.float32)
    
        ordered_box = order_points(box)

        width = int(np.linalg.norm(ordered_box[0] - ordered_box[1]))
        height = int(np.linalg.norm(ordered_box[0] - ordered_box[3]))
    
        if width <= 0 or height <= 0:
            raise ValueError("Zero width or height detected.")
    
        # Defining destination points for un-rotated rectangle
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
    
        # Get transformation matrix and warp
        M = cv2.getPerspectiveTransform(ordered_box, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))

        cv2.imwrite(output_path, warped)
    
        return width, height, ordered_box.tolist()      

# -------

class DefectProcessor:
    """
    This Processor does all the processes for defect detection.
    """
    def __init__(self, strategy: DefectStrategy = StandardDefectDetectionStrategy(), strategy_args: dict = {}) -> None:
        """
        Here we initialise our processor.
        """

        self._strategy = strategy
        self.strategy_args = strategy_args

    @property
    def strategy(self) -> DefectStrategy:
        """
        Maintaining reference to one of the DefectStrategy objects.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: DefectStrategy) -> None:
        """
        To change DefectStrategy object at runtime.
        """
        self._strategy = strategy

    @property
    def strategy_args(self) -> dict:
        """
        Maintaining reference to one of the strategy arguments.
        """
        return self._strategy_args

    @strategy.setter
    def strategy_args(self, strategy_args: dict) -> None:
        """
        To change strategy arguments at runtime.
        """
        self._strategy_args = strategy_args

    def detect(self, mapping, components_by_type) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Detect components in images using batched processing."""

        self._strategy_args["mapping"] = mapping
        self._strategy_args["components_by_type"] = components_by_type

        if "model" not in self._strategy_args:
            self._strategy_args["model"] = YOLO(self._strategy_args["model_path"])

        if "batch_size" not in self._strategy_args:
            self._strategy_args["batch_size"] = 4

        components_by_type, data = self._strategy.detect(**self._strategy_args)
        return components_by_type, data


# -------

class DefectStrategy(ABC):
    """
    The DefectStrategy interface declares operations common to all supported versions
    of detecting defects.

    The DefectStrategy uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    # def _process_batch(self, model, image_batch: List[str]) -> List:
    #     """Process a batch of images with memory cleanup."""
    #     try:
    #         with torch.cuda.amp.autocast():
    #             results = model(image_batch)
    #         torch.cuda.synchronize()
    #         gc.collect()
    #         torch.cuda.empty_cache()
    #         return results
    #     except RuntimeError as e:
    #         if "out of memory" in str(e):
    #             print("CUDA OOM detected, clearing cache and reducing batch...")
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #             results = []
    #             for img in image_batch:
    #                 try:
    #                     result = model([img])
    #                     results.extend(result)
    #                 except Exception as e:
    #                     print(f"Error processing single image: {str(e)}")
    #             return results
    #         raise e

    def _process_batch(self, model, image_batch: List[str]) -> List:
        """Process a batch of images on MPS with memory cleanup (no CUDA-specific code)."""
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        try:
            # Inference on MPS (no autocast support)
            results = model(image_batch)
            gc.collect()
            return results
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("MPS OOM detected, reducing batch size...")
                gc.collect()
                results = []
                for img in image_batch:
                    try:
                        result = model([img])
                        results.extend(result)
                    except Exception as e:
                        print(f"Error processing single image: {str(e)}")
                return results
            raise e

    @abstractmethod
    def detect(self, mapping: Dict[str, Dict], components_by_type: Dict[str, List], target_component: str, model, batch_size: int = 4, **kargs) -> Tuple[Dict[str, List], Dict[str, List]]:
        pass

class StandardDefectDetectionStrategy(DefectStrategy):
    """Standard way of detecting defects"""
    def detect(self, mapping: Dict[str, Dict], components_by_type: Dict[str, List], target_component: str, model, batch_size: int = 4, **kargs) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Detect defects in cropped components using batched processing."""
        combined_detailed = {}
        combined_simplified = {}

        # mapping --> {component_name : {original_image : image name/ path, cropped_image : [crpimg1, crpimg2,...]}}
        all_images = set()
        component_name = target_component
        if component_name in mapping:
            all_images.update(mapping[component_name]['original_image'])
        
        for img in all_images:
            combined_detailed[img] = []
            combined_simplified[img] = []

        print(f"\nProcessing defects for {component_name}...")

        cropped_images = mapping[component_name]['cropped_image']
        if not cropped_images:
            print(f"No cropped images of {component_name} found...")
            return combined_detailed, combined_simplified

        component_defect_data = {}
            
        print(f"Using model: {model} for {component_name}...")
        
        for i in tqdm(range(0, len(cropped_images), batch_size), 
                        desc=f"Processing {component_name} defect batches with model: {model}"):
            batch = cropped_images[i:i + batch_size]

            try:
                results = self._process_batch(model, batch)
                
                for r in results:
                    orients = r.obb
                    obboxes = orients.xyxyxyxy
                    classes = orients.cls
                    defect_path = r.path
                    defect_label_dict = r.names

                    preds = []
                    for idx in range(len(classes)):
                        defect_class = defect_label_dict[int(classes[idx])]
                        defect_bbox = obboxes[idx].cpu().numpy().tolist()
                        preds.append((defect_class, defect_bbox))

                    if defect_path not in component_defect_data:
                        component_defect_data[defect_path] = []
                    component_defect_data[defect_path].extend(preds)
                    
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue

            torch.cuda.empty_cache()
            gc.collect()
        
        #print(component_defect_data)
        detailed_result, simplified_result = self._map_defects(
            mapping[component_name],
            components_by_type[component_name],
            component_defect_data,
            component_name
        )
        #print(mapping[component_name])

        for image_path, detections in detailed_result.items():
            for detection in detections:
                detection['component_type'] = component_name
                combined_detailed[image_path].append(detection)

        for image_path, detections in simplified_result.items():
            for defect_class, defect_points in detections:
                combined_simplified[image_path].append((
                    component_name,
                    defect_class,
                    defect_points,
                ))

        torch.cuda.empty_cache()
        gc.collect()

        return combined_detailed, combined_simplified

    @staticmethod
    def _map_defects(
        component_mapping_data: Dict[str, List],
        component_original_detections: List[List],
        component_defect_detections: Dict[str, List],
        component_name: str
    ) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Map defect coordinates back to original image."""
        detailed_result = {}
        simplified_result = {}
        
        crop_to_original = {}
        for i in range(len(component_mapping_data['original_image'])):
            crop_to_original[component_mapping_data['cropped_image'][i]] = {
                'original_image': component_mapping_data['original_image'][i],
                'original_bbox': component_mapping_data['bbox'][i],
            }

        all_original_images = set(component_mapping_data['original_image'])
        for img in all_original_images:
            simplified_result[img] = []
            detailed_result[img] = []

        for cropped_path, defects in component_defect_detections.items():
            #print(defects)
            if cropped_path not in crop_to_original:
                continue

            original_info = crop_to_original[cropped_path]
            original_image = original_info['original_image']
            original_bbox = original_info['original_bbox']


            component_num = int(os.path.splitext(os.path.basename(cropped_path))[0])

            for defect_class, defect_points in defects:
                # old non-rotated logic
                projected_points = []
                for x, y in defect_points:
                    proj_x = x + original_bbox['x1']
                    proj_y = y + original_bbox['y1']
                    projected_points.append([proj_x, proj_y])
    
                detailed_result[original_image].append({
                    'defect_class': defect_class,
                    'original_coordinates': projected_points,
                    'component_number': component_num,
                    'cropped_image_path': cropped_path,
                    'original_component_bbox': original_bbox,
                    'relative_coordinates': defect_points
                })
    
                simplified_result[original_image].append((
                    defect_class,
                    projected_points
                ))

        return detailed_result, simplified_result


class TiledDefectDetectionStrategy(DefectStrategy):
    """This is used for first tiling the cropped image of component"""
    def detect(self, mapping: Dict[str, Dict], components_by_type: Dict[str, List], target_component: str, model, batch_size: int = 4, **kargs) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Detect defects in cropped components using batched processing."""
        combined_detailed = {}
        combined_simplified = {}
        
        tiles = kargs.get("tiles", 5)
        model_name = os.path.basename(kargs.get("model_path"))
        # mapping --> {component_name : {original_image : image name/ path, cropped_image : [crpimg1, crpimg2,...]}}
        all_images = set()
        component_name = target_component
        if component_name in mapping:
            all_images.update(mapping[component_name]['original_image'])
        
        for img in all_images:
            combined_detailed[img] = []
            combined_simplified[img] = []

        print(f"\nProcessing defects for {component_name}...")

        cropped_images = mapping[component_name]['cropped_image']
        if not cropped_images:
            print(f"No cropped images of {component_name} found...")
            return combined_detailed, combined_simplified

        component_defect_data = {}
            
        print(f"Using model: {model} for {component_name}...")
        
        for i in tqdm(range(0, len(cropped_images), batch_size), 
                        desc=f"Processing {component_name} defect batches with model: {model}"):
            batch = cropped_images[i:i + batch_size]

            tiled_batches = self._tile_parents(batch, num_tiles=tiles)

            for j in tqdm(range(0, len(tiled_batches), batch_size), 
                    desc=f"//Processing Tiled {component_name} defect batches with model: {model_name}//"):

                tiled_batch = tiled_batches[j:j + batch_size]
            
                try:
                    results = self._process_batch(model, tiled_batch)
                    
                    for r in results:
                        orients = r.obb
                        obboxes = orients.xyxyxyxy
                        classes = orients.cls
                        tiled_defect_path = r.path
                        defect_label_dict = r.names

                        if "_h" in tiled_defect_path:
                            defect_path = tiled_defect_path.rsplit("_h", 1)[0] + ".jpg"
                        elif "_v" in tiled_defect_path:
                            defect_path = tiled_defect_path.rsplit("_v", 1)[0] + ".jpg"
                        else:
                            print(f"[Error] Unexpected tile suffix: {tiled_defect_path}")
                            continue
            
                        preds = []
                        for idx in range(len(classes)):
                            defect_class = defect_label_dict[int(classes[idx])]
                            tiled_defect_bbox = obboxes[idx].cpu().numpy().tolist()
            
                            defect_bbox = self._map_tiled_defects(defect_path, tiled_defect_path, tiled_defect_bbox)
            
                            preds.append((defect_class, defect_bbox))

                        if defect_path not in component_defect_data:
                            component_defect_data[defect_path] = []
                        component_defect_data[defect_path].extend(preds)
            
                        
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    continue

            torch.cuda.empty_cache()
            gc.collect()
        
        #print(component_defect_data)
        detailed_result, simplified_result = self._map_defects(
            mapping[component_name],
            components_by_type[component_name],
            component_defect_data,
            component_name
        )
        #print(mapping[component_name])

        for image_path, detections in detailed_result.items():
            for detection in detections:
                detection['component_type'] = component_name
                combined_detailed[image_path].append(detection)

        for image_path, detections in simplified_result.items():
            for defect_class, defect_points in detections:
                combined_simplified[image_path].append((
                    component_name,
                    defect_class,
                    defect_points,
                ))

        torch.cuda.empty_cache()
        gc.collect()

        return combined_detailed, combined_simplified        

    @staticmethod
    def _map_defects(
        component_mapping_data: Dict[str, List],
        component_original_detections: List[List],
        component_defect_detections: Dict[str, List],
        component_name: str
    ) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Map defect coordinates back to original image."""
        detailed_result = {}
        simplified_result = {}

        crop_to_original = {}
        for i in range(len(component_mapping_data['original_image'])):
            crop_to_original[component_mapping_data['cropped_image'][i]] = {
                'original_image': component_mapping_data['original_image'][i],
                'original_bbox': component_mapping_data['bbox'][i],
            }

        all_original_images = set(component_mapping_data['original_image'])
        for img in all_original_images:
            simplified_result[img] = []
            detailed_result[img] = []

        for cropped_path, defects in component_defect_detections.items():
            #print(defects)
            if cropped_path not in crop_to_original:
                continue

            original_info = crop_to_original[cropped_path]
            original_image = original_info['original_image']
            original_bbox = original_info['original_bbox']


            component_num = int(os.path.splitext(os.path.basename(cropped_path))[0])

            for defect_class, defect_points in defects:
                ordered_box = np.array(original_bbox['ordered_box'], dtype=np.float32)
    
                width = original_bbox['width']
                height = original_bbox['height']
    
                dst_pts = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)
    
                # Inverse perspective transform
                M_inv = cv2.getPerspectiveTransform(dst_pts, ordered_box)
    
                # Map each defect point back to original
                defect_pts = np.array(defect_points, dtype=np.float32).reshape(-1, 1, 2)
                mapped_pts = cv2.perspectiveTransform(defect_pts, M_inv).reshape(-1, 2).tolist()
    
                detailed_result[original_image].append({
                    'defect_class': defect_class,
                    'original_coordinates': mapped_pts,
                    'component_number': component_num,
                    'cropped_image_path': cropped_path,
                    'original_component_bbox': original_bbox,
                    'relative_coordinates': defect_points
                })
    
                simplified_result[original_image].append((
                    defect_class,
                    mapped_pts
                ))

        return detailed_result, simplified_result


    @staticmethod
    def _tile_parents(batch: List[str], num_tiles: int = 5) -> List[str]:
        """
        Tile each image in the batch along the longer side into `num_tiles` slices.
        Saves images with suffix indicating axis (_h{index} or _v{index}).
        """
        tiled_image_paths = []
    
        for image_path in batch:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")
                continue
    
            h, w, _ = img.shape
            tile_axis = 'h' if w >= h else 'v'  # horizontal if width >= height
            tile_size = w // num_tiles if tile_axis == 'h' else h // num_tiles
    
            for i in range(num_tiles):
                if tile_axis == 'h':
                    x_start = i * tile_size
                    x_end = (i + 1) * tile_size if i < num_tiles - 1 else w
                    tile = img[:, x_start:x_end]
                else:
                    y_start = i * tile_size
                    y_end = (i + 1) * tile_size if i < num_tiles - 1 else h
                    tile = img[y_start:y_end, :]
    
                suffix = f"_{tile_axis}{i}.jpg"
                tiled_path = image_path.replace('.jpg', suffix)
                cv2.imwrite(tiled_path, tile)
                tiled_image_paths.append(tiled_path)
    
        return tiled_image_paths

    @staticmethod
    def _map_tiled_defects(original_path: str, tiled_path: str, bbox: List) -> List[float]:
        """
        Map bbox from tiled image back to the original.
        Supports both horizontal (_hX.jpg) and vertical (_vX.jpg) slicing.
        """
        try:
            base = os.path.basename(tiled_path).split('_')[-1].replace('.jpg', '')
            axis = base[0]
            tile_index = int(base[1:])
        except Exception:
            print(f"[Error] Invalid tile index in: {tiled_path}")
            return bbox
    
        original_img = cv2.imread(original_path)
        if original_img is None:
            print(f"[Error] Could not load image: {original_path}")
            return bbox
    
        orig_h, orig_w, _ = original_img.shape
        offset = 0
    
        if axis == 'h':
            tile_size = orig_w // 5
            offset = tile_index * tile_size
            mapped_bbox = [(pt[0] + offset, pt[1]) if isinstance(pt, list) else (bbox[2*i] + offset, bbox[2*i+1]) for i, pt in enumerate(bbox[:4])]
        elif axis == 'v':
            tile_size = orig_h // 5
            offset = tile_index * tile_size
            mapped_bbox = [(pt[0], pt[1] + offset) if isinstance(pt, list) else (bbox[2*i], bbox[2*i+1] + offset) for i, pt in enumerate(bbox[:4])]
        else:
            print(f"[Error] Unknown axis '{axis}' in: {tiled_path}")
            return bbox
    
        # Flatten result
        return [coord for point in mapped_bbox for coord in point]


class DirectDefectDetectionStrategy(DefectStrategy):
    """This strategy detects defects directly on original image"""
    def detect(self, mapping: Dict[str, Dict], components_by_type: Dict[str, List], target_component: str, model, batch_size: int = 4, **kargs) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Detect defects in images using batched processing."""

        combined_detailed = {}
        combined_simplified = {}

        images = mapping[target_component]['original_image']

        for i in tqdm(range(0, len(images), batch_size), desc="Processing direct defect batches"):
            batch = images[i:i + batch_size]
            
            try:
                results = self._process_batch(model, batch)
                
                for r in results:
                    orients = r.obb
                    obboxes = orients.xyxyxyxy
                    classes = orients.cls
                    defect_path = r.path
                    defect_label_dict = r.names

                    preds = []
                    for idx in range(len(classes)):
                        defect_class = defect_label_dict[int(classes[idx])]
                        defect_bbox = obboxes[idx].cpu().numpy().tolist()

                        preds.append((target_component, defect_class, defect_bbox))

                    combined_simplified[defect_path] = preds
                    
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue

            torch.cuda.empty_cache()
            gc.collect()

        return combined_detailed, combined_simplified

# -------

def merge_and_save_results(
    component_data: Dict[str, List],
    defect_results: Dict[str, List],
    target_components: List[str],
    output_path: str = None
) -> Dict[str, List]:
    """Merge component and defect results."""
    merged_results = {}

    for image_path, components in component_data.items():
        merged_results[image_path] = []
        
        for component in components:
            component_name, comp_bbox = component
            component_result = [component]  # Keep original component info
            
            if component_name in target_components and image_path in defect_results:
                comp_x_min = min(point[0] for point in comp_bbox)
                comp_x_max = max(point[0] for point in comp_bbox)
                comp_y_min = min(point[1] for point in comp_bbox)
                comp_y_max = max(point[1] for point in comp_bbox)
                
                for comp_type, defect_class, defect_bbox in defect_results[image_path]:
                    if comp_type != component_name:
                        continue
                        
                    defect_center_x = sum(point[0] for point in defect_bbox) / 4
                    defect_center_y = sum(point[1] for point in defect_bbox) / 4
                    
                    if (comp_x_min <= defect_center_x <= comp_x_max and
                        comp_y_min <= defect_center_y <= comp_y_max):
                        component_result.append((defect_class, defect_bbox))
            
            merged_results[image_path].append(component_result)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(merged_results, f, indent=2)

    return merged_results

# -------

def visualize_detections(
    image_path: str,
    detailed_results: Dict[str, List],
    simplified_results: Dict[str, List],
    show_detailed: bool = True,
    save_path: str = None
) -> None:
    """Visualize detections on image."""
    if image_path not in detailed_results:
        print(f"No results for {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}")
        return

    colors = {
        'Ok': (0, 255, 0),
        'Defect': (0, 0, 255),
        'Severe': (0, 165, 255),
        'Minor': (0, 255, 255)
    }

    if show_detailed:
        for detection in detailed_results[image_path]:
            points = np.array(detection['original_coordinates'], dtype=np.int32)
            defect_class = detection['defect_class']
            component_type = detection['component_type']

            color = colors.get(defect_class, (255, 255, 255))
            cv2.polylines(image, [points], True, color, 2)

            label = f"{component_type}({component_confidence:.2f})-{defect_class}({confidence:.2f})"
            cv2.putText(image, label, tuple(points[0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        for component_name, defect_class, points, confidence, component_confidence in simplified_results[image_path]:
            points = np.array(points, dtype=np.int32)
            color = colors.get(defect_class, (255, 255, 255))
            cv2.polylines(image, [points], True, color, 2)
            
            # Include both confidences in the label
            label = f"{component_name}({component_confidence:.2f})-{defect_class}({confidence:.2f})"
            cv2.putText(image, label, tuple(points[0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if save_path:
        cv2.imwrite(save_path, image)
        
# -------

def find_key(d, target):
    return next((k for k, v in d.items() if target in v), None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Component and Defect Detection Pipeline')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input JSON file containing image paths')
    parser.add_argument('--output', type=str, required=True,
                      help='Path for output JSON file with merged results')
    parser.add_argument('--temp', type=str, required=True,
                      help='Directory for temporary files')
    parser.add_argument('--vis', type=str, 
                      help='Optional: Directory for saving visualizations')
    parser.add_argument('--config', type=str, default="./base_model/config.json",
                      help='Path to component detection model')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
    	config = json.load(f)

    different_defect_steps = {
        "CroppingProcessor" : [
            "StandardCroppingStrategy",
            "RotatedCroppingStrategy"
            ],
        "DefectProcessor" : [
            "StandardDefectDetectionStrategy",
            "TiledDefectDetectionStrategy",
            "DirectDefectDetectionStrategy"
            ]
        }
    

    #Component processing....
    component_info = config['component']
    component_strategy, component_strategy_args = component_info
    component_processor = ComponentProcessor(component_strategy, strategy_args)


    try:
        image_paths = paths_from_json(args.input)
        if not image_paths:
            raise ValueError("No images found in input JSON")

        print("Detecting components...")
        components_by_type, data = component_processor.detect(image_paths)

        print("Detecting defects....")
        defects_info = config['defects']
        cropping_processor = CroppingProcessor(output_dir=args.temp)
        defect_processor = DefectProcessor()
        detailed_results, simplified_results = {}, {}
        
        TARGET_COMPONENTS = []
        for target_component, strategies in defects_info.items():
            TARGET_COMPONENTS.append(target_component)
            mapping = {target_component: {'original_image': image_paths, 'cropped_image': [], 'bbox': []}} # By default, mapping contains all the images in target_component
            for steps in strategies:
                for step in steps:
                    strategy, strategy_args = step
                    strategy_args["target_component"] = target_component

                    processor = find_key(different_defect_steps, "StandardCroppingStrategy")
                    if processor == "CroppingProcessor":
                        print(f"Cropping {target_component}...")
                        cropping_processor.strategy = strategy
                        cropping_processor.strategy_args = strategy_args

                        mapping = cropping_processor.crop(components_by_type)

                    else if processor == "DefectProcessor":
                        print(f"Detecting defects for {target_component} using {strategy}...")
                        defect_processor.strategy = strategy
                        defect_processor.strategy_args = strategy_args

                        detailed_results_iter, simplified_results_iter = defect_processor.detect(mapping, components_by_type)

                        for k, v in detailed_results_iter.items():
                            if k in detailed_results and detailed_results[k]:
                                detailed_results[k].extend(v)
                            else:
                                detailed_results[k] = v

                        for k, v in simplified_results_iter.items():
                            if k in simplified_results and simplified_results[k]:
                                simplified_results[k].extend(v)
                            else:
                                simplified_results[k] = v

                    else:
                        print("Correct processor for Defect Detection not found, Skipping defect detection.....")

        merged_results = merge_and_save_results(
            data, 
            simplified_results,
            TARGET_COMPONENTS,
            args.output
        )
        print("Saved merged results to:", args.output)

        if args.vis:
            print("\nGenerating visualizations...")
            os.makedirs(args.vis, exist_ok=True)
            
            for image_path in tqdm(detailed_results.keys(), desc="Visualizing"):
                save_path = os.path.join(
                    args.vis,
                    f"annotated_{os.path.basename(image_path)}"
                )
                
                visualize_detections(
                    image_path,
                    detailed_results,
                    simplified_results,
                    show_detailed=True,
                    save_path=save_path
                )
            print(f"Saved visualizations to: {args.vis}")

        print("\nProcessing Summary:")
        print(f"Total images processed: {len(detailed_results)}")
        for component_type in TARGET_COMPONENTS:
            component_count = sum(
                1 for image in detailed_results.values()
                for detection in image
                if detection.get('component_type') == component_type
            )
            print(f"Total {component_type} detections: {component_count}")
        
        print("\nDefect Statistics:")
        defect_counts = defaultdict(int)
        defect_conf_sums = defaultdict(float)
        
        for image in detailed_results.values():
        	for detection in image:
        		defect_class = detection['defect_class']
        		defect_counts[defect_class] += 1
        		
       	for defect_class in defect_counts:
       		avg_conf = defect_conf_sums[defect_class] / defect_counts[defect_class]
       		print(f"- {defect_class}: {defect_counts[defect_class]} detections")

    except Exception as e:
        print(f"Error in processing pipeline: {str(e)}")
        raise