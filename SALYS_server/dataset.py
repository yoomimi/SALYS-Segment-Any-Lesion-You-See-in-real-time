import os
import re
from pathlib import Path

import PIL.Image
import imageio
import numpy as np
import pickle as pkl
import pycocotools.mask
import torch
import torchvision.transforms as transforms
import yaml
import random
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2
from typing import Tuple


class PascalVOCDataset(Dataset):
    """
        PascalVOCDataset for image-based gaze version of the dataset, including the option of precomputed embeddings
        @param root_dir: Path to folder containing the gaze images, masks and original etc. as folders, per class
        @param prompt_types: Tuple containing the used prompt types. Choose out of: points, boxes, masks
        @param mask_type: Type of heatmap construction used for gaze mask. Default is 'gaussian', which applies a
            gaussian blur to the gaze mask image. Choose out of: gaussian, fixation, fixation_duration, gaze
    """

    def __init__(self, root_dir, inference=False, transform=None, return_path=False, use_embeddings=False,
                 prompt_types: Tuple = ("boxes",), mask_type="gaussian"):
        self.VERSION = 2
        self.root_dir = root_dir
        self.transform = transform
        self.return_path = return_path
        self.split = Path(self.root_dir).name
        self.use_embeddings = use_embeddings
        self.inference = inference
        # print(prompt_types,'prompt_types')
        assert all(prompt_type in ["boxes", "points", "masks", "fixations", "gaze_points"] for prompt_type in prompt_types), "Invalid prompt type"
        assert mask_type in ["gaussian", "fixation", "fixation_duration", "gaze"], "Invalid mask type"
        self.prompt_types = prompt_types
        self.mask_type = mask_type
        self.image_dict = self._get_image_dict()

        self.image_ids = list(self.image_dict.keys())
        self.image_ids.remove("__version")

    def _get_image_dict(self):

        # if Path(self.root_dir, "sam_dataset_info_dict.pkl").exists():
        #     data_dict = pkl.load(open(Path(self.root_dir, "sam_dataset_info_dict.pkl"), "rb"))
        #     if self.VERSION > data_dict.get("__version", -1):
        #         print("Version mismatch detected.")
        #         return self._construct_image_dict()
        #     return data_dict
        # else:
        return self._construct_image_dict()
        
        # 수정 전 (pkl 파일 고정)
        # if Path(self.root_dir, "sam_dataset_info_dict.pkl").exists():
        #     data_dict = pkl.load(open(Path(self.root_dir, "sam_dataset_info_dict.pkl"), "rb"))
        #     if self.VERSION > data_dict.get("__version", -1):
        #         print("Version mismatch detected.")
        #         return self._construct_image_dict()
        #     return data_dict
        # else:
        #     return self._construct_image_dict()
        
        

    def _get_bbox_regex(self):
        return re.compile("\d{4}_\d{6}_x_min=(\d+)_x_max=(\d+)_y_min=(\d+)_y_max=(\d+)")

    def _get_image_path(self, image_path):
        fileid = "_".join(image_path.stem.split("_")[:2])
        return Path(Path(self.root_dir).parent.parent, "JPEGImages", fileid + ".jpg").absolute().__str__()

    def _get_image_embedding(self, image_path, image_id):
        image_embedding = pkl.load(
            open(Path(Path(self.root_dir).parent.parent, "sam_embeddings", self.split, image_id + ".pkl"), "rb"))
        image_embedding = torch.tensor(image_embedding)
        return image_embedding

    def _construct_image_dict(self):
        """
            Constructs a dict, keys are image ids, dict contains list of bboxes and paths to additional info, and saves it to disk
        """
        image_dict = {}
        # print("Constructing image dict...")

        bbox_regex = self._get_bbox_regex()
        
        for class_dir in Path(self.root_dir).iterdir():
            if not class_dir.is_dir():
                continue
            image_paths = list(Path(class_dir, "original").glob("*.png"))
            # print(image_paths,'image_paths')
            for image_path in image_paths:
                gaze_path = Path(image_path.parent.parent, "gaze_images", image_path.name).absolute()
                hm_fixations_path = Path(image_path.parent.parent, "heatmaps_based_on_fixations_mask", image_path.name + ".npy").absolute()
                hm_fixations_duration_path = Path(image_path.parent.parent, "heatmaps_based_on_fixations_mask_using_duration", image_path.name + ".npy").absolute()
                hm_gaze_path = Path(image_path.parent.parent, "heatmaps_based_on_gaze_data_mask", image_path.name + ".npy").absolute()
                fixation_path = Path(image_path.parent.parent, "fixation_images", image_path.name).absolute()


                # fileid = "_".join(image_path.stem.split("_")[:2])
                            
                fileid_parts = image_path.stem.split("_obj")
                fileid = fileid_parts[0] + "_obj"  # "_obj"를 다시 추가하여 fileid 완성

                # print(image_path,'image_path')
                # print(fileid,'fileid')

                bbox_wrong_order = list(map(float, bbox_regex.match(str(image_path.stem)).groups()))
                bbox = [bbox_wrong_order[2], bbox_wrong_order[0], bbox_wrong_order[3],
                        bbox_wrong_order[1]]  # x_min, y_min, x_max, y_max

                mask_array = imageio.v3.imread(Path(image_path.parent.parent, "masks", image_path.name))
                if len(mask_array.shape) == 3:
                    mask_array = np.sum(mask_array, axis=2, dtype=int)
                mask_array = mask_array.astype(bool).astype(np.uint8)
                segmentation_rle_dict = pycocotools.mask.encode(np.asarray(mask_array, order="F"))
                assert type(segmentation_rle_dict) == dict, type(segmentation_rle_dict)

                fixation_img = imageio.v3.imread(fixation_path)
                if len(fixation_img.shape) == 3:
                    fixation_img = np.sum(fixation_img, axis=2, dtype=int)
                fixation_img = fixation_img.astype(bool)
                fixation_points = np.transpose(np.nonzero(fixation_img))
                fixation_points = fixation_points[:, ::-1]
                if len(fixation_points) == 0:
                    print("Skipping")
                    continue

                gaze_img = imageio.v3.imread(gaze_path)
                if len(gaze_img.shape) == 3:
                    gaze_img = np.sum(gaze_img, axis=2, dtype=int)
                gaze_img = gaze_img.astype(bool)
                gaze_points = np.transpose(np.nonzero(gaze_img))
                gaze_points = gaze_points[:, ::-1]

                if fileid not in image_dict.keys():
                    r = {
                        "original": self._get_image_path(image_path),
                        "image_id": fileid,
                        "bboxes": [],
                        "masks": [],
                        "gaze_paths": [],
                        "heatmap_paths": {"fixation": [],
                                          "fixation_duration": [],
                                          "gaze": []},
                        "fixations": [],
                        "gaze_points": [],
                    }
                    image_dict[fileid] = r

                image_dict[fileid]["gaze_paths"].append(str(gaze_path))
                image_dict[fileid]["heatmap_paths"]["fixation"].append(str(hm_fixations_path))
                image_dict[fileid]["heatmap_paths"]["fixation_duration"].append(str(hm_fixations_duration_path))
                image_dict[fileid]["heatmap_paths"]["gaze"].append(str(hm_gaze_path))
                image_dict[fileid]["bboxes"].append(bbox)
                image_dict[fileid]["masks"].append(segmentation_rle_dict)
                image_dict[fileid]["fixations"].append(fixation_points.tolist())
                image_dict[fileid]["gaze_points"].append(gaze_points.tolist())

        image_dict["__version"] = self.VERSION
        yaml.dump(image_dict, open(Path(self.root_dir, "sam_dataset_info_dict.yaml"), "w"))
        pkl.dump(image_dict, open(Path(self.root_dir, "sam_dataset_info_dict.pkl"), "wb"))
        return image_dict

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        
        
        image_id = self.image_ids[idx]
        image_info = self.image_dict[image_id]
        image_path = image_info["original"]
        image = cv2.imread(str(image_path))

        while image is None : 
            print("Image is None at  ", image_id ,"... resampling!")
            idx = random.randint(0, len(self.image_ids) - 1)
            image_id = self.image_ids[idx]
            image_info = self.image_dict[image_id]
            image_path = image_info["original"]
            image = cv2.imread(str(image_path))
        
        if self.use_embeddings:
            image_embedding = self._get_image_embedding(image_path, image_id)
        else:
            image_embedding = None

        og_h, og_w, _ = image.shape
        
        
        fixations = image_info["fixations"]
        gaze_points = image_info["gaze_points"]
        bboxes = image_info["bboxes"]
        masks_rle = image_info["masks"]
        masks = [pycocotools.mask.decode(rle) for rle in masks_rle]
        
        

        if "points" in self.prompt_types:
            all_points = [np.transpose(np.nonzero(mask)) for mask in masks]
            all_points = [points[np.random.choice(a=len(points), size=min(8, len(points)), replace=False), ::-1] for points in all_points]
            points = all_points
            
        elif "gaze_points" in self.prompt_types:
            # points = random.sample(gaze_points, 8)
            points = gaze_points
        else:
            points = fixations
            # points = fixations


        # print(image_path,'image_path')
        # print(image_info["heatmap_paths"][self.mask_type],'gaze_mask_paths')
        # print(image_info["masks"],'image_info["masks"]')

        if "masks" in self.prompt_types:
            if self.mask_type == "gaussian":
                gaze_mask_paths = image_info["gaze_paths"]
                gaze_masks = [imageio.v3.imread(gaze_path) for gaze_path in gaze_mask_paths]
                self.gaze_transform = transforms.GaussianBlur(kernel_size=7, sigma=5)
                gaze_masks = [np.array(self.gaze_transform(PIL.Image.fromarray(gaze_mask))) for gaze_mask in gaze_masks]
                gaze_masks = [gaze_mask / np.max(gaze_mask) for gaze_mask in gaze_masks]
            else:
                gaze_mask_paths = image_info["heatmap_paths"][self.mask_type]
                gaze_masks = [np.load(gaze_mask_path) for gaze_mask_path in gaze_mask_paths]
                gaze_masks = [mask / np.max(mask) for mask in gaze_masks]
        else:
            gaze_masks = None

        if self.inference:
            cls_list = [Path(path).parent.parent.name for path in image_info["gaze_paths"]]
            original_data = (image_info.copy(), image.copy(),  masks.copy(), points.copy(), bboxes.copy(),
                             gaze_masks.copy() if gaze_masks is not None else None, cls_list)

        if self.transform:
            image, masks, bboxes, gaze_masks, points, padding = self.transform(image, masks, np.array(bboxes), points, gaze_masks)
            if self.inference:
                original_data = (*original_data, padding)

        masks = np.stack(masks, axis=0)
        bboxes = np.stack(bboxes, axis=0)
        points, labels = points
        
        
        points_ =[torch.tensor(points_one_mask) for points_one_mask in points]
        labels_ = labels
        random_indices = random.sample(range(masks.shape[0]), min(masks.shape[0], 8))

        if gaze_masks is not None:
            gaze_masks = np.stack(gaze_masks, axis=0)
            # print(gaze_masks.shape,'gaze_masks.shape_before')
            gaze_masks = [gaze_masks[i] for i in random_indices]
            gaze_masks = np.stack(gaze_masks, axis=0)
            # print(gaze_masks.shape,'gaze_masks.shape_after')

        if "points" in self.prompt_types or "gaze_points" in self.prompt_types:
            points_=  [points_[i] for i in random_indices]
            labels_ = [labels_[i] for i in random_indices]
        elif "boxes" in self.prompt_types:
            bboxes = [bboxes[i] for i in random_indices]    
            
        # print(len(points_),'points_')
        # print(len(masks),'masks')
        masks = [masks[i] for i in random_indices]
        points = torch.nn.utils.rnn.pad_sequence(points_, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels_, batch_first=True, padding_value=-1)
        

        
        points = (points, labels)

        prompt_dict = {"points": None, "boxes": None, "masks": None}
        
        if "points" in self.prompt_types or "fixations" in self.prompt_types or "gaze_points" in self.prompt_types:
            prompt_dict["points"] = points
            
        if "boxes" in self.prompt_types:
            bboxes_np = np.array(bboxes)
            prompt_dict["boxes"] = torch.tensor(bboxes_np)
            
        if "masks" in self.prompt_types:
            gaze_masks_np = np.array(gaze_masks)
            prompt_dict["masks"] = torch.tensor(gaze_masks_np).float()

        masks_np = np.array(masks)  # 리스트를 NumPy 배열로 변환
        masks_tensor = torch.tensor(masks_np).float()  # NumPy 배열을 텐서로 변환

        # print(image.shape,'image_shape')
        # print(masks_tensor.shape,'masks_tensor_shape')
        
        
        return_list = [prompt_dict, masks_tensor]
        # return_list = [prompt_dict, torch.tensor(masks).float()]
        if self.return_path:
            return_list.append(image_path)
        if self.inference:
            return_list.append((og_h, og_w))
            return_list.append(original_data)

        image_return = image_embedding if self.use_embeddings else image

        return image_return, *return_list


class CellPose500Dataset(PascalVOCDataset):
    """
        CellPose500Dataset for image-based gaze version of the dataset, including the option of precomputed embeddings
        @param root_dir: Path to folder containing the gaze images, masks and original etc. as folders, per class
        @param prompt_types: Tuple containing the used prompt types. Choose out of: points, boxes, masks
    """

    def __init__(self, root_dir, inference=False, transform=None, return_path=False, use_embeddings=False,
                 prompt_types: Tuple = ("boxes",), mask_type="gaussian"):
        super().__init__(root_dir, inference, transform, return_path, use_embeddings, prompt_types, mask_type)
        print(root_dir,'root_dir-inCellPose500Dataset')

    def _get_bbox_regex(self):
        return re.compile("\d{3}_img.png_x_min=(\d+)_x_max=(\d+)_y_min=(\d+)_y_max=(\d+)")

    def _get_image_path(self, image_path):
        return image_path

    def _get_image_embedding(self, image_path, image_id):
        image_embedding = pkl.load(
            open(Path(Path(self.root_dir).parent, "sam_embeddings", self.split, image_path.stem + ".pkl"), "rb"))
        image_embedding = torch.tensor(image_embedding)
        return image_embedding
    
    
class T2SPIRDataset(PascalVOCDataset):


    def __init__(self, root_dir, inference=False, transform=None, return_path=False, use_embeddings=False,
                 prompt_types: Tuple = ("points",), mask_type="gaze"):
        super().__init__(root_dir, inference, transform, return_path, use_embeddings, prompt_types, mask_type)
        
    # def __init__(self, root_dir, inference=False, transform=None, return_path=False, use_embeddings=False,
    #              prompt_types: Tuple = ("points",), mask_type="gaussian"):
        # super().__init__(root_dir, inference, transform, return_path, use_embeddings, prompt_types, mask_type)
        
        # print(root_dir,'root_dir-int_T2SPIRdataset')
        
        

    def _get_bbox_regex(self):
        # 파일명에서 특정 형식을 찾기 위한 정규 표현식 정의
        # 앞부분은 자유롭게 변경될 수 있으며, 최소한 _x_min=... 형태를 포함하고 있어야 함
        return re.compile(r".*?_x_min=(\d+)_x_max=(\d+)_y_min=(\d+)_y_max=(\d+)")

    def _get_image_path(self, image_path):
        return image_path

    def _get_image_embedding(self, image_path, image_id):
        image_embedding = pkl.load(
            open(Path(Path(self.root_dir).parent, "sam_embeddings", self.split, image_path.stem + ".pkl"), "rb"))
        image_embedding = torch.tensor(image_embedding)
        return image_embedding

def collate_fn(batch):
    images, prompts, *rest = tuple(zip(*batch))
    images = torch.stack(images)
    return images, prompts, *rest


class ResizeAndPad:

    def __init__(self, target_size, inference=False):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()
        self.inference = inference

    def __call__(self, image, masks, bboxes, points=None, gaze_masks=None):
        # Resize image and masks
        og_h, og_w = image.shape[:2]
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        if gaze_masks is not None:
            gaze_masks = torch.tensor(np.array(gaze_masks), dtype=torch.float32).unsqueeze(1)
            gaze_masks = self.transform.apply_image_torch(gaze_masks) # bchw format

        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]
        if gaze_masks is not None:
            gaze_masks = transforms.Pad(padding)(gaze_masks)
            gaze_masks = transforms.Resize(self.target_size // 4, antialias=True)(gaze_masks)

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        # Adjust points
        if points is not None:
            points = [self.transform.apply_coords(np.array(points_one_mask), (og_h, og_w)) for points_one_mask in points]
            points = [[[point[0] + pad_w, point[1] + pad_h] for point in points_one_mask] for points_one_mask in points]
            labels = [torch.ones(len(points_one_mask)) for points_one_mask in points]
            points = (points, labels)

        return_list = [masks, bboxes]
        if gaze_masks is not None or points is not None:
            return_list.append(gaze_masks)
            return_list.append(points)

        return_list.append(padding)

        return image, *return_list


DATASETS = {"pascal": PascalVOCDataset,
            "cell": CellPose500Dataset,
            "t2spir": T2SPIRDataset
            }


def load_datasets(cfg, img_size, return_path=False):
    transform = ResizeAndPad(img_size)
    
    dataset_cls = DATASETS[cfg.dataset.type]

    # train_arg_dict = cfg.dataset.train.to_dict()
    # train_arg_dict["transform"] = transform
    # train_arg_dict["return_path"] = return_path
    
    
    # train = dataset_cls(**train_arg_dict)

    val_arg_dict = cfg.dataset.val.to_dict()
    val_arg_dict["transform"] = transform
    val_arg_dict["return_path"] = return_path
    val = dataset_cls(**val_arg_dict)
    # train_dataloader = DataLoader(train,
    #                               batch_size=cfg.batch_size,
    #                               shuffle=True,
    #                               num_workers=cfg.num_workers,
    #                               collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    return val_dataloader
