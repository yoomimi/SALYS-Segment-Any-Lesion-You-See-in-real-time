from itertools import chain

import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from segment_anything.modeling.sam import *


class Model(nn.Module):

    def __init__(self, cfg, inference=False):
        super().__init__()
        self.cfg = cfg
        self.inference = inference

    def setup(self):
        self.model = sam_model_registry[self.cfg.model.type](checkpoint=self.cfg.model.checkpoint if not self.inference else None)
        if self.inference:
                with open(self.cfg.model.checkpoint, "rb") as f:
                    state_dict = torch.load(f)
                self.load_state_dict(state_dict)
        self.model.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def forward(self, images, prompts):
        _, _, H, W = images.shape
        image_embeddings = self.model.image_encoder(images)
        pred_masks = []
        ious = []
        for embedding, prompt in zip(image_embeddings, prompts):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(**prompt)

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)

        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)

    def get_img_size(self):
        return self.model.image_encoder.img_size

    def get_parameters(self):
        return self.model.parameters()

    def get_full_model(self):
        return self.model


class PrecomputedEmbeddingModel(nn.Module):

    """
        This model excepts precomputed image embeddings and cannot train the image embedder
    """
    def __init__(self, cfg, inference=False):
        super().__init__()
        self.prompt_encoder: PromptEncoder = None
        self.mask_decoder: MaskDecoder = None
        self.full_model: Sam = None
        self.cfg = cfg
        self.inference = inference
        self.load_new_model_type = True if self.inference else cfg.model.get("load_new_model_type", False)

    def setup(self):
        self.full_model = sam_model_registry[self.cfg.model.type](checkpoint=self.cfg.model.checkpoint if not self.load_new_model_type else None)
        self.prompt_encoder = self.full_model.prompt_encoder
        self.mask_decoder = self.full_model.mask_decoder
        for param in self.full_model.image_encoder.parameters():
            param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.mask_decoder.parameters():
                param.requires_grad = False
        if self.load_new_model_type:
            with open(self.cfg.model.checkpoint, "rb") as f:
                state_dict = torch.load(f)
            self.load_state_dict(state_dict)
        self.prompt_encoder.train()
        self.mask_decoder.train()

    def forward(self, image_embeddings, prompts):
        H = W = self.full_model.image_encoder.img_size
        pred_masks = []
        ious = []
        for embedding, prompt in zip(image_embeddings, prompts):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(**prompt)

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)

        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.full_model)

    def get_img_size(self):
        return self.full_model.image_encoder.img_size

    def get_parameters(self):
        return chain(*[self.prompt_encoder.parameters(), self.mask_decoder.parameters()])

    def get_full_model(self):
        return self.full_model


class PrecomputeModel(nn.Module):
    """
        Use this model to precompute image embeddings on your dataset for faster training.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self):
        self.model = sam_model_registry[self.cfg.model.type](checkpoint=self.cfg.model.checkpoint)
        self.model.eval()
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.model.mask_decoder.parameters():
            param.requires_grad = False

    def forward(self, images):
        image_embeddings = self.model.image_encoder(images)
        return image_embeddings

    def get_predictor(self):
        return SamPredictor(self.model)

    def get_img_size(self):
        return self.model.image_encoder.img_size

    def get_parameters(self):
        return self.model.parameters()

    def get_full_model(self):
        return self.model


MODELS = {"PrecomputedEmbeddingModel": PrecomputedEmbeddingModel,
          "Model": Model,
          }
