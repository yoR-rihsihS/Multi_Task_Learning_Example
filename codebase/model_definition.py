import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet
from .aspp import ASPP
from .decoder import Decoder
from .classifier import Classifier
from .additional_modules import Gate
from .utils import select_and_normalize

class MultiTaskModel(nn.Module):
    def __init__(self, task_specs, backbone='resnet50', num_experts=5):
        super(MultiTaskModel, self).__init__()
        if backbone in ['resnet50', 'resnet101', 'resnet152']:
            low_level_channels = 256
            high_level_channels = 2048
        else:
            raise ValueError(f"backbone {backbone} is not supported. Choose from 'resnet50' or 'resnet101' or 'resnet152'")

        self.resnet = ResNet(model=backbone, replace_stride_with_dilation=[False, False, True]) # the output stride of resnet is 16
        self.aspp = ASPP(in_channels=high_level_channels, out_channels=256, atrous_rates=[6, 12, 18])
        modules = {}
        gates = {}
        for key, value in task_specs.items():
            if 'segmentation' in key:
                modules[key] = nn.ModuleList([Decoder(low_level_channels=low_level_channels, num_classes=value) for i in range(num_experts)])
                gates[key] = Gate(256, num_experts)
            elif 'classification' in key:
                modules[key] = nn.ModuleList([Classifier(in_channels=high_level_channels, num_classes=value) for i in range(num_experts)])
                gates[key] = Gate(high_level_channels, num_experts)
        self.prediction_heads = nn.ModuleDict(modules)
        self.gates = nn.ModuleDict(gates)

    def forward_classification(self, image=None, hl_features=None, p=0.8, force_all_experts=True):
        if image is not None:
            _, hl_features = self.resnet(image)
        gate_scores = {}
        for key, gate in self.gates.items():
            if 'classification' in key:
                gate_scores[key + '_logits'] = gate(hl_features)
        outputs = {}
        for key, prediction_head in self.prediction_heads.items():
            if 'classification' in key:
                outputs[key + '_logits'] = torch.stack([expert(hl_features) for expert in prediction_head], dim=1)

        if force_all_experts: # during initial phase of training
            for key, value in outputs.items():
                scores = gate_scores[key]
                weights = scores / (scores.sum(dim=1, keepdim=True) + 1e-6)
                weighted_value = value * weights.unsqueeze(-1)
                outputs[key] = weighted_value.sum(1)
            return outputs, gate_scores

        for key, value in outputs.items():
            scores = gate_scores[key]
            mask, weights = select_and_normalize(scores, p)
            weighted_value = value * weights.unsqueeze(-1)
            outputs[key] = weighted_value.sum(1)
        return outputs, gate_scores

    def forward_segmentation(self, image=None, size=None, ll_features=None, hl_features=None, p=0.8, force_all_experts=True):
        """
        Args:
            - image (tensor | None) : Image tensor of shape [bs, 3, h, w] or None.
            - ll_features (tensor | None) : Low level feature map of shape [bs, 256, h/4, w/4] or None.
            - hl_features (tensor | None) : High level feature map of shape [bs, 2048, h/16. w/16] or None.
        If image is None the function expects ll_features and hl_features.
        """
        if image is not None:
            size = image.shape[2:]
            ll_features, hl_features = self.resnet(image)
        hl_features = self.aspp(hl_features)
        gate_scores = {}
        for key, gate in self.gates.items():
            if 'segmentation' in key:
                gate_scores[key + '_logits'] = gate(hl_features)
        outputs = {}
        for key, prediction_head in self.prediction_heads.items():
            if 'segmentation' in key:
                outputs[key + '_logits'] = torch.stack([expert(ll_features, hl_features) for expert in prediction_head], dim=1)

        if force_all_experts: # during initial phase of training
            for key, value in outputs.items():
                scores = gate_scores[key]
                weights = scores / (scores.sum(dim=1, keepdim=True) + 1e-6)
                weights = weights.view(weights.size(0), weights.size(1), 1, 1, 1)
                weighted_value = value * weights
                outputs[key] = weighted_value.sum(1)
            for key, tensor in outputs.items():
                outputs[key] = F.interpolate(tensor, size=size, mode='bilinear', align_corners=True)
            return outputs, gate_scores

        for key, value in outputs.items():
            scores = gate_scores[key]
            mask, weights = select_and_normalize(scores, p)
            weights = weights.view(weights.size(0), weights.size(1), 1, 1, 1)
            weighted_value = value * weights
            outputs[key] = weighted_value.sum(1)
        for key, tensor in outputs.items():
            outputs[key] = F.interpolate(tensor, size=size, mode='bilinear', align_corners=True)
        return outputs, gate_scores

    def forward(self, image, p=0.8, force_all_experts=True):
        size = image.shape[-2:]
        low_level_features, high_level_features = self.resnet(image)
        outputs, gate_scores = {}, {}
        out, sco = self.forward_classification(image=None, hl_features=high_level_features, p=p, force_all_experts=force_all_experts)
        outputs.update(out)
        gate_scores.update(sco)
        out, sco = self.forward_segmentation(image=None, size=size, ll_features=low_level_features, hl_features=high_level_features, p=p, force_all_experts=force_all_experts)
        outputs.update(out)
        gate_scores.update(sco)
        return outputs, gate_scores
