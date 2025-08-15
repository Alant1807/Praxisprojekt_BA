import torch
import torch.nn as nn

from Scripts.loss import *
from anomalib.models.components import TimmFeatureExtractor

class AsymmetricSTFPM(nn.Module):
    """
    Das AsymmetricSTFPM-Modell verwendet ein Lehrer- und ein Schüler-Modell
    mit potenziell unterschiedlichen Architekturen.

    Args:
        teacher_architecture (str): Die Architektur des Lehrermodells.
        student_architecture (str): Die Architektur des Schülermodells.
        layers (list): Eine Liste von Schichten, die extrahiert werden sollen.
    """

    def __init__(self, teacher_architecture, student_architecture, layers):
        super().__init__()

        self.teacher_model = TimmFeatureExtractor(
            backbone=teacher_architecture, pre_trained=True, layers=layers
        ).eval()

        self.student_model = TimmFeatureExtractor(
            backbone=student_architecture, pre_trained=False, layers=layers, requires_grad=True
        )

        # Die Stem-Layer werden vom (stärkeren) Lehrer extrahiert
        self.stem_model = self.extract_stem_layers()

        for param in self.teacher_model.parameters():
            param.requires_grad = False
        for param in self.stem_model.parameters():
            param.requires_grad = False

        self.stem_model.eval()

    def extract_stem_layers(self):
        stem_layer_names = []
        for name, _ in self.teacher_model.feature_extractor.named_children():
            if 'layer' in name or 'blocks' in name:
                break
            stem_layer_names.append(name)
        
        stem_model = nn.Sequential(
            *[self.teacher_model.feature_extractor.get_submodule(name) for name in stem_layer_names]
        )

        for name in stem_layer_names:
            setattr(self.teacher_model.feature_extractor, name, nn.Identity())
            setattr(self.student_model.feature_extractor, name, nn.Identity())

        return stem_model

    def forward(self, x):
        stem_output = self.stem_model(x)
        with torch.no_grad():
            teacher_feature_maps = self.teacher_model(stem_output)
        student_feature_maps = self.student_model(stem_output)

        if isinstance(teacher_feature_maps, dict):
            teacher_feature_maps = list(teacher_feature_maps.values())
        if isinstance(student_feature_maps, dict):
            student_feature_maps = list(student_feature_maps.values())

        return teacher_feature_maps, student_feature_maps

    def anomaly_map(self, x):
        teacher_feature_maps, student_feature_maps = self.forward(x)
        batch_size, _, img_height, img_width = x.shape
        anomaly_map = torch.ones((batch_size, img_height, img_width), device=x.device)

        for t_map, s_map in zip(teacher_feature_maps, student_feature_maps):
            t_map_norm = torch.nn.functional.normalize(t_map, dim=1)
            s_map_norm = torch.nn.functional.normalize(s_map, dim=1)
            am = 0.5 * torch.sum(torch.pow(t_map_norm - s_map_norm, 2), dim=1)
            am = nn.Upsample(size=(img_height, img_width), mode="bilinear", align_corners=False)(am.unsqueeze(1))
            anomaly_map = torch.mul(anomaly_map, am.squeeze(1))

        return anomaly_map