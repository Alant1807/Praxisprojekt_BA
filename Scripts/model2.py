import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict
from Scripts.loss import *


class FeatureExtractor(nn.Module):
    def __init__(self, backbone, pretrained, layers, requires_grad=False):
        super().__init__()

        if backbone not in models.__dict__:
            raise ValueError(
                f"Backbone '{backbone}' not found in torchvision.models")

        self.model = models.__dict__[backbone](pretrained=pretrained)
        self.layers = layers

        self.features: Dict[str, torch.Tensor] = {}

        for param in self.model.parameters():
            param.requires_grad = requires_grad

        named_modules = dict([*self.model.named_modules()])
        for name in self.layers:
            if name not in named_modules:
                raise ValueError(f"Layer '{name}' not found in the model. "
                                 f"Available layers: {list(named_modules.keys())}")
            layer = named_modules[name]
            layer.register_forward_hook(self._get_hook(name))

    def _get_hook(self, name: str):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x):
        self.features.clear()

        if not next(self.model.parameters()).requires_grad:
            with torch.no_grad():
                _ = self.model(x)
        else:
            _ = self.model(x)

        return self.features


class STFPM(nn.Module):
    """
    Das STFPM-Modell (Student-Teacher Feature Pyramid Matching) verwendet ein Lehrer- und ein Schüler-Modell

    Args:
        architecture (str): Die Architektur des Modells, z.B. 'resnet50'.
        layers (list): Eine Liste von Schichten, die extrahiert werden sollen, z.B. [2, 3, 4].
    """

    def __init__(self, architecture, layers):
        super().__init__()

        self.teacher_model = FeatureExtractor(
            backbone=architecture, pretrained=True, layers=layers
        ).eval()

        self.student_model = FeatureExtractor(
            backbone=architecture, pretrained=False, layers=layers, requires_grad=True
        ).train()

        self.stem_model = self.extract_stem_layers()

        for parameters in self.stem_model.parameters():
            parameters.requires_grad = False

        self.stem_model.eval()

    def extract_stem_layers(self):
        teacher_backbone = self.teacher_model.model
        student_backbone = self.student_model.model

        stem_layers = []

        if 'resnet' in self.teacher_model.model.__class__.__name__.lower() or 'shufflenet' in self.teacher_model.model.__class__.__name__.lower():
            stem_layer_names = ['conv1', 'bn1', 'relu', 'maxpool']
            for name in stem_layer_names:
                if hasattr(teacher_backbone, name):
                    stem_layers.append(getattr(teacher_backbone, name))
                    setattr(teacher_backbone, name, nn.Identity())
                    setattr(student_backbone, name, nn.Identity())
        elif 'mobilenet' in self.teacher_model.model.__class__.__name__.lower():
            if hasattr(teacher_backbone, 'features') and isinstance(teacher_backbone.features, nn.Sequential):
                stem_module = teacher_backbone.features[0]
                stem_layers.append(stem_module)
                teacher_backbone.features[0] = nn.Identity()
                student_backbone.features[0] = nn.Identity()
        else:
            raise ValueError(
                f"Architecture '{self.teacher_model.model.__class__.__name__}' not supported for stem extraction.")

        if not stem_layers:
            raise ValueError(
                f"No stem layers were extracted for {self.teacher_model.model.__class__.__name__}. Please check the architecture and layer names.")

        return nn.Sequential(*stem_layers)

    def anomaly_map(self, x):
        """ 
        Berechnet die Anomalie-Karte für einen Batch von Bildern.

        Args:
            x (torch.Tensor): Ein Batch von Bildern, z.B. (B, C, H, W).

        Returns:
            torch.Tensor: Eine Anomalie-Karte der Form (B, H, W), die die Anomalien im Bild darstellt.
        """

        # x ist ein Batch von Bildern, z.B. (B, C, H, W)
        teacher_feature_maps, student_feature_maps = self.forward(x)
        batch_size, _, img_height, img_width = x.shape

        anomaly_map = torch.ones(
            (batch_size, img_height, img_width), device=x.device
        )
        for t_map, s_map in zip(teacher_feature_maps, student_feature_maps):
            # t_map, s_map Formen sind (B, C, H, W)

            # 1. Feature-Maps normalisieren
            t_map_norm = torch.nn.functional.normalize(t_map, dim=1)
            s_map_norm = torch.nn.functional.normalize(s_map, dim=1)

            # 2. Quadratische L2-Distanz berechnen, über Kanäle summieren
            # Dies resultiert in einer Karte der Form (B, H, W)
            am = 0.5 * torch.sum(torch.pow(t_map_norm - s_map_norm, 2), dim=1)

            # 3. Auf Bildgröße hochskalieren
            # self.upsample erwartet (B, C, H, W). Kanal-Dimension hinzufügen.
            am = nn.Upsample(size=(img_height, img_width), mode="bilinear", align_corners=False)(
                am.unsqueeze(1))  # am hat jetzt die Form (B, C, H, W)

            anomaly_map = torch.mul(anomaly_map, am.squeeze(1))

        return anomaly_map

    def get_img_size_on_first_layer(self, x):
        """
        Gibt die Bildgröße der ersten Schicht des Modells zurück.
        Dies ist nützlich, um die Eingangsgröße für das Modell zu bestimmen.

        Args:
            x (torch.Tensor): Ein Batch von Bildern, z.B. (B, C, H, W).

        Returns:
            tuple: Ein Tupel (H, W), das die Höhe und Breite der Feature-Map der ersten Schicht angibt.
        """

        teacher_maps, _ = self.forward(x)
        if len(teacher_maps) > 0:
            first_layer_map = teacher_maps[0]
            return first_layer_map.shape[-2], first_layer_map.shape[-1]

    def forward(self, x):
        """
        Führt das Modell für einen Batch von Bildern aus und gibt die Feature-Maps
        des Lehrers und des Schülers zurück.

        Args:
            x (torch.Tensor): Ein Batch von Bildern, z.B. (B, C, H, W).

        Returns:
            tuple: Ein Tupel, das die Feature-Maps des Lehrers und des Schülers enthält.
                   Jede Feature-Map ist ein Tensor der Form (B, C, H, W).
        """

        stem_output = self.stem_model(x)

        with torch.no_grad():
            teacher_feature_maps = self.teacher_model(stem_output)
        student_feature_maps = self.student_model(stem_output)

        teacher_feature_maps = list(teacher_feature_maps.values())
        student_feature_maps = list(student_feature_maps.values())

        return teacher_feature_maps, student_feature_maps
