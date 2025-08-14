import torch
import torch.nn as nn

from Scripts.loss import *

from anomalib.models.components import TimmFeatureExtractor


class STFPM(nn.Module):
    """
    Das STFPM-Modell (Student-Teacher Feature Pyramid Matching) verwendet ein Lehrer- und ein Schüler-Modell

    Args:
        architecture (str): Die Architektur des Modells, z.B. 'resnet50'.
        layers (list): Eine Liste von Schichten, die extrahiert werden sollen, z.B. [2, 3, 4].
    """

    def __init__(self, architecture, layers):
        super().__init__()

        self.teacher_model = TimmFeatureExtractor(
            backbone=architecture, pre_trained=True, layers=layers
        ).eval()

        self.student_model = TimmFeatureExtractor(
            backbone=architecture, pre_trained=False, layers=layers, requires_grad=True
        )

        self.stem_model = self.extract_stem_layers()

        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False
        for parameters in self.stem_model.parameters():
            parameters.requires_grad = False

        self.stem_model.eval()

    def extract_stem_layers(self):
        stem_layer_names = list()

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
        # teacher_feature_maps ist eine Liste von L Tensoren, jeder [B, C, H, W]
        batch_size = x.shape[0]

        # img_height, img_width = self.get_img_size_on_first_layer(x)
        img_height, img_width = x.shape[-2], x.shape[-1]

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

            # Kanal-Dimension entfernen
            am = am.squeeze(1)

            anomaly_map = torch.mul(anomaly_map, am)

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

        if isinstance(teacher_feature_maps, dict) and isinstance(
            student_feature_maps, dict
        ):
            teacher_feature_maps = list(teacher_feature_maps.values())
            student_feature_maps = list(student_feature_maps.values())

        return teacher_feature_maps, student_feature_maps
