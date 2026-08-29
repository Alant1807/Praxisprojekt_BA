import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any

from Scripts.losses import *
from anomalib.models.components import TimmFeatureExtractor


class STFPM(nn.Module):
    """PyTorch-Modell für STFPM (Student-Teacher Feature Pyramid Matching).

    Ein eingefrorenes Teacher-Modell liefert Referenz-Features für fehlerfreie Bilder.
    Ein Student-Modell lernt, diese Features nachzuahmen. Bei defekten Bildern
    entstehen Abweichungen zwischen beiden Modellen, welche die Anomalien aufzeigen.
    """

    def __init__(
        self,
        is_asymmetric: bool = False,
        architecture: str = None,
        layers: list = None,
        teacher_architecture: str = None,
        teacher_layers: list = None,
        student_architecture: str = None,
        student_layers: list = None,
        extract_stem: bool = False,
        partial_share_depth: int = 0,
        projection_head_type: str = "simple",
    ):
        """Initialisiert das STFPM-Modell.

        Args:
            is_asymmetric (bool): Ob Teacher und Student unterschiedliche Architekturen haben.
            architecture (str, optional): Backbone-Architektur (für symmetrischen Modus).
            layers (list, optional): Zu extrahierende Layer (für symmetrischen Modus).
            teacher_architecture (str, optional): Backbone des Teachers (für asymmetrischen Modus).
            teacher_layers (list, optional): Layer des Teachers (für asymmetrischen Modus).
            student_architecture (str, optional): Backbone des Students (für asymmetrischen Modus).
            student_layers (list, optional): Layer des Students (für asymmetrischen Modus).
            extract_stem (bool): Ob die ersten Schichten (Stem) geteilt werden.
            partial_share_depth (int): Wie viele tieferliegende Feature-Layer geteilt werden.
            projection_head_type (str): Art der Anpassungsschicht ("simple").
        """
        super().__init__()

        self.is_asymmetric = is_asymmetric
        self.projection_heads = nn.ModuleList()
        self.extract_stem = extract_stem
        self.partial_share_depth = partial_share_depth
        self.projection_head_type = projection_head_type

        if self.partial_share_depth > 0:
            if is_asymmetric:
                raise ValueError(
                    "Partial Sharing (PaSTe-Style) ist nur im symmetrischen Modus möglich."
                )
            if not extract_stem:
                raise ValueError(
                    "Partial Sharing setzt Stem-Sharing voraus (extract_stem=True)."
                )

        if is_asymmetric:
            self._init_asymmetric(
                teacher_architecture,
                teacher_layers,
                student_architecture,
                student_layers,
            )
        else:
            self._init_symmetric(architecture, layers)

        # Der Lehrer darf nicht lernen (eingefroren), da er die Ground Truth darstellt
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def _init_asymmetric(
        self,
        teacher_arch: str,
        teacher_layers: list,
        student_arch: str,
        student_layers: list,
    ):
        """Richtet unterschiedliche Netzwerke für Teacher und Student ein.

        Args:
            teacher_arch (str): Architektur des Teachers.
            teacher_layers (list): Zu vergleichende Schichten des Teachers.
            student_arch (str): Architektur des Students.
            student_layers (list): Zu vergleichende Schichten des Students.
        """
        if not all([teacher_arch, teacher_layers, student_arch, student_layers]):
            raise ValueError(
                "Asymmetrischer Modus benötigt vollständige Spezifikation für Teacher und Student."
            )
        if len(teacher_layers) != len(student_layers):
            raise ValueError(
                "Mismatch: Anzahl der Extraktions-Layer muss übereinstimmen."
            )

        self.teacher_model = TimmFeatureExtractor(
            backbone=teacher_arch, pre_trained=True, layers=teacher_layers
        ).eval()

        self.student_model = TimmFeatureExtractor(
            backbone=student_arch,
            pre_trained=False,
            layers=student_layers,
            requires_grad=True,
        )

        # Im asymmetrischen Modus ist ein geteilter Eingangsbereich (Stem) nicht möglich
        self.stem_model = nn.Identity()

        teacher_info = self.teacher_model.feature_extractor.feature_info
        student_info = self.student_model.feature_extractor.feature_info

        teacher_channels = {info["module"]: info["num_chs"] for info in teacher_info}
        student_channels = {info["module"]: info["num_chs"] for info in student_info}

        # Projektionsköpfe erstellen, um unterschiedliche Kanalanzahlen anzugleichen
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            t_ch = teacher_channels[t_layer]
            s_ch = student_channels[s_layer]

            head = self._create_projection_head(s_ch, t_ch)
            self.projection_heads.append(head)

    def _create_projection_head(self, in_channels: int, out_channels: int) -> nn.Module:
        """Erstellt eine Schicht zur Anpassung der Feature-Kanäle.

        Args:
            in_channels (int): Kanalanzahl des Students.
            out_channels (int): Kanalanzahl des Teachers.

        Returns:
            nn.Module: Die Anpassungsschicht (Identity oder 1x1 Convolution).
        """
        if self.projection_head_type == "simple":
            if in_channels == out_channels:
                return nn.Identity()
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            raise ValueError(f"Unbekannter projection_head_type: {self.projection_head_type}")

    def _get_shared_feature_layer_names(self, backbone_arch: str, partial_share_depth: int) -> list:
        """Sucht automatisch die ersten N Feature-Schichten zum Teilen.

        Args:
            backbone_arch (str): Name der Architektur.
            partial_share_depth (int): Anzahl der zu teilenden Schichten.

        Returns:
            list: Namen der identifizierten Schichten.
        """
        if partial_share_depth <= 0:
            return []

        import timm

        # Temporäres Modell erstellen, um nur die Struktur auszulesen
        temp_model = timm.create_model(backbone_arch, pretrained=False, features_only=False)

        stem_name_whitelist = [
            "conv_stem", "conv1", "conv2", "conv3", "bn1", "bn2", "bn3",
            "act1", "act2", "act3", "maxpool", "stem",
        ]
        
        feature_layer_prefixes = [
            "layer", "blocks", "stages", "features", "levels", "patch_embed",
        ]

        shared_feature_layers = []
        feature_count = 0

        for name, _ in temp_model.named_children():
            if any(s in name.lower() for s in stem_name_whitelist):
                continue

            if any(prefix in name.lower() for prefix in feature_layer_prefixes):
                if feature_count < partial_share_depth:
                    shared_feature_layers.append(name)
                    feature_count += 1
                else:
                    break

        del temp_model
        return shared_feature_layers

    def _init_symmetric(self, architecture: str, layers: list):
        """Richtet zwei baugleiche Netzwerke für Teacher und Student ein.

        Args:
            architecture (str): Backbone-Architektur.
            layers (list): Zu vergleichende Schichten.
        """
        if not all([architecture, layers]):
            raise ValueError("Symmetrischer Modus benötigt Architektur und Layer-Liste.")

        if self.partial_share_depth > 0:
            self.shared_feature_layers = layers[: self.partial_share_depth]
        else:
            self.shared_feature_layers = []

        # Entfernt geteilte Layer aus der Liste der zu vergleichenden Layer
        if self.shared_feature_layers:
            effective_layers = [l for l in layers if l not in self.shared_feature_layers]
            if not effective_layers:
                raise ValueError(
                    f"Alle Matching-Layer werden geteilt - keine Layer zum Vergleich übrig."
                )
            print(f"Partial Sharing aktiv (Tiefe={self.partial_share_depth}):")
            print(f"  Zusätzlich geteilte Feature-Layer: {self.shared_feature_layers}")
            print(f"  Matching-Layer: {layers} → {effective_layers}")
        else:
            effective_layers = layers

        self.effective_layers = effective_layers

        self.teacher_model = TimmFeatureExtractor(
            backbone=architecture, pre_trained=True, layers=effective_layers
        ).eval()

        self.student_model = TimmFeatureExtractor(
            backbone=architecture,
            pre_trained=False,
            layers=effective_layers,
            requires_grad=True,
        )

        if self.extract_stem:
            print("Extracting stem layers for efficiency.")
            self.stem_model = self._extract_stem_layers(
                additional_shared_layers=self.shared_feature_layers
            )

            for param in self.stem_model.parameters():
                param.requires_grad = False
            self.stem_model.eval()
        else:
            print("No Stem-Layer extraction.")
            self.stem_model = nn.Identity()

    def _extract_stem_layers(self, additional_shared_layers: list = None) -> nn.Sequential:
        """Kapselt die ersten Schichten ab, damit sie von Teacher und Student gemeinsam genutzt werden.

        Args:
            additional_shared_layers (list, optional): Zusätzliche Feature-Schichten.

        Returns:
            nn.Sequential: Das extrahierte Modul oder nn.Identity bei Fehlschlag.
        """
        additional_shared_layers = additional_shared_layers or []

        stem_name_whitelist = [
            "conv_stem", "conv1", "conv2", "conv3", "bn1", "bn2", "bn3",
            "act1", "act2", "act3", "maxpool", "stem",
        ]

        all_children = list(self.teacher_model.feature_extractor.named_children())
        all_child_names = [name for name, _ in all_children]

        if additional_shared_layers:
            last_shared = additional_shared_layers[-1]
            last_shared_top = last_shared.split(".")[0]
            last_shared_flat = last_shared.replace(".", "_")

            flat_match = False
            if last_shared_flat in all_child_names:
                cutoff_idx = all_child_names.index(last_shared_flat)
                flat_match = True
            elif last_shared_top in all_child_names:
                cutoff_idx = all_child_names.index(last_shared_top)
            else:
                 print(f"Warnung: '{last_shared}' nicht in Top-Level-Modulen gefunden.")
                 cutoff_idx = -1

            if cutoff_idx >= 0:
                stem_layer_names = []

                for idx, (name, _) in enumerate(all_children):
                    if idx < cutoff_idx:
                        stem_layer_names.append(name)
                    elif idx == cutoff_idx:
                        if flat_match:
                            stem_layer_names.append(name)
                        elif "." in last_shared:
                            parent_module = self.teacher_model.feature_extractor.get_submodule(name)
                            sub_idx = int(last_shared.split(".")[-1])

                            for sub_name, _ in parent_module.named_children():
                                full_name = f"{name}.{sub_name}"
                                stem_layer_names.append(full_name)
                                if sub_name.isdigit() and int(sub_name) >= sub_idx:
                                    break
                        else:
                            stem_layer_names.append(name)
                        break 
        else:
            stem_layer_names = []
            for name, _ in all_children:
                if any(s in name.lower() for s in stem_name_whitelist):
                    stem_layer_names.append(name)

        if not stem_layer_names:
            print("Warnung: Keine Stem-Layer identifiziert, verwende Identity.")
            return nn.Identity()

        print(f"Extrahierte Stem-Layer (inkl. Partial Sharing): {stem_layer_names}")

        try:
            stem_model = nn.Sequential(
                *[self.teacher_model.feature_extractor.get_submodule(name) for name in stem_layer_names]
            )

            def replace_submodule(model, target_name, new_module):
                parts = target_name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                last_part = parts[-1]
                if hasattr(parent, "__getitem__") and last_part.isdigit():
                    parent[int(last_part)] = new_module
                else:
                    setattr(parent, last_part, new_module)

            # Entferne die ausgelagerten Schichten aus den Originalmodellen
            for name in stem_layer_names:
                replace_submodule(self.teacher_model.feature_extractor, name, nn.Identity())
                replace_submodule(self.student_model.feature_extractor, name, nn.Identity())

            return stem_model

        except Exception as e:
            print(f"Stem-Extraktion fehlgeschlagen: {e}. Verwende Identity.")
            return nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cached_teacher_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Verarbeitet ein Bild im Modell.

        Args:
            x (torch.Tensor): Das Eingabebild.
            cached_teacher_features (list, optional): Zuvor berechnete Features des Teachers, um Rechenzeit zu sparen.

        Returns:
            tuple: (Teacher-Features, angepasste Student-Features)
        """
        stem_output = self.stem_model(x)

        if cached_teacher_features is not None:
            teacher_feature_maps = cached_teacher_features
        else:
            with torch.no_grad():
                teacher_feature_maps = self.teacher_model(stem_output)

            if isinstance(teacher_feature_maps, dict):
                teacher_feature_maps = list(teacher_feature_maps.values())

        student_feature_maps = self.student_model(stem_output)
        if isinstance(student_feature_maps, dict):
            student_feature_maps = list(student_feature_maps.values())

        aligned_student_maps = []

        for i, (t_map, s_map) in enumerate(zip(teacher_feature_maps, student_feature_maps)):
            if self.is_asymmetric:
                s_map_projected = self.projection_heads[i](s_map)
            else:
                s_map_projected = s_map

            # Gegebenenfalls räumliche Anpassung durchführen, falls Architekturen leicht abweichen
            if s_map_projected.shape[-2:] != t_map.shape[-2:]:
                s_map_aligned = F.interpolate(
                    s_map_projected,
                    size=t_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                s_map_aligned = s_map_projected

            aligned_student_maps.append(s_map_aligned)

        return teacher_feature_maps, aligned_student_maps

    def anomaly_map(self, x: torch.Tensor, use_original_paper: bool) -> torch.Tensor:
        """Berechnet die finale Anomalie-Heatmap.

        Args:
            x (torch.Tensor): Das Eingabebild.
            use_original_paper (bool): Ob die Original-MSE-Berechnung oder Cosine-Similarity genutzt wird.

        Returns:
            torch.Tensor: Die fertig aufsummierte Anomalie-Map.
        """
        teacher_feature_maps, student_feature_maps = self.forward(x)
        batch_size, _, img_height, img_width = x.shape

        anomaly_map = torch.ones(
            (batch_size, img_height, img_width), device=x.device, dtype=x.dtype
        )

        if use_original_paper:
            for t_map, s_map in zip(teacher_feature_maps, student_feature_maps):
                level_anomaly_map = compute_level_anomaly_mse(t_map, s_map)

                level_anomaly_map = F.interpolate(
                    level_anomaly_map.unsqueeze(1),
                    size=(img_height, img_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                anomaly_map = anomaly_map * level_anomaly_map
        else:
            for t_map, s_map in zip(teacher_feature_maps, student_feature_maps):
                level_anomaly_map = compute_level_anomaly_cosine(t_map, s_map)

                # In-place Operationen für besseres Speichermanagement
                level_anomaly_map = F.interpolate(
                    level_anomaly_map.unsqueeze(1),
                    size=(img_height, img_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze_(1)

                anomaly_map.mul_(level_anomaly_map)

        return anomaly_map

    def get_trainable_state_dict(self) -> Dict[str, Any]:
        """Gibt die gelernten Modellgewichte zurück.

        Returns:
            dict: Modellgewichte (verschachtelt bei Asymmetrie, sonst flach).
        """
        if self.is_asymmetric and len(self.projection_heads) > 0:
            return {
                "student": self.student_model.state_dict(),
                "projection_heads": self.projection_heads.state_dict(),
            }
        return self.student_model.state_dict()

    def load_trainable_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Lädt die Modellgewichte, automatisch passend für alte (flache) oder neue Formate.

        Args:
            state_dict (dict): Die zu ladenden Gewichte.
        """
        is_nested = (
            isinstance(state_dict, dict)
            and "student" in state_dict
            and isinstance(state_dict["student"], dict)
        )

        if is_nested:
            self.student_model.load_state_dict(state_dict["student"])
            if self.is_asymmetric and "projection_heads" in state_dict:
                 self.projection_heads.load_state_dict(state_dict["projection_heads"])
        else:
            self.student_model.load_state_dict(state_dict)

    def get_trainable_parameters(self):
        """Gibt alle Parameter zurück, die im Training optimiert werden sollen.

        Returns:
            list: Liste der trainierbaren Parameter.
        """
        params = list(self.student_model.parameters())
        if self.is_asymmetric:
            params += list(self.projection_heads.parameters())
        return params

    def get_model_info(self) -> Dict[str, Any]:
        """Zählt die Parameter der einzelnen Modellbestandteile.

        Returns:
            dict: Informationen zur Modellgröße und Architektur.
        """
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        projection_params = sum(p.numel() for p in self.projection_heads.parameters())
        stem_params = sum(p.numel() for p in self.stem_model.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "is_asymmetric": self.is_asymmetric,
            "extract_stem": self.extract_stem,
            "teacher_params": f"{teacher_params:,}",
            "student_params": f"{student_params:,}",
            "projection_params": f"{projection_params:,}",
            "projection_head_type": self.projection_head_type,
            "stem_params": f"{stem_params:,}",
            "total_params": f"{teacher_params + student_params + projection_params + stem_params:,}",
            "trainable_params": f"{trainable_params:,}",
            "num_projection_heads": len(self.projection_heads),
         }


@torch.jit.script
def compute_level_anomaly_cosine(t_map: torch.Tensor, s_map: torch.Tensor) -> torch.Tensor:
    """Berechnet die Abweichung pro Schicht über Cosine Similarity.

    Args:
        t_map (torch.Tensor): Teacher-Features.
        s_map (torch.Tensor): Student-Features.

    Returns:
        torch.Tensor: Anomaliewert der Schicht.
    """
    cos_sim = F.cosine_similarity(t_map, s_map, dim=1)
    return 2 * (1.0 - cos_sim)


@torch.jit.script
def compute_level_anomaly_mse(t_map: torch.Tensor, s_map: torch.Tensor) -> torch.Tensor:
    """Berechnet die Abweichung pro Schicht über MSE (wie im Original-Paper).

    Args:
        t_map (torch.Tensor): Teacher-Features.
        s_map (torch.Tensor): Student-Features.

    Returns:
        torch.Tensor: Anomaliewert der Schicht.
    """
    t_map_norm = F.normalize(t_map, p=2.0, dim=1)
    s_map_norm = F.normalize(s_map, p=2.0, dim=1)

    diff = t_map_norm - s_map_norm
    return diff.pow(2).sum(dim=1)