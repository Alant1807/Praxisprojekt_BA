"""
Dieses Skript definiert die Kernkomponenten des STFPM (Student-Teacher Feature Pyramid Matching) Modells.
Es besteht aus zwei Hauptklassen:
1. FeatureExtractor: Ein generischer Wrapper, um Zwischen-Feature-Maps aus jedem torchvision-Modell zu extrahieren.
2. STFPM: Die Hauptmodellklasse, die ein vortrainiertes Lehrer-Modell und ein trainierbares Schüler-Modell
   verwendet, um Anomalien durch den Vergleich ihrer jeweiligen Feature-Maps zu erkennen.
"""

# --- Importe ---
import torch
import torch.nn as nn
import torchvision.models as models
# Notwendig für den Zugriff auf quantisierte Modelle
import torchvision.models.quantization
from typing import List, Dict, Tuple


class STFPM(nn.Module):
    """
    Implementiert das STFPM-Modell, das auf dem Vergleich der Merkmale eines
    festen, vortrainierten Lehrer-Modells und eines trainierbaren Schüler-Modells basiert.

    Die Architektur trennt die ersten "Stem"-Layer (z.B. die erste Konvolution) ab,
    friert sie ein und verwendet sie als gemeinsamen Feature-Extractor für Lehrer und Schüler.
    Dies reduziert die Trainingszeit und stellt sicher, dass beide Modelle die gleichen
    Low-Level-Features als Eingabe erhalten.

    Args:
        architecture (str): Der Name der Backbone-Architektur aus torchvision (z.B. 'resnet18').
        layers (list): Eine Liste von Layer-Namen, deren Feature-Maps extrahiert werden sollen.
        quantize (bool): Gibt an, ob ein quantisiertes Modell verwendet werden soll.
    """

    def __init__(self, architecture: str, layers: List[str], quantize: bool = False):
        super().__init__()

        # Lehrer-Modell: Nutzt vortrainierte Gewichte und wird nicht trainiert (eval mode).
        # Es dient als Referenz für "normale" Merkmale.
        self.teacher_model = FeatureExtractor(
            backbone=architecture, pretrained=True, layers=layers, quantize=quantize
        ).eval()

        # Schüler-Modell: Hat die gleiche Architektur, aber ohne vortrainierte Gewichte.
        # Es wird trainiert, die Merkmale des Lehrers nachzuahmen (train mode).
        self.student_model = FeatureExtractor(
            backbone=architecture, pretrained=False, layers=layers, quantize=quantize, requires_grad=True
        ).train()

        # Extrahiert die initialen Layer (den "Stem") aus dem Backbone.
        # Diese Layer werden für Lehrer und Schüler geteilt.
        self.stem_model = self.extract_stem_layers()

        # Friere die Parameter des Stems ein, da sie nicht trainiert werden sollen.
        for parameters in self.stem_model.parameters():
            parameters.requires_grad = False
        self.stem_model.eval()

    def extract_stem_layers(self) -> nn.Sequential:
        """
        Trennt die initialen Layer ("Stem") vom Backbone des Lehrer- und Schülermodells.
        Diese Layer werden durch `nn.Identity` ersetzt, um Doppelverarbeitung zu vermeiden.
        Dies ist architekturspezifisch.

        Returns:
            nn.Sequential: Ein Modul, das die extrahierten Stem-Layer enthält.
        """
        teacher_backbone = self.teacher_model.model
        student_backbone = self.student_model.model
        stem_layers = []
        model_name = teacher_backbone.__class__.__name__.lower()

        # Spezifische Logik für ResNet- und ShuffleNet-Architekturen
        if 'resnet' in model_name or 'shufflenet' in model_name:
            stem_layer_names = ['conv1', 'bn1', 'relu', 'maxpool']
            for name in stem_layer_names:
                if hasattr(teacher_backbone, name):
                    # Füge den Layer zum Stem hinzu
                    stem_layers.append(getattr(teacher_backbone, name))
                    # Ersetze den extrahierten Layer im Originalmodell durch eine Identity-Funktion
                    setattr(teacher_backbone, name, nn.Identity())
                    setattr(student_backbone, name, nn.Identity())

        # Spezifische Logik für MobileNet
        elif 'mobilenet' in model_name:
            if hasattr(teacher_backbone, 'features') and isinstance(teacher_backbone.features, nn.Sequential):
                stem_module = teacher_backbone.features[0]
                stem_layers.append(stem_module)
                teacher_backbone.features[0] = nn.Identity()
                student_backbone.features[0] = nn.Identity()
        else:
            raise ValueError(
                f"Architektur '{teacher_backbone.__class__.__name__}' wird für die Stem-Extraktion nicht unterstützt.")

        if not stem_layers:
            raise ValueError(
                f"Für {teacher_backbone.__class__.__name__} konnten keine Stem-Layer extrahiert werden.")

        return nn.Sequential(*stem_layers)

    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Berechnet die Anomalie-Karte durch den Vergleich der Feature-Maps von Lehrer und Schüler.

        Args:
            x (torch.Tensor): Der Eingabebild-Tensor (B, C, H, W).

        Returns:
            torch.Tensor: Eine Anomalie-Karte (B, H, W), bei der höhere Werte auf Anomalien hindeuten.
        """
        teacher_feature_maps, student_feature_maps = self.forward(x)
        batch_size, _, img_height, img_width = x.shape

        # Initialisiere die Anomalie-Karte mit Einsen
        anomaly_map = torch.ones(
            (batch_size, img_height, img_width), device=x.device
        )

        # Iteriere über die Feature-Maps der verschiedenen Ebenen
        for t_map, s_map in zip(teacher_feature_maps, student_feature_maps):
            # 1. Normalisiere die Feature-Maps entlang der Kanal-Dimension.
            #    Dies entspricht der Berechnung der Kosinus-Ähnlichkeit und macht den Vergleich
            #    unabhängig von der Magnitude der Aktivierungen.
            t_map_norm = torch.nn.functional.normalize(t_map, dim=1)
            s_map_norm = torch.nn.functional.normalize(s_map, dim=1)

            # 2. Berechne die quadratische L2-Distanz (pixelweise) und summiere über die Kanäle.
            #    Das Ergebnis ist eine Distanzkarte der Form (B, H, W).
            distance_map = 0.5 * \
                torch.sum(torch.pow(t_map_norm - s_map_norm, 2), dim=1)

            # 3. Skaliere die Distanzkarte auf die ursprüngliche Bildgröße hoch.
            #    unsqueeze(1) fügt eine Kanal-Dimension hinzu, die für Upsample benötigt wird.
            upsampled_map = nn.Upsample(size=(img_height, img_width), mode="bilinear", align_corners=False)(
                distance_map.unsqueeze(1))

            # 4. Multipliziere die Anomalie-Karten der verschiedenen Ebenen elementweise.
            #    Dies kombiniert die Informationen aus den unterschiedlichen Abstraktionsebenen.
            anomaly_map = torch.mul(anomaly_map, upsampled_map.squeeze(1))

        return anomaly_map

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Führt den Forward-Pass für das gesamte STFPM-Modell aus.

        Args:
            x (torch.Tensor): Der Eingabebild-Tensor (B, C, H, W).

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Ein Tupel mit zwei Listen:
                - Die Feature-Maps des Lehrer-Modells.
                - Die Feature-Maps des Schüler-Modells.
        """
        # 1. Leite das Bild durch den gemeinsamen, eingefrorenen Stem.
        stem_output = self.stem_model(x)

        # 2. Berechne die Feature-Maps des Lehrers (ohne Gradientenberechnung).
        with torch.no_grad():
            teacher_feature_maps = self.teacher_model(stem_output)

        # 3. Berechne die Feature-Maps des Schülers (mit Gradientenberechnung, falls im Trainingsmodus).
        student_feature_maps = self.student_model(stem_output)

        # Konvertiere die Dictionaries der Feature-Maps in Listen.
        teacher_feature_maps = list(teacher_feature_maps.values())
        student_feature_maps = list(student_feature_maps.values())

        return teacher_feature_maps, student_feature_maps


class FeatureExtractor(nn.Module):
    """
    Ein Wrapper um ein torchvision-Modell, der die Aktivierungen von
    Zwischen-Layern mithilfe von Forward Hooks extrahiert.

    Args:
        backbone (str): Name des Modells aus torchvision.
        pretrained (bool): Ob vortrainierte Gewichte geladen werden sollen.
        layers (list): Liste der Layer-Namen, von denen Features extrahiert werden sollen.
        quantize (bool): Ob eine quantisierte Version des Modells geladen werden soll.
        requires_grad (bool): Ob die Parameter des Modells trainierbar sein sollen.
    """

    def __init__(self, backbone: str, pretrained: bool, layers: List[str], quantize: bool, requires_grad: bool = False):
        super().__init__()

        if backbone not in models.__dict__ and backbone not in torchvision.models.quantization.__dict__:
            raise ValueError(
                f"Backbone '{backbone}' nicht in torchvision.models gefunden.")

        # Argumente für das Laden des Modells vorbereiten
        weights_arg = "DEFAULT" if pretrained else None
        quantize_arg = quantize and pretrained

        # Lade das Modell (entweder regulär oder quantisiert)
        self.model = torchvision.models.quantization.__dict__[backbone](
            weights=weights_arg,
            quantize=quantize_arg
        )

        self.layers = layers
        self.features: Dict[str, torch.Tensor] = {}

        # Setze den `requires_grad`-Status für alle Parameter des Modells.
        for param in self.model.parameters():
            param.requires_grad = requires_grad

        # Registriere einen Forward Hook für jeden spezifizierten Layer.
        named_modules = dict([*self.model.named_modules()])
        for name in self.layers:
            if name not in named_modules:
                raise ValueError(
                    f"Layer '{name}' nicht im Modell gefunden. Verfügbare Layer: {list(named_modules.keys())}")
            layer = named_modules[name]
            layer.register_forward_hook(self._get_hook(name))

    def _get_hook(self, name: str):
        """Erstellt eine Hook-Funktion, die die Ausgabe eines Layers speichert."""
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Führt einen Forward-Pass aus, um die Hooks auszulösen und die Feature-Maps zu sammeln.

        Args:
            x (torch.Tensor): Der Eingabe-Tensor.

        Returns:
            Dict[str, torch.Tensor]: Ein Dictionary, das die extrahierten Feature-Maps enthält.
        """
        # Lösche Features aus dem vorherigen Forward-Pass.
        self.features.clear()

        # Der eigentliche Output des Modells wird ignoriert (`_`).
        # Der Zweck dieses Passes ist es, die Hooks zu aktivieren, welche `self.features` füllen.
        # Wenn das Modell nicht trainiert wird, kann dies in einem `no_grad`-Kontext geschehen, um Speicher zu sparen.
        if not next(self.model.parameters()).requires_grad:
            with torch.no_grad():
                _ = self.model(x)
        else:
            _ = self.model(x)

        return self.features
