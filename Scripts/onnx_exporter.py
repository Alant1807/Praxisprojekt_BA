import os
import shutil
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx import shape_inference, checker

from Scripts.utils import (
    load_config,
    setup_gpu_optimizations,
    setup_cpu_optimizations,
)
from Scripts.stfpm_arch import *
from Scripts.results_aggregator import merge_and_save_variant_summaries


def setup_export_optimizations(device, config):
    """Konfiguriert Hardware-Optimierungen für den Export.

    Args:
        device (torch.device): Das verwendete Gerät (CPU/GPU).
        config (dict): Konfigurations-Wörterbuch.

    Returns:
        dict: Konfiguration der Hardware-Optimierungen.
    """
    if device.type == "cuda":
        hw_config = setup_gpu_optimizations(device, config)
        hw_config["device_type"] = "GPU"
        return hw_config
    else:
        hw_config = setup_cpu_optimizations()
        hw_config["device_type"] = "CPU"
        return hw_config


# Standard ImageNet-Werte
MVTEC_MEAN = [0.485, 0.456, 0.406]
MVTEC_STD = [0.229, 0.224, 0.225]

# Domänenspezifische Werte für GKD
GKD_MEAN = [0.4847, 0.4847, 0.4847]
GKD_STD = [0.3220, 0.3220, 0.3220]


def get_normalization_values(config):
    """Ermittelt die passenden Normalisierungswerte (Mean/Std) basierend auf dem Datensatz.

    Args:
        config (dict): Das Konfigurations-Wörterbuch.

    Returns:
        tuple: (Listen für Mean, Listen für Std)
    """
    norm_config = config.get("preprocessing", {})
    if "mean" in norm_config and "std" in norm_config:
        return norm_config["mean"], norm_config["std"]

    dataset_name = config.get("dataset", {}).get("name", "")

    if dataset_name == "MVTecAD":
        return MVTEC_MEAN, MVTEC_STD
    elif dataset_name == "GKD":
        return GKD_MEAN, GKD_STD
    else:
        print(
            f"Unbekanntes Dataset '{dataset_name}' - verwende ImageNet-Normalisierung"
        )
        return MVTEC_MEAN, MVTEC_STD


class ONNX_Exporter:
    """Kapselt den gesamten Prozess zum Exportieren eines Modells ins ONNX-Format."""

    def __init__(self, best_model_path: str, collected_paths: dict):
        """Initialisiert den Exporter.

        Args:
            best_model_path (str): Pfad zur Modelldatei (.pt).
            collected_paths (dict): Wörterbuch mit benötigten Pfaden (muss 'yaml' enthalten).
        
        Raises:
            ValueError: Wenn 'yaml' nicht in collected_paths vorhanden ist.
        """
        yaml_path = collected_paths.get("yaml")
        if yaml_path is None:
            raise ValueError("'yaml' Pfad fehlt in collected_paths")

        self.config = load_config(yaml_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.hw_config = setup_export_optimizations(self.device, self.config)

        self._print_export_config()

        is_asymmetric = self.config.get("model_settings", {}).get(
            "is_asymmetric", False
        )
        self.use_original_paper = self.config.get("model_settings", {}).get(
            "use_original_paper", False
        )

        model_args = {"is_asymmetric": is_asymmetric}

        if is_asymmetric:
            model_args.update(
                {
                    "teacher_architecture": self.config["teacher_model"]["architecture"],
                    "teacher_layers": self.config["teacher_model"]["layer"],
                    "student_architecture": self.config["student_model"]["architecture"],
                    "student_layers": self.config["student_model"]["layer"],
                    "projection_head_type": self.config["model_settings"].get("projection_head_type", "simple"),
                }
            )
        else:
            model_args.update(
                {
                    "architecture": self.config["model"]["architecture"],
                    "layers": self.config["model"]["layers"],
                    "extract_stem": self.config.get("model_settings", {}).get("shared_stem", True),
                    "partial_share_depth": self.config.get("model_settings", {}).get("partial_share_depth", 0),
                }
            )

        print("Lade STFPM-Modell")
        model = STFPM(**model_args)

        weights = torch.load(
            best_model_path, map_location=self.device, weights_only=True
        )
        model.load_trainable_state_dict(weights)
        model.eval()

        is_mvtec = self.config["dataset"]["name"] == "MVTecAD"
        self.dataset_identifier = (
            f"{self.config['dataset']['name']}_{self.config['dataset']['class']}"
            if is_mvtec
            else self.config["dataset"]["name"]
        )

        img_size = self.config["dataset"]["img_size"]
        mean_values, std_values = get_normalization_values(self.config)

        print(f"  Normalisierung: mean={mean_values}, std={std_values}")

        # Wickelt das Modell mit Vor- und Nachverarbeitungsschichten ein
        self.wrapped_model = (
            ONNXWrapper(
                model=model,
                img_size=img_size,
                mean=mean_values,
                std=std_values,
                use_original_paper=self.use_original_paper,
            )
            .to(self.device)
            .eval()
        )

        impl_type = (
            "Original Paper (L2 + Norm)"
            if self.use_original_paper
            else "Optimiert (Cosine Similarity)"
        )
        print(f"Modell von {best_model_path} geladen und für den ONNX-Export vorbereitet.")
        print(f"Implementierung: {impl_type}")

    def _print_export_config(self):
        """Gibt die aktuelle ONNX-Export-Konfiguration aus."""
        print("\n" + "=" * 50)
        print("ONNX EXPORT KONFIGURATION")
        print("=" * 50)
        if self.hw_config["device_type"] == "GPU":
            print(f"  Device: {self.device} ({self.hw_config['gpu_name']})")
            print(f"  VRAM: {self.hw_config['gpu_memory_gb']:.1f} GB")
            print(f"  TF32: aktiviert")
            print(f"  cuDNN Benchmark: {self.config['training']['cudnn_benchmark']}")
        else:
            print(f"  Device: {self.device}")
            print(f"  CPU Threads: {self.hw_config['num_threads']}")
            mkldnn_status = "aktiviert" if self.hw_config["mkldnn_enabled"] else "nicht verfügbar"
            print(f"  MKL/oneDNN: {mkldnn_status}")
        print("=" * 50 + "\n")

    def export_onnx(self, onnx_output_path: str) -> dict:
        """Exportiert das Modell nach ONNX und führt anschließende Überprüfungen durch.

        Args:
            onnx_output_path (str): Zielpfad für die ONNX-Datei.

        Returns:
            dict: Konfigurations-Metadaten für das exportierte Modell.
        
        Raises:
            RuntimeError: Wenn die ONNX-Datei nicht erstellt werden konnte.
        """
        img_size = self.config["dataset"]["img_size"]

        dummy_input = torch.randint(
            0,
            256,
            (1, img_size, img_size, 3),
            dtype=torch.uint8,
            device=self.device,
        )

        print("Führe Warmup durch...")
        with torch.no_grad():
            for _ in range(3):
                _ = self.wrapped_model(dummy_input)

        input_names = ["input_image"]
        output_names = ["anomaly_map", "anomaly_score"]

        print(f"Exportiere ONNX-Modell nach: {onnx_output_path}")
        torch.onnx.export(
            self.wrapped_model,
            dummy_input,
            onnx_output_path,
            training=torch.onnx.TrainingMode.EVAL,
            export_params=True,
            opset_version=18,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "input_image": {0: "batch_size"},
                "anomaly_map": {0: "batch_size"},
                "anomaly_score": {0: "batch_size"},
            },
            verbose=False,
            export_modules_as_functions=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )

        if not os.path.exists(onnx_output_path):
            raise RuntimeError(f"ONNX-Export fehlgeschlagen: {onnx_output_path}")

        self._infer_and_check(onnx_output_path)
        return self.create_onnx_yaml(input_names, output_names)

    def _infer_and_check(self, path: str):
        """Überprüft und vereinfacht den exportierten ONNX-Graphen.

        Args:
            path (str): Pfad zur gespeicherten ONNX-Datei.
        """
        print("\nFühre ONNX Shape Inference und Validation durch...")
        m = onnx.load(path)

        # Dimensionen der Knoten berechnen
        try:
            m = shape_inference.infer_shapes(m, data_prop=True, strict_mode=True)
            print("Shape Inference erfolgreich.")
        except TypeError:
            m = shape_inference.infer_shapes(m, data_prop=True)
        except Exception:
            m = shape_inference.infer_shapes(m)

        # Graph wenn möglich vereinfachen
        try:
            import onnxsim
            m_simplified, check = onnxsim.simplify(m)
            if check:
                m = m_simplified
                print("Graph-Simplifikation erfolgreich.")
                m = shape_inference.infer_shapes(m, data_prop=True)
            else:
                print("Warnung: Graph-Simplifikation fehlgeschlagen, Original wird verwendet.")
        except ImportError:
            print("Warnung: onnxsim nicht installiert — pip install onnxsim")
        except Exception as e:
            print(f"Warnung: Graph-Simplifikation übersprungen: {e}")

        # Ungenutzte Parameter entfernen
        try:
            all_nodes = m.graph.node
            used_initializers = set()
            for node in all_nodes:
                used_initializers.update(node.input)
            for output in m.graph.output:
                used_initializers.add(output.name)

            unused_count = 0
            new_initializers = []
            for init in m.graph.initializer:
                if init.name in used_initializers:
                    new_initializers.append(init)
                else:
                    unused_count += 1
            if unused_count > 0:
                m.graph.ClearField("initializer")
                m.graph.initializer.extend(new_initializers)
                print(f"Entfernte {unused_count} ungenutzte Initializer.")
        except Exception as e:
            print(f"Warnung beim Entfernen ungenutzter Initializer: {e}")

        # Modellvalidierung abschließen
        try:
            checker.check_model(m)
            print("ONNX Model Check erfolgreich: Das Modell ist gültig.")
        except Exception as e:
            print(f"ONNX Model Check Warnung: {e}")
        
        onnx.save(m, path)

    def create_onnx_yaml(self, input_names: list, output_names: list) -> dict:
        """Erstellt die Metadaten für das exportierte Modell.

        Args:
            input_names (list): Namen der Input-Knoten.
            output_names (list): Namen der Output-Knoten.

        Returns:
            dict: Zusammengefasste Konfiguration als Dictionary.
        """
        import datetime

        mean_values = self.wrapped_model.mean.cpu().numpy().flatten().tolist()
        std_values = self.wrapped_model.std.cpu().numpy().flatten().tolist()

        is_asymmetric = self.config.get("model_settings", {}).get("is_asymmetric", False)
        img_size = self.config["dataset"]["img_size"]

        if is_asymmetric:
            architecture_info = {
                "model_type": "STFPM (Asymmetrisch)",
                "method": "Feature-Based Knowledge Distillation",
                "teacher": {
                    "backbone": self.config["teacher_model"]["architecture"],
                    "matching_layers": self.config["teacher_model"]["layer"],
                    "status": "Eingefroren (pretrained auf ImageNet)",
                },
                "student": {
                    "backbone": self.config["student_model"]["architecture"],
                    "matching_layers": self.config["student_model"]["layer"],
                    "status": "Trainiert auf Normaldaten",
                },
                "projection_heads": self.config["model_settings"].get("projection_head_type", "simple"),
            }
        else:
            shared_stem = self.config.get("model_settings", {}).get("shared_stem", False)
            partial_depth = self.config.get("model_settings", {}).get("partial_share_depth", 0)
            architecture_info = {
                "model_type": "STFPM (Symmetrisch)",
                "method": "Feature-Based Knowledge Distillation",
                "backbone": self.config["model"]["architecture"],
                "matching_layers": self.config["model"]["layers"],
                "shared_stem": shared_stem,
                "partial_share_depth": partial_depth,
            }

        if self.use_original_paper:
            scoring_info = {
                "feature_comparison": "L2-Normalisierung + quadrierte euklidische Distanz (Wang et al., 2021)",
                "anomaly_map": "Multiplikative Fusion der Anomaliekarten aller Matching-Layer nach bilinearem Upsampling",
                "anomaly_score": "Maximum der Anomaly Map",
            }
        else:
            scoring_info = {
                "feature_comparison": "Cosine Similarity (optimierte Variante)",
                "anomaly_map": "Multiplikative Fusion der Anomaliekarten aller Matching-Layer nach bilinearem Upsampling",
                "anomaly_score": "Maximum der Anomaly Map",
            }

        preprocessing_info = {
            "beschreibung": "Vollständig im ONNX-Graphen eingebettet — keine externe Vorverarbeitung nötig",
            "schritte": [
                f"1. Eingabe: Rohes RGB-Bild als uint8 (0-255) im Format [Batch, {img_size}, {img_size}, 3]",
                "2. Konvertierung: uint8 → float32, Multiplikation mit 1/255",
                "3. Layout: Permutation von NHWC zu NCHW",
                f"4. Resize: Bilineare Interpolation auf {img_size}×{img_size} (falls Eingabe abweicht)",
                f"5. Normalisierung: (x - mean) * (1/std) mit mean={[round(v, 4) for v in mean_values]}, std={[round(v, 4) for v in std_values]}",
            ],
        }

        postprocessing_info = {
            "beschreibung": "Vollständig im ONNX-Graphen eingebettet",
            "schritte": [
                "1. Feature-Extraktion durch Teacher und Student",
                "2. Anomaliekarten-Berechnung pro Matching-Layer",
                f"3. Upsampling auf {img_size}×{img_size} (bilinear)",
                "4. Multiplikative Fusion aller Layer-Anomaliekarten",
                "5. Score-Aggregation: Maximum der Anomaly Map",
            ]
        }

        usage_info = {
            "eingabe": f"Ein RGB-Bild als numpy-Array mit dtype=uint8 und Shape [1, H, W, 3]. Empfohlen: {img_size}×{img_size} Pixel. Andere Größen werden intern skaliert.",
            "ausgabe": {
                output_names[0]: f"Anomaly Map [Batch, {img_size}, {img_size}] — höhere Werte = stärkere Anomalie",
                output_names[1]: "Anomaly Score [Batch] — einzelner Wert pro Bild, vergleichbar mit dem Schwellenwert",
            },
            "schwellenwert": "Wird separat in der inference_summary.json bereitgestellt (Feld: 'quantile_threshold'). Score > Schwellenwert → Anomalie erkannt.",
            "beispiel": {
                "python": [
                    "import onnxruntime as ort",
                    "import numpy as np",
                    "from PIL import Image",
                    "",
                    "session = ort.InferenceSession('modell.onnx')",
                    f"img = np.array(Image.open('bild.png').convert('RGB').resize(({img_size}, {img_size})), dtype=np.uint8)",
                    "img = img[np.newaxis, ...]  # Batch-Dimension hinzufügen",
                    "anomaly_map, score = session.run(None, {'input_image': img})",
                    "print(f'Anomaly Score: {score[0]:.4f}')",
                ],
            },
        }

        dataset_name = self.config.get("dataset", {}).get("name", "Unbekannt")
        normalization_source = (
            "Berechnet auf dem GKD-Trainingssplit"
            if dataset_name == "GKD"
            else "Standard ImageNet-Werte (pretrained Backbone)"
        )

        dataset_info = {
            "name": self.dataset_identifier,
            "normalisierung": normalization_source,
            "mean": [round(v, 4) for v in mean_values],
            "std": [round(v, 4) for v in std_values],
        }

        inference_config = {
            "modell_id": self.dataset_identifier,
            "export_datum": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "quell_framework": "PyTorch → ONNX (Opset 18)",
            "architektur": architecture_info,
            "datensatz": dataset_info,
            "scoring": scoring_info,
            "eingabe": {
                "namen": input_names,
                "shape": [1, img_size, img_size, 3],
                "datentyp": "uint8 (0-255)",
                "layout": "Batch, Höhe, Breite, Kanäle (NHWC)",
            },
            "ausgabe": {
                "namen": output_names,
                "beschreibungen": {
                    output_names[0]: f"Anomaly Map [{img_size}×{img_size}] — pixelweise Anomaliewerte (float32)",
                    output_names[1]: "Anomaly Score — globaler Bildstatus (float32, höher = anomaler)",
                },
            },
            "eingebettete_vorverarbeitung": preprocessing_info,
            "eingebettete_nachverarbeitung": postprocessing_info,
            "graph_optimierungen": {
                "constant_folding": "Konstante Teilgraphen (Mean/Std-Buffer) zur Exportzeit vorberechnet",
                "graph_simplifikation": "Redundante Operationen über onnxsim entfernt",
                "dead_code_elimination": "Ungenutzte Initializer (z.B. BatchNorm-Tracking-Parameter) entfernt",
                "shape_inference": "Dimensionen aller Zwischentensoren statisch berechnet",
            },
            "nutzung": usage_info,
        }

        return inference_config


class ONNXWrapper(nn.Module):
    """Fügt dem PyTorch-Modell feste Vor- und Nachverarbeitungsschritte hinzu, damit das ONNX-Modell sofort einsatzbereit ist."""

    def __init__(
        self,
        model,
        img_size: int,
        mean: list = None,
        std: list = None,
        use_original_paper: bool = False,
    ):
        """Initialisiert den End-to-End ONNX-Wrapper.

        Args:
            model (nn.Module): Das zugrundeliegende PyTorch-Modell.
            img_size (int): Die Zielgröße der Bilder (Breite und Höhe).
            mean (list, optional): Mittelwerte für die Normalisierung.
            std (list, optional): Standardabweichungen für die Normalisierung.
            use_original_paper (bool, optional): Ob der originale Loss zur Inferenz genutzt wird. Standard ist False.
        """
        super().__init__()

        self.model = model
        self.img_size = img_size
        self.use_original_paper = use_original_paper

        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.register_buffer("inv_std", 1.0 / torch.tensor(std).view(1, 3, 1, 1))
        self.register_buffer("scale_factor", torch.tensor(1.0 / 255.0))

    def forward(self, x: torch.Tensor):
        """Verarbeitet das Bild vom Roh-Input bis zur fertigen Anomalie-Bewertung.

        Args:
            x (torch.Tensor): Unverarbeitetes Bild als Tensor.

        Returns:
            tuple: (Anomaly Map, Anomaly Score)
        """
        x = self.preprocess_image(x)
        teacher_map, student_map = self.model(x)
        anomaly_map, anomaly_score = self.postprocessing(teacher_map, student_map, x)
        return anomaly_map, anomaly_score

    def preprocess_image(self, x: torch.Tensor) -> torch.Tensor:
        """Skaliert, rotiert und normalisiert das rohe Eingangsbild.

        Args:
            x (torch.Tensor): Rohes Eingabebild.

        Returns:
            torch.Tensor: Normalisierter Tensor für das Modell.
        """
        x = x.to(torch.float32)
        x = x.permute(0, 3, 1, 2)
        x = x * self.scale_factor
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.mean) * self.inv_std
        return x

    def postprocessing(self, teacher_map, student_map, preprocessed_x):
        """Berechnet die finale Anomalie-Map und den Score aus den Ausgaben des Modells.

        Args:
            teacher_map (list): Feature-Maps des Teacher-Modells.
            student_map (list): Feature-Maps des Student-Modells.
            preprocessed_x (torch.Tensor): Das vorverarbeitete Eingabebild zur Größenreferenz.

        Returns:
            tuple: (Berechnete Anomaly Map, berechneter Score)
        """
        b, _, H, W = preprocessed_x.shape
        anomaly_map = torch.ones(
            (b, H, W), device=preprocessed_x.device, dtype=torch.float32
        )

        if self.use_original_paper:
            for t_map, s_map in zip(teacher_map, student_map):
                t_map_norm = F.normalize(t_map, p=2.0, dim=1)
                s_map_norm = F.normalize(s_map, p=2.0, dim=1)
                diff = t_map_norm - s_map_norm
                am = diff.pow(2).sum(dim=1)
                am = F.interpolate(
                    am.unsqueeze(1),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                anomaly_map = anomaly_map * am
        else:
            for t_map, s_map in zip(teacher_map, student_map):
                cos_sim = F.cosine_similarity(t_map, s_map, dim=1)
                am = 2 * (1.0 - cos_sim)
                am = F.interpolate(
                    am.unsqueeze(1),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                anomaly_map.mul_(am)

        anomaly_score = anomaly_map.flatten(start_dim=1).max(dim=1)[0]
        return anomaly_map, anomaly_score