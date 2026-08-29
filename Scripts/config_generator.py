import os
import yaml
import copy
import time

# =============================================================================
# KONFIGURATIONSVORLAGEN (Templates)
# =============================================================================
# Hier werden zentrale Hardware- und Trainingsprofile definiert. 
# Dies ermöglicht es, Modelle schnell für verschiedene Ziel-Systeme (GPU, CPU, Baseline) 
# zu generieren, ohne hunderte Zeilen Code anpassen zu müssen.

def get_gpu_optimized_settings():
    """
    Konfigurationsvorlage für High-Performance Training auf modernen NVIDIA GPUs.
    
    Diese Einstellungen zielen auf maximalen Datendurchsatz und minimale Latenz ab.
    
    Optimierungen:
    - Mixed Precision (FP16): Reduziert Speicherbandbreite und erhöht Rechenleistung auf Tensor Cores.
    - Channels Last: Optimiertes Speicherlayout (NHWC) für effizientere Faltungsoperationen.
    - Caching & Prefetching: Aggressives Vorladen von Daten zur Vermeidung von GPU-Leerlauf (Starvation).
    - Asynchroner Transfer: Entkoppelt CPU-Datenbereitstellung von GPU-Berechnung.
    """
    return {
        "dataloader": {
            "batch_size": "auto",       # Dynamische Anpassung an verfügbaren VRAM
            "num_workers": 6,           # Hohe Parallelität für I/O
            "pin_memory": True,         # Page-locked Memory für schnelleren PCIe-Transfer
            "persistent_workers": True, # Vermeidet Overhead durch Worker-Neustart (Prozesse bleiben im RAM)
            "prefetch_factor": 2        # Puffergröße pro Worker (Lädt 2 Batches im Voraus)
        },
        "training": {
            "cache_teacher_features": True,  # Vermeidung redundanter Inferenz des eingefrorenen Lehrers
            "fast_zero_grad": True,          # Optimierter Gradienten-Reset (set_to_none=True spart Zeit)
            "async_host_to_device": True,    # Asynchroner Daten-Transfer CPU -> GPU (Pipelining)
            "force_cpu_caching": False,      # Cached Teacher-Features liegen im schnellen GPU-VRAM
            "cudnn_benchmark": True          # Auto-Tuner für Convolution-Algorithmen (schneller bei fester Input-Größe)
        },
        "model_settings": {
            "channels_last": True,           # NHWC Layout (Hardwarenäher für NVIDIA GPUs)
            "amp_mixed_precision": True,     # Automatic Mixed Precision
            "shared_stem": True,             # Gemeinsamer Feature-Extraktor für Teacher/Student (spart Rechenzeit)
            "partial_share_depth": 1,
            "use_original_paper": False       # Loss-Berechnung gemäß Original-Publikation (MSE)
        },
        "loss": {
            "params": {
                "use_original_paper": False
            }
        }
    }


def get_baseline_settings():
    """
    Baseline-Konfiguration ohne spezielle Hardware-Beschleunigung.
    Dient als Referenzpunkt für Benchmarks (Ablation Studies), um den Einfluss
    der Optimierungen in isolierten Vergleichen zu quantifizieren.
    """
    return {
        "dataloader": {
            "batch_size": 32,
            "num_workers": 6,
            "pin_memory": False,
            "persistent_workers": True,
            "prefetch_factor": 2
        },
        "training": {
            "cache_teacher_features": False, # Teacher wird jede Epoche neu berechnet
            "fast_zero_grad": False,
            "async_host_to_device": False,
            "force_cpu_caching": False,
            "cudnn_benchmark": False
        },
        "model_settings": {
            "channels_last": False,          # Klassisches NCHW Layout
            "amp_mixed_precision": False,    # Reines Float32 Training
            "shared_stem": False,            # Teacher und Student laufen komplett getrennt
            "partial_share_depth": 0,
            "use_original_paper": True
        },
        "loss": {
            "params": {
                "use_original_paper": True
            }
        }
    }


def get_cpu_optimized_settings():
    """
    Spezialisierte Konfiguration für reines CPU-Training (z.B. auf Intel Xeon Servern).
    Fokus liegt auf Overhead-Reduktion, da CPUs oft empfindlicher auf 
    Context-Switches und Speicherverwaltung reagieren als GPUs.
    
    Anpassungen:
    - Deaktiviertes Pinning/Async Transfer (bringt auf CPU keinen Vorteil und kostet Overhead).
    - Caching im System-RAM (da kein VRAM vorhanden).
    """
    return {
        "dataloader": {
            "batch_size": 32,
            "num_workers": 6,
            "pin_memory": False,          # Macht auf CPUs keinen Sinn
            "persistent_workers": True,   
            "prefetch_factor": 2 
        },
        "training": {
            "cache_teacher_features": True,   
            "fast_zero_grad": True,     
            "async_host_to_device": False,  # Blockierender Transfer, da ohnehin alles auf CPU passiert
            "force_cpu_caching": True,       
            "cudnn_benchmark": False
        },
        "model_settings": {
            "channels_last": True,        # Auch moderne CPUs (z.B. mit AVX512) profitieren oft von NHWC
            "amp_mixed_precision": False, # FP16 Support auf CPUs ist oft fehlerhaft oder langsam
            "shared_stem": True, 
            "partial_share_depth": 1,
            "use_original_paper": True   # Nutzt optimierten Cosine-Loss statt Original     
        },
        "loss": {
            "params": {
                "use_original_paper": True    
            }
        }
    }


def get_settings_by_mode(mode: str) -> dict:
    """
    Factory-Methode zur Auswahl der Konfigurationsstrategie.
    Ermöglicht das einfache Umschalten zwischen Experimentier-Umgebungen
    über ein zentrales String-Interface.
    """
    modes = {
        "gpu": get_gpu_optimized_settings,
        "baseline": get_baseline_settings,
        "cpu": get_cpu_optimized_settings
    }

    if mode not in modes:
        raise ValueError(
            f"Unbekannter Modus: {mode}. Erlaubt: {list(modes.keys())}")

    return modes[mode]()


# =============================================================================
# SYMMETRISCHE KONFIGURATIONS-GENERATOREN
# =============================================================================

def generate_gkdConfigs(mode: str = "baseline", append_suffix: bool = False, output_dir_name: str = "Configs_GKD"):
    """
    Generiert YAML-Konfigurationsdateien für den GKD-Datensatz.
    Hierbei handelt es sich um symmetrische Netze (Teacher und Student haben 
    die exakt selbe Architektur).
    
    Args:
        mode: "gpu", "baseline", oder "cpu"
    """
    # Laden der Basiseinstellungen
    settings = get_settings_by_mode(mode)
    settings_label = mode

    # Definition der zu evaluierenden Architekturen und ihrer Extraktions-Layer (Feature-Pyramiden).
    # 'layers' definiert, an welchen Stellen des Netzes Features für den Teacher-Student Vergleich abgegriffen werden.
    base_configs = {
        "resnet10t": {
            "model": {
                "architecture": "resnet10t",
                "layers": ["layer1", "layer2", "layer3", "layer4"],
            }
        },
        "fasternet_t0": {
            "model": {
                "architecture": "fasternet_t0",
                "layers": ["stages.0", "stages.1", "stages.2", "stages.3"],
            }
        },
        "mobilenetv4_conv_small": {
            "model": {
                "architecture": "mobilenetv4_conv_small",
                "layers": ["blocks.0", "blocks.1", "blocks.2", "blocks.4"],
            }
        },
        "mobilenetv4_conv_small_050": {
            "model": {
                "architecture": "mobilenetv4_conv_small_050",
                "layers": ["blocks.0", "blocks.1", "blocks.2", "blocks.4"],
            }
        },
        "lcnet_050": {
            "model": {
                "architecture": "lcnet_050",
                "layers": ["blocks.0", "blocks.1", "blocks.2", "blocks.4", "blocks.5"],
            }
        },
        "mobilenetv3_small_100": {
            "model": {
                "architecture": "mobilenetv3_small_100",
                "layers": ["blocks.0", "blocks.1", "blocks.3", "blocks.5"],
            }
        }
    }

    # Template für die vollständige Experiment-Konfiguration.
    # Hier werden unsere Hardware-Profile aus der Funktion oben mit Parametern wie
    # Epochen, Datensatz-Infos, und LR-Scheduler verheiratet.
    default_config = {
        "dataset": {
            "name": "GKD",
            "base_path": "GKD_Bilder",
            "img_size": 256,
            "enable_timing": False
        },
        "dataloader": {
            "batch_size": settings["dataloader"]["batch_size"],
            "num_workers": 6,
            "pin_memory": settings["dataloader"]["pin_memory"],
            "persistent_workers": settings["dataloader"]["persistent_workers"],
            "prefetch_factor": settings["dataloader"]["prefetch_factor"]
        },
        "epochs": 100,
        "optimizer": {
            "active": "AdamW",
            "configs": {
                "SGD": {"lr": 0.4, "momentum": 0.9, "weight_decay": 0.0001, "nesterov": True},
                "AdamW": {"lr": 0.001, "weight_decay": 0.01},
            },
        },
        "training": {
            "cache_teacher_features": settings["training"]["cache_teacher_features"],
            "fast_zero_grad": settings["training"]["fast_zero_grad"],
            "async_host_to_device": settings["training"]["async_host_to_device"],
            "force_cpu_caching": settings["training"]["force_cpu_caching"],
            "cudnn_benchmark": settings["training"]["cudnn_benchmark"],
            "detailed_timing": False,
            "profiling": False,
        },
        "scheduler": {
            "type": "OneCycleLR",
            "params": {
                "max_lr": 0.01,
                "epochs": 100,
                "pct_start": 0.3,
                "anneal_strategy": "cos",
            },
        },
        "model_settings": {
            "channels_last": settings["model_settings"]["channels_last"],
            "amp_mixed_precision": settings["model_settings"]["amp_mixed_precision"],
            "is_asymmetric": False,      # Hier rein Symmetrisch
            "shared_stem": settings['model_settings']['shared_stem'],
            "partial_share_depth": settings['model_settings']['partial_share_depth'],
            "use_original_paper": settings['model_settings']['use_original_paper']
        },
        "loss": {
            "params": {
                "epsilon": 1e-08, 
                "reduction": "sum", 
                "use_original_paper": settings['loss']['params']['use_original_paper']
            }
        }
    }

    # Iteration und Dateierzeugung
    output_dir = output_dir_name
    suffix = f"_{int(time.time())}" if append_suffix else ""
    # Erzeugt für jede in `base_configs` gelistete Architektur ein eigenes YAML-Config-File.
    for model_name, model_config in base_configs.items():
        # Deep Copy ist essenziell, um Seiteneffekte (Mutations) zwischen den Iterationen zu vermeiden, 
        # da Dictionaries in Python mutable sind (Call by Reference).
        config = copy.deepcopy(default_config)
        config["model"] = model_config["model"]
            
        mode_folder = f"{model_name}"
        save_path = os.path.join(output_dir, mode_folder)
        os.makedirs(save_path, exist_ok=True)

        config_filename = f"STFPM_Config_{model_name}_{suffix}_GKD.yaml"
        # Schreibt die Dictionary-Struktur geordnet auf die Festplatte
        with open(os.path.join(save_path, config_filename), "w") as f:
            yaml.dump(config, f, sort_keys=False,
                      indent=4, default_flow_style=False)

    print(
        f"✅ Alle GKD-Konfigurationen ({settings_label}) wurden im Ordner '{output_dir}' erstellt.")


def generate_MVTec_configs(mode: str = "baseline", append_suffix: bool = False):
    """
    Generiert YAML-Konfigurationsdateien für verschiedene Modelle und ALLE MVTec-Klassen.
    Dies multipliziert die Anzahl der Dateien um den Faktor 15 (Anzahl der MVTec Klassen).
    
    Args:
        mode: "gpu", "baseline", oder "cpu"
    """
    settings = get_settings_by_mode(mode)
    settings_label = mode

    mvtec_classes = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    # Identische Basis-Architekturen wie bei GKD
    base_configs = {
        "resnet10t": {
            "model": {
                "architecture": "resnet10t",
                "layers": ["layer1", "layer2", "layer3", "layer4"],
            }
        },
        "resnet18": {
            "model": {
                "architecture": "resnet18",
                "layers": ["layer1", "layer2", "layer3", "layer4"],
            }
        },
        "tf_efficientnet_lite0": {
            "model": {
                "architecture": "tf_efficientnet_lite0",
                "layers": ["blocks.1", "blocks.2", "blocks.4", "blocks.6"],
            }
        },
        "mobilenetv4_conv_small": {
            "model": {
                "architecture": "mobilenetv4_conv_small",
                "layers": ["blocks.0", "blocks.1", "blocks.2", "blocks.4"],
            }
        },
        "edgenext_x_small": {
            "model": {
                "architecture": "edgenext_x_small",
                "layers": ["stages.0", "stages.1", "stages.2", "stages.3"],
            }
        },
        "edgenext_xx_small": {
            "model": {
                "architecture": "edgenext_xx_small",
                "layers": ["stages.0", "stages.1", "stages.2", "stages.3"],
            }
        },
        "convnextv2_femto": {
            "model": {
                "architecture": "convnextv2_femto",
                "layers": ["stages.0", "stages.1", "stages.2", "stages.3"],
            }
        }
    }

    # Template für MVTec-Spezifika anpassen
    default_config = {
        "dataset": {
            "name": "MVTecAD",
            "base_path": "Images",
            "img_size": 256,
            "enable_timing": False
        },
        "dataloader": {
            "batch_size": settings["dataloader"]["batch_size"],
            "num_workers": 6,
            "pin_memory": settings["dataloader"]["pin_memory"],
            "persistent_workers": settings["dataloader"]["persistent_workers"],
            "prefetch_factor": settings["dataloader"]["prefetch_factor"]
        },
        "epochs": 100,
        "optimizer": {
            "active": "AdamW",
            "configs": {
                "SGD": {"lr": 0.4, "momentum": 0.9, "weight_decay": 1e-4},
                "AdamW": {"lr": 0.0003, "weight_decay": 0.01},
            },
        },
        "training": {
            "cache_teacher_features": settings["training"]["cache_teacher_features"],
            "fast_zero_grad": settings["training"]["fast_zero_grad"],
            "async_host_to_device": settings["training"]["async_host_to_device"],
            "force_cpu_caching": settings["training"]["force_cpu_caching"],
            "cudnn_benchmark": settings["training"]["cudnn_benchmark"],
            "detailed_timing": False,
            "profiling": False,
            "use_qat": False,            # QAT hier für MVTec deaktiviert
            "qat_backend": "fbgemm"
        },
        "scheduler": {
            "type": "OneCycleLR",
            "params": {
                "max_lr": 0.001,
                "epochs": 100,
                "pct_start": 0.3,
                "anneal_strategy": "cos"
            },
        },
        "model_settings": {
            "channels_last": settings["model_settings"]["channels_last"],
            "amp_mixed_precision": settings["model_settings"]["amp_mixed_precision"],
            "is_asymmetric": False,
            "shared_stem": settings['model_settings']['shared_stem'],
            "partial_share_depth": 0,
            "use_original_paper": settings['model_settings']['use_original_paper']
        },
        "loss": {
            "params": {
                "epsilon": 1e-08, 
                "reduction": "sum", 
                "use_original_paper": settings['loss']['params']['use_original_paper']
            }
        }
    }

    output_dir = f"Configs_MVTec"

    suffix = f"_{int(time.time())}" if append_suffix else ""

    # Verschachtelte Schleife: Für jedes Modell und für jede MVTec-Klasse eine Config generieren
    for model_name, model_config in base_configs.items():
        for cls in mvtec_classes:
            
            # WICHTIG: Deepcopy in der inneren Schleife, damit sich die Klassen nicht gegenseitig überschreiben
            config = copy.deepcopy(default_config)
            config["dataset"]["class"] = cls
            config["model"] = model_config["model"]
            
            if "partial_share_depth" in model_config:
                config["model_settings"]["partial_share_depth"] = model_config["partial_share_depth"]

            mode_folder = f"{model_name}"
            # Sub-Ordner pro Klasse erstellen
            save_path = os.path.join(output_dir, mode_folder, cls)
            os.makedirs(save_path, exist_ok=True)

            config_filename = f"Config_{model_name}_{cls}{suffix}.yaml"    
            with open(os.path.join(save_path, config_filename), "w") as f:
                yaml.dump(config, f, sort_keys=False,
                          indent=4, default_flow_style=False)

    print(
        f"✅ Alle MVTec-Konfigurationen ({settings_label}) wurden im Ordner '{output_dir}' erstellt.")


# =============================================================================
# ASYMMETRISCHE KONFIGURATIONS-GENERATOREN
# =============================================================================

def generate_asymmetric_MVTEC_configs(mode: str = "baseline", append_suffix: bool = False, output_dir: str = "Configs_GKD_Asymmetric"):
    """
    Generiert Konfigurationen für asymmetrische Lehrer-Studenten-Paare (MVTec).
    
    Hierbei haben Teacher und Student explizit unterschiedliche Architekturen
    (z.B. schwerer Teacher, extrem leichter Student). Dies ist die eigentliche
    "Knowledge Distillation".
    
    Args:
        mode: "gpu", "baseline", oder "cpu"
    """
    settings = get_settings_by_mode(mode)
    settings_label = mode

    mvtec_classes = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    # Definition der Asymmetrischen Paare.
    # Es muss zwingend ein kompatibles Schicht-Mapping definiert werden, damit Teacher
    # und Student semantisch auf derselben "Höhe" des Modells verglichen werden können.
    asymmetric_pairs = {
        "resnet18_teacher_mobilenetv4_conv_small_050_student": {
            "student_model": {
                "architecture": "mobilenetv4_conv_small_050",
                "layer": ["blocks.0", "blocks.1", "blocks.2", "blocks.4"]
            },
            "teacher_model": {
                "architecture": "resnet10t",
                "layer": ["layer1", "layer2", "layer3", "layer4"]
            }
        },
        "resnet18_teacher_fasternet_t0_student": {
            "student_model": {
                "architecture": "fasternet_t0",
                "layer": ["stages.0", "stages.1", "stages.2", "stages.3"]
            },
            "teacher_model": {
                "architecture": "resnet10t",
                "layer": ["layer1", "layer2", "layer3", "layer4"]
            }
        },
        "resnet18_teacher_lcnet_050_student": {
            "student_model": {
                "architecture": "lcnet_050",
                "layer": ["blocks.1", "blocks.2", "blocks.4", "blocks.5"]
            },
            "teacher_model": {
                "architecture": "resnet10t",
                "layer": ["layer1", "layer2", "layer3", "layer4"]
            }
        },
        "resnet18_teacher_mobilenetv4_conv_small_student": {
            "student_model": {
                "architecture": "mobilenetv4_conv_small",
                "layer": ["blocks.0", "blocks.1", "blocks.2", "blocks.4"]
            },
            "teacher_model": {
                "architecture": "resnet10t",
                "layer": ["layer1", "layer2", "layer3", "layer4"]
            }
        },
        "resnet18_teacher_mobilenetv3_small_100_student": {
            "student_model": {
                "architecture": "mobilenetv3_small_100",
                "layer": ["blocks.0", "blocks.1", "blocks.3", "blocks.5"]
            },
            "teacher_model": {
                "architecture": "resnet10t",
                "layer": ["layer1", "layer2", "layer3", "layer4"]
            }
        }
    }

    # Template für asymmetrische MVTec Modelle
    default_config = {
        "dataset": {
            "name": "MVTecAD",
            "base_path": "Images",
            "img_size": 256,
            "enable_timing": False
        },
        "dataloader": {
            "batch_size": settings['dataloader']['batch_size'],
            "num_workers": 6,
            "pin_memory": settings["dataloader"]["pin_memory"],
            "persistent_workers": settings["dataloader"]["persistent_workers"],
            "prefetch_factor": settings["dataloader"]["prefetch_factor"]
        },
        "epochs": 100,
        "optimizer": {
            "active": "AdamW",
            "configs": {
                "SGD": {"lr": 0.4, "momentum": 0.9, "weight_decay": 0.0001},
                "AdamW": {"lr": 0.001, "weight_decay": 0.01},
            },
        },
        "training": {
            "cache_teacher_features": settings["training"]["cache_teacher_features"],
            "fast_zero_grad": settings["training"]["fast_zero_grad"],
            "async_host_to_device": settings["training"]["async_host_to_device"],
            "force_cpu_caching": settings["training"]["force_cpu_caching"],
            "cudnn_benchmark": settings["training"]["cudnn_benchmark"],
            "detailed_timing": False,
            "profiling": False
        },
        "scheduler": {
            "type": "OneCycleLR",
            "params": {
                "max_lr": 0.01,
                "epochs": 100,
                "pct_start": 0.3,
                "anneal_strategy": "cos",
            },
        },
        "model_settings": {
            "channels_last": settings["model_settings"]["channels_last"],
            "amp_mixed_precision": settings["model_settings"]["amp_mixed_precision"],
            "is_asymmetric": True,         # WICHTIG: Flag aktivieren!
            "shared_stem": settings['model_settings']['shared_stem'],
            "use_original_paper": settings['model_settings']['use_original_paper'],
            "projection_head_type": "simple" # Einfache 1x1 Convolution zum Angleichen der Feature-Kanäle
        },
        "loss": {
            "params": {
                "epsilon": 1e-08, 
                "reduction": "sum", 
                "use_original_paper": settings['loss']['params']['use_original_paper']
            }
        }
    }

    output_dir = output_dir
    suffix = f"_{int(time.time())}" if append_suffix else ""

    for pair_name, models in asymmetric_pairs.items():
        for cls in mvtec_classes:

            config = copy.deepcopy(default_config)

            config["dataset"]["class"] = cls
            
            # Anstatt nur "model" zu setzen, werden hier "teacher_model" und "student_model" separat geschrieben
            config["teacher_model"] = models["teacher_model"]
            config["student_model"] = models["student_model"]

            teacher_name = models['teacher_model']['architecture']
            student_name = models['student_model']['architecture']
            
            # Ordnerstruktur anpassen: Ordnername ist jetzt "Teacher_vs_Student"
            mode_folder = f"{teacher_name}_vs_{student_name}"
            save_path = os.path.join(output_dir, mode_folder, cls)
            os.makedirs(save_path, exist_ok=True)

            config_filename = f"Asymmetric_STFPM_Config_{pair_name}_{cls}_{suffix}.yaml"
            with open(os.path.join(save_path, config_filename), "w") as f:
                yaml.dump(config, f, sort_keys=False,
                          indent=4, default_flow_style=False)

    print(
        f"✅ Alle asymmetrischen MVTecAD-Konfigurationen ({settings_label}) wurden im Ordner '{output_dir}' erstellt.")


def generate_asymmetric_gkd_configs(mode: str = "baseline", append_suffix: bool = False, output_dir: str = "Configs_GKD_Asymmetric"):
    """
    Generiert Konfigurationen für asymmetrische Lehrer-Studenten-Paare speziell für GKD.
    
    Args:
        mode: "gpu", "baseline", oder "cpu"
        append_suffix: Ob ein Zeitstempel als Suffix hinzugefügt werden soll
        output_dir: Das Verzeichnis, in dem die Konfigurationsdateien gespeichert werden
    """
    settings = get_settings_by_mode(mode)
    settings_label = mode

    # Die Struktur der Asymmetrischen Paare ist identisch zur MVTec-Konfiguration, 
    # aber auf GKD fokussiert (ohne Klassen-Iteration im Anschluss).
    asymmetric_pairs = {
        "resnet18_teacher_mobilenetv4_conv_small_050_student": {
            "student_model": {
                "architecture": "mobilenetv4_conv_small_050",
                "layer": ["blocks.0", "blocks.1", "blocks.2", "blocks.4"]
            },
            "teacher_model": {
                "architecture": "resnet10t",
                "layer": ["layer1", "layer2", "layer3", "layer4"]
            }
        },
        "resnet18_teacher_fasternet_t0_student": {
            "student_model": {
                "architecture": "fasternet_t0",
                "layer": ["stages.0", "stages.1", "stages.2", "stages.3"]
            },
            "teacher_model": {
                "architecture": "resnet10t",
                "layer": ["layer1", "layer2", "layer3", "layer4"]
            }
        },
        "resnet18_teacher_lcnet_050_student": {
            "student_model": {
                "architecture": "lcnet_050",
                "layer": ["blocks.1", "blocks.2", "blocks.4", "blocks.5"]
            },
            "teacher_model": {
                "architecture": "resnet10t",
                "layer": ["layer1", "layer2", "layer3", "layer4"]
            }
        },
        "resnet18_teacher_mobilenetv4_conv_small_student": {
            "student_model": {
                "architecture": "mobilenetv4_conv_small",
                "layer": ["blocks.0", "blocks.1", "blocks.2", "blocks.4"]
            },
            "teacher_model": {
                "architecture": "resnet10t",
                "layer": ["layer1", "layer2", "layer3", "layer4"]
            }
        },
        "resnet18_teacher_mobilenetv3_small_100_student": {
            "student_model": {
                "architecture": "mobilenetv3_small_100",
                "layer": ["blocks.0", "blocks.1", "blocks.3", "blocks.5"]
            },
            "teacher_model": {
                "architecture": "resnet10t",
                "layer": ["layer1", "layer2", "layer3", "layer4"]
            }
        }
    }

    # Template für asymmetrische GKD Modelle
    default_config = {
        "dataset": {
            "name": "GKD",
            "base_path": "GKD_Bilder",
            "img_size": 256,
            "enable_timing": False
        },
        "dataloader": {
            "batch_size": settings['dataloader']['batch_size'],
            "num_workers": 6,
            "pin_memory": settings["dataloader"]["pin_memory"],
            "persistent_workers": settings["dataloader"]["persistent_workers"],
            "prefetch_factor": settings["dataloader"]["prefetch_factor"]
        },
        "epochs": 100,
        "optimizer": {
            "active": "AdamW",
            "configs": {
                "SGD": {"lr": 0.4, "momentum": 0.9, "weight_decay": 0.0001},
                "AdamW": {"lr": 0.001, "weight_decay": 0.01},
            },
        },
        "training": {
            "cache_teacher_features": settings["training"]["cache_teacher_features"],
            "fast_zero_grad": settings["training"]["fast_zero_grad"],
            "async_host_to_device": settings["training"]["async_host_to_device"],
            "force_cpu_caching": settings["training"]["force_cpu_caching"],
            "cudnn_benchmark": settings["training"]["cudnn_benchmark"],
            "detailed_timing": False,
            "profiling": False,
        },
        "scheduler": {
            "type": "OneCycleLR",
            "params": {
                "max_lr": 0.001,
                "epochs": 100,
                "pct_start": 0.3,
                "anneal_strategy": "cos",
            },
        },
        "model_settings": {
            "channels_last": settings["model_settings"]["channels_last"],
            "amp_mixed_precision": settings["model_settings"]["amp_mixed_precision"],
            "is_asymmetric": True,         # WICHTIG: Flag aktivieren!
            "shared_stem": settings['model_settings']['shared_stem'],
            "use_original_paper": settings['model_settings']['use_original_paper'],
            "projection_head_type": "simple" # Einfache 1x1 Convolution zum Angleichen der Feature-Kanäle
        },
        "loss": {
            "params": {
                "epsilon": 1e-08, 
                "reduction": "sum", 
                "use_original_paper": settings['loss']['params']['use_original_paper']
            }
        }
    }

    output_dir = output_dir
    suffix = f"_{int(time.time())}" if append_suffix else ""

    # Generierungs-Schleife ohne innere Klassen-Schleife (da GKD nur einen generischen Datensatz hat)
    for pair_name, models in asymmetric_pairs.items():
       
        config = copy.deepcopy(default_config)

        config["teacher_model"] = models["teacher_model"]
        config["student_model"] = models["student_model"]

        teacher_name = models['teacher_model']['architecture']
        student_name = models['student_model']['architecture']
        mode_folder = f"{teacher_name}_vs_{student_name}"
        save_path = os.path.join(output_dir, mode_folder)
        os.makedirs(save_path, exist_ok=True)

        config_filename = f"Asymmetric_STFPM_Config_{pair_name}_{suffix}.yaml"
        with open(os.path.join(save_path, config_filename), "w") as f:
            yaml.dump(config, f, sort_keys=False,
                      indent=4, default_flow_style=False)

    print(
        f"✅ Alle asymmetrischen GKD-Konfigurationen ({settings_label}) wurden im Ordner '{output_dir}' erstellt.")