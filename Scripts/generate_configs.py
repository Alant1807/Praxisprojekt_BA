import os
import yaml


def generate_configs():
    """
    Generiert YAML-Konfigurationsdateien für verschiedene Modelle und alle MVTec-Klassen.
    """
    mvtec_classes = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    base_configs = {
        "resnet18": {
            "model": {
                "architecture": "resnet18",
                "layers": ["layer1", "layer2", "layer3", "layer4"],
            },
        },
        "mobilenetv3_large_100": {
            "model": {
                "architecture": "mobilenetv3_large_100",
                "layers": ["blocks.0", "blocks.1", "blocks.2", "blocks.4", "blocks.6"],
            },
        },
        "tf_efficientnet_lite0": {
            "model": {
                "architecture": "tf_efficientnet_lite0",
                "layers": ["blocks.1", "blocks.2", "blocks.4", "blocks.6"],
            },
        },
        "tf_efficientnet_lite1": {
            "model": {
                "architecture": "tf_efficientnet_lite1",
                "layers": ["blocks.1", "blocks.2", "blocks.4", "blocks.6"],
            },
        },
        "tf_efficientnet_lite2": {
            "model": {
                "architecture": "tf_efficientnet_lite2",
                "layers": ["blocks.1", "blocks.2", "blocks.4", "blocks.6"],
            },
        },
        "tf_efficientnet_lite3": {
            "model": {
                "architecture": "tf_efficientnet_lite3",
                "layers": ["blocks.1", "blocks.2", "blocks.4", "blocks.6"],
            },
        },
        "tf_efficientnet_lite4": {
            "model": {
                "architecture": "tf_efficientnet_lite4",
                "layers": ["blocks.1", "blocks.2", "blocks.4", "blocks.6"],
            },
        },
        "mobilenetv4_conv_large": {
            "model": {
                "architecture": "mobilenetv4_conv_large",
                "layers": ["blocks.0", "blocks.1", "blocks.2", "blocks.4"],
            },
        },
        "mobilenetv4_conv_medium": {
            "model": {
                "architecture": "mobilenetv4_conv_medium",
                "layers": ["blocks.0", "blocks.1", "blocks.2", "blocks.4"],
            },
        },
        "mobilenetv4_conv_small": {
            "model": {
                "architecture": "mobilenetv4_conv_small",
                "layers": ["blocks.0", "blocks.1", "blocks.2", "blocks.4"],
            },
        }
    }

    default_config = {
        "dataset": {"name": "MVTecAD", "base_path": "Images", "img_size": 256},
        "dataloader": {"batch_size": 64},
        "epochs": 200,
        "optimizer": {
            "active": "AdamW",
            "configs": {
                "SGD": {"lr": 0.4, "momentum": 0.9, "weight_decay": 0.0001},
                "AdamW": {"lr": 0.001, "weight_decay": 0.01},
            },
        },
        "device": {"type": "cpu"},
        "loss": {"params": {"epsilon": 1e-08}},
        "scheduler": {
            "type": "OneCycleLR",
            "params": {"max_lr": 0.01, "epochs": 200},
        },
        "model_settings": {"use_channels_last": True, "use_amp_mixed_precision": False},
    }

    output_dir = "Configs_timm"

    for model_name, model_config in base_configs.items():
        for cls in mvtec_classes:
            config = default_config.copy()
            config["dataset"]["class"] = cls
            config["model"] = model_config["model"]
            save_path = os.path.join(output_dir, model_name, cls)
            os.makedirs(save_path, exist_ok=True)
            config_filename = f"STFPM_Config_{model_name}_{cls}.yaml"
            with open(os.path.join(save_path, config_filename), "w") as f:
                yaml.dump(config, f, sort_keys=False)

    print(f"Alle Konfigurationen wurden im Ordner '{output_dir}' erstellt.")


def generate_ymlConfigs():
    """
    Generiert YAML-Konfigurationsdateien für verschiedene Modelle und alle MVTec-Klassen.
    """
    mvtec_classes = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    base_configs = {
        "resnet18": {
            "model": {
                "architecture": "resnet18",
                "layers": ["layer1", "layer2", "layer3", "layer4"],
            },
        },
        "mobilenet_v3_large": {
            "model": {
                "architecture": "mobilenet_v3_large",
                "layers": ['features.3', 'features.6', 'features.12']
            }
        },
        "shufflenet_v2_x1_0": {
            "model": {
                "architecture": "shufflenet_v2_x1_0",
                "layers": ["stage2", "stage3", "stage4"]
            }
        }
    }

    default_config = {
        "dataset": {"name": "MVTecAD", "base_path": "Images", "img_size": 256},
        "dataloader": {"batch_size": 64},
        "epochs": 200,
        "optimizer": {
            "active": "AdamW",
            "configs": {
                "SGD": {"lr": 0.4, "momentum": 0.9, "weight_decay": 0.0001},
                "AdamW": {"lr": 0.001, "weight_decay": 0.01},
            },
        },
        "device": {"type": "cpu"},
        "loss": {"params": {"epsilon": 1e-08}},
        "scheduler": {
            "type": "OneCycleLR",
            "params": {"max_lr": 0.01, "epochs": 200},
        },
        "model_settings": {"use_channels_last": True, "use_amp_mixed_precision": False},
    }

    output_dir = "Configs"

    for model_name, model_config in base_configs.items():
        for cls in mvtec_classes:
            config = default_config.copy()
            config["dataset"]["class"] = cls
            config["model"] = model_config["model"]
            save_path = os.path.join(output_dir, model_name, cls)
            os.makedirs(save_path, exist_ok=True)
            config_filename = f"STFPM_Config_{model_name}_{cls}.yaml"
            with open(os.path.join(save_path, config_filename), "w") as f:
                yaml.dump(config, f, sort_keys=False)

    print(f"Alle Konfigurationen wurden im Ordner '{output_dir}' erstellt.")


if __name__ == "__main__":
    generate_configs()
