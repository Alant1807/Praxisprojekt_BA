import os
import yaml

def generate_asymmetric_configs():
    """
    Generiert YAML-Konfigurationsdateien für verschiedene Modelle und alle MVTec-Klassen.
    """
    mvtec_classes = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    base_configs = {
        "mobilenetv4large_teacher_mobilenetv4small_student": {
            "model": {
                "teacher_architecture": "mobilenetv4_conv_large",
                "student_architecture": "mobilenetv4_conv_small",
                "layers": ["blocks.0", "blocks.1", "blocks.2", "blocks.4"],
                "asymmetric": True
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

    output_dir = "Asymmetric_Configs"

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
    generate_asymmetric_configs()
