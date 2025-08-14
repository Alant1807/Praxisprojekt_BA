import torch
from torch import nn
import torch.quantization
import yaml
import os
import json
import copy

from Scripts.model import *
from torch.utils.data import DataLoader
from Scripts.dataset import *


class QuantizableWrapper(nn.Module):
    def __init__(self, model: STFPM):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.stem = model.stem_model

        self.teacher_model = model.teacher_model
        self.student_model = model.student_model

    def forward(self, x):
        stem_out = self.stem(x)
        with torch.no_grad():
            teacher_feature_maps = self.teacher_model(stem_out)

        student_input_quant = self.quant(stem_out)
        student_features_quant = self.student_model(student_input_quant)
        student_feature_maps = [self.dequant(f) for f in student_features_quant]

        if isinstance(teacher_feature_maps, dict):
            teacher_feature_maps = list(teacher_feature_maps.values())

        return teacher_feature_maps, student_feature_maps

    def anomaly_map(self, x):
        teacher_feature_maps, student_feature_maps = self.forward(x)

        batch_size = x.shape[0]
        img_height, img_width = x.shape[-2], x.shape[-1]
        anomaly_map_result = torch.ones(
            (batch_size, img_height, img_width), device=x.device
        )

        if len(teacher_feature_maps) != len(student_feature_maps):
            raise ValueError(f"Teacher and Student feature map counts do not match: "
                             f"{len(teacher_feature_maps)} vs {len(student_feature_maps)}")

        for t_map, s_map in zip(teacher_feature_maps, student_feature_maps):
            t_map_norm = torch.nn.functional.normalize(t_map, dim=1)
            s_map_norm = torch.nn.functional.normalize(s_map, dim=1)
            am = 0.5 * torch.sum(torch.pow(t_map_norm - s_map_norm, 2), dim=1)
            am = nn.Upsample(size=(img_height, img_width), mode="bilinear", align_corners=False)(
                am.unsqueeze(1)
            )
            am = am.squeeze(1)
            anomaly_map_result = torch.mul(anomaly_map_result, am)

        return anomaly_map_result


def fuse_resnet_student_model(model: QuantizableWrapper):
    feature_extractor = model.student_model.feature_extractor

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(feature_extractor, layer_name, None)
        if layer is None:
            print(
                f"[WARN] Layer '{layer_name}' nicht gefunden im student_model.")
            continue

        for i, block in enumerate(layer):
            if hasattr(block, 'conv1') and hasattr(block, 'bn1') and hasattr(block, 'act1'):
                torch.quantization.fuse_modules(
                    block,
                    ["conv1", "bn1", "act1"],
                    inplace=True
                )
            if hasattr(block, 'conv2') and hasattr(block, 'bn2') and hasattr(block, 'act2'):
                torch.quantization.fuse_modules(
                    block,
                    ["conv2", "bn2", "act2"],
                    inplace=True
                )


def quantize_model(best_student_weight_path, config, summary_metric):
    cpu_device = torch.device("cpu")
    
    original_device = torch.device(config['device']['type'])

    # 2. Create the base model.
    model = STFPM(
        architecture=config['model']['architecture'],
        layers=config['model']['layers']
    )

    model.student_model.load_state_dict(torch.load(
        best_student_weight_path, map_location=cpu_device
    ))

    model.to(cpu_device).eval()

    quantizable_model = QuantizableWrapper(model).to(cpu_device).eval()

    fuse_resnet_student_model(quantizable_model)
    quantizable_model.student_model.qconfig = torch.quantization.get_default_qconfig(
        'fbgemm')

    torch.quantization.prepare(quantizable_model.student_model, inplace=True)

    calibrated_model = calibrate_model(quantizable_model, config, cpu_device)

    model_quantized = torch.quantization.convert(
        calibrated_model.student_model, inplace=True)

    save_quantized_model(model_quantized, config, summary_metric)


def calibrate_model(model, config, device):
    model.eval()
    train_set = MVTecDataset(
        img_size=config['dataset']['img_size'],
        base_path=config['dataset']['base_path'],
        cls=config['dataset']['class'],
        mode='train',
        subfolders=['good']
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config['dataloader']['batch_size'],
        shuffle=True
    )
    with torch.no_grad():
        for i, (images, _, _, _) in enumerate(train_loader):
            if i >= 10:
                print("Calibration complete.")
                break
            images = images.to(device)
            model(images)
    return model


def save_quantized_model(model, config, summary_metric):
    save_path = os.path.join(
        'quantized_models',
        f"{config['model']['architecture']}",
        f"{summary_metric['training_id']}"
    )
    os.makedirs(save_path, exist_ok=True)
    yaml_path = os.path.join(
        save_path, f"STFPM_Config_{config['model']['architecture']}.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)

    summary_metric_path = os.path.join(save_path, 'summary_metric.json')
    with open(summary_metric_path, 'w') as f:
        json.dump(summary_metric, f)

    torch.save(model.state_dict(), os.path.join(
        save_path, 'quantized_model.pth'))
