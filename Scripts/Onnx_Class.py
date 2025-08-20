import os
from Scripts.model import *
import yaml


class ONNX_Exporter:
    """
    ONNX_Exporter ist eine Klasse, die ein PyTorch-Modell in das ONNX-Format exportiert.

    Args:
        best_model_path (str): Der Pfad zum besten gespeicherten Modell.
        config (dict): Konfigurationseinstellungen, die das Modellarchitektur und andere Parameter enthalten.
    """

    def __init__(self, best_model_path, config_path):
        self.config = self._load_config(config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        stfpm_model = STFPM(
            architecture=self.config['model']['architecture'],
            layers=self.config['model']['layers'],
        )
        stfpm_model.student_model.load_state_dict(torch.load(
            best_model_path, map_location=self.device))
        stfpm_model.eval()
        self.wrapped_model = ONNXWrapper(stfpm_model).to(self.device).eval()
        print(
            f"Model loaded from {best_model_path} and wrapped for ONNX export.")

    def _load_config(self, config_path):
        """
        Lädt die Konfiguration aus einer YAML-Datei.

        Args:
            config_path (str): Der Pfad zur YAML-Konfigurationsdatei.

        Returns:
            dict: Die geladene Konfiguration als Dictionary.
        """

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def export_onnx(self, output_path='onnx_models'):
        """
        Exportiert das PyTorch-Modell in das ONNX-Format und speichert es im angegebenen Verzeichnis.
        Args:
            output_path (str): Der Pfad, in dem das ONNX-Modell gespeichert werden soll.
        """

        img_size = self.config['dataset']['img_size']
        dummy_input = torch.randint(
            0, 256, (1, img_size, img_size, 3), dtype=torch.uint8, device=self.device)

        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(
            output_path, f'STFPM_{self.config["dataset"]["class"]}_{self.config["model"]["architecture"]}.onnx')

        onnx_program = torch.onnx.export(
            self.wrapped_model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        print(f"ONNX model exported to {save_path}")

class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """
        Forward-Methode für das ONNX-Modell.
        Diese Methode wird aufgerufen, wenn das Modell in ONNX exportiert wird.

        Args:
            x (torch.Tensor): Eingabebild, erwartet die Form (B, H,
        """

        preprocessed_x = self.preprocess_image(x)

        teacher_map, student_map = self.model(preprocessed_x)

        anomaly_map, anomaly_score = self.postprocessing(
            teacher_map, student_map, preprocessed_x)

        return anomaly_map, anomaly_score

    def preprocess_image(self, x):
        """
        Preprocessiert ein Eingabebild für das ONNX-Modell.

        Args:
            x (torch.Tensor): Eingabebild, erwartet die Form (B, H, W, C).

        Returns:
            torch.Tensor: Preprocessiertes Bild, bereit für die Inferenz.
        """

        x = x.to(torch.float32)
        x = x.permute(0, 3, 1, 2)
        x /= 255.0
        x = torch.nn.functional.interpolate(x,
                                            size=(256, 256),
                                            mode='bicubic',
                                            align_corners=False)
        x = (x - self.mean) / self.std

        return x

    def postprocessing(self, teacher_map, student_map, preprocessed_x):
        """
        Postprocessing der Anomalie-Karten, um die finale Anomalie-Karte und den Anomalie-Score zu berechnen.

        Args:
            teacher_map (list): Liste von Feature-Maps des Lehrermodells.
            student_map (list): Liste von Feature-Maps des Schülermodells.
            preprocessed_x (torch.Tensor): Preprocessiertes Eingabebild.

        Returns:
            tuple: Enthält die finale Anomalie-Karte und den Anomalie-Score.
        """

        batch_size, _, img_height, img_width = preprocessed_x.shape
        anomaly_map = torch.ones(
            (batch_size, img_height, img_width), device=preprocessed_x.device)

        for t_map, s_map in zip(teacher_map, student_map):
            t_map_norm = torch.nn.functional.normalize(t_map, dim=1)
            s_map_norm = torch.nn.functional.normalize(s_map, dim=1)

            am = 0.5 * torch.sum(torch.pow(t_map_norm - s_map_norm, 2), dim=1)
            am = torch.nn.functional.interpolate(
                am.unsqueeze(1),
                size=(img_height, img_width),
                mode='bilinear',
                align_corners=False
            )
            anomaly_map = torch.mul(anomaly_map, am.squeeze(1))

        anomaly_score = torch.amax(anomaly_map, dim=(1, 2))

        return anomaly_map, anomaly_score
