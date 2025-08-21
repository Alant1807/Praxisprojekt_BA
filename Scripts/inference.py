import torch
import torch.quantization
import time
import numpy as np

from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from Scripts.loss import *
from Scripts.results_manager import *
from Scripts.plots import *


class Inference:
    """
    Führt die Inferenz für das STFPM-Modell durch und berechnet die Anomalie-Karte.

    Args:
        model (STFPM): Das STFPM-Modell, das die Anomalie-Karte berechnet.
        test_loader (DataLoader): Der DataLoader für den Testdatensatz.
        config (dict): Konfigurationsparameter, die das Gerät und andere Einstellungen enthalten.
        output_dir (str): Der Ordner, in dem die Ergebnisse gespeichert werden sollen.
        path_to_student_weight (str, optional): Pfad zu den Gewichten des Schüler-Modells. 
                                                Wenn None, werden die Standardgewichte verwendet.
        trainings_id (str): Eindeutige ID für den Trainingslauf.
        inferenz (bool): Gibt an, ob Inferenz durchgeführt werden soll. Standard ist True.
    """

    def __init__(self, model, test_loader, config, device, output_dir="Inference_Runs", path_to_student_weight=None, trainings_id=None, inferenz=True):
        self.config = config
        self.device = device

        if self.config.get('model_settings', {}).get('use_channels_last', False):
            self.actual_memory_format = torch.channels_last
        else:
            self.actual_memory_format = torch.preserve_format

        self.model = model.to(
            self.device, memory_format=self.actual_memory_format)

        self.test_loader = test_loader
        self.trainings_id = trainings_id

        if path_to_student_weight is not None:
            if not os.path.exists(path_to_student_weight):
                raise FileNotFoundError(
                    f"Path to student weights does not exist: {path_to_student_weight}")
            student_weights = torch.load(
                path_to_student_weight, map_location=self.device) if path_to_student_weight else None
            self.model.student_model.load_state_dict(student_weights)

        if inferenz:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

            self.run_base_path = os.path.join(
                self.output_dir,
                f"{self.config['dataset']['name']}_{self.config['dataset']['class']}",
                self.config['model']['architecture'],
                self.trainings_id
            )

            self.anomaly_maps_dir_for_run, self.plots_save_dir_for_run = self.create_dir_for_run(
                self.run_base_path)

    def evaluate_per_epoch(self):
        """
        Führt die Inferenz auf dem Testdatensatz durch und berechnet die AUC-ROC-Score pro Epoche.
        Gibt den AUC-ROC-Score und die Gesamt-Inferenzzeit zurück.

        Returns:
            tuple: AUC-ROC-Score und die Gesamt-Inferenzzeit in Sekunden.
        """

        self.model.eval()

        get_labels = list()
        get_anomaly_scores = list()
        total_inference_time = 0.0
        with torch.no_grad():
            for images, _, _, labels in tqdm(self.test_loader, desc="Evaluating", leave=False):
                img_t = images.to(
                    self.device, memory_format=self.actual_memory_format)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                # with torch.autocast(device_type=self.device.type):
                anomaly_map = self.model.anomaly_map(img_t)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                total_inference_time += (end_time - start_time)

                get_labels.extend(labels.cpu().numpy().tolist())
                get_anomaly_scores.extend(torch.amax(
                    anomaly_map, dim=(1, 2)).detach().cpu().numpy())

        return roc_auc_score(get_labels, get_anomaly_scores), total_inference_time

    def evaluate_loaded_model(self):
        """
        Führt die Inferenz auf dem Testdatensatz durch und berechnet den AUC-ROC-Score für ein geladenes Modell.
        Speichert die Anomalie-Karten

        Returns:
            tuple: AUC-ROC-Score und die Gesamt-Inferenzzeit in Sekunden.
        """

        self.model.eval()

        get_labels = list()
        get_anomaly_scores = list()

        total_inference_time = 0.0
        with torch.no_grad():
            for images, names, _, labels in tqdm(self.test_loader, desc="Evaluating", leave=False):
                img_t = images.to(
                    self.device, memory_format=self.actual_memory_format)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                # with torch.autocast(device_type=self.device.type):
                anomaly_map = self.model.anomaly_map(img_t)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                total_inference_time += (end_time - start_time)

                get_labels.extend(labels.cpu().numpy().tolist())
                get_anomaly_scores.extend(torch.amax(
                    anomaly_map, dim=(1, 2)).detach().cpu().numpy())

                self.save_anomaly_maps(
                    anomaly_map, names, self.anomaly_maps_dir_for_run)

        return roc_auc_score(get_labels, get_anomaly_scores), total_inference_time

    def create_dir_for_run(self, run_base_path):
        """
        Erstellt die Verzeichnisse für die Anomalie-Karten und die Plots für den aktuellen Lauf.

        Returns:
            tuple: Pfade zu den Verzeichnissen für Anomalie-Karten und Plots.
        """

        anomaly_maps_dir_for_run = os.path.join(run_base_path, "anomaly_maps")
        plots_save_dir_for_run = os.path.join(run_base_path, "plots")

        os.makedirs(anomaly_maps_dir_for_run, exist_ok=True)
        os.makedirs(plots_save_dir_for_run, exist_ok=True)

        return anomaly_maps_dir_for_run, plots_save_dir_for_run

    def save_anomaly_maps(self, anomaly_maps, names, anomaly_maps_dir):
        numpy_anomaly_maps = anomaly_maps.detach().cpu().numpy()
        for i in range(numpy_anomaly_maps.shape[0]):
            map_to_save = numpy_anomaly_maps[i]
            image_name = names[i]
            np.save(os.path.join(anomaly_maps_dir,
                    f"{os.path.splitext(image_name)[0]}.npy"), map_to_save)

    def generate_heatmaps_from_saved_maps(self):
        """
        Erstellt Heatmaps für den aktuellen Inferenz-Lauf.
        Dazu werden die zuvor berechneten und als .npy-Dateien gespeicherten
        Anomalie-Karten aus dem 'anomaly_maps'-Verzeichnis geladen.
        """

        print(
            f"\nStarte Heatmap-Generierung aus gespeicherten Karten für: {self.run_base_path}")

        # Prüfen, ob das Verzeichnis mit den Anomalie-Karten existiert
        if not os.path.isdir(self.anomaly_maps_dir_for_run):
            print(
                f"Fehler: Anomalie-Karten-Verzeichnis nicht gefunden unter {self.anomaly_maps_dir_for_run}")
            return

        # Wir benötigen die Originalbilder. Der test_loader ist in der Klasse verfügbar (self.test_loader).
        # Daraus greifen wir auf das Dataset zu, um die Bildpfade zu erhalten.
        test_set = self.test_loader.dataset
        img_size = test_set.img_size

        # Erstelle ein Mapping von Dateinamen (ohne Endung) zu den skalierten Originalbildern
        image_mapping = {}
        labels_mapping = {}
        for sample in test_set.samples:
            base_name = os.path.splitext(sample['name'])[0]
            # Lade das Originalbild und bringe es auf die korrekte Größe
            pil_img = Image.open(sample['path']).convert(
                "RGB").resize((img_size, img_size))
            image_mapping[base_name] = np.array(pil_img)
            labels_mapping[base_name] = sample['label']

        # Lade alle .npy-Dateien aus dem Anomalie-Verzeichnis
        npy_files = [f for f in os.listdir(
            self.anomaly_maps_dir_for_run) if f.endswith('.npy')]
        if not npy_files:
            print(
                f"Keine .npy Anomalie-Karten in '{self.anomaly_maps_dir_for_run}' gefunden.")
            return

        # Iteriere durch die .npy-Dateien und erstelle für jede eine Heatmap
        for npy_file in tqdm(npy_files, desc="Erstelle Heatmaps"):
            base_name = os.path.splitext(npy_file)[0]

            if labels_mapping.get(base_name) != 1:
                continue

            if base_name in image_mapping:
                # Lade die Anomalie-Karte und das zugehörige Originalbild
                anomaly_map_data = np.load(os.path.join(
                    self.anomaly_maps_dir_for_run, npy_file))
                original_image_np = image_mapping[base_name]

                # Definiere den Speicherpfad und rufe die Plot-Funktion auf
                heatmap_save_path = os.path.join(
                    self.plots_save_dir_for_run, f"{base_name}_heatmap.png")
                plot_heatmap(original_image_np, anomaly_map_data,
                             heatmap_save_path, title=base_name)
            else:
                print(
                    f"Warnung: Kein Originalbild für die Karte '{npy_file}' gefunden.")

        print(
            f"Heatmap-Generierung abgeschlossen. Ergebnisse in '{self.plots_save_dir_for_run}' gespeichert.")

    def create_inference_summary(self, json_file, auroc_score, inference_time):
        """
        Erstellt eine Zusammenfassung der Inferenz und speichert sie in einer JSON-Datei.

        Args:
            json_file (dict): JSON-Daten, die Informationen über das Dataset und das Modell enthalten.
            auroc_score (float): Der AUC-ROC-Score der Inferenz.
            inference_time (float): Die Gesamt-Inferenzzeit in Sekunden.
        """

        inference_summary = {
            'dataset_name': json_file.get('dataset_name'),
            "model_used": {
                'training_id': json_file.get('training_id'),
                'model_architecture': json_file.get('model_architecture'),
                'student_model_weights_path': json_file.get('Pfad_der_Gewichte', {}).get('Pfad_bester_Gewichte')
            },
            'performance_metrics': {
                'auroc_score': auroc_score,
                'inference_time': inference_time,
            }
        }

        inference_summary_path = os.path.join(
            self.run_base_path,
            "inference_summary.json"
        )

        with open(inference_summary_path, 'w') as f:
            json.dump(inference_summary, f, indent=4)
        print(f"Inference summary saved to {inference_summary_path}")
