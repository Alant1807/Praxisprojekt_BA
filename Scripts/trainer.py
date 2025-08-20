import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
import time
import yaml
import json
import uuid

from Scripts.loss import *
from Scripts.results_manager import *
from Scripts.inference import *
from Scripts.plots import *
from tqdm import tqdm


class Trainer:
    """
    Trainer-Klasse für das Training des STFPM-Modells (Student-Teacher Feature Pyramid Matching).

    Args:
        model (STFPM): Das zu trainierende Modell.
        train_loader (DataLoader): DataLoader für die Trainingsdaten.
        test_loader (DataLoader): DataLoader für die Testdaten.
        config (dict): Konfigurationseinstellungen, die das Modellarchitektur, Optimizer, Scheduler und andere Parameter enthalten.
        train_folder_dir (str): Verzeichnis, in dem Trainingsordner erstellt werden sollen. Standard ist "Training_Runs".
    """

    def __init__(self, model, train_loader, test_loader, config, train_folder_dir="Training_Runs"):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if self.config.get('model_settings', {}).get('use_channels_last', False):
            self.actual_memory_format = torch.channels_last
        else:
            self.actual_memory_format = torch.preserve_format

        self.model = model.to(
            self.device, memory_format=self.actual_memory_format)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = Loss_function(**self.config['loss']['params'])

        optimizer_params = self.config['optimizer']['configs'][self.config['optimizer']['active']]
        optimizer_class = getattr(optim, self.config['optimizer']['active'])
        self.optimizer = optimizer_class(
            self.model.student_model.parameters(), **optimizer_params)

        self.scheduler = None
        if 'scheduler' in self.config and self.config['scheduler']:
            scheduler_name = self.config['scheduler']['type']
            if hasattr(lr_scheduler, scheduler_name):
                scheduler_class = getattr(lr_scheduler, scheduler_name)
                params = self.config['scheduler']['params']
                if scheduler_name == "OneCycleLR":
                    params['steps_per_epoch'] = len(self.train_loader)
            self.scheduler = scheduler_class(self.optimizer, **params)
        else:
            raise ValueError(
                f"Unbekannter Scheduler-Typ: {scheduler_name}")

        self.model.teacher_model.eval()
        self.train_folder_dir = train_folder_dir
        os.makedirs(self.train_folder_dir, exist_ok=True)

        self.training_id = str(uuid.uuid4())

        self.use_amp = False
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.evaluate = Inference(
            self.model, self.test_loader, self.config, trainings_id=self.training_id, inferenz=False)

        self.teacher_features_cache = dict()

    def _cache_teacher_features(self):
        """
        Berechnet die Feature-Maps des Teacher-Modells für den gesamten Trainingsdatensatz
        und speichert sie im Cache.
        """
        self.model.teacher_model.eval()
        self.model.stem_model.eval()
        print("Caching teacher features...")
        with torch.no_grad():
            for images, _, image_paths, _ in tqdm(self.train_loader, desc="Caching Features"):
                img_t = images.to(
                    self.device, memory_format=self.actual_memory_format)
                stem_output = self.model.stem_model(img_t)
                teacher_maps_dict = self.model.teacher_model(stem_output)
                teacher_maps = list(teacher_maps_dict.values())
                for i, path in enumerate(image_paths):
                    self.teacher_features_cache[path] = [
                        t_map[i].detach() for t_map in teacher_maps]

    def train_per_epoch(self, epoch, num_epochs):
        """
        Führt das Training für eine Epoche durch und berechnet den Verlust.
        Args:
            epoch (int): Die aktuelle Epoche.
            num_epochs (int): Die Gesamtanzahl der Epochen.
        Returns:
            float: Der durchschnittliche Verlust für die Epoche.
        """

        self.model.student_model.train()
        self.model.stem_model.eval()
        train_loss = 0.0
        for images, _, image_paths, _ in tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
            img_t = images.to(
                self.device, memory_format=self.actual_memory_format)
            self.optimizer.zero_grad(set_to_none=True)

            teacher_map_b = [list() for _ in range(
                len(self.teacher_features_cache[image_paths[0]]))]
            for path in image_paths:
                cached_maps = self.teacher_features_cache[path]
                for i, t_map in enumerate(cached_maps):
                    teacher_map_b[i].append(t_map)

            teacher_map = [torch.stack(maps) for maps in teacher_map_b]
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                with torch.no_grad():
                    stem_output = self.model.stem_model(img_t)
                student_features_dict = self.model.student_model(stem_output)
                student_map_list = list(student_features_dict.values())
                loss = self.criterion(teacher_map, student_map_list)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            train_loss += loss.item()
        return train_loss

    def train(self):
        """
        Startet das Training des Modells über die konfigurierte Anzahl von Epochen.
        Speichert die besten Gewichte und Metriken während des Trainings.
        """
        # Teacher-Features vor dem Training cachen
        self._cache_teacher_features()

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        print(f"Starte Training mit ID: {self.training_id}")
        if self.config.get('model', {}).get('asymmetric', False):
            print(
                f"Asymmetrisches Modell: Teacher-Modell:{self.config['model']['teacher_architecture']},  Student-Modell:{self.config['model']['student_architecture']}, Dataset-Klasse: {self.config['dataset']['class']}, Epochen: {self.config['epochs']}")
        else:
            print(
                f"Modell: {self.config['model']['architecture']},  Dataset-Klasse: {self.config['dataset']['class']}, Epochen: {self.config['epochs']}")

        metrics = {
            "train_loss": [],
            "epoch_durations": [],
            "best_epoch_loss": float('inf'),
            "best_epoch": -1,
            "auroc_scores": [],
            "best_auc_score": 0.0,
            "inference_times": [],
            "total_training_time": 0,
            "avg_epoch_duration": 0
        }

        best_model_student_weights = None

        total_training_time = time.time()
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start_time = time.time()
            train_loss = self.train_per_epoch(epoch, self.config['epochs'])
            metrics["train_loss"].append(train_loss)
            epoch_duration = time.time() - epoch_start_time
            metrics["epoch_durations"].append(epoch_duration)
            print(
                f"Epoch {epoch}/{self.config['epochs']} abgeschlossen. Verlust: {train_loss}, Dauer: {epoch_duration:.2f} Sekunden")

            if train_loss < metrics["best_epoch_loss"]:
                metrics["best_epoch_loss"] = train_loss
                metrics["best_epoch"] = epoch
                best_model_student_weights = self.model.student_model.state_dict()
                print(
                    f"Neuer bester Verlust: {metrics['best_epoch_loss']} bei Epoche {metrics['best_epoch']}. Gewichte zwischengespeichert.")

            device_type = "GPU" if torch.cuda.is_available() else "CPU"
            auc_score, inference_time = self.evaluate.evaluate_per_epoch()
            metrics.update({
                "auroc_scores": metrics.get("auroc_scores", []) + [auc_score],
                "inference_times": metrics.get('inference_times', []) + [inference_time]
            })
            print(
                f"ROC-AUC: {auc_score:.4f}, {device_type} Inference Time: {inference_time:.4f} seconds\n")

            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
                print(
                    f"Scheduler Schritt nach Epoche {epoch}. Aktuelle Lernrate: {self.optimizer.param_groups[0]['lr']:.6f}")

        metrics.update({
            "total_training_time": time.time() - total_training_time,
            "avg_epoch_duration": sum(metrics["epoch_durations"]) / len(metrics["epoch_durations"])
        })

        print(
            f"\nTraining abgeschlossen. Gesamtdauer: {metrics['total_training_time']:.2f} Sekunden, Durchschnittliche Epochendauer: {metrics['avg_epoch_duration']:.2f} Sekunden")

        run_base_path, weights_save_dir_for_run, logs_save_dir_for_run, plots_save_dir_for_run = self.create_dir_for_run(
            self.training_id)

        base_filename = f"{self.config['model']['architecture']}_{self.config['dataset']['class']}_{self.training_id}"

        final_weights_path = os.path.join(
            weights_save_dir_for_run, f"{base_filename}_final_student.pth")
        torch.save(self.model.student_model.state_dict(), final_weights_path)
        print(f"\nEndgültige Gewichte gespeichert unter: {final_weights_path}")

        if best_model_student_weights:
            best_weight_path = os.path.join(
                weights_save_dir_for_run, f"{base_filename}_best_student.pth")
            torch.save(best_model_student_weights, best_weight_path)
            print(f"Beste Gewichte gespeichert unter: {best_weight_path}")
        else:
            print("Keine besseren Gewichte gefunden während des Trainings.")

        self.create_log_csv(self.training_id, timestamp,
                            metrics, logs_save_dir_for_run, base_filename)
        config_save_path = self.create_yaml_config(run_base_path)
        self.create_summary_metric(self.training_id, timestamp, metrics,
                                   config_save_path, run_base_path, final_weights_path, best_weight_path)
        self.create_plots_for_run(
            metrics, plots_save_dir_for_run, base_filename)

    def create_dir_for_run(self, training_id):
        """
        Erstellt ein Verzeichnis für den aktuellen Trainingslauf.

        Args:
            training_id (str): Eindeutige ID für den Trainingslauf.

        Returns:
            tuple: Ein Tupel mit den Pfaden für das Basisverzeichnis, das Verzeichnis für Gewichte, Logs und Plots.
        """

        run_base_path = os.path.join(
            self.train_folder_dir,
            f"{self.config['dataset']['name']}_{self.config['dataset']['class']}",
            self.config['model']['architecture'],
            training_id
        )

        weights_save_dir_for_run = os.path.join(run_base_path, "weights")
        logs_save_dir_for_run = os.path.join(run_base_path, "logs")
        plots_save_dir_for_run = os.path.join(run_base_path, "plots")

        os.makedirs(weights_save_dir_for_run, exist_ok=True)
        os.makedirs(logs_save_dir_for_run, exist_ok=True)
        os.makedirs(plots_save_dir_for_run, exist_ok=True)

        return run_base_path, weights_save_dir_for_run, logs_save_dir_for_run, plots_save_dir_for_run

    def create_log_csv(self, training_id, timestamp, metrics, logs_save_dir_for_run, base_filename):
        """
        Erstellt eine CSV-Datei mit den Trainingsmetriken.

        Args:
            training_id (str): Eindeutige ID für den Trainingslauf.
            timestamp (str): Zeitstempel des Trainingsstarts.
            metrics (dict): Metriken, die während des Trainings gesammelt wurden.
        """

        result_data = {
            "training_id": training_id,
            "timestamp": timestamp,
            "model": self.config['model']['architecture'],
            "used_memory_format": self.actual_memory_format,
            "used_amp_mices_precision": self.config['model_settings']['use_amp_mixed_precision'],
            "dataset_class": self.config['dataset']['class'],
            "used_layer": self.config['model']['layers'],
            "img_size": self.config['dataset']['img_size'],
            "batch_size": self.config['dataloader']['batch_size'],
            "optimizer_type": self.config['optimizer']['active'],
            "learning_rate": self.config['optimizer']['configs'][self.config['optimizer']['active']].get('lr', 'N/A'),
            "scheduler_type": self.config['scheduler']['type'] if 'scheduler' in self.config else None,
            "num_epochs": self.config['epochs'],
            "total_training_time": metrics["total_training_time"],
            "avg_epoch_duration": metrics["avg_epoch_duration"],
            "final_train_loss": metrics["train_loss"][-1],
            "best_train_loss": metrics["best_epoch_loss"],
            "best_epoch": metrics["best_epoch"],
            "auroc_score": metrics["auroc_scores"][-1],
            "best_auc_score": max(metrics["auroc_scores"]),
            "inference_times": sum(metrics["inference_times"]) / len(metrics["inference_times"])
        }

        create_result_df(
            result_data,
            target_filename=os.path.join(
                logs_save_dir_for_run, f"{base_filename}_results.csv")
        )

    def create_yaml_config(self, run_base_path):
        """
        Speichert die Konfiguration in einer YAML-Datei.

        Args:
            run_base_path (str): Basisverzeichnis für den aktuellen Trainingslauf.

        Returns:
            str: Pfad zur gespeicherten YAML-Konfigurationsdatei.
        """

        config_safe_path = os.path.join(
            run_base_path, f"STFPM_Config_{self.config['model']['architecture']}.yaml")
        with open(config_safe_path, 'w') as config_file:
            yaml.dump(self.config, config_file, sort_keys=False, indent=4)
        print(f"Konfiguration gespeichert unter: {config_safe_path}")
        return config_safe_path

    def create_summary_metric(self, training_id, timestamp, metrics, config_save_path, run_base_path, final_weights_path, best_weight_path):
        """
        Erstellt eine Zusammenfassung der Metriken des Trainingslaufs.

        Args:
            training_id (str): Eindeutige ID für den Trainingslauf.
            timestamp (str): Zeitstempel des Trainingsstarts.
            metrics (dict): Metriken, die während des Trainings gesammelt wurden.
            run_base_path (str): Basisverzeichnis für den aktuellen Trainingslauf.
            final_weights_path (str): Pfad zu den endgültigen Gewichten.
            best_weight_path (str): Pfad zu den besten Gewichten.
        """

        summary_metrics = {
            "training_id": training_id,
            "timestamp": timestamp,
            "dataset_name": f"{self.config['dataset']['name']}_{self.config['dataset']['class']}",
            "model_architecture": self.config['model']['architecture'],
            "used_memory_format": str(self.actual_memory_format),
            "used_amp_mixed_precision": self.config['model_settings']['use_amp_mixed_precision'],
            "config_used_path": config_save_path,
            "key_parameters": {
                "num_epochs": self.config['epochs'],
                "lernrate": self.config['optimizer']['configs'][self.config['optimizer']['active']].get('lr', 'N/A'),
                "optimizer": self.config['optimizer']['active'],
                "scheduler": self.config['scheduler']['type'] if 'scheduler' in self.config else 'N/A'
            },
            "training_summary": {
                "training_duration": metrics["total_training_time"],
                "final_train_loss": metrics["train_loss"][-1],
                "best_train_loss": metrics["best_epoch_loss"],
                "best_epoch": metrics["best_epoch"],
                "avg_epoch_duration": metrics["avg_epoch_duration"]
            },
            "evaluation_performance": {
                "auroc_score": metrics["auroc_scores"][-1],
                "best_auroc_score": max(metrics["auroc_scores"]),
                "inference_time": sum(metrics["inference_times"]) / len(metrics["inference_times"])
            },
            "Pfad_der_Gewichte": {
                "Pfad_finale_Gewichte": final_weights_path,
                "Pfad_bester_Gewichte": best_weight_path
            }
        }

        summary_json_path = os.path.join(run_base_path, "summary_metrics.json")
        with open(summary_json_path, 'w') as f:
            json.dump(summary_metrics, f, indent=4)
        print(
            f"\nZusammenfassungsmetriken gespeichert unter: {summary_json_path}")

    def create_plots_for_run(self, metrics, plots_save_dir_for_run, base_filename):
        """
        Erstellt und speichert Plots für die Trainingsmetriken.

        Args:
            metrics (dict): Metriken, die während des Trainings gesammelt wurden.
            plots_save_dir_for_run (str): Verzeichnis, in dem die Plots gespeichert werden sollen.
            base_filename (str): Basisname für die Plot-Dateien.
        """

        plot_loss_curves(metrics, self.config,
                         plots_save_dir_for_run, base_filename)
        plot_auroc_scores(metrics, self.config,
                          plots_save_dir_for_run, base_filename)
