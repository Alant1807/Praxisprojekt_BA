import torch
import torch.optim as optim
from torch.amp import GradScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from tqdm import tqdm
import os
import gc
import datetime
import csv
import torch.nn as nn

from Scripts.dataset_mvtec import MVTecDataset, MVTecDatasetCached
from Scripts.dataset_gkd import GKDDataset, GKDDatasetCached
from torch.utils.data import DataLoader, random_split
from Scripts.utils import (
    setup_gpu_optimizations,
    setup_cpu_optimizations,
    get_teacher_features_cache,
    load_config,
    cache_teacher_features,
    get_optimal_batch_size,
    get_optimal_num_workers
)
from Scripts.stfpm_arch import *
from Scripts.losses import Loss_function


class LRRangeTestFinder:
    """
    Findet die optimale Lernrate, indem sie schrittweise erhöht und der Loss gemessen wird.
    Hilft dabei, die perfekte 'base_lr' (Start) und 'max_lr' (Maximum) für das Training zu finden.
    """

    def __init__(self, config_path: str, use_cached_tensors: bool):
        """Initialisiert den LR-Range-Test.

        Args:
            config_path (str): Pfad zur YAML-Konfigurationsdatei.
            use_cached_tensors (bool): Ob vorverarbeitete Tensoren (.pt) genutzt werden sollen.
        """
        self.config = load_config(config_path)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.is_gpu = self.device.type == "cuda"
        self.is_cpu = self.device.type == "cpu"

        # Hardware-Optimierungen aktivieren
        if self.is_gpu:
            self.hw_config = setup_gpu_optimizations(self.device, self.config)
            self.cudnn_benchmark = self.hw_config["cudnn_benchmark"]
        else:
            self.hw_config = setup_cpu_optimizations()
            self.cudnn_benchmark = False

        config_amp = self.config.get("model_settings", {}).get(
            "amp_mixed_precision", False
        )
        self.use_amp = config_amp and self.is_gpu
     
        if config_amp and self.is_cpu:
            print("AMP auf CPU deaktiviert.")

        if self.config.get("model_settings", {}).get("channels_last", True):
            self.actual_memory_format = torch.channels_last
        else:
            self.actual_memory_format = torch.contiguous_format

        self.non_blocking = self.config.get("training", {}).get(
            "async_host_to_device", True
        )

        self.use_cached_tensors = use_cached_tensors

        self.is_asymmetric = self.config.get("model_settings", {}).get(
            "is_asymmetric", False
        )
        self.model = self._init_model()
        self.train_loader = self._init_dataloader()

        # Teacher-Features cachen, um den Forward-Pass während des Tests zu beschleunigen
        if self.config.get("training", {}).get("cache_teacher_features", True):
            print("Caching der Teacher-Features aktiviert.")
            config_cache_on_cpu = self.config.get("training", {}).get(
                "force_cpu_caching", False
            )
            self.cache_on_cpu = config_cache_on_cpu and self.is_gpu

            self.teacher_features_cache = cache_teacher_features(
                self.model,
                self.train_loader,
                self.device,
                self.actual_memory_format,
                self.use_amp,
                self.non_blocking,
                cache_on_cpu=self.cache_on_cpu,
            )
        else:
            self.teacher_features_cache = None
            self.cache_on_cpu = False

        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        self.model_name = self._get_model_architecture_name()
        self.save_dir = f"lr_finder_plots_{self.config['dataset']['name']}"
        self.model_save_dir = os.path.join(self.save_dir, self.model_name)
        os.makedirs(self.model_save_dir, exist_ok=True)

        print(
            f"Initialization complete for {self._get_model_architecture_name()}. GPU: {self.is_gpu}"
        )

    def _init_model(self):
        """Erstellt das STFPM-Modell anhand der Konfiguration.

        Returns:
            nn.Module: Das initialisierte Modell.
        """
        model_args = {"is_asymmetric": self.is_asymmetric}

        if self.is_asymmetric:
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
                    "extract_stem": self.config["model_settings"]["shared_stem"],
                    "partial_share_depth": self.config["model_settings"].get("partial_share_depth", 0),
                }
            )

        print("Using feature-based distillation mode (STFPM).")
        model = STFPM(**model_args).to(self.device, memory_format=self.actual_memory_format)
        return model

    def _init_dataloader(self) -> DataLoader:
        """Lädt den Datensatz und erstellt den PyTorch-DataLoader.

        Returns:
            DataLoader: Der vorbereitete DataLoader für das Training.
        """
        dataset_name = self.config["dataset"]["name"]

        if dataset_name == "MVTecAD":
            if not self.use_cached_tensors:
                train_set = MVTecDataset(
                    img_size=self.config["dataset"]["img_size"],
                    base_path=self.config["dataset"]["base_path"],
                    cls=self.config["dataset"]["class"],
                    mode="train",
                )
            else:
                train_set = MVTecDatasetCached(
                    base_path=self.config["dataset"]["base_path"],
                    cls=self.config["dataset"]["class"],
                    mode="train",
                )
        elif dataset_name == "GKD":
            if not self.use_cached_tensors:
                full_train_set = GKDDataset(
                    img_size=self.config["dataset"]["img_size"],
                    data_path=self.config["dataset"]["base_path"],
                    mode="train",
                )
            else:
                full_train_set = GKDDatasetCached(
                    data_path=self.config["dataset"]["base_path"], mode="train"
                )
        else:
            raise ValueError(f"Unbekannter Datensatz: {dataset_name}")
        
        if dataset_name == "GKD":
            val_size = int(0.2 * len(full_train_set))
            train_size = len(full_train_set) - val_size
            training_set, _ = random_split(
                full_train_set,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            training_set = train_set

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_gpu = device.type == "cuda"

        config_batch_size = self.config.get("dataloader", {}).get("batch_size")
        if config_batch_size is None or config_batch_size == "auto":
            img_size = self.config.get("dataset", {}).get("img_size", 256)
            model_name = self.config.get("model", {}).get("architecture", "resnet18")
            batch_size = get_optimal_batch_size(device, img_size, model_name)
            print(f"Auto Batch-Size: {batch_size}")
        else:
            batch_size = config_batch_size

        config_num_workers = self.config.get("dataloader", {}).get("num_workers")
        if config_num_workers is None or config_num_workers == "auto":
            num_workers = get_optimal_num_workers(device)
            print(f"Auto Workers: {num_workers}")
        else:
            num_workers = min(config_num_workers, os.cpu_count())

        pin_memory = self.config.get("dataloader", {}).get("pin_memory", True) and is_gpu
        prefetch_factor = 2 if num_workers > 0 else None

        return DataLoader(
            training_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor,
            drop_last=False,
        )

    def _get_model_architecture_name(self) -> str:
        """Ermittelt den Modellnamen als String für Datei-Outputs.

        Returns:
            str: Der Modellname.
        """
        if self.is_asymmetric:
            teacher = self.config["teacher_model"]["architecture"]
            student = self.config["student_model"]["architecture"]
            return f"Teacher_{teacher}_Student_{student}"
        else:
            return self.config["model"]["architecture"]

    def run_test(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        num_epochs: int = 5
    ):
        """Führt den eigentlichen Lerngeschwindigkeits-Test aus.

        Args:
            start_lr (float, optional): Niedrige Start-Lernrate. Standard ist 1e-7.
            end_lr (float, optional): Hohe End-Lernrate. Standard ist 1.0.
            num_epochs (int, optional): Dauer des Tests in Epochen. Standard ist 5.

        Returns:
            tuple: (Liste der Lernraten, Liste der entsprechenden Losses)
        """
        print(f"\n{'='*60}")
        print(f"LR RANGE TEST - Nach Leslie Smith's Paper")
        print(f"{'='*60}")
        print(f"  Start LR:    {start_lr:.2e}")
        print(f"  End LR:      {end_lr:.2e}")
        print(f"  Epochen:     {num_epochs}")
        print(f"{'='*60}\n")

        self.model.student_model.train()

        stem_is_active = (
            self.model.stem_model is not None
            and not isinstance(self.model.stem_model, nn.Identity)
            and sum(p.numel() for p in self.model.stem_model.parameters()) > 0
        )
        if stem_is_active:
            self.model.stem_model.eval()

        criterion = Loss_function(**self.config["loss"]["params"])

        optimizer_config = self.config["optimizer"]
        optimizer_params = optimizer_config["configs"][optimizer_config["active"]].copy()
        if "lr" in optimizer_params:
            del optimizer_params["lr"]

        trainable_params = self.model.get_trainable_parameters()
        optimizer = getattr(optim, optimizer_config["active"])(
            trainable_params, lr=start_lr, **optimizer_params
        )

        total_steps = num_epochs * len(self.train_loader)
        # Berechnet den Wachstumsfaktor, um im letzten Schritt genau end_lr zu erreichen
        gamma = (end_lr / start_lr) ** (1 / total_steps)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        lrs_recorded = []
        losses_recorded = []
        best_loss = float("inf")

        for epoch in range(num_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for images, _, image_paths, _ in pbar:
                img_t = images.to(
                    self.device,
                    memory_format=self.actual_memory_format,
                    non_blocking=self.non_blocking,
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                    cache_enabled=True,
                ):
                    if self.teacher_features_cache is not None:
                        teacher_map_cached = get_teacher_features_cache(
                            self.teacher_features_cache,
                            image_paths,
                            device=self.device if self.cache_on_cpu else None,
                            non_blocking=self.non_blocking,
                        )
                        teacher_output, student_output = self.model(
                            img_t, cached_teacher_features=teacher_map_cached
                        )
                    else:
                        teacher_output, student_output = self.model(img_t)

                    loss = criterion(teacher_output, student_output)

                current_loss = loss.item()
              
                # Test abbrechen, wenn die Lernrate so hoch ist, dass das Modell "explodiert"
                if torch.isnan(loss) or current_loss > best_loss * 10:
                    print(f"\nLoss explodiert bei LR={optimizer.param_groups[0]['lr']:.2e}. Test beendet.")
                    break

                if current_loss < best_loss:
                    best_loss = current_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                lrs_recorded.append(optimizer.param_groups[0]["lr"])
                losses_recorded.append(current_loss)

                scheduler.step()
                pbar.set_postfix(
                    loss=f"{current_loss:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
            else:
                # Weiterlaufen, wenn es kein "break" gab
                continue
            # Bricht die äußere Schleife ab, wenn die innere mit "break" beendet wurde
            break

        dataset_label = (
            f"{self.config['dataset']['name']}/{self.config['dataset']['class']}"
            if "class" in self.config["dataset"]
            else self.config["dataset"]["name"]
        )
        print(f"\nLR Range Test abgeschlossen für {dataset_label}. {len(lrs_recorded)} Datenpunkte gesammelt.")

        self.plot_results(lrs_recorded, losses_recorded)
        self._save_csv(lrs_recorded, losses_recorded)

        return lrs_recorded, losses_recorded

    def plot_results(self, lrs: list, losses: list):
        """Zeichnet den Plot (LR vs. Loss) auf einer logarithmischen Skala.

        Args:
            lrs (list): Die aufgezeichneten Lernraten.
            losses (list): Die dazugehörigen Verluste.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        lrs = np.array(lrs)
        losses = np.array(losses)

        ax.plot(lrs, losses, color="blue", linewidth=2, label="Loss")

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        
        ax.grid(True, which="major", color="gray", alpha=0.6, linewidth=0.8, linestyle="-")
        ax.grid(True, which="minor", color="gray", alpha=0.3, linewidth=0.5, linestyle="--")

        ax.set_xlabel("Learning Rate (log scale)", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("LR Range Test - Loss vs Learning Rate", fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)

        dataset_name = self.config["dataset"]["name"]
        dataset_class = self.config["dataset"].get("class", "")
        fig.suptitle(
            f"LR Range Test - {dataset_name} {dataset_class} | {self.model_name}",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"lr_range_test_visual_{timestamp}.png"
        save_path = os.path.join(self.model_save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot gespeichert: {save_path}")

        plt.show()
        plt.close()

    def _save_csv(self, lrs: list, losses: list):
        """Speichert die Rohdaten des Tests in einer CSV-Datei.

        Args:
            lrs (list): Die aufgezeichneten Lernraten.
            losses (list): Die dazugehörigen Verluste.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
        csv_filename = f"lr_data_{timestamp}.csv"
        csv_save_path = os.path.join(self.model_save_dir, csv_filename)

        with open(csv_save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Learning Rate", "Loss"])
            for lr, loss in zip(lrs, losses):
                writer.writerow([lr, loss])

        print(f"Zeitreihendaten gespeichert: {csv_save_path}")


def run_lr_range_test(
    config_path: str,
    use_cached_tensors: bool,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    num_epochs: int = 5
):
    """
    Hauptfunktion: Führt den LR-Range-Test aus und räumt den Speicher danach komplett auf.

    Args:
        config_path (str): Pfad zur Konfiguration.
        use_cached_tensors (bool): Tensoren aus dem Cache laden?
        start_lr (float, optional): Start-Lernrate. Standard ist 1e-7.
        end_lr (float, optional): End-Lernrate. Standard ist 1.0.
        num_epochs (int, optional): Dauer des Tests in Epochen. Standard ist 5.

    Returns:
        tuple: (Liste der Lernraten, Liste der Losses)
    """
    finder = LRRangeTestFinder(config_path, use_cached_tensors)
    lrs, losses = finder.run_test(start_lr, end_lr, num_epochs)

    # Speicher aggressiv aufräumen, damit ein direkt anschließendes Training genug RAM/VRAM hat
    del finder.model
    del finder.train_loader
    if hasattr(finder, 'teacher_features_cache'):
        del finder.teacher_features_cache
    del finder
    
    gc.collect() 
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

    return lrs, losses