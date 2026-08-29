from concurrent.futures import process

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import GradScaler
import time
import yaml
import json
import uuid
import os
import math
import copy
import psutil

from Scripts.losses import *
from Scripts.results_aggregator import *
from Scripts.inference import *
from Scripts.plots import *
from tqdm import tqdm
from Scripts.utils import (
    cache_teacher_features,
    get_teacher_features_cache,
    print_system_info,
    setup_cpu_optimizations,
    setup_gpu_optimizations,
)


class Trainer:
    """Verwaltet den Trainingsprozess für STFPM-Modelle."""

    def __init__(
        self,
        model,
        train_loader,
        config,
        test_loader=None,
        run_evaluation=True,
        train_folder_dir="Training_Runs",
        pretrained_weights=None,
    ):
        """Initialisiert den Trainer.

        Args:
            model (nn.Module): Das zu trainierende Modell.
            train_loader (DataLoader): Daten für das Training.
            config (dict): Konfigurations-Dictionary.
            test_loader (DataLoader, optional): Daten für die Evaluierung.
            run_evaluation (bool, optional): Ob während des Trainings evaluiert werden soll. Standard ist True.
            train_folder_dir (str, optional): Ausgabeordner. Standard ist "Training_Runs".
            pretrained_weights (str, optional): Pfad zu vortrainierten Startgewichten.

        Raises:
            ValueError: Wenn run_evaluation True ist, aber kein test_loader übergeben wurde.
        """
        self.config = config
        self.run_evaluation = run_evaluation

        if self.run_evaluation and test_loader is None:
            raise ValueError(
                "Ein 'test_loader' muss bereitgestellt werden, wenn 'run_evaluation' auf True gesetzt ist."
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.is_gpu = self.device.type == "cuda"
        self.is_cpu = self.device.type == "cpu"

        # Hardware-spezifische Optimierungen einrichten
        if self.is_gpu:
            self.hw_config = setup_gpu_optimizations(self.device, self.config)
            self.cudnn_benchmark = self.config["training"]["cudnn_benchmark"]
        else:
            self.hw_config = setup_cpu_optimizations()
            self.cudnn_benchmark = False
     
        print_system_info()

        # Mixed Precision (AMP) spart Speicher und beschleunigt Berechnungen auf GPUs
        config_amp = self.config.get("model_settings", {}).get(
            "amp_mixed_precision", False
        )
        self.use_amp = config_amp and self.is_gpu

        if config_amp and self.is_cpu:
            print("AMP ist auf der CPU nicht verfügbar - wird deaktiviert.")

        # Channels Last Format für effizientere Berechnungen
        config_channels_last = self.config.get("model_settings", {}).get(
            "channels_last", True
        )

        if config_channels_last:
            self.actual_memory_format = torch.channels_last
        else:
            self.actual_memory_format = torch.contiguous_format

        # Erlaubt asynchrone Datenübertragung zur GPU (Pipelining)
        self.non_blocking = self.config.get("training", {}).get(
            "async_host_to_device", True
        )

        self.model = model.to(
            self.device, memory_format=self.actual_memory_format
        )

        # Lade vortrainierte Gewichte, falls angegeben
        if pretrained_weights is not None:
            if not os.path.exists(pretrained_weights):
                raise ValueError(f"Vorab trainierte Gewichte nicht gefunden: {pretrained_weights}")

            print(f"Vorab trainierte Gewichte werden geladen: {pretrained_weights}")
            weights = torch.load(pretrained_weights, map_location=self.device)
            self.model.load_trainable_state_dict(weights)
            print("Vorab trainierte Gewichte erfolgreich geladen.")

        self.train_loader = train_loader
        self.train_dataset = train_loader.dataset
        self.test_loader = test_loader

        self.cache_teacher_features = self.config.get("training", {}).get(
            "cache_teacher_features", True
        )
        self.criterion = Loss_function(**self.config["loss"]["params"])

        # Berechnet die Teacher-Features einmalig vorab, um im Training Zeit zu sparen
        if self.cache_teacher_features:
            print("Caching der Teacher-Features aktiviert.")

            config_cache_on_cpu = self.config.get("training", {}).get(
                "force_cpu_caching", False
            )
            self.cache_on_cpu = config_cache_on_cpu and self.is_gpu

            if config_cache_on_cpu and self.is_cpu:
                print("force_cpu_caching ignoriert (CPU-Training braucht keinen CPU-Cache)")

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

        # Optimizer vorbereiten (berücksichtigt nur trainierbare Parameter wie den Student)
        optimizer_config = self.config["optimizer"]
        optimizer_params = optimizer_config["configs"][optimizer_config["active"]]
        optimizer_class = getattr(optim, optimizer_config["active"])

        trainable_params = self.model.get_trainable_parameters()
        self.optimizer = optimizer_class(trainable_params, **optimizer_params)

        # Lernraten-Scheduler vorbereiten
        self.scheduler = None
        if "scheduler" in self.config and self.config["scheduler"]:
            scheduler_name = self.config["scheduler"]["type"]
            if hasattr(lr_scheduler, scheduler_name):
                scheduler_class = getattr(lr_scheduler, scheduler_name)
                params = self.config["scheduler"]["params"].copy()
                
                if scheduler_name == "OneCycleLR":
                    params["steps_per_epoch"] = len(self.train_loader)
                self.scheduler = scheduler_class(self.optimizer, **params)

        # Teacher-Netzwerk zwingend in den Evaluierungs-Modus setzen
        if hasattr(self.model, "teacher_model") and self.model.teacher_model:
            self.model.teacher_model.eval()
        if hasattr(self.model, "teacher_full"):
            self.model.teacher_full.eval()

        self.train_folder_dir = train_folder_dir
        os.makedirs(self.train_folder_dir, exist_ok=True)
        
        self.training_id = str(uuid.uuid4())

        # Hilft gegen zu kleine Gradienten bei float16 (AMP)
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        self.evaluate = None
        if self.run_evaluation:
            self.evaluate = Inference(
                self.model,
                self.test_loader,
                self.config,
                trainings_id=self.training_id,
                inferenz=False,
            )

        is_mvtec = self.config["dataset"]["name"] == "MVTecAD"
        if is_mvtec:
            self.dataset_identifier = f"{self.config['dataset']['name']}_{self.config['dataset']['class']}"
            self.log_dataset_info = {"dataset_class": self.config["dataset"]["class"]}
            self.display_dataset_info = f"Dataset-Klasse: {self.config['dataset']['class']}"
        else:
            self.dataset_identifier = self.config["dataset"]["name"]
            self.log_dataset_info = {"dataset": self.config["dataset"]["name"]}
            self.display_dataset_info = f"Dataset: {self.config['dataset']['name']}"

        # set_to_none=True spart Zeit beim Nullen der Gradienten
        self.use_set_to_none = self.config.get("training", {}).get("fast_zero_grad", True)

    def _is_stem_layer_active(self) -> bool:
        """Prüft, ob ein gemeinsamer Stem-Layer (Eingangsblock) aktiv ist.

        Returns:
            bool: True, wenn Stem-Layer geteilt wird.
        """
        if self.model.stem_model is None:
            return False
        if isinstance(self.model.stem_model, nn.Identity):
            return False
        stem_params = sum(p.numel() for p in self.model.stem_model.parameters())
        return stem_params > 0

    def _print_optimization_status(self):
        """Gibt eine Übersicht über aktive Hardware- und Trainings-Optimierungen aus."""
        print(f"\n{'='*60}")
        print(f"{'GPU-MODUS' if self.is_gpu else 'CPU-MODUS':^60}")
        print(f"{'='*60}")

        if self.is_gpu:
            print(f"GPU: {self.hw_config.get('gpu_name', 'Unknown')}")
            print(f"VRAM: {self.hw_config.get('gpu_memory_gb', 0):.1f} GB")
            print(f"TF32: {self.hw_config.get('tf32_enabled', False)}")
            print(f"BF16: {self.hw_config.get('BF16', False)}")
        else:
            print(f"CPU Threads: {self.hw_config.get('num_threads', 'Unknown')}")
            print(f"MKL/oneDNN: {'Ja' if self.hw_config.get('mkldnn_enabled') else 'Nein'}")

        print(f"\n{'─'*60}")
        print(f"{'CONFIG-WERTE vs TATSÄCHLICHE WERTE':^60}")
        print(f"{'─'*60}")

        config_amp = self.config.get("model_settings", {}).get("amp_mixed_precision", False)
        config_channels_last = self.config.get("model_settings", {}).get("channels_last", False)
        config_cudnn_benchmark = self.config.get("training", {}).get("cudnn_benchmark", False)
        config_non_blocking = self.config.get("training", {}).get("async_host_to_device", False)
        config_teacher_cache = self.config.get("training", {}).get("cache_teacher_features", False)
        config_set_to_none = self.config.get("training", {}).get("fast_zero_grad", False)
        config_stem = self.config.get("model_settings", {}).get("shared_stem", False)

        print(f"\n  {'Optimierung':<35} {'Config':<10} {'Aktiv':<10}")
        print(f"  {'-'*55}")
        print(f"  {'AMP (Mixed Precision)':<35} {str(config_amp):<10} {'Aktiviert' if self.use_amp else 'Deaktiviert':<10}")
        print(f"  {'Channels Last Format':<35} {str(config_channels_last):<10} {'Aktiviert' if self.actual_memory_format == torch.channels_last else 'Deaktiviert':<10}")
        print(f"  {'cudnn_benchmark + TF32':<35} {str(config_cudnn_benchmark):<10} {'Aktiviert' if self.cudnn_benchmark else 'Deaktiviert':<10}")
        print(f"  {'Non-blocking Transfer':<35} {str(config_non_blocking):<10} {'Aktiviert' if self.non_blocking else 'Deaktiviert':<10}")
        print(f"  {'Teacher Feature Caching':<35} {str(config_teacher_cache):<10} {'Aktiviert' if self.teacher_features_cache is not None else 'Deaktiviert':<10}")
        print(f"  {'Gradient set_to_none':<35} {str(config_set_to_none):<10} {'Aktiviert' if self.use_set_to_none else 'Deaktiviert':<10}")
        print(f"  {'Shared Stem Layer':<35} {str(config_stem):<10} {'Aktiviert' if self._is_stem_layer_active() else 'Deaktiviert':<10}")

        print(f"\n{'─'*60}")
        print(f"{'DEBUG: OPTIMIERUNGS-VERIFIKATION':^60}")
        print(f"{'─'*60}")

        if self.teacher_features_cache is not None:
            if isinstance(self.teacher_features_cache, dict) and self.teacher_features_cache.get("is_indexed", False):
                num_cached = len(self.teacher_features_cache["path_to_index"])
                num_layers = len(self.teacher_features_cache["features"])
                cache_device = self.teacher_features_cache.get("device", "unknown")
                first_shape = self.teacher_features_cache["features"][0].shape
                print(f"  Teacher Cache: AKTIV (Index-basiert)")
                print(f"    └─ {num_cached} Bilder, {num_layers} Layer")
                print(f"    └─ Device: {cache_device}")
                print(f"    └─ Shape Layer 0: {first_shape}")
            else:
                print(f"  Teacher Cache: AKTIV (Legacy-Format)")
        else:
            print(f"  Teacher Cache: DEAKTIVIERT")
            print(f"    └─ Teacher-Features werden JEDE EPOCHE neu berechnet!")

        stem_is_identity = isinstance(self.model.stem_model, nn.Identity)
        stem_is_none = self.model.stem_model is None
        if not stem_is_none and not stem_is_identity:
            stem_params = sum(p.numel() for p in self.model.stem_model.parameters())
            print(f"  Shared Stem: AKTIV ({stem_params:,} Parameter)")
            print(f"    └─ Typ: {type(self.model.stem_model).__name__}")
        else:
            print(f"  Shared Stem: DEAKTIVIERT")
            if stem_is_identity:
                print(f"    └─ Grund: nn.Identity() (kein echter Stem)")
            elif stem_is_none:
                print(f"    └─ Grund: None")

        use_original_paper = self.config.get("loss", {}).get("params", {}).get("use_original_paper", True)
        print(f"  {'L2+MSE (Original Paper)' if use_original_paper else 'Cosine'} Loss Function: {'Cosine' if not use_original_paper else 'L2+MSE (Original Paper)'}")

        dl_config = self.config.get("dataloader", {})
        print(f"\n  DataLoader Config:")
        print(f"    └─ num_workers: {dl_config.get('num_workers', 'auto')}")
        print(f"    └─ pin_memory: {dl_config.get('pin_memory', 'default')}")
        print(f"    └─ persistent_workers: {dl_config.get('persistent_workers', 'default')}")

        print(f"{'='*60}\n")

    def train_per_epoch(self, epoch: int, num_epochs: int, profiler=None, detailed_timing: bool = False) -> float:
        """Wählt automatisch die schnelle oder die detaillierte Trainingsschleife.

        Args:
            epoch (int): Aktuelle Epoche.
            num_epochs (int): Gesamtanzahl der Epochen.
            profiler (optional): PyTorch Profiler-Instanz.
            detailed_timing (bool, optional): Ob Detail-Latenzen gemessen werden sollen.

        Returns:
            float: Der durchschnittliche Trainingsfehler (Loss) der Epoche.
        """
        if detailed_timing:
            return self._train_per_epoch_with_timing(epoch, num_epochs, profiler)
        else:
            return self._train_per_epoch_fast(epoch, num_epochs, profiler)

    def _train_per_epoch_fast(self, epoch: int, num_epochs: int, profiler=None) -> float:
        """Schneller Trainingsablauf (Ohne Zeitmessungen und GPU-Synchronisation)."""
        self.model.student_model.train()

        stem_is_active = self._is_stem_layer_active()
        if stem_is_active:
            self.model.stem_model.eval()

        train_loss = 0.0
        batch_iterator = tqdm(
            self.train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False
        )

        for images, _, image_paths, _ in batch_iterator:
            img_t = images.to(
                self.device,
                memory_format=self.actual_memory_format,
                non_blocking=self.non_blocking,
            )

            if self.use_set_to_none:
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.zero_grad()

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
                cache_enabled=True,
            ):
                if self.teacher_features_cache is not None:
                    cached_features = get_teacher_features_cache(
                        self.teacher_features_cache,
                        image_paths,
                        device=self.device if self.cache_on_cpu else None,
                        non_blocking=self.non_blocking,
                    )
                    teacher_output, student_output = self.model(
                        img_t, cached_teacher_features=cached_features
                    )
                else:
                    teacher_output, student_output = self.model(img_t)

                loss = self.criterion(teacher_output, student_output)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            train_loss += loss.item()

            if profiler:
                profiler.step()
             
        # Peak RAM pro Epoche nachverfolgen
        process = psutil.Process(os.getpid())
        current_ram = process.memory_info().rss / (1024 * 1024)
        if not hasattr(self, 'global_peak_ram') or current_ram > self.global_peak_ram:
            self.global_peak_ram = current_ram
    
        if self.is_gpu:
            current_vram = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            if not hasattr(self, 'global_peak_vram') or current_vram > self.global_peak_vram:
                self.global_peak_vram = current_vram

        return train_loss / len(self.train_loader)

    def _train_per_epoch_with_timing(self, epoch: int, num_epochs: int, profiler=None) -> float:
        """Trainingsschleife mit detaillierten Zeitmessungen pro Teilschritt (langsamer!)."""
        self.model.student_model.train()

        stem_is_active = self._is_stem_layer_active()
        if stem_is_active:
            self.model.stem_model.eval()

        train_loss = 0.0

        time_data_loading = 0.0
        time_to_device = 0.0
        time_stem = 0.0
        time_teacher = 0.0
        time_student = 0.0
        time_loss = 0.0
        time_backward = 0.0

        if self.is_gpu:
            torch.cuda.reset_peak_memory_stats(self.device)
        
        process = psutil.Process(os.getpid())
        epoch_peak_ram = process.memory_info().rss / (1024 * 1024)  # in MB

        if hasattr(self.train_dataset, "reset_timing"):
            self.train_dataset.reset_timing()

        batch_iterator = tqdm(
            self.train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False
        )

        data_start = time.time()

        for images, _, image_paths, _ in batch_iterator:
            current_ram = process.memory_info().rss / (1024 * 1024)
            if current_ram > epoch_peak_ram:
                epoch_peak_ram = current_ram

            time_data_loading += time.time() - data_start

            t0 = time.time()
            img_t = images.to(
                self.device,
                memory_format=self.actual_memory_format,
                non_blocking=self.non_blocking,
            )
            
            if self.is_gpu:
                torch.cuda.synchronize()
            time_to_device += time.time() - t0

            if self.use_set_to_none:
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.zero_grad()

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
                cache_enabled=True,
            ):
                if stem_is_active:
                    t0 = time.time()
                    stem_output = self.model.stem_model(img_t)
                    if self.is_gpu:
                        torch.cuda.synchronize()
                    time_stem += time.time() - t0
                else:
                    stem_output = img_t

                t0 = time.time()
                if self.teacher_features_cache is not None:
                    cached_features = get_teacher_features_cache(
                        self.teacher_features_cache,
                        image_paths,
                        device=self.device if self.cache_on_cpu else None,
                        non_blocking=self.non_blocking,
                    )
                    if self.is_gpu:
                        torch.cuda.synchronize()
                    time_teacher += time.time() - t0
                    teacher_feature_maps = cached_features
                else:
                    with torch.no_grad():
                        teacher_feature_maps = self.model.teacher_model(stem_output)
                    if isinstance(teacher_feature_maps, dict):
                        teacher_feature_maps = list(teacher_feature_maps.values())
                    if self.is_gpu:
                        torch.cuda.synchronize()
                    time_teacher += time.time() - t0

                t0 = time.time()
                student_feature_maps = self.model.student_model(stem_output)
                if isinstance(student_feature_maps, dict):
                    student_feature_maps = list(student_feature_maps.values())

                aligned_student_maps = []
                for i, (t_map, s_map) in enumerate(zip(teacher_feature_maps, student_feature_maps)):
                    if self.model.is_asymmetric:
                        s_map = self.model.projection_heads[i](s_map)
                    if s_map.shape[-2:] != t_map.shape[-2:]:
                        s_map = torch.nn.functional.interpolate(
                            s_map,
                            size=t_map.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    aligned_student_maps.append(s_map)

                if self.is_gpu:
                    torch.cuda.synchronize()
                time_student += time.time() - t0

                t0 = time.time()
                loss = self.criterion(teacher_feature_maps, aligned_student_maps)
                if self.is_gpu:
                    torch.cuda.synchronize()
                time_loss += time.time() - t0

            t0 = time.time()
            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.is_gpu:
                torch.cuda.synchronize()
            time_backward += time.time() - t0

            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            train_loss += loss.item()

            if profiler:
                profiler.step()

            data_start = time.time()

        epoch_peak_vram = 0.0
        if self.is_gpu:
            epoch_peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        if not hasattr(self, 'global_peak_ram'):
            self.global_peak_ram = 0.0
        if not hasattr(self, 'global_peak_vram'):
            self.global_peak_vram = 0.0

        if epoch_peak_ram > self.global_peak_ram:
            self.global_peak_ram = epoch_peak_ram
            
        if self.is_gpu and epoch_peak_vram > self.global_peak_vram:
            self.global_peak_vram = epoch_peak_vram

        total = (
            time_data_loading + time_to_device + time_stem + time_teacher +
            time_student + time_loss + time_backward
        )

        if epoch <= 3 or epoch % 10 == 0:
            cache_status = "CACHED" if self.teacher_features_cache else "COMPUTED"
            stem_status = "SHARED" if stem_is_active else "IDENTITY"
            method_status = "MSE (Original)" if self.config.get("loss", {}).get("params", {}).get("use_original_paper", True) else "COSINE (Optimiert)"

            print(f"\n  Timing Breakdown (Epoch {epoch}):")
            print(f"     {'Component':<20} {'Time':>8} {'Percent':>8}  {'Status':<12}")
            print(f"     {'-'*52}")
            print(f"     {'Data Loading':<20} {time_data_loading:>7.2f}s {100*time_data_loading/total:>7.1f}%")
            print(f"     {'To Device':<20} {time_to_device:>7.2f}s {100*time_to_device/total:>7.1f}%")
            print(f"     {'Stem Layer':<20} {time_stem:>7.2f}s {100*time_stem/total:>7.1f}%  ← {stem_status}")
            print(f"     {'Teacher Forward':<20} {time_teacher:>7.2f}s {100*time_teacher/total:>7.1f}%  ← {cache_status}")
            print(f"     {'Student Forward':<20} {time_student:>7.2f}s {100*time_student/total:>7.1f}%")
            print(f"     {'Loss Compute':<20} {time_loss:>7.2f}s {100*time_loss/total:>7.1f}%  ← {method_status}")
            print(f"     {'Backward + Step':<20} {time_backward:>7.2f}s {100*time_backward/total:>7.1f}%")
            print(f"     {'-'*52}")
            print(f"     {'TOTAL':<20} {total:>7.2f}s")

            compute_time = time_stem + time_teacher + time_student + time_loss + time_backward
            io_time = time_data_loading + time_to_device
            ratio = io_time / compute_time if compute_time > 0 else float("inf")
            device_label = "GPU" if self.is_gpu else "CPU"

            print(f"\n      Analyse:")
            print(f"        I/O Zeit:     {io_time:.2f}s ({100*io_time/total:.1f}%)")
            print(f"        {device_label} Compute:  {compute_time:.2f}s ({100*compute_time/total:.1f}%)")
            bound_msg = "(I/O-bound!)" if ratio > 2 else f"({device_label}-bound)" if ratio < 0.5 else ""
            print(f"        I/O:{device_label} Ratio: {ratio:.2f}x {bound_msg}")

            print(f"\n      Speicher (Peak DIESER Epoche):")
            print(f"        CPU RAM:      {epoch_peak_ram:.2f} MB")
            if self.is_gpu:
                print(f"        GPU VRAM:     {epoch_peak_vram:.2f} MB")
                
            print(f"\n      Speicher (GLOBALER Peak bisher):")
            print(f"        CPU RAM:      {self.global_peak_ram:.2f} MB")
            if self.is_gpu:
                print(f"        GPU VRAM:     {self.global_peak_vram:.2f} MB")

            if stem_is_active and not self.teacher_features_cache:
                stem_savings = time_stem
                print(f"\n        Stem Ersparnis: ~{stem_savings:.2f}s/Epoche (Shared statt 2x)")
        else:
            compute_time = time_stem + time_teacher + time_student + time_loss + time_backward
            io_time = time_data_loading + time_to_device
            ratio = io_time / compute_time if compute_time > 0 else float("inf")
            cache_status = "C" if self.teacher_features_cache else "NC"
            stem_status = "S" if stem_is_active else "-"
            
            mem_log = f" | VRAM:{epoch_peak_vram:.0f}MB" if self.is_gpu else ""
            print(
                f"     Data:{time_data_loading:.1f}s | Stem:{time_stem:.1f}s[{stem_status}] | T:{time_teacher:.1f}s[{cache_status}] | S:{time_student:.1f}s | Bwd:{time_backward:.1f}s | Ratio:{ratio:.1f}x | RAM:{epoch_peak_ram:.0f}MB{mem_log}"
            )

        if hasattr(self.train_dataset, "print_timing_report"):
            self.train_dataset.print_timing_report(epoch)

        return train_loss / len(self.train_loader)

    def train(self, set_profiler: bool = False, detailed_timing: bool = None):
        """Hauptmethode zum Starten des Trainings.

        Args:
            set_profiler (bool, optional): Ob der PyTorch Profiler genutzt wird.
            detailed_timing (bool, optional): Ob detaillierte Zeiten ausgegeben werden sollen.
        """
        if detailed_timing is None:
            detailed_timing = self.config.get("training", {}).get("detailed_timing", True)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        print(f"Starte Training mit ID: {self.training_id}")

        if self.config.get("model_settings", {}).get("is_asymmetric", False):
            print(
                f"Asymmetrisches Modell: Teacher:{self.config['teacher_model']['architecture']}, "
                f"Student:{self.config['student_model']['architecture']}, "
                f"{self.display_dataset_info}, Epochen: {self.config['epochs']}"
            )
        else:
            print(
                f"Modell: {self.config['model']['architecture']}, "
                f"{self.display_dataset_info}, Epochen: {self.config['epochs']}"
            )

        self._print_optimization_status()

        run_base_path, weights_save_dir, logs_save_dir, plots_save_dir = self.create_dir_for_run(self.training_id)

        if set_profiler:
            print("\nExtrahiere ein reales Bild aus dem DataLoader für die Komplexitätsanalyse...")
            data_iter = iter(self.train_loader)
            images, _, _, _ = next(data_iter)
            
            sample_image = images[0:1].to(
                self.device, memory_format=self.actual_memory_format, non_blocking=self.non_blocking
            )

            if self.config.get("model_settings", {}).get("is_asymmetric", False):
                t_name = self.config['teacher_model']['architecture']
                s_name = self.config['student_model']['architecture']
            else:
                t_name = self.config['model']['architecture']
                s_name = self.config['model']['architecture']
            
            analyze_real_stfpm_complexity(
                 self.model, 
                sample_input=sample_image, 
                device=self.device,
                teacher_name=t_name,
                student_name=s_name
            ) 

        metrics = {
            "train_loss": [],
            "epoch_durations": [],
            "best_epoch_loss": float("inf"),
            "best_epoch": -1,
            "auroc_scores": [],
            "aupr_scores": [],
            "inference_times": [],
            "peak_ram_mb": 0,
            "peak_vram_mb": 0
        }
        best_model_weights = None
        total_training_time_start = time.time()

        prof = None
        if set_profiler:
            profiler_log_dir = os.path.join(logs_save_dir, "profiler_traces")
            os.makedirs(profiler_log_dir, exist_ok=True)

            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.is_gpu:
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            prof = torch.profiler.profile(
                 activities=activities,
                schedule=torch.profiler.schedule(wait=3, warmup=1, active=5),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            )

        for epoch in range(1, self.config["epochs"] + 1):
            epoch_start_time = time.time()

            # Starte Profiler typischerweise ab Epoche 5 für saubere Messungen (Warm-up ist vorbei)
            if set_profiler and epoch == 5:
                print(f"\n--- Starte Profiler für Epoche {epoch} ---")
                prof.start()
                train_loss = self.train_per_epoch(
                    epoch,
                    self.config["epochs"],
                    profiler=prof,
                    detailed_timing=detailed_timing,
                )
                prof.stop()

                sort_key = "cuda_time_total" if self.is_gpu else "cpu_time_total"
                print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

                trace_filename = f"trace_epoch_{epoch}_{self.dataset_identifier}_{self.training_id}.json"
                trace_path = os.path.join(profiler_log_dir, trace_filename)
                prof.export_chrome_trace(trace_path)
                print(f"  -> Profiler-Trace wurde gespeichert in: {trace_path}")
            else:
                train_loss = self.train_per_epoch(
                     epoch,
                    self.config["epochs"],
                    detailed_timing=detailed_timing,
                )

            metrics["train_loss"].append(train_loss)
            epoch_duration = time.time() - epoch_start_time
            metrics["epoch_durations"].append(epoch_duration)

            print(f"Epoch {epoch}/{self.config['epochs']} - Loss: {train_loss:.6f}, Zeit: {epoch_duration:.2f}s")

            if train_loss < metrics["best_epoch_loss"]:
                metrics["best_epoch_loss"] = train_loss
                metrics["best_epoch"] = epoch
                
                # Nur trainierbare Gewichte sichern, um Speicher zu sparen
                current_weights = self.model.get_trainable_state_dict()
                best_model_weights = copy.deepcopy(current_weights)
                print(f"  -> Neuer bester Loss: {metrics['best_epoch_loss']:.6f}")

            if self.run_evaluation:
                auc_score, aupr, inference_time = self.evaluate.evaluate_per_epoch()
                metrics["auroc_scores"].append(auc_score)
                metrics["aupr_scores"].append(aupr)
                metrics["inference_times"].append(inference_time)
                print(f"  -> AUROC: {auc_score:.4f}, AUPR: {aupr:.4f}, Inferenz: {inference_time:.4f}s")

            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()

        metrics["total_training_time"] = time.time() - total_training_time_start
        metrics["avg_epoch_duration"] = sum(metrics["epoch_durations"]) / len(metrics["epoch_durations"])

        print(f"\nTraining abgeschlossen!")
        print(f"  Gesamtdauer: {metrics['total_training_time']:.2f}s")
        print(f"  Ø Epochendauer: {metrics['avg_epoch_duration']:.2f}s")

        if self.config.get("training", {}).get("use_qat", False):
            from torch.ao.quantization.quantize_fx import convert_fx

            print("Konvertiere QAT-Student-Modell in finale INT8-Gewichte...")
            self.model.eval().to("cpu")
            self.model.student_model = convert_fx(self.model.student_model)
            print("QAT-Modell erfolgreich in INT8 konvertiert.")

        base_filename = f"{self._get_model_name()}_{self.dataset_identifier}_{self.training_id}"
        weight_suffix = "student"

        final_weights_path = os.path.join(weights_save_dir, f"{base_filename}_final_{weight_suffix}.pth")
        torch.save(self.model.get_trainable_state_dict(), final_weights_path)
        metrics["final_model_size_mb"] = os.path.getsize(final_weights_path) / (1024 * 1024)

        best_weight_path = None
        if best_model_weights:
            best_weight_path = os.path.join(weights_save_dir, f"{base_filename}_best_{weight_suffix}.pth")
            torch.save(best_model_weights, best_weight_path)
            metrics["best_model_size_mb"] = os.path.getsize(best_weight_path) / (1024 * 1024)

        # Berechne Modellgröße in MB
        total_params = sum(p.numel() for p in self.model.parameters())
        b = next(self.model.parameters()).element_size()
        metrics["model_param_size_mb"] = total_params * b / (1024 * 1024)

        metrics["peak_ram_mb"] = getattr(self, "global_peak_ram", "N/A")
        metrics["peak_vram_mb"] = getattr(self, "global_peak_vram", "N/A")

        self.create_log_csv(self.training_id, timestamp, metrics, logs_save_dir, base_filename)
        config_save_path = self.create_yaml_config(run_base_path)
        
        self.create_summary_metric(
            self.training_id,
            timestamp,
            metrics,
            config_save_path,
            run_base_path,
            final_weights_path,
            best_weight_path,
        )
        self.create_plots_for_run(metrics, plots_save_dir, base_filename)

    def create_dir_for_run(self, training_id: str) -> tuple:
        """Erstellt alle nötigen Output-Ordner für den aktuellen Lauf.

        Args:
            training_id (str): Eindeutige ID des Laufs.

        Returns:
            tuple: Die generierten Verzeichnispfade.
        """
        model_folder_name = self._get_model_name()

        run_base_path = os.path.join(
            self.train_folder_dir,
            self.dataset_identifier,
            model_folder_name,
            training_id,
        )

        weights_save_dir = os.path.join(run_base_path, "weights")
        logs_save_dir = os.path.join(run_base_path, "logs")
        plots_save_dir = os.path.join(run_base_path, "plots")

        os.makedirs(weights_save_dir, exist_ok=True)
        os.makedirs(logs_save_dir, exist_ok=True)
        os.makedirs(plots_save_dir, exist_ok=True)
        return run_base_path, weights_save_dir, logs_save_dir, plots_save_dir

    def create_log_csv(self, training_id: str, timestamp: str, metrics: dict, logs_save_dir: str, base_filename: str):
        """Exportiert Trainingsstatistiken als CSV.

        Args:
            training_id (str): ID des Trainings.
            timestamp (str): Startzeitpunkt.
            metrics (dict): Die gesammelten Metriken.
            logs_save_dir (str): Ordnerpfad für die Logs.
            base_filename (str): Name der Datei.
        """
        if self.config.get("model_settings", {}).get("is_asymmetric", False):
            model_info = {
                "teacher_model": self.config["teacher_model"]["architecture"],
                "student_model": self.config["student_model"]["architecture"],
            }
        else:
            model_info = {"model": self.config["model"]["architecture"]}

        result_data = {
            "training_id": training_id,
            "timestamp": timestamp,
            **self.log_dataset_info,
            **model_info,
            "device": str(self.device),
            "is_gpu": self.is_gpu,
            **self.hw_config,
            "channels_last": str(self.actual_memory_format),
            "teacher_caching": self.cache_teacher_features,
            "used_asynchronous_data_transfer": self.non_blocking,
            "used_shared_stem_layer": self._is_stem_layer_active(),
            "gradient_set_to_none": self.use_set_to_none,
            "cudnn_Benchmark": self.cudnn_benchmark,
            "amp": self.use_amp,
            "img_size": self.config["dataset"]["img_size"],
            "batch_size": self.config["dataloader"]["batch_size"],
            "optimizer": self.config["optimizer"]["active"],
            "lr": self.config["optimizer"]["configs"][self.config["optimizer"]["active"]].get("lr", "N/A"),
            "scheduler": self.config.get("scheduler", {}).get("type"),
            "epochs": self.config["epochs"],
            "total_time": metrics["total_training_time"],
            "avg_epoch_time": metrics["avg_epoch_duration"],
            "final_loss": (metrics["train_loss"][-1] if metrics["train_loss"] else 0.0),
            "best_loss": metrics["best_epoch_loss"],
            "best_epoch": metrics["best_epoch"],
            "final_auroc": (metrics["auroc_scores"][-1] if metrics["auroc_scores"] else 0.0),
            "best_auroc": (max(metrics["auroc_scores"]) if metrics["auroc_scores"] else 0.0),
            "avg_inference_time": (
                (sum(metrics["inference_times"]) / len(metrics["inference_times"]))
                if metrics["inference_times"] else 0.0
            ),
            "best_model_mb": metrics.get("best_model_size_mb"),
            "model_param_size_mb": metrics.get("model_param_size_mb"),
            "peak_ram_mb": metrics.get("peak_ram_mb"),   
            "peak_vram_mb": metrics.get("peak_vram_mb"),
            "final_aupr": (metrics["aupr_scores"][-1] if metrics["aupr_scores"] else 0.0),
            "best_aupr": (max(metrics["aupr_scores"]) if metrics["aupr_scores"] else 0.0),
        }
 
        create_result_df(
            result_data,
            target_filename=os.path.join(logs_save_dir, f"{base_filename}_results.csv"),
        )

    def create_yaml_config(self, run_base_path: str) -> str:
        """Kopiert/Speichert die YAML Konfiguration im Ordner des Trainingslaufs.

        Args:
            run_base_path (str): Zielordner für die Config.

        Returns:
            str: Der genaue Pfad der gespeicherten YAML Datei.
        """
        model_name = self._get_model_name()
        config_save_path = os.path.join(run_base_path, f"Config_{model_name}.yaml")
        with open(config_save_path, "w") as f:
            yaml.dump(
                self.config,
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=4,
            )
        return config_save_path

    def create_summary_metric(
        self,
        training_id: str,
        timestamp: str,
        metrics: dict,
        config_save_path: str,
        run_base_path: str,
        final_weights_path: str,
        best_weight_path: str,
    ):
        """Erzeugt ein JSON mit einer Zusammenfassung aller wichtigen Laufzeit-Daten.

        Args:
            training_id (str): ID des Trainings.
            timestamp (str): Startzeitpunkt.
            metrics (dict): Gesammelte Leistungsdaten.
            config_save_path (str): Wo die Config liegt.
            run_base_path (str): Zielordner.
            final_weights_path (str): Pfad der letzten Epoche.
            best_weight_path (str): Pfad der besten Epoche.
        """
        model_arch_info = self._get_model_name()
        weight_suffix = "student"

        summary_metrics = {
            "training_id": training_id,
            "timestamp": timestamp,
            "model_architecture": model_arch_info,
            "config_path": config_save_path,
            "hardware": {
                "device": str(self.device),
                "is_gpu": self.is_gpu,
                **self.hw_config,
            },
            "dataset/dataloader": {
                "dataset_name": self.dataset_identifier,
                "img_size": self.config["dataset"]["img_size"],
                "batch_size": self.config["dataloader"]["batch_size"],
            },
            "optimizations_used": {
                "amp": self.use_amp,
                "channels_last": self.actual_memory_format == torch.channels_last,
                "cudnn_benchmark": self.cudnn_benchmark,
                "used_asynchronous_data_transfer": self.non_blocking,
                "teacher_caching": self.teacher_features_cache is not None,
                "gradient_set_to_none": self.use_set_to_none,
                "used_shared_stem_layer": self._is_stem_layer_active(),
            },
            "config_values": {
                "amp_mixed_precision": self.config.get("model_settings", {}).get("amp_mixed_precision", False),
                "channels_last": self.config.get("model_settings", {}).get("channels_last", False),
                "cudnn_benchmark": self.config.get("training", {}).get("cudnn_benchmark", False),
                "async_host_to_device": self.config.get("training", {}).get("async_host_to_device", False),
                "cache_teacher_features": self.config.get("training", {}).get("cache_teacher_features", False),
                "fast_zero_grad": self.config.get("training", {}).get("fast_zero_grad", False),
                "shared_stem": self.config.get("model_settings", {}).get("shared_stem", False),
                "use_original_paper": self.config.get("model_settings", {}).get("use_original_paper", True),
                "num_workers": self.config.get("dataloader", {}).get("num_workers", "auto"),
                "pin_memory": self.config.get("dataloader", {}).get("pin_memory", False),
                "persistent_workers": self.config.get("dataloader", {}).get("persistent_workers", False),
            },
            "training_params": {
                "optimizer": self.config["optimizer"]["active"],
                "lr": self.config["optimizer"]["configs"][self.config["optimizer"]["active"]].get("lr", "N/A"),
                "scheduler": self.config.get("scheduler", {}).get("type"),
                "max_lr": self.config.get("scheduler", {}).get("params", {}).get("max_lr"),
                "epochs": self.config["epochs"],
            },
            "training_summary": {
                "duration": metrics["total_training_time"],
                "avg_epoch_time": metrics["avg_epoch_duration"],
                "final_loss": metrics["train_loss"][-1],
                "best_loss": metrics["best_epoch_loss"],
                "best_epoch": metrics["best_epoch"],
                "peak_ram_mb": metrics.get("peak_ram_mb"),
                "peak_vram_mb": metrics.get("peak_vram_mb"),
                "loss_per_epoch": metrics["train_loss"]
            },
            "Speicherbedarf auf der Festplatte (MB)": {
                "final": metrics.get("final_model_size_mb", 0.0),
                "best": metrics.get("best_model_size_mb", 0.0),
             },
            "Speicherbedarf der Modellparameter (MB)": metrics.get("model_param_size_mb"),
            "weight_paths": {
                "final_" + weight_suffix: final_weights_path,
                "best_" + weight_suffix: best_weight_path,
            },
        }

        if self.run_evaluation:
            summary_metrics["evaluation"] = {
                "avg_inference_time": (
                    (sum(metrics["inference_times"]) / len(metrics["inference_times"]))
                    if metrics["inference_times"] else 0.0
                ),
                "final_auroc": (metrics["auroc_scores"][-1] if metrics["auroc_scores"] else 0.0),
                "best_auroc": (max(metrics["auroc_scores"]) if metrics["auroc_scores"] else 0.0),
                "final_aupr": (metrics["aupr_scores"][-1] if metrics["aupr_scores"] else 0.0),
                "best_aupr": (max(metrics["aupr_scores"]) if metrics["aupr_scores"] else 0.0),
            }

        summary_json_path = os.path.join(run_base_path, "summary_metrics.json")
        with open(summary_json_path, "w") as f:
            json.dump(summary_metrics, f, indent=4)

    def create_plots_for_run(self, metrics: dict, plots_save_dir: str, base_filename: str):
        """Generiert und speichert Loss- sowie AUROC-Plots.

        Args:
            metrics (dict): Wörterbuch mit Metriken.
            plots_save_dir (str): Speicherordner für die Bilder.
            base_filename (str): Name der Datei.
        """
        plot_loss_curves(metrics, self.config, plots_save_dir, base_filename)
        if self.run_evaluation:
            plot_auroc_scores(
                metrics, self.config, plots_save_dir, base_filename
            )

    def _get_model_name(self) -> str:
        """Gibt den Basisnamen der genutzten Modell-Architektur zurück.

        Returns:
            str: Der Modellname.
        """
        if self.config.get("model_settings", {}).get("is_asymmetric", False):
            return (
                f"teacher-{self.config['teacher_model']['architecture']}_"
                f"student-{self.config['student_model']['architecture']}"
            )
        else:
            return self.config["model"]["architecture"]