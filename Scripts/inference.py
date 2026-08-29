import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import yaml
import os
import json
import thop
from fvcore.nn import FlopCountAnalysis
from Scripts.utils import *
from memory_profiler import memory_usage
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from Scripts.losses import *
from Scripts.results_aggregator import *
from torch.profiler import profile, record_function, ProfilerActivity
from Scripts.stfpm_arch import *


class Inference:
    """Kapselt den Inferenz- und Evaluierungsprozess für STFPM-Modelle."""

    def __init__(
        self,
        model,
        test_loader,
        config,
        output_dir="Inference_Runs",
        path_to_student_weight=None,
        trainings_id=None,
        inferenz=True,
    ):
        """Initialisiert die Inferenz-Pipeline.

        Args:
            model: Das zu evaluierende PyTorch-Modell.
            test_loader: DataLoader mit den Testdaten.
            config (dict): Konfigurations-Wörterbuch.
            output_dir (str, optional): Zielordner für Ergebnisse. Standard ist "Inference_Runs".
            path_to_student_weight (str, optional): Pfad zu vortrainierten Gewichten.
            trainings_id (str, optional): Eindeutige ID des Trainingslaufs.
            inferenz (bool, optional): Ob der Inferenz-Modus aktiv ist (erstellt Ausgabeordner). Standard ist True.
        """
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_gpu = self.device.type == "cuda"
        self.is_cpu = self.device.type == "cpu"

        # Setzt Backend-spezifische Hardware-Optimierungen
        if self.is_gpu:
            self.hw_config = setup_gpu_optimizations(self.device, self.config)
        else:
            self.hw_config = setup_cpu_optimizations()

        # Speicherformat optimieren (Channels Last ist auf modernen GPUs oft schneller)
        if self.config.get("model_settings", {}).get("channels_last", True):
            self.actual_memory_format = torch.channels_last
        else:
            self.actual_memory_format = torch.contiguous_format

        # Mixed Precision (AMP) aktivieren, falls konfiguriert
        if self.config.get("model_settings", {}).get("amp_mixed_precision", False) and self.is_gpu:
            self.use_amp = True
        else:
            self.use_amp = False

        # Asynchrone Datenübertragung beschleunigt das Pipelining
        self.non_blocking = self.config.get("training", {}).get("async_host_to_device", True)

        self.model = model.to(self.device, memory_format=self.actual_memory_format)

        # PyTorch Compiler aktivieren (ab PyTorch 2.0 verfügbar)
        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                print("torch.compile aktiviert.")
            except Exception as e:
                print(f"torch.compile nicht verfügbar: {e}")

        self.optimal_threshold = None
       
        self.test_loader = test_loader
        self.trainings_id = trainings_id
        self.path_to_student_weight = path_to_student_weight

        self.model_name = self._get_model_name()

        # Lade vorhandene Gewichte, falls angegeben
        if path_to_student_weight is not None:
            if not os.path.exists(path_to_student_weight):
                raise FileNotFoundError(f"Pfad existiert nicht: {path_to_student_weight}")

            print(f"Lade Student-Gewichte von: {path_to_student_weight}")
            weights = torch.load(path_to_student_weight, map_location=self.device)
            self.model.load_trainable_state_dict(weights)

        # Teacher-Netzwerke immer einfrieren (Eval-Modus)
        if getattr(self.model, "teacher_model", None) is not None:
            self.model.teacher_model.eval()
        if getattr(self.model, "teacher_full", None) is not None:
            self.model.teacher_full.eval()

        is_mvtec = self.config["dataset"]["name"] == "MVTecAD"
        self.dataset_identifier = (
            f"{self.config['dataset']['name']}_{self.config['dataset']['class']}"
            if is_mvtec
            else self.config["dataset"]["name"]
        )

        # Erstelle Ausgabeordner und speichere die Konfiguration
        if inferenz:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
            model_folder = self._get_model_name()
            self.run_base_path = os.path.join(
                self.output_dir,
                self.dataset_identifier,
                model_folder,
                self.trainings_id,
            )
            os.makedirs(self.run_base_path, exist_ok=True)

            config_save_path = os.path.join(self.run_base_path, f"Config_{self.model_name}.yaml")
            try:
                with open(config_save_path, "w") as f:
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False, indent=4)
                    print(f"Konfiguration gespeichert: {config_save_path}")
            except Exception as e:
                print(f"Fehler beim Speichern der Konfiguration: {e}")

            self._print_inference_config()

    def _print_inference_config(self):
        """Druckt eine Übersicht der aktuellen Inferenz-Einstellungen."""
        print("\n" + "=" * 50)
        print("INFERENZ-KONFIGURATION")
        print("=" * 50)

        if self.is_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  Device: {self.device} ({gpu_name}, {gpu_mem:.1f} GB)")
        else:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            print(f"  Device: {self.device} ({cpu_count} CPU Cores)")

        print(f"\n  Optimierungen:")
        print(f"    {'Aktiviert' if self.actual_memory_format == torch.channels_last else 'Deaktiviert'} Channels Last")
        print(f"    {'Aktiviert' if self.use_amp else 'Deaktiviert'} Mixed Precision (AMP)")
        print(f"    {'Aktiviert' if self.non_blocking else 'Deaktiviert'} Async Data Transfer")
        print("=" * 50 + "\n")

    def evaluate(self, detailed_profiling=False, measure_memory=False):
        """Startet den Evaluierungsprozess.

        Args:
            detailed_profiling (bool, optional): Ob eine detaillierte Performance-Analyse durchgeführt werden soll.
            measure_memory (bool, optional): Ob der maximale Speicherbedarf gemessen werden soll.

        Returns:
            Gibt Ergebnisse basierend auf dem gewählten Modus zurück.
        """
        if detailed_profiling:
            print("\n" + "=" * 60)
            print("LAUF 1/2: Detailliertes Timing-Breakdown")
            print("=" * 60)
            timing_inference_time = self._evaluate_with_timing()

            print("\n" + "=" * 60)
            print("LAUF 2/2: PyTorch Profiler & Map-Speicherung")
            print("=" * 60)
            labels, scores = self.evaluate_with_PytorchProfiler()
            return labels, scores, timing_inference_time
        else:
            return self._evaluate_fast(measure_memory=measure_memory)

    def evaluate_per_epoch(self):
        """Schnelle Evaluierungsschleife für Zwischenchecks während des Trainings.

        Returns:
            tuple: (AUROC Score, AUPR Score, Gesamte Inferenzzeit in Sekunden)
        """
        self.model.eval()
        get_labels = []
        get_anomaly_scores = []
        total_inference_time = 0.0

        with torch.inference_mode():
            for images, _, _, labels in tqdm(self.test_loader, desc="Evaluating", leave=False):
                img_t = images.to(
                    self.device,
                    memory_format=self.actual_memory_format,
                    non_blocking=self.non_blocking,
                )

                if self.is_gpu:
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    anomaly_map = self.model.anomaly_map(
                        img_t,
                        self.config.get("model_settings", {}).get("use_original_paper", True),
                    )
                    
                    # Der stärkste Fehlerwert im Bild bestimmt den Gesamt-Score
                    scores = torch.amax(anomaly_map, dim=(1, 2))

                if self.is_gpu:
                    torch.cuda.synchronize()
                total_inference_time += time.perf_counter() - start_time

                get_labels.extend(labels.cpu().numpy().tolist())
                get_anomaly_scores.extend(scores.detach().cpu().numpy())

        if not get_labels or not get_anomaly_scores:
            return 0.0, 0.0, total_inference_time

        return (
            roc_auc_score(get_labels, get_anomaly_scores),
            average_precision_score(get_labels, get_anomaly_scores),
            total_inference_time,
        )

    def _evaluate_fast(self, measure_memory=False):
        """Auf Durchsatz optimierte Inferenz ohne Zwischenmessungen.

        Args:
            measure_memory (bool): Aktiviert die Speichermessung.

        Returns:
            tuple: (Array mit wahren Labels, Array mit vorhergesagten Scores, Gesamte Inferenzzeit)
        """
        self.model.eval()
        get_labels = []
        get_anomaly_scores = []
        total_inference_time = 0.0
        
        peak_memory_mb = 0.0
        process = None

        if measure_memory:
            if self.is_gpu:
                torch.cuda.reset_peak_memory_stats(self.device)
            else:
                import psutil
                process = psutil.Process(os.getpid())
                peak_memory_mb = process.memory_info().rss / (1024 * 1024)

        with torch.inference_mode():
            # Warmup (Erste Durchläufe werden für realistischere Zeiten ignoriert)
            warmup_batch = next(iter(self.test_loader))
            img_warmup = warmup_batch[0].to(self.device, memory_format=self.actual_memory_format)
            for _ in range(5):
                _ = self.model.anomaly_map(img_warmup, use_original_paper=self.config.get("model_settings", {}).get("use_original_paper", True))
            if self.is_gpu:
                torch.cuda.synchronize()

            for images, names, _, labels in tqdm(self.test_loader, desc="Evaluating", leave=False):
                img_t = images.to(
                    self.device,
                    memory_format=self.actual_memory_format,
                    non_blocking=self.non_blocking,
                )

                if self.is_gpu:
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    anomaly_map = self.model.anomaly_map(
                        img_t,
                        self.config.get("model_settings", {}).get("use_original_paper", True),
                    )

                scores = torch.amax(anomaly_map, dim=(1, 2))

                if self.is_gpu:
                    torch.cuda.synchronize()
                total_inference_time += time.perf_counter() - start_time

                get_labels.append(labels.cpu())
                get_anomaly_scores.append(scores.detach().cpu())
                
                # CPU-Speicher manuell überwachen
                if measure_memory and self.is_cpu:
                    current_mem = process.memory_info().rss / (1024 * 1024)
                    if current_mem > peak_memory_mb:
                        peak_memory_mb = current_mem
        
        if measure_memory:
            if self.is_gpu:
                peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            
            self.peak_memory_mb = peak_memory_mb
            mem_type = "VRAM" if self.is_gpu else "RAM"
            print(f"\n  Peak {mem_type} während _evaluate_fast: {self.peak_memory_mb:.2f} MB")

        if not get_labels or not get_anomaly_scores:
            return np.array([]), np.array([]), 0.0

        labels_concat = torch.cat(get_labels).numpy()
        scores_concat = torch.cat(get_anomaly_scores).numpy()
        
        return labels_concat, scores_concat, total_inference_time

    def _evaluate_with_timing(self):
        """Diagnostischer Analyse-Modus zur Isolierung der Teil-Latenzen.
        
        Achtung: Erfordert ständige Synchronisation, was die GPU ausbremst.

        Returns:
            float: Die gemessene Gesamtlaufzeit.
        """
        self.model.eval()
        get_labels = []
        get_anomaly_scores = []

        time_data_loading = 0.0
        time_to_device = 0.0
        time_stem = 0.0
        time_teacher = 0.0
        time_student = 0.0
        time_anomaly_compute = 0.0
        time_upsampling = 0.0
        time_scoring = 0.0

        use_original_paper = self.config.get("model_settings", {}).get("use_original_paper", True)
        stem_is_active = self._is_stem_layer_active()
        num_images = 0

        batch_iterator = tqdm(self.test_loader, desc="Evaluating (Timing)", leave=False)
        data_start = time.time()

        with torch.inference_mode():
            warmup_batch = next(iter(self.test_loader))
            img_warmup = warmup_batch[0].to(self.device, memory_format=self.actual_memory_format)
            for _ in range(5):
                _ = self.model.anomaly_map(img_warmup, use_original_paper=use_original_paper)
            if self.is_gpu:
                 torch.cuda.synchronize()
                 
            for images, _, _, labels in batch_iterator:
                batch_size = images.shape[0]
                num_images += batch_size

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

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    t0 = time.time()
                    if stem_is_active:
                        stem_output = self.model.stem_model(img_t)
                        if self.is_gpu:
                            torch.cuda.synchronize()
                        time_stem += time.time() - t0
                    else:
                        stem_output = img_t

                    t0 = time.time()
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

                    # Passt asymmetrische Feature-Maps (z.B. Teacher vs Student) in der Größe an
                    aligned_student_maps = []
                    for i, (t_map, s_map) in enumerate(zip(teacher_feature_maps, student_feature_maps)):
                        if self.model.is_asymmetric:
                            s_map = self.model.projection_heads[i](s_map)
                        if s_map.shape[-2:] != t_map.shape[-2:]:
                            s_map = F.interpolate(
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
                    _, _, img_height, img_width = img_t.shape
                    anomaly_map = torch.ones(
                        (batch_size, img_height, img_width),
                        device=img_t.device,
                        dtype=img_t.dtype,
                    )

                    level_maps = []
                    for t_map, s_map in zip(teacher_feature_maps, aligned_student_maps):
                        if use_original_paper:
                            level_anomaly = compute_level_anomaly_mse(t_map, s_map)
                        else:
                            level_anomaly = compute_level_anomaly_cosine(t_map, s_map)
                        level_maps.append(level_anomaly)

                    if self.is_gpu:
                        torch.cuda.synchronize()
                    time_anomaly_compute += time.time() - t0

                    t0 = time.time()
                    for level_anomaly in level_maps:
                        level_upsampled = F.interpolate(
                            level_anomaly.unsqueeze(1),
                            size=(img_height, img_width),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)
                        anomaly_map = anomaly_map * level_upsampled

                    if self.is_gpu:
                        torch.cuda.synchronize()
                    time_upsampling += time.time() - t0

                t0 = time.time()
                scores = torch.amax(anomaly_map, dim=(1, 2))
                if self.is_gpu:
                    torch.cuda.synchronize()
                time_scoring += time.time() - t0

                get_labels.extend(labels.cpu().numpy().tolist())
                get_anomaly_scores.extend(scores.detach().cpu().numpy())
                data_start = time.time()

        total = (
            time_data_loading + time_to_device + time_stem + time_teacher +
            time_student + time_anomaly_compute + time_upsampling + time_scoring
        )

        stem_status = "SHARED" if stem_is_active else "IDENTITY"
        method_status = "MSE (Original)" if use_original_paper else "COSINE (Optimiert)"

        print(f"\n  ⏱️ Inference Timing Breakdown ({num_images} Bilder):")
        print(f"     {'Component':<20} {'Time':>8} {'Percent':>8}  {'Info':<20}")
        print(f"     {'-'*60}")
        print(f"     {'Data Loading':<20} {time_data_loading:>7.3f}s {100*time_data_loading/total:>7.1f}%")
        print(f"     {'To Device':<20} {time_to_device:>7.3f}s {100*time_to_device/total:>7.1f}%")
        print(f"     {'Stem Layer':<20} {time_stem:>7.3f}s {100*time_stem/total:>7.1f}%  ← {stem_status}")
        print(f"     {'Teacher Forward':<20} {time_teacher:>7.3f}s {100*time_teacher/total:>7.1f}%")
        print(f"     {'Student Forward':<20} {time_student:>7.3f}s {100*time_student/total:>7.1f}%")
        print(f"     {'Anomaly Compute':<20} {time_anomaly_compute:>7.3f}s {100*time_anomaly_compute/total:>7.1f}%  ← {method_status}")
        print(f"     {'Upsampling':<20} {time_upsampling:>7.3f}s {100*time_upsampling/total:>7.1f}%")
        print(f"     {'Scoring (amax)':<20} {time_scoring:>7.3f}s {100*time_scoring/total:>7.1f}%")
        print(f"     {'-'*60}")
        print(f"     {'TOTAL':<20} {total:>7.3f}s")

        io_time = time_data_loading + time_to_device
        compute_time = time_stem + time_teacher + time_student + time_anomaly_compute + time_upsampling
        ratio = io_time / compute_time if compute_time > 0 else float("inf")
        device_label = "GPU" if self.is_gpu else "CPU"

        print(f"\n      Analyse:")
        print(f"        I/O Zeit:        {io_time:.3f}s ({100*io_time/total:.1f}%)")
        print(f"        {device_label} Compute:    {compute_time:.3f}s ({100*compute_time/total:.1f}%)")
        bound_msg = "(I/O-bound!)" if ratio > 2 else f"({device_label}-bound)" if ratio < 0.5 else "(balanced)"
        print(f"        I/O:{device_label} Ratio:   {ratio:.2f}x {bound_msg}")

        print(f"\n      Pro-Bild Statistiken:")
        print(f"        Durchschnitt:    {1000*total/num_images:.2f} ms/Bild")
        print(f"        Throughput:      {num_images/total:.1f} Bilder/s")

        if get_labels and get_anomaly_scores:
            auroc = roc_auc_score(get_labels, get_anomaly_scores)
            aupr = average_precision_score(get_labels, get_anomaly_scores)
            print(f"\nAUROC: {auroc:.4f}")
            print(f"AUPR:  {aupr:.4f}")

        return total

    def evaluate_with_PytorchProfiler(self):
        """Nutzt den PyTorch Profiler für detaillierte Leistungsanalysen.
        
        Exportiert einen Chrome Trace (.json).

        Returns:
            tuple: (Array mit wahren Labels, Array mit vorhergesagten Scores)
        """
        self.peak_memory_mb = self.measure_memory_usage()
        self.gmacs, self.gflops, self.mparams = self.analyze_model_complexity()

        self.model.eval()
        get_labels = []
        get_anomaly_scores = []

        if self.is_gpu:
            profiler_activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        else:
            profiler_activities = [ProfilerActivity.CPU]

        profiler_ctx = profile(
            activities=profiler_activities, record_shapes=True, with_stack=True
        )

        use_original_paper = self.config.get("model_settings", {}).get("use_original_paper", True)

        # Warmup außerhalb des Profilers, um Verzerrungen zu vermeiden
        with torch.inference_mode():
            warmup_batch = next(iter(self.test_loader))
            img_warmup = warmup_batch[0].to(self.device, memory_format=self.actual_memory_format)
            for _ in range(5):
                _ = self.model.anomaly_map(img_warmup, use_original_paper=use_original_paper)
            if self.is_gpu:
                torch.cuda.synchronize()

        with profiler_ctx as prof:
            with torch.inference_mode():
                for images, names, _, labels in tqdm(self.test_loader, desc="Evaluating (Profiler)", leave=False):
                    img_t = images.to(
                        self.device,
                        memory_format=self.actual_memory_format,
                        non_blocking=self.non_blocking,
                    )

                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                        profiler_label = "stfpm_anomaly_map"
                        with record_function(profiler_label):
                            anomaly_map = self.model.anomaly_map(img_t, use_original_paper=use_original_paper)

                        scores = torch.amax(anomaly_map, dim=(1, 2))

                    get_labels.append(labels.cpu())
                    get_anomaly_scores.append(scores.detach().cpu())

        print("\n--- PyTorch Profiler-Analyse ---")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

        profiler_output_path = os.path.join(
            self.run_base_path,
            f"{self.dataset_identifier}_{self.model_name}_{self.trainings_id}_profiler_trace.json",
        )
        prof.export_chrome_trace(profiler_output_path)
        print(f"  -> Profiler-Trace gespeichert in: {profiler_output_path}")

        if not get_labels or not get_anomaly_scores:
            return np.array([]), np.array([])

        labels_concat = torch.cat(get_labels).numpy()
        scores_concat = torch.cat(get_anomaly_scores).numpy()
        return labels_concat, scores_concat

    def _is_stem_layer_active(self):
        """Prüft, ob Teacher und Student einen gemeinsamen Eingangsblock (Stem) nutzen.

        Returns:
            bool: True, wenn ein gemeinsamer Stem existiert, sonst False.
        """
        if self.model.stem_model is None:
            return False
        if isinstance(self.model.stem_model, nn.Identity):
            return False
        stem_params = sum(p.numel() for p in self.model.stem_model.parameters())
        return stem_params > 0

    def create_inference_summary(self, json_file, y_true, y_pred_scores, inference_time):
        """Erstellt eine JSON-Zusammenfassung der Konfiguration und Evaluierungs-Metriken.

        Args:
            json_file (dict): Basis-JSON mit Architekturen und Pfaden.
            y_true (list/array): Wahre Labels.
            y_pred_scores (list/array): Vorhergesagte Anomalie-Scores.
            inference_time (float): Gesamte Inferenzzeit.
        """
        if len(y_true) > 0 and len(y_pred_scores) > 0:
            auroc_score = roc_auc_score(y_true, y_pred_scores)
            aupr_score = average_precision_score(y_true, y_pred_scores)
        else:
            auroc_score = 0.0
            aupr_score = 0.0
            self.optimal_threshold = 0.0

        weight_key = "best_student"
        weights_path = json_file.get("weight_paths", {})

        model_size_mb = (
            os.path.getsize(weights_path.get(weight_key)) / (1024 * 1024)
            if weights_path and weights_path.get(weight_key) and os.path.exists(weights_path.get(weight_key))
            else 0.0
        )

        if self.config.get("model_settings", {}).get("is_asymmetric", False):
            model_arch = {
                "teacher_architecture": json_file.get("teacher_architecture"),
                "student_architecture": json_file.get("student_architecture"),
            }
        else:
            model_arch = {
                "model_architecture": json_file.get("model_architecture")
            }

        inference_summary = {
            "dataset_name": json_file.get("dataset/dataloader", {}).get("dataset_name"),
            "model_used": {
                "training_id": json_file.get("training_id"),
                "model_weights_path": self.path_to_student_weight,
                **model_arch,
            },
            "performance_metrics": {
                "auroc_score": auroc_score,
                "aupr_score": aupr_score,
                "quantile_threshold": float(self.optimal_threshold) if self.optimal_threshold is not None else None,
                "total_inference_time_sec": inference_time,
                "avg_inference_time_per_image_ms": (
                    (inference_time / len(self.test_loader.dataset)) * 1000
                    if len(self.test_loader.dataset) > 0 else 0
                ),
                "model_size_mb": model_size_mb,
                "peak_ram_mb" if self.is_cpu else "peak_vram_mb": getattr(self, "peak_memory_mb", "N/A")
            },
            "model_complexity": {
                "student_gmacs": getattr(self, "gmacs", "N/A"),
                "student_gflops": getattr(self, "gflops", "N/A"),
                "student_mparams": getattr(self, "mparams", "N/A"),
                "teacher_gmacs": getattr(self, "teacher_gmacs", "N/A"),
                "teacher_gflops": getattr(self, "teacher_gflops", "N/A"),
                "teacher_mparams": getattr(self, "teacher_mparams", "N/A"),
            },
        }

        inference_summary_path = os.path.join(self.run_base_path, "inference_summary.json")
        with open(inference_summary_path, "w") as f:
            json.dump(inference_summary, f, indent=4)
        print(f"Inference summary saved to {inference_summary_path}")

    def analyze_model_complexity(self):
        """Berechnet Parameteranzahl, FLOPs und MACs des Modells.

        Returns:
            tuple: (Student GMACs, Student GFLOPs, Student Parameter in Millionen)
        """
        img_size = self.config["dataset"]["img_size"]
        dummy_input = torch.randn(1, 3, img_size, img_size).to(self.device)

        analysis_input = (
            self.model.stem_model(dummy_input)
            if getattr(self.model, "stem_model", None) else dummy_input
        )

        def count_parameters(m):
            return sum(p.numel() for p in m.parameters())

        # Student auswerten
        model_to_analyse = self.model.student_model
        model_to_analyse.eval()

        params = count_parameters(model_to_analyse)
        mparams = params / 1e6

        try:
            flops = FlopCountAnalysis(model_to_analyse, analysis_input).total()
            gflops = flops / 1e9
            gmacs = gflops / 2.0  # 1 MAC entspricht i.d.R. 2 FLOPs
        except Exception:
            macs, _ = thop.profile(model_to_analyse, inputs=(analysis_input,), verbose=False)
            gmacs = macs / 1e9
            gflops = gmacs * 2.0

        print(f"\n--- Komplexitäts-Analyse (Student) ---")
        print(f"  Parameter: {mparams:.2f}M | GMACs: {gmacs:.2f} | GFLOPs: {gflops:.2f}")

        # Teacher auswerten (Referenz)
        self.model.teacher_model.eval()
        t_params = count_parameters(self.model.teacher_model)
        self.teacher_mparams = t_params / 1e6

        try:
            t_flops = FlopCountAnalysis(self.model.teacher_model, analysis_input).total()
            self.teacher_gflops = t_flops / 1e9
            self.teacher_gmacs = self.teacher_gflops / 2.0
        except Exception:
            t_macs, _ = thop.profile(self.model.teacher_model, inputs=(analysis_input,), verbose=False)
            self.teacher_gmacs = t_macs / 1e9
            self.teacher_gflops = self.teacher_gmacs * 2.0

        print(f"\n--- Komplexitäts-Analyse (Teacher, Referenz) ---")
        print(f"  Parameter: {self.teacher_mparams:.2f}M | GMACs: {self.teacher_gmacs:.2f} | GFLOPs: {self.teacher_gflops:.2f}")

        return gmacs, gflops, mparams

    def measure_memory_usage(self):
        """Ermittelt den maximalen RAM- oder VRAM-Verbrauch während der Inferenz.

        Returns:
            float oder str: Peak Memory in MB, oder "N/A" wenn der Ladevorgang fehlschlägt.
        """
        if self.is_cpu:
            print("\nSpeichermessung (CPU)...")
            try:
                images, _, _, _ = next(iter(self.test_loader))
                img_t = images[:1].to(self.device, memory_format=self.actual_memory_format)
            except StopIteration:
                return "N/A"

            def inference_step():
                self.model.eval()
                with torch.inference_mode():
                    _ = self.model.anomaly_map(
                        img_t,
                        self.config.get("model_settings", {}).get("use_original_paper", True),
                    )

            peak_memory_mb = memory_usage((inference_step, (), {}), interval=0.1, max_usage=True)
            print(f"  Peak RAM: {peak_memory_mb:.2f} MB")
            return peak_memory_mb
        else:
            print("\nSpeichermessung (GPU)...")
            torch.cuda.reset_peak_memory_stats(self.device)
            try:
                images, _, _, _ = next(iter(self.test_loader))
                img_t = images[:1].to(self.device, memory_format=self.actual_memory_format)
            except StopIteration:
                return "N/A"

            self.model.eval()
            with torch.inference_mode():
                _ = self.model.anomaly_map(
                    img_t,
                    self.config.get("model_settings", {}).get("use_original_paper", True),
                )

            max_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            print(f"  Peak VRAM: {max_memory_mb:.2f} MB")
            return max_memory_mb

    def _get_model_name(self) -> str:
        """Generiert einen Namen aus der Architektur.

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

    def calculate_threshold_on_val_set(self, val_loader, fpr=0.05):
        """Berechnet den Schwellenwert auf Basis einer gewünschten False-Positive-Rate.

        Args:
            val_loader: DataLoader für das Validierungsset (nur fehlerfreie Bilder).
            fpr (float, optional): Akzeptierte False-Positive-Rate. Standard ist 0.05 (5%).

        Returns:
            float: Der berechnete Schwellenwert.
        """
        self.model.eval()
        scores = []
        with torch.inference_mode():
            for images, _, _, _ in tqdm(val_loader, desc="Berechne Validierungs-Threshold", leave=False):
                img_t = images.to(self.device, memory_format=self.actual_memory_format)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    anomaly_map = self.model.anomaly_map(
                        img_t,
                        use_original_paper=self.config.get("model_settings", {}).get("use_original_paper", True),
                    )
                    
                    batch_scores = torch.amax(anomaly_map, dim=(1, 2))
                scores.append(batch_scores.detach().cpu())

        final_scores = torch.cat(scores).numpy()
        target_quantile = (1.0 - fpr) * 100
        threshold = np.percentile(final_scores, target_quantile)
        return float(threshold)