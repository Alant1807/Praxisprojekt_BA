import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple
import json


def plot_combined_loss_curves(run_configs: list, output_path: str = "combined_loss_curves.png"):
    """Zeichnet mehrere Loss-Kurven zum Vergleich in ein gemeinsames Diagramm.

    Args:
        run_configs (list): Liste mit Konfigurationen (müssen 'json_path' und 'label' enthalten).
        output_path (str, optional): Speicherpfad für das fertige Diagramm. Standard ist "combined_loss_curves.png".
    """
    default_colors = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd"]
    
    plt.figure(figsize=(12, 6))
    
    for i, config in enumerate(run_configs):
        json_path = config["json_path"]
        label = config["label"]
        color = config.get("color", default_colors[i % len(default_colors)])
        
        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        loss_per_epoch = summary.get("training_summary", {}).get("loss_per_epoch")
        
        if loss_per_epoch is None:
            print(f"Warnung: 'loss_per_epoch' nicht in {json_path} gefunden. Überspringe.")
            continue
        
        epochs = range(1, len(loss_per_epoch) + 1)
        
        final_loss = loss_per_epoch[-1]
        best_auroc = summary.get("evaluation", {}).get("best_auroc", "N/A")
        
        if isinstance(best_auroc, float):
            full_label = f"{label} (Loss: {final_loss:.3f}, AUROC: {best_auroc:.4f})"
        else:
            full_label = f"{label} (Loss: {final_loss:.3f})"
        
        plt.plot(epochs, loss_per_epoch, color=color, linewidth=2, label=full_label)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.title("Vergleich der Loss-Kurven: Scheduler & Lernraten-Strategien", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Berechne maximale Epochenanzahl für eine dynamische Anpassung der X-Achse
    max_epochs = max(len(summary.get("training_summary", {}).get("loss_per_epoch", [])) 
                     for config in run_configs 
                     for summary in [json.load(open(config["json_path"]))])
    if max_epochs > 20:
        plt.xticks(np.arange(0, max_epochs + 1, 10))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Kombinierter Plot gespeichert: {output_path}")


def plot_auroc_scores(metrics: dict, config: dict, plots_save_dir_for_run: str, base_filename: str):
    """Zeichnet die Entwicklung der Erkennungsleistung (AUROC) über alle Epochen.

    Hilft zu erkennen, ob das Modell noch lernt oder bereits overfittet.

    Args:
        metrics (dict): Wörterbuch mit den gesammelten Metriken (muss "auroc_scores" enthalten).
        config (dict): Konfigurations-Wörterbuch (für Modellnamen im Titel).
        plots_save_dir_for_run (str): Zielordner für das Diagramm.
        base_filename (str): Basisname der zu speichernden Datei.
    """
    plt.figure(figsize=(10, 6))

    epochs_range = range(len(metrics["auroc_scores"]))

    sns.lineplot(
        x=epochs_range, y=metrics["auroc_scores"], label="AUROC Score"
    )

    # Titel basierend auf dem Modelltyp anpassen (Asymmetrisch vs. Symmetrisch)
    if config.get("model_settings", {}).get("is_asymmetric", False):
        title = f"AUROC Scores - T: {config['teacher_model']['architecture']} / S: {config['student_model']['architecture']}"
    else:
        title = f"AUROC Scores over Epochs - {config['model']['architecture'] if config['model_settings']['model_type'] == 'stfpm' else config['model_settings']['model_type']}"

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("AUROC Score")
    plt.legend()

    # Verhindert, dass die Zahlen auf der X-Achse bei vielen Epochen überlappen
    num_epochs = len(metrics["auroc_scores"])
    if num_epochs > 0:
        if num_epochs > 20:
            step = max(1, num_epochs // 10)
            plt.xticks(ticks=np.arange(0, num_epochs, step))
        else:
            plt.xticks(ticks=epochs_range)

    plt.grid(True)
    plt.tight_layout()

    auroc_scores_path = os.path.join(plots_save_dir_for_run, "auroc_scores")
    os.makedirs(auroc_scores_path, exist_ok=True)
    plt.savefig(os.path.join(auroc_scores_path, f"{base_filename}_auroc.png"))

    # Wichtig: Speicher freigeben, um RAM-Lecks in langen Trainingsschleifen zu vermeiden
    plt.close()


def plot_loss_curves(metrics: dict, config: dict, plots_save_dir_for_run: str, base_filename: str):
    """Zeichnet die Entwicklung des Trainingsfehlers (Loss) über alle Epochen.

    Args:
        metrics (dict): Wörterbuch mit den gesammelten Metriken (muss "train_loss" enthalten).
        config (dict): Konfigurations-Wörterbuch (für Modellnamen im Titel).
        plots_save_dir_for_run (str): Zielordner für das Diagramm.
        base_filename (str): Basisname der zu speichernden Datei.
    """
    plt.figure(figsize=(10, 6))

    epochs_range = range(len(metrics["train_loss"]))

    sns.lineplot(
        x=epochs_range, y=metrics["train_loss"], label="Training Loss"
    )

    if config.get("model_settings", {}).get("is_asymmetric", False):
        title = f"Loss Curves - T: {config['teacher_model']['architecture']} / S: {config['student_model']['architecture']}"
    else:
        title = f"Loss Curves - {config['model']['architecture']}"

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    num_epochs = len(metrics["train_loss"])
    if num_epochs > 0:
        if num_epochs > 20:
            step = max(1, num_epochs // 10)
            plt.xticks(ticks=np.arange(0, num_epochs, step))
        else:
            plt.xticks(ticks=epochs_range)

    plt.grid(True)
    plt.tight_layout()

    loss_curves_path = os.path.join(plots_save_dir_for_run, "loss_curves")
    os.makedirs(loss_curves_path, exist_ok=True)
    plt.savefig(os.path.join(loss_curves_path, f"{base_filename}_loss.png"))

    plt.close()