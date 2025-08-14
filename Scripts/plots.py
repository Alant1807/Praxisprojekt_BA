import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from PIL import Image

def plot_auroc_scores(metrics, config, plots_save_dir_for_run, base_filename):
    """
    Plottet die AUROC-Scores über die Epochen.

    Args:
        metrics (dict): Ein Dictionary, das die AUROC-Scores enthält.
        config (dict): Konfiguration des Modells, einschließlich der Architektur und der Anzahl der Epochen.
        plots_save_dir_for_run (str): Verzeichnis, in dem die Plots gespeichert werden sollen.
        base_filename (str): Basisname für die gespeicherten Plot-Dateien.
    """
    
    plt.figure(figsize=(10, 6))
    
    epochs_range = range(config['epochs'])

    sns.lineplot(x=epochs_range, y=metrics["auroc_scores"], label="AUROC Score")

    plt.title(f"AUROC Scores over Epochs - {config['model']['architecture']}")
    plt.xlabel("Epoch") 
    plt.ylabel("AUROC Score")
    plt.legend()

    if config['epochs'] > 0:
        if config['epochs'] > 20:
            step = max(1, config['epochs'] // 10)
            plt.xticks(ticks=np.arange(0, config['epochs'], step))
        else:
            plt.xticks(ticks=epochs_range)

    plt.grid(True)
    plt.tight_layout()

   
    auroc_scores_path = os.path.join(plots_save_dir_for_run, "auroc_scores")
    os.makedirs(auroc_scores_path, exist_ok=True)
    plt.savefig(os.path.join(auroc_scores_path, f"{base_filename}_auroc.png"))

    plt.close()

def plot_loss_curves(metrics, config, plots_save_dir_for_run, base_filename):
    """
    Plottet die Verlustkurven für das Training.

    Args:
        metrics (dict): Ein Dictionary, das die Trainingsverluste enthält.
        config (dict): Konfiguration des Modells, einschließlich der Architektur und der Anzahl der Epochen.
        plots_save_dir_for_run (str): Verzeichnis, in dem die Plots gespeichert werden sollen.
        base_filename (str): Basisname für die gespeicherten Plot-Dateien.
    """
    
    plt.figure(figsize=(10, 6))

    epochs_range = range(config['epochs'])

    sns.lineplot(x=epochs_range, y=metrics["train_loss"], label="Training Loss")

    plt.title(f"Loss Curves - {config['model']['architecture']}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    if config['epochs'] > 0:
        if config['epochs'] > 20:
            step = max(1, config['epochs'] // 10)
            plt.xticks(ticks=np.arange(0, config['epochs'], step))
        else:
            plt.xticks(ticks=epochs_range)

    plt.grid(True)
    plt.tight_layout()

    loss_curves_path = os.path.join(plots_save_dir_for_run, "loss_curves")
    os.makedirs(loss_curves_path, exist_ok=True)
    plt.savefig(os.path.join(loss_curves_path, f"{base_filename}_loss.png"))

    plt.close()

# In Praxisprojekt/Scripts/plots.py

def plot_heatmap(original_image, anomaly_map, save_path, title):
    """
    Erzeugt und speichert eine Gegenüberstellung des Originalbildes 
    und der Anomalie-Heatmap.

    Args:
        original_image (np.ndarray): Das Originalbild (als NumPy-Array).
        anomaly_map (np.ndarray): Die 2D-Anomalie-Karte.
        save_path (str): Der vollständige Pfad zum Speichern des Ausgabebildes.
        title (str): Der Basistitel für den Plot (z.B. der Dateiname).
    """
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Erstelle eine Figur mit zwei Subplots nebeneinander (1 Zeile, 2 Spalten)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- Erster Subplot: Nur das Originalbild ---
    axes[0].imshow(original_image)
    axes[0].set_title("Originalbild")
    axes[0].axis('off')  # Achsen ausblenden

    # --- Zweiter Subplot: Bild mit Heatmap-Überlagerung ---
    axes[1].imshow(original_image)  # Zuerst das Originalbild als Hintergrund
    # Dann die Heatmap mit Transparenz darüberlegen
    heatmap = axes[1].imshow(anomaly_map, cmap='jet', alpha=0.5)
    axes[1].set_title("Anomalie-Heatmap")
    axes[1].axis('off')  # Achsen ausblenden

    # Füge einen Farbbalken für den Heatmap-Subplot hinzu
    fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

    # Setze einen übergeordneten Titel für die gesamte Figur
    fig.suptitle(f"Vergleich für Bild: {title}", fontsize=16)

    # Passe das Layout an, damit alles gut aussieht, und speichere die Figur
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Platz für den Haupttitel schaffen
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)