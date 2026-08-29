import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from adjustText import adjust_text

    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


def find_pareto_front(df, x_col, y_col):
    """Berechnet die Pareto-Front (Minimierung von X, Maximierung von Y).

    Ein Punkt gehört zur Pareto-Front, wenn er bei gleicher Leistung (Y)
    am wenigsten Ressourcen (X) verbraucht, oder bei gleichen Ressourcen
    die höchste Leistung erzielt.

    Args:
        df (pd.DataFrame): Der DataFrame mit den Datenpunkten.
        x_col (str): Spaltenname für die X-Achse (wird minimiert, z.B. Inferenzzeit).
        y_col (str): Spaltenname für die Y-Achse (wird maximiert, z.B. AUROC).

    Returns:
        pd.DataFrame: Ein DataFrame, der nur die Punkte der Pareto-Front enthält.
    """
    df_copy = df.copy()
    df_copy[x_col] = pd.to_numeric(df_copy[x_col], errors="coerce")
    df_copy[y_col] = pd.to_numeric(df_copy[y_col], errors="coerce")

    # Unvollständige oder doppelte Daten entfernen
    df_copy = df_copy.dropna(subset=[x_col, y_col])
    df_copy = df_copy.drop_duplicates(subset=[x_col, y_col])

    # Sortiere aufsteigend nach X (Ressourcen) und absteigend nach Y (Leistung)
    df_copy = df_copy.sort_values(by=[x_col, y_col], ascending=[True, False])

    pareto_front = []
    max_y = -float("inf")

    # Ein Punkt ist nur dann auf der Pareto-Front, wenn sein Y-Wert besser ist 
    # als der bisherige Bestwert (da X bereits aufsteigend sortiert ist)
    for index, row in df_copy.iterrows():
        if row[y_col] > max_y:
            pareto_front.append(row)
            max_y = row[y_col]

    return pd.DataFrame(pareto_front)


def _get_model_label(row):
    """Erstellt ein gut lesbares Label für ein Modell aus einer DataFrame-Zeile.

    Besteht normalerweise aus dem Architekturnamen oder 'Teacher→Student'.
    Hängt außerdem die letzten 4 Ziffern der Trainings-ID an.

    Args:
        row (pd.Series): Eine Zeile aus dem DataFrame.

    Returns:
        str: Das formatierte Label für den Plot.
    """
    label = row.get("model_architecture", "Model")

    if label == "N/A" or pd.isna(label):
        teacher = row.get("teacher_architecture", "")
        student = row.get("student_architecture", "")
        if (
            teacher
            and student
            and str(teacher) != "N/A"
            and str(student) != "N/A"
        ):
            label = f"{teacher}→{student}"
        else:
            label = "Unknown"

    # Trainings-ID zur besseren Unterscheidung der Punkte anhängen
    training_id = row.get("training_id")
    if pd.notna(training_id) and str(training_id).strip() != "":
        t_id_str = str(training_id).strip()
        
        if t_id_str.endswith(".0"):
            t_id_str = t_id_str[:-2]
            
        suffix = t_id_str[-4:]
        label = f"{label}_{suffix}"

    return label


def plot_pareto(df, x_col, y_col, title, output_path):
    """Erstellt und speichert ein Scatter-Plot mit hervorgehobener Pareto-Front.

    Args:
        df (pd.DataFrame): Die zu plottenden Daten.
        x_col (str): Metrik für die X-Achse (Ressourcen).
        y_col (str): Metrik für die Y-Achse (Leistung).
        title (str): Überschrift des Diagramms.
        output_path (Path): Speicherpfad für das fertige PNG.
    """
    if df.empty:
        print(
            f"Warnung: Keine Daten für '{title}' vorhanden. Plot wird übersprungen."
        )
        return

    pareto_df = find_pareto_front(df, x_col, y_col)
    pareto_indices = set(pareto_df.index)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 10))

    # Alle Datenpunkte leicht ausgegraut plotten
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        ax=ax,
        color="grey",
        alpha=0.4,
        s=60,
        label="Alle Modelle (Dominiert)",
    )

    # Die effizienten Punkte (Pareto-Front) groß und rot hervorheben
    if not pareto_df.empty:
        sns.scatterplot(
            data=pareto_df,
            x=x_col,
            y=y_col,
            ax=ax,
            color="#d62728",
            s=120,
            label="Pareto-Front (Effizient)",
            zorder=5,
        )

        # Rote Linie durch die Pareto-Front ziehen
        ax.plot(
            pareto_df[x_col],
            pareto_df[y_col],
            color="#d62728",
            alpha=0.6,
            linestyle="--",
            linewidth=2,
        )

    texts = []

    # Beschriftungen (Labels) für die einzelnen Punkte hinzufügen
    for index, row in df.iterrows():
        is_pareto = index in pareto_indices
        label_text = _get_model_label(row)

        if is_pareto:
            t = ax.text(
                row[x_col],
                row[y_col],
                label_text,
                fontsize=10,
                weight="bold",
                color="black",
            )
        else:
            t = ax.text(
                row[x_col],
                row[y_col],
                label_text,
                fontsize=8,
                weight="normal",
                color="dimgrey",
                alpha=0.8,
            )

        texts.append(t)

    # Überschneidende Beschriftungen automatisch verschieben (falls Bibliothek installiert)
    if HAS_ADJUST_TEXT:
        adjust_text(
            texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
        )

    ax.set_title(title, fontsize=16, weight="bold", pad=20)

    xlabel = x_col.replace("_", " ").replace("avg", "Ø").title()
    ylabel = y_col.replace("_", " ").title()

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.legend(loc="lower right", frameon=True)

    y_min, y_max = df[y_col].min(), df[y_col].max()
    y_buffer = (y_max - y_min) * 0.1 if y_max != y_min else 0.05
    ax.set_ylim(bottom=y_min - y_buffer, top=min(y_max + y_buffer, 1.05))

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Diagramm gespeichert: {output_path}")

    # Speichere die rohen Pareto-Daten als separate CSV
    if not pareto_df.empty:
        pareto_csv_path = Path(
            str(output_path).replace(".png", "_pareto_data.csv")
        )
        pareto_df.to_csv(pareto_csv_path, index=False)
        print(f"Pareto-Front-Daten gespeichert: {pareto_csv_path}")


def execute_pareto_plot(base_path, output_path):
    """Sucht nach CSV-Dateien, fasst sie zusammen und erstellt verschiedene Pareto-Plots.

    Args:
        base_path (str | Path): Quellordner, in dem nach den Ergebnissen gesucht wird.
        output_path (str | Path): Zielordner für die generierten Diagramme.
    """
    base_path = Path(base_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_files = list(base_path.rglob("*_model_variant_summary.csv"))

    merged_files = []
    training_only_files = []
    inference_only_files = []

    # Unterteilt die gefundenen Dateien danach, ob sie Inferenzdaten, 
    # Trainingsdaten oder beides (merged) enthalten
    for f in summary_files:
        fname = f.name.lower()
        if "inference" in fname:
            inference_only_files.append(f)
        else:
            try:
                cols = pd.read_csv(f, nrows=0).columns.tolist()
                if "auroc_score" in cols and "training_id" in cols:
                    merged_files.append(f)
                else:
                    training_only_files.append(f)
            except Exception:
                training_only_files.append(f)

    if merged_files:
        print(
            f"\n{len(merged_files)} zusammengeführte CSV-Dateien gefunden (Training + Inferenz)."
        )
        files_to_use = merged_files
    elif training_only_files:
        print(
            f"\nWarnung: Keine zusammengeführten CSVs gefunden. "
            f"Nutze {len(training_only_files)} reine Trainings-CSVs (Inferenzmetriken fehlen!)."
        )
        files_to_use = training_only_files
    else:
        print(f"Fehler: Keine Ergebnis-Dateien in {base_path} gefunden.")
        return

    df_list = []
    for f in files_to_use:
        try:
            tmp_df = pd.read_csv(f)
            df_list.append(tmp_df)
        except Exception as e:
            print(f"Fehler beim Laden von {f}: {e}")

    if not df_list:
        return

    full_df = pd.concat(df_list, ignore_index=True)

    print(f"{len(full_df)} Ergebnisse aggregiert. Starte Plotting...")

    available_cols = set(full_df.columns.tolist())
    print(f"Verfügbare Spalten: {sorted(available_cols)}")

    # Liste von Diagrammen, die standardmäßig generiert werden sollen
    metrics_to_plot = [
        (
            "avg_inference_time_per_image_ms",
            "auroc_score",
            "Pareto: AUROC vs. Inferenzzeit",
        ),
        ("gflops", "auroc_score", "Pareto: AUROC vs. GFLOPs"),
        ("best_model_mb", "auroc_score", "Pareto: AUROC vs. Modellgröße"),
        (
            "training_duration",
            "auroc_score",
            "Pareto: AUROC vs. Trainingszeit",
        ),
        (
            "peak_ram_mb",
            "auroc_score",
            "Pareto: AUROC vs. Peak RAM",
        ),
    ]

    for x, y, title in metrics_to_plot:
        if x in available_cols and y in available_cols:
            plot_df = full_df.dropna(subset=[x, y]).copy()
            if plot_df.empty:
                print(
                    f"Überspringe '{title}': Keine gültigen Datenpunkte nach NaN-Filterung."
                )
                continue

            safe_name = f"pareto_{y}_vs_{x}".replace(" ", "_").lower() + ".png"
            plot_pareto(plot_df, x, y, title, output_path / safe_name)
        else:
            missing = []
            if x not in available_cols:
                missing.append(x)
            if y not in available_cols:
                missing.append(y)
            print(f"Überspringe '{title}': Fehlende Spalten: {missing}")

    overview_path = output_path / "pareto_all_data.csv"
    full_df.to_csv(overview_path, index=False)
    print(f"\nGesamtdaten exportiert: {overview_path}")