# =============================================================================
# IMPORTS
# =============================================================================
import os
import glob
import requests
import tarfile
import numpy as np
import torchvision.transforms as T
import time
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

# =============================================================================
# MVTec DOWNLOAD URLS
# =============================================================================
# URLs zu den einzelnen Fehlerklassen des MVTec AD Datensatzes.
# Werden genutzt, um fehlende Daten automatisch herunterzuladen.
SUBSET_URLS = {
    "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz",
}

# =============================================================================
# STANDARD DATASET (ON-THE-FLY)
# =============================================================================


class MVTecDataset(Dataset):
    """
    Dataset-Klasse für MVTec.
    Übernimmt:
      - Prüfung, ob Daten vorhanden sind. Falls nicht, Download + Entpacken.
      - Lädt Bildpfade aus base_path/cls/mode/[subfolders oder subfolders_except].
      - Erzeugt Label gemäß label_dict (z.B. 'good' -> 0, Rest -> 1).
      - Lädt die Bilder erst in __getitem__ (Lazy Loading).
      - Gibt im __getitem__ ein 4-Tuple zurück:
        (img_t, name, path, label)
    """

    def __init__(
        self,
        img_size,
        base_path,
        cls,
        mode="train",
        subfolders=None,
        subfolders_except=None,
        label_dict=None,
        transform=None,
        download_if_missing=True,
    ):
        """
        Parameter:
          base_path        : Hauptverzeichnis, wo die MVTec-Daten liegen sollen.
          cls              : Name der MVTec-Klasse ('bottle','cable',...).
          mode             : 'train' oder 'test' (oder andere Modi).
          subfolders       : Liste der Ordner in 'mode/', z.B. ['good','defect_1'].
                             - Falls None, werden alle Ordner genommen.
                             - Du kannst NICHT gleichzeitig subfolders und subfolders_except verwenden.
          subfolders_except: Liste der Ordner in 'mode/', die NICHT genommen werden.
                             - Falls None, werden alle Ordner genommen (sofern subfolders=None).
                             - Du kannst NICHT gleichzeitig subfolders und subfolders_except verwenden.
          label_dict       : z.B. {'good': 0, 'defect_1': 1, ...}
                             Falls None, gilt: 'good' -> 0, sonst -> 1.
          transform        : torchvision.Transform-Pipeline für das Bild.
          download_if_missing: Falls True, wird automatisch heruntergeladen, wenn nicht vorhanden.
        """

        super().__init__()

        self.img_size = img_size

        # 1) Allgemeine Parameter initialisieren
        self.base_path = os.path.abspath(base_path)
        self.cls = cls
        self.mode = mode
        # Standard-Transformation nutzen, falls der Nutzer keine eigene übergibt
        self.transform = (
            transform if transform is not None else self.default_transform()
        )

        # 2) Logik zur Filterung der Unterordner prüfen
        # Whitelist (subfolders) und Blacklist (subfolders_except) schließen sich gegenseitig aus.
        if subfolders and subfolders_except:
            raise ValueError(
                "Bitte entweder 'subfolders' ODER 'subfolders_except' angeben, nicht beides."
            )
        self.subfolders = subfolders
        self.subfolders_except = subfolders_except

        # 3) Label-Dictionary setzen (falls None => Standard: 'good' ist 0, alles andere (Anomalien) ist 1)
        self.label_dict = label_dict if label_dict is not None else {}

        # 4) Existenz des Datenordners prüfen und ggf. Download anstoßen
        self.cls_path = os.path.join(self.base_path, self.cls)
        if not os.path.isdir(self.cls_path):
            if download_if_missing:
                if self.cls in SUBSET_URLS:
                    print(
                        f"[INFO] {self.cls_path} existiert nicht. Starte Download."
                    )
                    download_and_extract(SUBSET_URLS[self.cls], self.base_path)
                else:
                    raise ValueError(
                        f"Keine Download-URL für Klasse '{self.cls}' gefunden. "
                        "Bitte SUBSET_URLS erweitern oder manuell Daten bereitstellen."
                    )
            else:
                raise FileNotFoundError(
                    f"Die Klasse '{self.cls}' liegt nicht unter {self.cls_path} und Download ist deaktiviert."
                )

        # 5) Bildpfade sammeln (Scannt das Dateisystem nach den entsprechenden Bildern)
        self.image_paths = self._collect_image_paths()

        # 6) Metadaten-Liste erstellen
        # Erzeugt ein Dictionary für jedes Bild, um später schnell in __getitem__ darauf zugreifen zu können
        self.samples = []
        for p in self.image_paths:
            # Extrahiert den Namen des übergeordneten Ordners (z.B. 'good' oder den Namen des Defekts)
            subf = p.replace("\\", "/").split("/")[-2]

            # Label-Zuweisung basierend auf dem Ordnernamen
            if subf in self.label_dict:
                label = self.label_dict[subf]
            else:
                label = 0 if subf == "good" else 1

            self.samples.append(
                {
                    "path": p,
                    "subcategory": subf,
                    "label": label,
                    # Kombinierter Name aus Kategorie und Dateiname zur eindeutigen Identifikation (z.B. in Plots)
                    "name": f"{subf}_{os.path.basename(p)}",
                }
            )

    def default_transform(self):
        """Standard-Bildvorverarbeitung (Resizing, Tensor-Cast, Normalisierung)"""
        return T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],  # ImageNet Standard-Werte für Mean
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],  # ImageNet Standard-Werte für Std
                ),
            ]
        )

    def _collect_image_paths(self):
        """
        Sammelt alle Bildpfade unter:
            base_path/cls/mode/[subfolders oder subfolders_except].
        Falls beides None => alle Subordner nehmen.
        """
        mode_path = os.path.join(self.cls_path, self.mode)
        if not os.path.isdir(mode_path):
            raise FileNotFoundError(
                f"Unterordner '{mode_path}' existiert nicht."
            )

        # Alle Ordner in mode_path ermitteln
        all_subfolders = [
            d
            for d in os.listdir(mode_path)
            if os.path.isdir(os.path.join(mode_path, d))
        ]

        # Falls Whitelist (subfolders) angegeben: Nur diese Ordner nutzen
        if self.subfolders is not None:
            chosen_subfolders = [
                sf for sf in self.subfolders if sf in all_subfolders
            ]
        # Falls Blacklist (subfolders_except) angegeben: alles außer diese
        elif self.subfolders_except is not None:
            chosen_subfolders = [
                sf for sf in all_subfolders if sf not in self.subfolders_except
            ]
        else:
            # Falls gar nichts angegeben => alle Unterordner nutzen
            chosen_subfolders = all_subfolders

        # Nun durchsuchen wir die ausgewählten Ordner nach Bildern
        patterns = ["*.png", "*.jpg", "*.jpeg"]
        collected = []
        for sf in chosen_subfolders:
            sf_path = os.path.join(mode_path, sf)
            if not os.path.isdir(sf_path):
                print(
                    f"[WARN] Subfolder {sf_path} existiert nicht. Überspringe."
                )
                continue

            # Glob sucht rekursiv nach allen Dateiendungen im aktuellen Unterordner
            for pat in patterns:
                full_pattern = os.path.join(sf_path, "**", pat)
                files = glob.glob(full_pattern, recursive=True)
                collected.extend(files)

        return sorted(collected)

    def __len__(self):
        """Gibt die Gesamtanzahl der gefundenen Bilder zurück."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Lädt das Bild zum Zeitpunkt der Anfrage (Lazy Loading) und wendet Transformationen an.
        Dies spart RAM, kostet aber I/O-Zeit während des Trainings.
        """
        data = self.samples[idx]
        path = data["path"]
        label = data["label"]
        name = data["name"]

        # Lazy Loading: Bild wird jetzt erst von der Festplatte gelesen
        pil_img = Image.open(path).convert("RGB")

        # Transformiertes Bild (Resize, Normalize)
        img_t = self.transform(pil_img)

        # Rückgabe des Tupels für das Training/Testing
        return (
            img_t,  # Transformierter Bild-Tensor
            name,  # Eindeutiger Dateiname
            path,  # Absoluter Pfad auf der Festplatte
            label,  # 0 (good) oder 1 (defect)
        )


# =============================================================================
# HILFSFUNKTIONEN FÜR DOWNLOAD
# =============================================================================


def download_and_extract(url, target_dir):
    """
    Download einer tar.xz-Datei mit Fortschrittsanzeige via tqdm und anschließendes Entpacken.
    """
    os.makedirs(target_dir, exist_ok=True)

    filename = os.path.basename(url)  # z.B. 'bottle.tar.xz'
    download_path = os.path.join(target_dir, filename)

    # 1. Download mit Fortschrittsanzeige (Streaming in Blöcken)
    print(f"[INFO] Lade {url} herunter...")
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024  # 1 KB Blöcke

    with open(download_path, "wb") as f, tqdm(
        desc=f"Downloading {filename}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in r.iter_content(block_size):
            f.write(data)
            bar.update(len(data))

    # 2. Entpacken mit Fortschrittsanzeige
    print(f"[INFO] Entpacke {download_path}...")
    with tarfile.open(download_path, "r:xz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting", unit="files"):
            tar.extract(member, path=target_dir)

    # 3. Archiv nach erfolgreichem Entpacken löschen, um Platz zu sparen
    os.remove(download_path)
    print("[INFO] Download und Entpacken abgeschlossen.")


# =============================================================================
# OPTIMIERTES DATASET (CACHED)
# =============================================================================


class MVTecDatasetCached(Dataset):
    """
    High-Performance Dataset für MVTec (lädt vorberechnete .pt Tensoren).
    Erwartet, dass 'create_tensor_cache_mvtec' zuvor ausgeführt wurde.

    Vorteil: Umgeht das teure Dekodieren von PNG-Dateien und das bilineare
    Resizing während des Trainings, was den I/O-Flaschenhals massiv reduziert.

    Struktur: base_path/cached_uint8/cls/mode/subfolder/*.pt
    """

    def __init__(
        self,
        base_path,
        cls,
        mode="train",
        subfolders=None,
        subfolders_except=None,
        label_dict=None,
        enable_timing=False,
    ):
        """
        Args:
            base_path (str): Pfad zum MVTec-Root (darin wird 'cached_uint8' erwartet).
            cls (str): Klasse (z.B. 'bottle').
            mode (str): 'train' oder 'test'.
            subfolders (list): Whitelist von Unterordnern (z.B. ['good']).
            subfolders_except (list): Blacklist von Unterordnern.
            label_dict (dict): Map von Ordnername zu Label.
            enable_timing (bool): Aktiviert Performance-Messung für Benchmarks.
        """
        super().__init__()

        self.cls = cls
        self.mode = mode
        self.enable_timing = enable_timing

        # Benchmarking Variablen zur Messung der Lade- und Verarbeitungszeit
        self._load_time = 0.0
        self._process_time = 0.0
        self._call_count = 0

        # Pfad zum Cache definieren
        # Wir gehen davon aus, dass der Cache unter base_path/cached_uint8 liegt
        self.cache_base = os.path.join(
            os.path.abspath(base_path), "cached_uint8"
        )
        self.cls_mode_path = os.path.join(self.cache_base, cls, mode)

        # Sicherstellen, dass der vorberechnete Cache existiert
        if not os.path.isdir(self.cls_mode_path):
            raise FileNotFoundError(
                f"Cache-Pfad nicht gefunden: {self.cls_mode_path}\n"
                f"Bitte zuerst 'create_tensor_cache_mvtec' ausführen!"
            )

        # Labels Logik und Ordner-Filterung
        self.label_dict = label_dict if label_dict is not None else {}
        self.subfolders = subfolders
        self.subfolders_except = subfolders_except

        # Validierung der Whitelist/Blacklist Argumente
        if subfolders and subfolders_except:
            raise ValueError(
                "Bitte entweder 'subfolders' ODER 'subfolders_except' angeben."
            )

        # --- Bildpfade sammeln ---
        self.samples = self._collect_samples()

        # --- Normalisierung ---
        # Da Tensoren bereits resized und gecroppt auf der Platte liegen,
        # brauchen wir hier nur noch die statistische Normalisierung (Mean/Std).
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        print(
            f"[{cls.upper()}] Cached Dataset ({mode}) initialisiert: {len(self.samples)} Samples."
        )

    def _collect_samples(self):
        """
        Sucht rekursiv nach vorberechneten .pt Dateien in den erlaubten Unterordnern.
        """
        samples = []

        # 1. Alle verfügbaren Unterordner (z.B. 'good', 'broken_large') im Cache finden
        all_subfolders = [
            d
            for d in os.listdir(self.cls_mode_path)
            if os.path.isdir(os.path.join(self.cls_mode_path, d))
        ]

        # 2. Ordner filtern (Whitelist/Blacklist anwenden)
        if self.subfolders is not None:
            chosen_subfolders = [
                sf for sf in self.subfolders if sf in all_subfolders
            ]
        elif self.subfolders_except is not None:
            chosen_subfolders = [
                sf for sf in all_subfolders if sf not in self.subfolders_except
            ]
        else:
            chosen_subfolders = all_subfolders

        # 3. Tensoren in den ausgewählten Ordnern sammeln
        for sf in chosen_subfolders:
            sf_path = os.path.join(self.cls_mode_path, sf)

            # Glob für PyTorch-Tensoren (.pt Dateien)
            pt_files = glob.glob(os.path.join(sf_path, "*.pt"))

            # Label aus dem Ordnernamen ableiten
            if sf in self.label_dict:
                label = self.label_dict[sf]
            else:
                label = 0 if sf == "good" else 1

            for pt_path in pt_files:
                # Ursprünglichen Dateinamen rekonstruieren (für sauberes Logging/Plotting)
                # Aus `bild_001.pt` wird virtuell wieder `good_bild_001.png`
                filename = os.path.splitext(os.path.basename(pt_path))[0]
                original_name = f"{sf}_{filename}.png"

                samples.append(
                    {
                        "path": pt_path,  # Pfad zur optimierten .pt Datei
                        "original_name": original_name,
                        "subcategory": sf,
                        "label": label,
                    }
                )

        # Sortieren garantiert deterministische Reihenfolge über verschiedene Läufe hinweg
        return sorted(samples, key=lambda x: x["path"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Effizientes Laden eines Datensatzes.
        """
        sample = self.samples[idx]
        pt_path = sample["path"]

        if self.enable_timing:
            # --- Profiling Modus ---
            # Misst exakt, wie lange der Festplattenzugriff im Vergleich zur CPU-Rechnung dauert
            t0 = time.perf_counter()
            # weights_only=True als Sicherheitsmaßnahme beim Laden von PyTorch-Dateien
            img_uint8 = torch.load(pt_path, weights_only=True)
            t1 = time.perf_counter()

            # CPU-Operation: uint8 (0-255) -> float32 (0.0-1.0) -> Normalize
            img_float = img_uint8.float().div(255.0)
            img_t = self.normalize(img_float)
            t2 = time.perf_counter()

            # Zeiten für die spätere Auswertung akkumulieren
            self._load_time += t1 - t0
            self._process_time += t2 - t1
            self._call_count += 1
        else:
            # --- Fast Path (Produktivbetrieb) ---
            img_uint8 = torch.load(pt_path, weights_only=True)
            # Effiziente Konvertierung und Normalisierung
            img_t = self.normalize(img_uint8.float().div(255.0))

        # Einheitliche Rückgabe-Struktur zu MVTecDataset
        return (img_t, sample["original_name"], pt_path, sample["label"])

    # --- Reporting Tools  ---
    def get_timing_stats(self):
        """Berechnet Durchschnittszeiten für Laden und Verarbeiten, falls Profiling aktiv war."""
        if self._call_count == 0:
            return None
        return {
            "load_ms": (self._load_time / self._call_count) * 1000,
            "proc_ms": (self._process_time / self._call_count) * 1000,
            "total_ms": (
                (self._load_time + self._process_time) / self._call_count
            )
            * 1000,
        }


# =============================================================================
# CACHE ERSTELLUNGS-FUNKTION (OFFLINE PREPROCESSING)
# =============================================================================


def create_tensor_cache_mvtec(data_path, img_size=256):
    """
    Erstellt den Cache und behält die MVTec-typische Unterordner-Struktur (Defekt-Typen) bei.

    Zweck: Bilder werden einmalig skaliert und als uint8-Tensoren gespeichert.
    Das spart während des Trainings extrem viel Zeit, da PNG-Dekodierung entfällt
    und uint8 auf der SSD 4x weniger Platz/Bandbreite benötigt als float32.
    """
    # Zielordner für die vorberechneten Daten
    output_base_dir = os.path.join(data_path, "cached_uint8")

    # MVTec Klassen iterieren (z.B. 'bottle', 'cable')
    # Wir nehmen an, data_path enthält direkt die Klassen-Ordner
    classes = [
        d
        for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and d != "cached_uint8"
    ]

    for cls in classes:
        # Jeden Modus einzeln verarbeiten
        for mode in ["train", "test"]:
            src_mode_path = os.path.join(data_path, cls, mode)

            # Überspringen, falls der Ordner fehlt
            if not os.path.exists(src_mode_path):
                continue

            # Wir suchen rekursiv nach allen Bildern (.png und .jpg)
            image_paths = glob.glob(
                os.path.join(src_mode_path, "**", "*.png"), recursive=True
            )
            image_paths += glob.glob(
                os.path.join(src_mode_path, "**", "*.jpg"), recursive=True
            )

            if not image_paths:
                continue

            # tqdm für visuelles Feedback in der Konsole
            for img_path in tqdm(image_paths, desc=f"Processing {cls}/{mode}"):
                try:
                    # 1. Relativen Pfad berechnen, um Struktur (z.B. '/good/' oder '/defect_1/') zu erhalten
                    rel_path = os.path.relpath(img_path, src_mode_path)

                    # 2. Zielpfad für die .pt Datei aufbauen
                    target_path = os.path.join(
                        output_base_dir,
                        cls,
                        mode,
                        os.path.splitext(rel_path)[0] + ".pt",
                    )

                    # Zielordner erstellen (exist_ok=True verhindert Fehler, wenn er schon da ist)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)

                    # 3. Bild laden & Vorverarbeiten (Offline)
                    # Sichert 3 Farbkanäle (RGB), auch bei Graustufen
                    img = Image.open(img_path).convert("RGB")

                    # Bilineares Resizing auf die Ziel-Auflösung des Netzes
                    img_resized = img.resize(
                        (img_size, img_size), resample=Image.BILINEAR
                    )
                    img_array = np.array(img_resized)

                    # 4. In Tensor umwandeln
                    # .permute(2,0,1) wandelt Format von HWC (PIL Standard) zu CHW (PyTorch Standard)
                    # .contiguous() stellt Speicher-Kontinuität für schnelles Lesen sicher
                    tensor_uint8 = (
                        torch.from_numpy(img_array)
                        .permute(2, 0, 1)
                        .contiguous()
                    )

                    # 5. Speichern auf Festplatte
                    torch.save(tensor_uint8, target_path)

                except Exception as e:
                    print(f"[ERROR] Fehler bei {img_path}: {e}")
