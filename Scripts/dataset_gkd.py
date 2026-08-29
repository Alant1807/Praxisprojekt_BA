"""
Dataset-Implementierungen für den GKD-Datensatz.
- GKDDataset: Lädt Bilder (PNGs) direkt von der Festplatte.
- GKDDatasetCached: Lädt vorbereitete Tensoren (.pt) für mehr Geschwindigkeit.
"""

import os
import glob
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch
import time
from tqdm import tqdm
import numpy as np

class GKDDataset(Dataset):
    """Lädt Bilder einzeln zur Laufzeit. Gut für Entwicklung und Experimente."""

    def __init__(self, img_size: int, data_path: str, mode: str = "train"):
        """Initialisiert das Dataset.

        Args:
            img_size (int): Die Zielgröße der Bilder (Breite und Höhe).
            data_path (str): Der Hauptpfad zu den Bilddaten.
            mode (str, optional): Der Modus ("train", "test" oder "val"). Standard ist "train".
        """
        super().__init__()
        self.img_size = img_size
        self.data_path = os.path.abspath(data_path)
        self.mode = mode

        self.transform = self.default_transform()

        image_paths = []
        labels = []
        subcategories = []

        # Prüft, ob ein 'train'-Ordner existiert (unterscheidet lokale vs. Cluster-Ordnerstruktur)
        has_train_folder = os.path.exists(os.path.join(self.data_path, "train"))

        if has_train_folder:
            if mode == "train":
                search_paths = os.path.join(self.data_path, "train", "good", "*.png")
                image_paths = sorted(glob.glob(search_paths))
                labels = [0] * len(image_paths)
                subcategories = ["good"] * len(image_paths)

            elif mode == "test":
                good_images = sorted(glob.glob(os.path.join(self.data_path, "test", "good", "*.png")))
                anomaly_images = sorted(glob.glob(os.path.join(self.data_path, "test", "anomaly", "*.png")))
                image_paths = good_images + anomaly_images
                labels = [0] * len(good_images) + [1] * len(anomaly_images)
                subcategories = ["good"] * len(good_images) + ["anomaly"] * len(anomaly_images)

            elif mode == "val":
                image_paths = sorted(glob.glob(os.path.join(self.data_path, "test", "good", "*.png")))
                labels = [0] * len(image_paths)
                subcategories = ["good"] * len(image_paths)
        else:
            if mode == "train":
                search_paths = os.path.join(self.data_path, "good", "*.png")
                image_paths = sorted(glob.glob(search_paths))
                labels = [0] * len(image_paths)
                subcategories = ["good"] * len(image_paths)

            elif mode == "test":
                good_images = sorted(glob.glob(os.path.join(self.data_path, "good", "*.png")))
                anomaly_images = sorted(glob.glob(os.path.join(self.data_path, "anomaly", "*.png")))
                image_paths = good_images + anomaly_images
                labels = [0] * len(good_images) + [1] * len(anomaly_images)
                subcategories = ["good"] * len(good_images) + ["anomaly"] * len(anomaly_images)

            elif mode == "val":
                image_paths = sorted(glob.glob(os.path.join(self.data_path, "good", "*.png")))
                labels = [0] * len(image_paths)
                subcategories = ["good"] * len(image_paths)

        self.samples = []
        for i in range(len(image_paths)):
            self.samples.append(
                {
                    "path": image_paths[i],
                    "label": labels[i],
                    "subcategory": subcategories[i],
                    # Eindeutiger Name für die spätere Zuordnung der Ergebnisse
                    "name": f"{subcategories[i]}_{os.path.basename(image_paths[i])}",
                }
            )

    def default_transform(self) -> T.Compose:
        """Nutzt GKD-spezifische Normalisierungswerte (nicht den ImageNet-Standard).

        Returns:
            T.Compose: Die kombinierte Bildtransformation.
        """
        return T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.4847, 0.4847, 0.4847], std=[0.3220, 0.3220, 0.3220]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Lädt und transformiert ein einzelnes Bild.

        Args:
            idx (int): Index des gewünschten Bildes.

        Returns:
            tuple: Ein Tupel aus (Bild-Tensor, Bildname, Dateipfad, Label).
        """
        sample = self.samples[idx]
        pil_img = Image.open(sample["path"]).convert("RGB")
        img_t = self.transform(pil_img)

        return (img_t, sample["name"], sample["path"], sample["label"])


class GKDDatasetCached(Dataset):
    """
    Lädt vorverarbeitete Tensoren (.pt Dateien). 
    Ist schneller als GKDDataset, erfordert aber, dass die Bilder vorher mit 
    'create_tensor_cache' umgewandelt wurden.
    """

    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        enable_timing: bool = False,
        load_to_ram: bool = False,
    ):
        """Initialisiert das gecachte Dataset.

        Args:
            data_path (str): Hauptpfad, der den Ordner 'cached_uint8' enthält.
            mode (str, optional): Der Modus ("train", "test" oder "val"). Standard ist "train".
            enable_timing (bool, optional): Wenn True, werden Ladezeiten gemessen. Standard ist False.
            load_to_ram (bool, optional): Wenn True, wird alles in den RAM geladen. Standard ist False.
        """
        super().__init__()
        self.data_root = os.path.abspath(data_path)
        self.cache_path = os.path.join(self.data_root, "cached_uint8")
        self.mode = mode
        self.enable_timing = enable_timing
        self.load_to_ram = load_to_ram

        if not os.path.isdir(self.cache_path):
            raise FileNotFoundError(f"Der Cache-Pfad '{self.cache_path}' existiert nicht.")

        self._load_time = 0.0
        self._process_time = 0.0
        self._call_count = 0

        self.normalize = T.Normalize(mean=[0.4847, 0.4847, 0.4847], std=[0.3220, 0.3220, 0.3220])

        tensor_paths = []
        labels = []
        subcategories = []

        # Prüft wieder auf Ordnerstruktur (Lokal vs. Cluster)
        has_train_folder = os.path.exists(os.path.join(self.cache_path, "train"))

        if has_train_folder:
            if mode == "train":
                tensor_paths = sorted(glob.glob(os.path.join(self.cache_path, "train", "good", "*.pt")))
                labels = [0] * len(tensor_paths)
                subcategories = ["good"] * len(tensor_paths)
            elif mode == "test":
                good_tensors = sorted(glob.glob(os.path.join(self.cache_path, "test", "good", "*.pt")))
                anomaly_tensors = sorted(glob.glob(os.path.join(self.cache_path, "test", "anomaly", "*.pt")))
                tensor_paths = good_tensors + anomaly_tensors
                labels = [0] * len(good_tensors) + [1] * len(anomaly_tensors)
                subcategories = ["good"] * len(good_tensors) + ["anomaly"] * len(anomaly_tensors)
            elif mode == "val":
                tensor_paths = sorted(glob.glob(os.path.join(self.cache_path, "test", "good", "*.pt")))
                labels = [0] * len(tensor_paths)
                subcategories = ["good"] * len(tensor_paths)
        else:
            if mode == "train":
                tensor_paths = sorted(glob.glob(os.path.join(self.cache_path, "good", "*.pt")))
                labels = [0] * len(tensor_paths)
                subcategories = ["good"] * len(tensor_paths)
            elif mode == "test":
                good_tensors = sorted(glob.glob(os.path.join(self.cache_path, "good", "*.pt")))
                anomaly_tensors = sorted(glob.glob(os.path.join(self.cache_path, "anomaly", "*.pt")))
                tensor_paths = good_tensors + anomaly_tensors
                labels = [0] * len(good_tensors) + [1] * len(anomaly_tensors)
                subcategories = ["good"] * len(good_tensors) + ["anomaly"] * len(anomaly_tensors)
            elif mode == "val":
                tensor_paths = sorted(glob.glob(os.path.join(self.cache_path, "good", "*.pt")))
                labels = [0] * len(tensor_paths)
                subcategories = ["good"] * len(tensor_paths)

        if not tensor_paths:
            print(f"WARNUNG: Keine .pt Dateien in {self.cache_path} für mode='{mode}' gefunden!")

        self.samples = []
        for i, pt_path in enumerate(tensor_paths):
            original_name = os.path.splitext(os.path.basename(pt_path))[0] + ".png"
            self.samples.append({
                "pt_path": pt_path,
                "label": labels[i],
                "name": f"{subcategories[i]}_{original_name}",
                "path": pt_path,
            })

        print(f"GKDDatasetCached ({mode}): {len(self.samples)} Tensoren-Pfade registriert.")

        self.ram_cache = None
        # Lädt bei Bedarf alles direkt in den Arbeitsspeicher (RAM) für maximale Geschwindigkeit
        if self.load_to_ram:
            print(f"Lade alle {mode}-Tensoren in den RAM (Inferenz-Modus)...")
            self.ram_cache = []
            for sample in tqdm(self.samples, desc=f"Caching {mode} to RAM"):
                # weights_only=True ist wichtig aus Sicherheitsgründen beim Laden
                img_uint8 = torch.load(sample["pt_path"], weights_only=True)
                img_t = self.normalize(img_uint8.float().div(255.0))
                self.ram_cache.append(img_t)
            print(f"✅ Alle {mode}-Tensoren erfolgreich im RAM gecacht!")
        else:
            print(f"Festplatten-Modus aktiv: Tensoren werden on-the-fly geladen (Trainings-Modus).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Gibt einen gecachten Bild-Tensor zurück.

        Args:
            idx (int): Index des gewünschten Tensors.

        Returns:
            tuple: Ein Tupel aus (Bild-Tensor, Bildname, Dateipfad, Label).
        """
        sample = self.samples[idx]

        if self.enable_timing:
            t0 = time.perf_counter()

        # Bild kommt aus dem RAM
        if self.load_to_ram:
            img_t = self.ram_cache[idx]

            if self.enable_timing:
                self._load_time += time.perf_counter() - t0
                self._process_time += 0.0
                self._call_count += 1

        # Bild wird von der Festplatte geladen
        else:
            img_uint8 = torch.load(sample["pt_path"], weights_only=True)

            if self.enable_timing:
                t1 = time.perf_counter()
                self._load_time += t1 - t0

            # Wandelt den platzsparenden uint8 Tensor in float32 um, was das Netz benötigt
            img_t = self.normalize(img_uint8.float().div(255.0))

            if self.enable_timing:
                self._process_time += time.perf_counter() - t1
                self._call_count += 1

        return (img_t, sample["name"], sample["pt_path"], sample["label"])

    def reset_timing(self):
        self._load_time = 0.0
        self._process_time = 0.0
        self._call_count = 0

    def get_timing_stats(self):
        if self._call_count == 0:
            return None
        return {
            "load_ms": (self._load_time / self._call_count) * 1000,
            "proc_ms": (self._process_time / self._call_count) * 1000,
            "total_ms": ((self._load_time + self._process_time) / self._call_count) * 1000,
            "count": self._call_count,
        }

    def print_timing_report(self, epoch=None):
        stats = self.get_timing_stats()
        if stats is None:
            return

        prefix = f"[Epoch {epoch}]" if epoch is not None else "[Dataset]"

        if self.load_to_ram:
            print(f"\n{prefix} RAM-Zugriffs-Stats (n={stats['count']}):")
            print(f"  ├─ RAM Fetch (Load): {stats['load_ms']:6.3f} ms/img")
            print(f"  ├─ CPU (Norm/Cast):  {stats['proc_ms']:6.3f} ms/img  <- (Sollte 0 sein)")
            print(f"  └─ TOTAL:            {stats['total_ms']:6.3f} ms/img")
        else:
            print(f"\n{prefix} Timing Stats (n={stats['count']}):")
            print(f"  ├─ Disk I/O (Load): {stats['load_ms']:6.3f} ms/img")
            print(f"  ├─ CPU (Norm/Cast): {stats['proc_ms']:6.3f} ms/img")
            print(f"  └─ TOTAL:           {stats['total_ms']:6.3f} ms/img")


def create_tensor_cache(data_path: str, img_size: int = 256) -> str:
    """Wandelt PNG-Bilder einmalig in .pt-Dateien um, um spätere Ladezeiten zu verkürzen.

    Args:
        data_path (str): Der Hauptpfad mit den Rohbildern.
        img_size (int, optional): Die Zielgröße der Bilder (Breite und Höhe). Standard ist 256.

    Returns:
        str: Der Pfad zum erstellten Cache-Ordner.
    """
    output_dir = os.path.join(data_path, "cached_uint8")

    print(f"\n{'='*60}")
    print(f"        TENSOR CACHE (UINT8) ERSTELLEN")
    print(f"{'='*60}")

    has_train_folder = os.path.exists(os.path.join(data_path, "train"))

    if has_train_folder:
        splits = [
            ("Lokal train/good", os.path.join(data_path, "train", "good"), os.path.join(output_dir, "train", "good")),
            ("Lokal test/good", os.path.join(data_path, "test", "good"), os.path.join(output_dir, "test", "good")),
            ("Lokal test/anomaly", os.path.join(data_path, "test", "anomaly"), os.path.join(output_dir, "test", "anomaly")),
        ]
    else:
        splits = [
            ("Cluster good", os.path.join(data_path, "good"), os.path.join(output_dir, "good")),
            ("Cluster anomaly", os.path.join(data_path, "anomaly"), os.path.join(output_dir, "anomaly")),
        ]

    for desc, input_folder, output_folder in splits:
        if not os.path.exists(input_folder):
            print(f"  Überspringe (nicht gefunden): {input_folder}")
            continue

        os.makedirs(output_folder, exist_ok=True)
        image_paths = sorted(glob.glob(os.path.join(input_folder, "*.png")))

        if not image_paths:
            print(f"  Keine Bilder in: {input_folder}")
            continue

        print(f"\n  Verarbeite '{desc}': {len(image_paths)} Bilder")

        for path in tqdm(image_paths, desc=f"  {desc}"):
            pil_img = Image.open(path).convert("RGB")
            pil_img = pil_img.resize((img_size, img_size), Image.BILINEAR)
            img_array = np.array(pil_img)

            # Bilder werden bewusst als uint8 (statt float32) gespeichert, 
            # da das ca. 75% Festplattenspeicher spart.
            tensor_uint8 = torch.from_numpy(img_array).permute(2, 0, 1).contiguous()

            filename = os.path.splitext(os.path.basename(path))[0] + ".pt"
            torch.save(tensor_uint8, os.path.join(output_folder, filename))

    print(f"\nCache erstellt in: {output_dir}")
    return output_dir