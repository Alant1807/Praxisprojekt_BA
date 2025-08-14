import os
import glob
import requests
import tarfile
import numpy as np
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset

SUBSET_URLS = {
        'bottle': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz',
        'cable': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz',
        'capsule': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz',
        'carpet': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz',
        'grid': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz',
        'hazelnut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz',
        'leather': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz',
        'metal_nut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz',
        'pill': 'https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz',
        'screw': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz',
        'tile': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz',
        'toothbrush': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz',
        'transistor': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz',
        'wood': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz',
        'zipper': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz'
    
}

class MVTecDataset(Dataset):
    """
    Dataset-Klasse für MVTec. Übernimmt:
      - Prüfung, ob Daten vorhanden sind. Falls nicht, Download + Entpacken.
      - Lädt Bildpfade aus base_path/cls/mode/[subfolders oder subfolders_except].
      - Erzeugt Label gemäß label_dict (z.B. 'good' -> 0, Rest -> 1).
      - Lädt die Bilder erst in __getitem__ (Lazy Loading).
      - Gibt im __getitem__ ein 6-Tuple zurück:
        (img_t, np_img, name, path, label, subcategory)
    """

    def __init__(
        self,
        img_size,
        base_path,
        cls,
        mode='train',
        subfolders=None,
        subfolders_except=None,
        label_dict=None,
        transform=None,
        download_if_missing=True
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

        # 1) Allgemeine Parameter
        self.base_path = os.path.abspath(base_path)
        self.cls = cls
        self.mode = mode
        self.transform = transform if transform is not None else self.default_transform()

        # 2) subfolders-Logik prüfen
        if subfolders and subfolders_except:
            raise ValueError("Bitte entweder 'subfolders' ODER 'subfolders_except' angeben, nicht beides.")
        self.subfolders = subfolders
        self.subfolders_except = subfolders_except

        # 3) label_dict setzen (falls None => default 'good': 0, rest: 1)
        self.label_dict = label_dict if label_dict is not None else {}

        # 4) Existenz des Datenordners prüfen (ggf. Download)
        self.cls_path = os.path.join(self.base_path, self.cls)
        if not os.path.isdir(self.cls_path):
            if download_if_missing:
                if self.cls in SUBSET_URLS:
                    print(f"[INFO] {self.cls_path} existiert nicht. Starte Download.")
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

        # 5) Bildpfade sammeln
        self.image_paths = self._collect_image_paths()

        # 6) Metadaten-Liste
        self.samples = []
        for p in self.image_paths:
            subf = p.replace("\\", "/").split("/")[-2]  # z.B. 'good', 'defect_1', ...
            if subf in self.label_dict:
                label = self.label_dict[subf]
            else:
                label = 0 if subf == 'good' else 1

            self.samples.append({
                "path": p,
                "subcategory": subf,
                "label": label,
                "name": f"{subf}_{os.path.basename(p)}"
            })

    def default_transform(self):
        return T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

    def _collect_image_paths(self):
        """
        Sammelt alle Bildpfade unter:
            base_path/cls/mode/[subfolders oder subfolders_except].
        Falls beides None => alle Subordner nehmen.
        """
        
        mode_path = os.path.join(self.cls_path, self.mode)
        if not os.path.isdir(mode_path):
            raise FileNotFoundError(f"Unterordner '{mode_path}' existiert nicht.")

        # Alle Ordner in mode_path ermitteln
        all_subfolders = [
            d for d in os.listdir(mode_path)
            if os.path.isdir(os.path.join(mode_path, d))
        ]

        # Falls subfolders angegeben: Nur diese
        if self.subfolders is not None:
            chosen_subfolders = [sf for sf in self.subfolders if sf in all_subfolders]
        # Falls subfolders_except angegeben: alles außer diese
        elif self.subfolders_except is not None:
            chosen_subfolders = [sf for sf in all_subfolders if sf not in self.subfolders_except]
        else:
            # Falls gar nichts angegeben => alle
            chosen_subfolders = all_subfolders

        # Nun sammeln wir die Bilder
        patterns = ["*.png", "*.jpg", "*.jpeg"]
        collected = []
        for sf in chosen_subfolders:
            sf_path = os.path.join(mode_path, sf)
            if not os.path.isdir(sf_path):
                print(f"[WARN] Subfolder {sf_path} existiert nicht. Überspringe.")
                continue

            for pat in patterns:
                full_pattern = os.path.join(sf_path, "**", pat)
                files = glob.glob(full_pattern, recursive=True)
                collected.extend(files)

        return sorted(collected)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        path = data["path"]
        label = data["label"]
        name = data["name"]

        # Lazy Loading
        pil_img = Image.open(path).convert("RGB")

        # Transformiertes Bild
        img_t = self.transform(pil_img)

        # -> Rückgabe analog Ursprungs-Implementierung:
        return (
            img_t,         # transformierter Tensor
            name,          # Dateiname
            path,          # voller Pfad
            label          # 0 (good) oder 1 (defect)
        )
    
def download_and_extract(url, target_dir):
    """
    Download einer tar.xz-Datei mit Fortschrittsanzeige via tqdm und anschließendes Entpacken.
    """
    
    os.makedirs(target_dir, exist_ok=True)

    filename = os.path.basename(url)  # z.B. 'bottle.tar.xz'
    download_path = os.path.join(target_dir, filename)

    # 1. Download mit Fortschrittsanzeige
    print(f"[INFO] Lade {url} herunter...")
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024

    with open(download_path, 'wb') as f, tqdm(
        desc=f"Downloading {filename}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for data in r.iter_content(block_size):
            f.write(data)
            bar.update(len(data))

    # 2. Entpacken mit Fortschrittsanzeige
    print(f"[INFO] Entpacke {download_path}...")
    with tarfile.open(download_path, 'r:xz') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting", unit="files"):
            tar.extract(member, path=target_dir)

    # 3. Archiv entfernen
    os.remove(download_path)
    print("[INFO] Download und Entpacken abgeschlossen.")