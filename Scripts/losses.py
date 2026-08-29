"""
Verlustfunktionen (Loss) für STFPM.
Vergleicht die Feature-Maps von Teacher- und Student-Netzwerk.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# JIT-Kompilierung macht die Ausführung auf der GPU schneller.
@torch.jit.script
def compute_mse_loss_original(
    t_map: torch.Tensor, s_map: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Berechnet den MSE-Loss nach Original-Paper (mit L2-Normalisierung).

    Args:
        t_map (torch.Tensor): Feature-Map des Teacher-Modells.
        s_map (torch.Tensor): Feature-Map des Student-Modells.
        eps (float, optional): Kleiner Wert zur Vermeidung von Division durch Null. Standard ist 1e-12.

    Returns:
        torch.Tensor: Der berechnete Loss-Wert.
    """
    # Längen der Vektoren normieren, um nur die Struktur zu vergleichen
    t_norm = F.normalize(t_map, p=2.0, dim=1, eps=eps)
    s_norm = F.normalize(s_map, p=2.0, dim=1, eps=eps)

    diff = t_norm - s_norm

    return diff.pow(2).sum(dim=1).mean()


@torch.jit.script
def compute_cosine_loss_optimized(
    t_map: torch.Tensor, s_map: torch.Tensor
) -> torch.Tensor:
    """
    Schnellere Alternative zum Original-MSE-Loss mittels Cosine Similarity.

    Args:
        t_map (torch.Tensor): Feature-Map des Teacher-Modells.
        s_map (torch.Tensor): Feature-Map des Student-Modells.

    Returns:
        torch.Tensor: Der berechnete Loss-Wert.
    """
    # Berechnet Ähnlichkeit direkt ohne extra Normalisierungsschritte
    cos_sim = F.cosine_similarity(t_map, s_map, dim=1)

    # Umrechnung in einen Distanzwert (wird mit 2.0 multipliziert, um auf gleicher Skala wie MSE zu sein)
    return (1.0 - cos_sim).mean() * 2.0


class Loss_function(nn.Module):
    """
    Fasst den Loss über alle Ebenen der Feature-Pyramide zusammen.
    """

    def __init__(
        self,
        epsilon: float = 1e-12,
        alpha_l: Optional[List[float]] = None,
        use_original_paper: bool = False,
    ):
        """
        Initialisiert die Verlustfunktion.

        Args:
            epsilon (float, optional): Wert zur numerischen Stabilisierung. Standard ist 1e-12.
            alpha_l (list of float, optional): Gewichtungen für die einzelnen Layer. Standard ist None (alle gleich gewichtet).
            use_original_paper (bool, optional): Ob die Original-MSE-Methode (True) oder die schnellere Cosine-Methode (False) genutzt wird. Standard ist False.
        """
        super().__init__()
        self.epsilon = epsilon
        self.alpha_l = alpha_l
        self.use_original_paper = use_original_paper

    def forward(
        self, teacher_maps: List[torch.Tensor], student_maps: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Berechnet den gesamten Loss für alle Feature-Maps.

        Args:
            teacher_maps (list of torch.Tensor): Feature-Maps des Teacher-Modells.
            student_maps (list of torch.Tensor): Feature-Maps des Student-Modells.

        Returns:
            torch.Tensor: Der aufsummierte und gewichtete Gesamt-Loss.

        Raises:
            ValueError: Wenn die Anzahl der Feature-Maps oder Gewichte nicht übereinstimmt.
        """
        if len(teacher_maps) != len(student_maps):
            raise ValueError(
                "Mismatch: Teacher und Student müssen gleich viele Feature-Maps haben."
            )

        num_layers = len(teacher_maps)
        device = teacher_maps[0].device

        # Wenn keine Gewichte übergeben wurden, zählen alle Layer gleich viel
        if self.alpha_l is None:
            alphas = [1.0] * num_layers
        elif len(self.alpha_l) == num_layers:
            alphas = self.alpha_l
        else:
            raise ValueError(
                f"Alpha-Liste ({len(self.alpha_l)}) muss exakt der Layer-Anzahl ({num_layers}) entsprechen."
            )

        # Wichtig: total_loss direkt auf der Grafikkarte (device) erstellen, um Ladezeiten zu sparen
        total_loss = torch.tensor(
            0.0, device=device, dtype=teacher_maps[0].dtype
        )

        for idx, (t_map, s_map) in enumerate(zip(teacher_maps, student_maps)):
            if self.use_original_paper:
                loss = compute_mse_loss_original(t_map, s_map, self.epsilon)
            else:
                loss = compute_cosine_loss_optimized(t_map, s_map)

            total_loss = total_loss + alphas[idx] * loss

        return total_loss