import torch
import torch.nn as nn
import torch.nn.functional


class Loss_function(nn.Module):
    """
    Verwaltet die Verlustfunktion für das STFPM-Modell (Student-Teacher Feature Pyramid Matching).
    
    Args:
        epsilon (float): Ein kleiner Wert, um Division durch Null zu vermeiden.
        alpha_l (list, optional): Eine Liste von Gewichten für die verschiedenen Feature-Maps. 
                                  Wenn None, werden alle Gewichte auf 1.0 gesetzt.
    """

    def __init__(self, epsilon=1e-12, alpha_l=None):
        super().__init__()
        self.epsilon = epsilon
        self.alpha_l = alpha_l
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, teacher_maps, student_maps):
        """
        Berechnet den Verlust zwischen den Feature-Maps des Lehrers und des Schülers.
        
        Args:
            teacher_maps (list of torch.Tensor): Eine Liste von Feature-Maps des Lehrers.
            student_maps (list of torch.Tensor): Eine Liste von Feature-Maps des Schülers.
        
        Returns:
            torch.Tensor: Der berechnete Verlustwert.
        """

        if len(teacher_maps) != len(student_maps):
            raise ValueError(
                "Die Anzahl der Studenten- und Teacher-Feature-Maps muss übereinstimmen."
            )

        num_layers = len(teacher_maps)

        # kann man die Gewichte lernen ?
        if self.alpha_l is None:
            alphas = [1.0] * num_layers
        elif len(self.alpha_l) == num_layers:
            alphas = self.alpha_l
        else:
            raise ValueError(
                "Die Länge der Alpha-Liste muss der Anzahl der Feature-Maps entsprechen."
            )

        total_loss = 0.0

        for idx, (t_map, s_map) in enumerate(zip(teacher_maps, student_maps)):
            t_map_norm = torch.nn.functional.normalize(
                t_map, dim=1, eps=self.epsilon)
            s_map_norm = torch.nn.functional.normalize(
                s_map, dim=1, eps=self.epsilon)

            loss = self.mse(t_map_norm, s_map_norm)

            _, _, height, width = t_map.shape
            total_loss += alphas[idx] * loss / (2 * height * width)

        return total_loss
