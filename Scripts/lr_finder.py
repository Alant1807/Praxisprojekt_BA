import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Scripts.dataset import *
from Scripts.model import *
from Scripts.loss import *
from torch.utils.data import *


class LRRangeTestFinder:
    """
    A class to find the optimal learning rate range for a given model and configuration.

    This class encapsulates the logic for the "LR Range Test" as described in
    "Cyclical Learning Rates for Training Neural Networks" (arXiv:1506.01186v6).

    Args:
        config_path (str): Path to the YAML configuration file containing model, dataset, and optimizer settings.
    """

    def __init__(self, config_path: str):
        print("Initializing LR Range Test environment...")
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['device']['type'])
        self.model = self._init_model()
        self.train_loader = self._init_dataloader()
        # self.scaler = torch.amp.GradScaler(device=self.device.type)

        print("Initialization complete.")

    def _load_config(self, config_path: str) -> dict:
        """
        Loads the configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            dict: The loaded configuration as a dictionary.
        """

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _init_model(self) -> STFPM:
        """
        Initializes the STFPM model based on the configuration.

        Returns:
            STFPM: An instance of the STFPM model configured as per the YAML file.
        """

        model = STFPM(
            architecture=self.config['model']['architecture'],
            layers=self.config['model']['layers']
        ).to(self.device, memory_format=torch.channels_last)
        return model

    def _init_dataloader(self) -> DataLoader:
        """
        Initializes the training DataLoader.

        Returns:
            DataLoader: A DataLoader instance for the training dataset.
        """

        train_set = MVTecDataset(
            img_size=self.config['dataset']['img_size'],
            base_path=self.config['dataset']['base_path'],
            cls=self.config['dataset']['class'],
            mode='train',
            subfolders=['good']
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.config['dataloader']['batch_size'],
            shuffle=True
        )
        return train_loader

    def run_test(self, start_lr: float = 1e-7, end_lr: float = 1.0, num_epochs: int = 5):
        """
        Runs the LR Range Test.

        Args:
            start_lr (float): The learning rate to start the test with.
            end_lr (float): The learning rate to end the test with.
            num_epochs (int): The number of epochs to run the test for.
        """

        print(
            f"Starting LR Range Test: from {start_lr} to {end_lr} over {num_epochs} epochs.")

        criterion = Loss_function(**self.config['loss']['params'])

        optimizer_params = self.config['optimizer']['configs'][self.config['optimizer']['active']]
        optimizer_class = getattr(optim, self.config['optimizer']['active'])
        if 'lr' in optimizer_params:
            del optimizer_params['lr']
        optimizer = optimizer_class(
            self.model.student_model.parameters(), lr=start_lr, **optimizer_params)

        num_iter = num_epochs * len(self.train_loader)
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda x: np.exp(
                x * (np.log(end_lr) - np.log(1e-7)) / (num_iter - 1))
        )

        lrs_recorded = list()
        losses_recorded = list()
        self.model.student_model.train()

        for epoch in range(num_epochs):
            pbar = tqdm(self.train_loader,
                        desc=f"Epoch {epoch+1}/{num_epochs}")
            for images, _, _, _ in pbar:
                img_t = images.to(
                    self.device, memory_format=torch.channels_last)

                optimizer.zero_grad(set_to_none=True)

                # with torch.autocast(device_type=self.device.type):
                with torch.no_grad():
                    teacher_map, _ = self.model(img_t)
                _, student_map = self.model(img_t)
                loss = criterion(teacher_map, student_map)

                if torch.isnan(loss) or loss > (min(losses_recorded, default=1e9) * 4):
                    print("Loss exploded. Stopping test.")
                    self.plot_results(lrs_recorded, losses_recorded)
                    return lrs_recorded, losses_recorded

                loss.backward()
                optimizer.step()

                lrs_recorded.append(optimizer.param_groups[0]['lr'])
                losses_recorded.append(loss.item())

                scheduler.step()
                pbar.set_postfix(loss=loss.item(),
                                 lr=optimizer.param_groups[0]['lr'])

        print("LR Range Test finished.")
        self.plot_results(lrs_recorded, losses_recorded)

    def plot_results(self, lrs: list, losses: list):
        """

        Plots the results of the LR Range Test.

        Args:
            lrs (list): A list of learning rates recorded during the test.
            losses (list): A list of corresponding losses recorded during the test.
        """

        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("LR Range Test")
        plt.grid(True)
        plt.show()
