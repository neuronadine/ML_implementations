# %%
import random
import numpy as np
import torch
from typing import Tuple, List, NamedTuple
from tqdm import tqdm
import torchvision
from torchvision import transforms

# %%


# Seed all random number generators
np.random.seed(197331)
torch.manual_seed(197331)
random.seed(197331)


class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    dense_hiddens: Tuple[int, ...] = (256, 256)


class Trainer:
    def __init__(
        self,
        network_type: str = "mlp",
        net_config: NetworkConfiguration = NetworkConfiguration(),
        lr: float = 0.001,
        batch_size: int = 128,
        activation_name: str = "relu",
    ):

        self.lr = lr
        self.batch_size = batch_size
        self.train, self.test = self.load_dataset(self)
        dataiter = iter(self.train)
        images, labels = next(dataiter)
        input_dim = images.shape[1:]
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        if network_type == "mlp":
            self.network = self.create_mlp(
                input_dim[0] * input_dim[1] * input_dim[2],
                net_config,
                activation_function,
            )
        elif network_type == "cnn":
            self.network = self.create_cnn(
                input_dim[0], net_config, activation_function
            )
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.train_logs = {
            "train_loss": [],
            "test_loss": [],
            "train_mae": [],
            "test_mae": [],
        }

    @staticmethod
    def load_dataset(self):
        transform = transforms.ToTensor()

        trainset = torchvision.datasets.SVHN(
            root="./data", split="train", download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True
        )

        testset = torchvision.datasets.SVHN(
            root="./data", split="test", download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False
        )

        return trainloader, testloader

    @staticmethod
    def create_mlp(
        input_dim: int, net_config: NetworkConfiguration, activation: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Create a multi-layer perceptron (MLP) network.

        :param net_config: a NetworkConfiguration named tuple. Only the field 'dense_hiddens' will be used.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the MLP.
        """
        layers = []

        # Flattening layer
        layers.append(torch.nn.Flatten())

        # hidden layers
        prev_layer_dim = input_dim
        for hidden_layer_dim in net_config.dense_hiddens:
            layers.append(torch.nn.Linear(prev_layer_dim, hidden_layer_dim))
            layers.append(activation)
            prev_layer_dim = hidden_layer_dim

        # output layer
        layers.append(torch.nn.Linear(prev_layer_dim, 1))

        return torch.nn.Sequential(*layers)

    @staticmethod
    def create_cnn(
        in_channels: int, net_config: NetworkConfiguration, activation: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Create a convolutional network.

        :param in_channels: The number of channels in the input image.
        :param net_config: a NetworkConfiguration specifying the architecture of the CNN.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the CNN.
        """
        # convolutional layers
        layers = []
        prev_layer_channels = in_channels
        conv_layers_num = len(net_config.n_channels)

        for i, (out_channels, kernel_size, stride) in enumerate(
            zip(net_config.n_channels, net_config.kernel_sizes, net_config.strides)
        ):
            layers.append(
                torch.nn.Conv2d(prev_layer_channels, out_channels, kernel_size, stride)
            )
            if i < conv_layers_num - 1:
                layers.append(activation)
                layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            else : 
                layers.append(torch.nn.AdaptiveAvgPool2d((4, 4)))
            prev_layer_channels = out_channels

        # Flattening layer
        layers.append(torch.nn.Flatten())
        
        # Fully connected layers
        dense_input_dim = prev_layer_channels * 4 * 4
        for hidden_dim in net_config.dense_hiddens:
            layers.append(torch.nn.Linear(dense_input_dim, hidden_dim))
            layers.append(activation)
            dense_input_dim = hidden_dim
        
        # output layer
        layers.append(torch.nn.Linear(dense_input_dim, 1))
        
        # combine all layers into sequential model
        return torch.nn.Sequential(*layers)

    @staticmethod
    def create_activation_function(activation_str: str) -> torch.nn.Module:
        if activation_str == "relu":
            return torch.nn.ReLU()
        elif activation_str == "tanh":
            return torch.nn.Tanh()
        elif activation_str == "sigmoid":
            return torch.nn.Sigmoid()

    def compute_loss_and_mae(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # get predictions
        y_pred = self.network(X)

        # Convert labels to float
        y = y.float().unsqueeze(1)

        # compute mean squared error loss
        loss = torch.nn.functional.mse_loss(y_pred, y)

        # compute mean absolute error
        mae = torch.nn.functional.l1_loss(y_pred, y)

        return loss, mae

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        # compute mse and mae
        loss, mae = self.compute_loss_and_mae(X_batch, y_batch)

        # backpropagate the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # return the loss and mae
        return loss.item(), mae.item()

    def train_loop(self, n_epochs: int) -> dict:
        N = len(self.train)
        for epoch in tqdm(range(n_epochs)):
            train_loss = 0.0
            train_mae = 0.0
            for i, data in enumerate(self.train):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                loss, mae = self.training_step(inputs, labels)
                train_loss += loss
                train_mae += mae

            # Log data every epoch
            self.train_logs["train_mae"].append(train_mae / N)
            self.train_logs["train_loss"].append(train_loss / N)
            self.evaluation_loop()

        return self.train_logs

    def evaluation_loop(self) -> None:
        self.network.eval()
        N = len(self.test)
        with torch.inference_mode():
            test_loss = 0.0
            test_mae = 0.0
            for data in self.test:
                inputs, labels = data
                loss, mae = self.compute_loss_and_mae(inputs, labels)
                test_loss += loss.item()
                test_mae += mae.item()

        self.train_logs["test_mae"].append(test_mae / N)
        self.train_logs["test_loss"].append(test_loss / N)

    def evaluate(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        self.network.eval()

        # Initiate the total loss and mae
        tot_loss = torch.tensor(0.0)
        tot_mae = torch.tensor(0.0)
        tot_samples = 0

        # split the data into batches
        batch_size = self.batch_size
        X_batches = torch.split(X, batch_size)
        y_batches = torch.split(y, batch_size)

        # loop through the data and compute the loss and mae
        for X_batch, y_batch in zip(X_batches, y_batches):
            loss, mae = self.compute_loss_and_mae(X_batch, y_batch)
            batch_size = X_batch.size(0)
            tot_loss += loss * batch_size
            tot_mae += mae * batch_size
            tot_samples += batch_size

        if tot_samples > 0:
            average_loss = tot_loss / tot_samples
            average_mae = tot_mae / tot_samples
        else:
            average_loss = torch.tensor(0.0)
            average_mae = torch.tensor(0.0)

        self.network.train()

        # return the average loss and mae
        return average_loss, average_mae
