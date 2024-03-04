from typing import Optional, List, Union, Tuple
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import torch
from torch import nn, optim
from utils.metrics import Metrics
from dataflow.dataloaders import DatasetLoader
from components.encoder_decoder import AutoFormer


class EarlyStopper:
    """
     A simple class for early stopping during training based on the loss.

     Args:
         patience: Number of epochs to wait for improvement before stopping.
         min_delta: Minimum change in the monitored quantity to qualify as an improvement.

     Attributes:
         patience: Number of epochs to wait for improvement before stopping.
         min_delta Minimum change in the monitored quantity to qualify as an improvement.
         counter: Counter to track the number of epochs without improvement.
         min_loss: Minimum observed loss during training.

     """

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, current_loss: float):
        """
        Check if early stopping criteria are met based on the current loss.

        Args:
            current_loss: The current value of the monitored quantity (e.g., validation loss).

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """

        if (self.min_loss - current_loss) < self.min_delta:
            self.min_loss = current_loss
            self.counter += 1
            if self.counter >= self.patience:
                return True

        elif current_loss < self.min_loss:
            self.min_loss = current_loss
            self.counter = 0

        return False


class Autoformer(nn.Module):
    def __init__(self, input_len: int = 120, mask_len: int = 48, output_len: int = 24,
                 input_features: int = 1, embedding_features: int = 16, hidden_features: int = 64,
                 convolution_features: int = 64 * 4, output_features: int = 1, decomposition_kernel: int = 25,
                 heads: int = 4, factor: int = 1):
        super(Autoformer, self).__init__()
        self.input_len = input_len
        self.mask_len = mask_len
        self.output_len = output_len
        self.input_features = input_features
        self.embedding_features = embedding_features
        self.hidden_features = hidden_features
        self.convolution_features = convolution_features
        self.output_features = output_features
        self.decomposition_kernel = decomposition_kernel
        self.heads = heads
        self.factor = factor

    def train_model(self, path: str, features: Optional[Union[List[str], None]], target: List[str], dates: List[str],
                    scale_type: str, learning_rate: float, epochs: int, batch_size: int, shuffle: bool,
                    save_folder: str) -> Tuple[AutoFormer, Union[MinMaxScaler, StandardScaler]]:
        """
        Train the model.

        Args:
            path: Path to the training CSV file.
            features: List of feature names. If provided, features will be selected along with the target.
                                             If empty or None, only the target column will be selected.
            target: Name of the target column.
            dates: Name of the column containing dates.
            scale_type: Scale the input data using MinMax scaling or Standard scaling.
            learning_rate: Learning rate for optimization.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            shuffle: Whether to shuffle the training data.
            save_folder: The folder where the trained scaler and trained AutoFormer models will be saved.

        Returns:
            Trained model and scaler used for training.
        """

        dataset_loader = DatasetLoader(path=path, target=target, features=features, dates=dates, scale_type=scale_type,
                                       input_len=self.input_len, mask_len=self.mask_len, output_len=self.output_len,
                                       flag='train')
        train_loader, train_scaler = dataset_loader(batch_size=batch_size, shuffle=shuffle)
        dump(train_scaler, os.path.join(save_folder, 'scaler.joblib'))

        model = AutoFormer(input_features=self.input_features, hidden_features=self.hidden_features,
                           convolution_features=self.convolution_features,
                           decomposition_kernel=self.decomposition_kernel, mask_len=self.mask_len,
                           output_len=self.output_len, output_features=self.output_features,
                           heads=self.heads, factor=self.factor, flag='train')

        if scale_type == 'minmax':
            es = EarlyStopper(patience=3, min_delta=0.0005)
        else:
            es = EarlyStopper(patience=3, min_delta=0.0005)

        model_optim = optim.Adam(model.parameters(), lr=learning_rate)
        model.train()
        for epoch in range(epochs):
            train_loss = []

            for i, (x, y, x_time, y_time) in enumerate(train_loader):
                model_optim.zero_grad()
                model_output = model(x, x_time, y_time)
                y = y[:, -self.output_len:, :]
                loss = nn.MSELoss()(model_output, y)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
            print("Epoch: {} | Train Loss: {}".format(epoch + 1, np.mean(train_loss)))

            early_stopping = es.early_stop(current_loss=np.mean(train_loss))
            if early_stopping:
                torch.save(model.state_dict(), os.path.join(save_folder, 'checkpoint.pth'))
                print("Early stopping")
                break

        torch.save(model.state_dict(), os.path.join(save_folder, 'checkpoint.pth'))

        return model, train_scaler

    def test_model(self, path: str, features: Optional[Union[List[str], None]], target: List[str], dates: List[str],
                   trained_model_path: str, trained_scaler_path: str, batch_size: int, shuffle: bool) \
            -> Tuple[List, List, List]:
        """
        Test the model.

        Args:
            path: Path to the testing CSV file.
            features: List of feature names. If provided, features will be selected along with the target.
                                             If empty or None, only the target column will be selected.
            target: Name of the target column.
            dates: Name of the column containing dates.
            trained_model_path: Path to the trained model checkpoint.
            trained_scaler_path: Path to the scaler used for training.
            batch_size: Batch size for testing.
            shuffle: Whether to shuffle the testing data.

        Returns:
            Actual and predicted results.
        """

        scaler = load(trained_scaler_path)
        dataset_loader = DatasetLoader(path=path, target=target, features=features, dates=dates,
                                       input_len=self.input_len, mask_len=self.mask_len, output_len=self.output_len,
                                       flag='test', scaler=scaler)
        test_loader, _ = dataset_loader(batch_size=batch_size, shuffle=shuffle)

        model = AutoFormer(input_features=self.input_features, hidden_features=self.hidden_features,
                           convolution_features=self.convolution_features,
                           decomposition_kernel=self.decomposition_kernel, mask_len=self.mask_len,
                           output_len=self.output_len, output_features=self.output_features,
                           heads=self.heads, factor=self.factor, flag='test')

        checkpoint = torch.load(trained_model_path)
        model.load_state_dict(checkpoint)

        previous = []  # previous moments
        results = []
        actuals = []
        model.eval()
        with torch.no_grad():
            for i, (x, y, x_time, y_time) in enumerate(test_loader):
                prediction = model(x, x_time, y_time)
                y = y[:, -self.output_len:, :]

                previous.append(x.detach().cpu().numpy().squeeze(0))
                results.append(prediction.detach().cpu().numpy().squeeze(0))
                actuals.append(y.detach().cpu().numpy().squeeze(0))

        metrics = Metrics(np.array(actuals), np.array(results))

        print("RMSE is equal to: {}".format(metrics.rmse()))

        previous_inverse = [scaler.inverse_transform(sequence) for sequence in previous]
        actuals_inverse = [scaler.inverse_transform(sequence) for sequence in actuals]
        results_inverse = [scaler.inverse_transform(sequence) for sequence in results]

        return previous_inverse, actuals_inverse, results_inverse
