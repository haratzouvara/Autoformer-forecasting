import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
from utils.data import ReadData, ScaleData
from typing import List, Optional, Union, Tuple
import torch


class CreateSample(Dataset):
    """
    Data sampling
    """

    def __init__(self, data: np.ndarray, time_features: np.ndarray, input_len: int, mask_len: int, output_len: int):
        """

        Args:
            data: The input data.
            input_len: Length of the encoder input sequence.
            mask_len: Length of the mask sequence. The part of the encoder input used for the decoder input
            output_len: Length of the decoder output sequence.
        """

        super(CreateSample, self).__init__()
        self.data = data
        self.time_features = time_features
        self.input_len = input_len
        self.mask_len = mask_len
        self.output_len = output_len

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample.

        Returns:
            Tuple containing encoder and decoder sequences.

        """

        encoder_sequence = self.data[index: index + self.input_len]
        encoder_time = self.time_features[index: index + self.input_len]

        decoder_sequence = self.data[index + self.input_len - self.mask_len:
                                     index + self.input_len + self.output_len]
        decoder_time = self.time_features[index + self.input_len - self.mask_len:
                                          index + self.input_len + self.output_len]

        return (torch.Tensor(encoder_sequence), torch.Tensor(decoder_sequence),
                torch.Tensor(encoder_time), torch.Tensor(decoder_time))

    def __len__(self) -> int:
        return len(self.data) - self.input_len - self.output_len + 1


class DatasetLoader:
    def __init__(self, path: str, features: Optional[Union[List[str], None]], target: List[str], dates: List[str],
                 scale_type=None, input_len: int = 120, mask_len: int = 48, output_len: int = 24,
                 flag: str = 'train', scaler=None):
        """
        Create Dataloader

        Args:
            path: Path to the data file.
            features: List of feature names. An emtpy List considers target as the unique feature
            target: Name of the target column.
            dates: Name of the column containing dates.
            input_len: Length of the encoder input sequence.
            mask_len: Length of the mask sequence. The part of the encoder input used for the decoder input
            output_len: Length of the decoder output sequence.
            flag: A flag indicating the dataset type (e.g., "train", "test", "vali")
            scaler: Optional scaler instance. Required when flag is 'test' or 'vali'.

        Raises:
            ValueError: If flag is 'test' or 'vali' and scaler is not provided.
        """

        self.data, self.time_features = ReadData.read(path=path, features=features, target=target, dates=dates)
        if flag == 'train':
            self.data_scaled, self.scaler = ScaleData.scale_train(data=self.data, method=scale_type)
        elif flag == 'test':
            if scaler is None:
                raise ValueError("When flag is 'test', you have to provide a scaler instance")
            self.data_scaled, self.scaler = ScaleData.scale_test(data=self.data,  scaler=scaler)
        else:
            raise ValueError("The flag should be set to either 'train' or 'test'.")
        self.dataset = CreateSample(self.data_scaled, self.time_features, input_len, mask_len, output_len)

    def __call__(self, batch_size: int, shuffle: bool) -> Tuple[DataLoader, MinMaxScaler | StandardScaler]:
        """
        Create a DataLoader instance.

        Args:
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data.

        Returns:
             DataLoader instance and scaler.

        """

        dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader, self.scaler