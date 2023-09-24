"""
Custom tokenizer model.
Author: https://www.github.com/gitmylo/
License: MIT
"""

import json
import os.path
from zipfile import ZipFile

import numpy
import math
import torch
from torch import Tensor, nn, optim
from torch.serialization import MAP_LOCATION
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def masked_loss(lossfunc, y_pred, y_true):
    """
    Computes the loss while ignoring padded values.

    Args:
        lossfunc: The loss function to use.
        y_pred: The predicted values of shape (seq_len, batch_size, embedding_dim).
        y_true: The true values of shape (batch_size, seq_len).
    """

    # Create a mask for non-padded values
    mask = y_true != 0
    
    # Check and handle mismatch in sequence length
    if y_pred.size(0) != y_true.size(1):
        seq_len = min(y_pred.size(0), y_true.size(1))
        y_pred = y_pred[:seq_len]
        y_true = y_true[:, :seq_len]
        mask = mask[:, :seq_len]
    
    # Adjust mask dimensions to match y_pred's dimensions
    mask_expanded = mask.unsqueeze(-1).expand_as(y_pred.permute(1, 0, 2))
    
    # Apply the mask and compute the loss
    loss = lossfunc(y_pred.permute(1, 0, 2)[mask_expanded].view(-1, y_pred.size(-1)), y_true[mask].view(-1))
    return loss


class CustomTokenizer(nn.Module):
    def __init__(self, hidden_size=1024, input_size=768, output_size=10000, batch_size=8, transformer=False):
        super(CustomTokenizer, self).__init__()
        next_size = input_size

        self.transformer = transformer

        if transformer:
            # Replace LSTM with Transformer Encoder
            self.pos_encoder = PositionalEncoding(d_model=input_size)
            encoder_layers = TransformerEncoderLayer(d_model=input_size, nhead=8, dim_feedforward=hidden_size)
            self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=10)
            next_size = input_size  # Transformer does not change the feature dimension
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=False)
            self.intermediate = nn.Linear(hidden_size, 4096)
            next_size = 4096

        # for layer in self.transformer_encoder.layers:
        #     nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
        #     nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            
        #     nn.init.kaiming_normal_(layer.linear1.weight, nonlinearity='relu')
        #     nn.init.kaiming_normal_(layer.linear2.weight, nonlinearity='relu')
        #     if layer.self_attn.in_proj_bias is not None:
        #         nn.init.zeros_(layer.self_attn.in_proj_bias)
        #     if layer.self_attn.out_proj.bias is not None:
        #         nn.init.zeros_(layer.self_attn.out_proj.bias)
        
        self.batch_size = batch_size

        self.fc = nn.Linear(next_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.optimizer: optim.Optimizer = None
        self.scheduler: optim.lr_scheduler = None
        self.lossfunc = nn.CrossEntropyLoss()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x):
        x = x.permute(1, 0, 2) # (batch_size, seq_len, input_size) -> (seq_len, batch_size, input_size)
        
        if self.transformer:
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
        else:
            x, _ = self.lstm(x)
            x = self.intermediate(x)
        
        x = self.fc(x)
        x = self.softmax(x)
        return x


    @torch.no_grad()
    def get_token(self, x):
        """
        Used to get the token for the first
        :param x: An array with shape (N, input_size) where N is a whole number greater or equal to 1, and input_size is the input size used when creating the model.
        :return: An array with shape (N,) where N is the same as N from the input. Every number in the array is a whole number in range 0...output_size - 1 where output_size is the output size used when creating the model.
        """
        return torch.argmax(self(x), dim=1)

    def prepare_training(self):
        self.optimizer = optim.AdamW(self.parameters(), 1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5, threshold=1e-3, verbose=True)

    def train_step(self, x_train, y_train):
        # y_train = y_train[:-1]
        # y_train = y_train[1:]

        optimizer = self.optimizer
        lossfunc = self.lossfunc

        # Forward pass
        y_pred = self(x_train)

        # Calculate the loss
        loss = masked_loss(lossfunc, y_pred, y_train)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

        return loss

    def save(self, path):
        info_path = '.'.join(os.path.basename(path).split('.')[:-1]) + '/.info'
        torch.save(self.state_dict(), path)
        data_from_model = Data(self.input_size, self.hidden_size, self.output_size)
        with ZipFile(path, 'a') as model_zip:
            model_zip.writestr(info_path, data_from_model.save())
            model_zip.close()

    @staticmethod
    def load_from_checkpoint(path, map_location: MAP_LOCATION = None):
        old = True
        with ZipFile(path) as model_zip:
            filesMatch = [file for file in model_zip.namelist() if file.endswith('/.info')]
            file = filesMatch[0] if filesMatch else None
            if file:
                old = False
                data_from_model = Data.load(model_zip.read(file).decode('utf-8'))
            model_zip.close()
        if old:
            model = CustomTokenizer()
        else:
            model = CustomTokenizer(data_from_model.hidden_size, data_from_model.input_size, data_from_model.output_size)
        model.load_state_dict(torch.load(path, map_location=map_location))
        if map_location:
            model = model.to(map_location)
        return model

class CustomDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = list(data_x.values())
        self.data_y = list(data_y.values())

    def collate_fn(self, batch):
        x_list, y_list = zip(*batch)
        x_padded = pad_sequence(x_list, batch_first=True)
        y_padded = pad_sequence(y_list, batch_first=True)
        return x_padded, y_padded

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return torch.tensor(self.data_x[index]), torch.tensor(self.data_y[index])


class Data:
    input_size: int
    hidden_size: int
    output_size: int

    def __init__(self, input_size=768, hidden_size=1024, output_size=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    @staticmethod
    def load(string):
        data = json.loads(string)
        return Data(data['input_size'], data['hidden_size'], data['output_size'])

    def save(self):
        data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
        }
        return json.dumps(data)


def auto_train(data_path, save_path='model.pth', load_model: str | None = None):
    data_x, data_y = {}, {}

    if load_model and os.path.isfile(load_model):
        print('Loading model from', load_model)
        model_training = CustomTokenizer.load_from_checkpoint(load_model, 'cuda')
    else:
        print('Creating new model.')
        model_training = CustomTokenizer().to('cuda')
    save_path = os.path.join(data_path, save_path)
    base_save_path = '.'.join(save_path.split('.')[:-1])

    sem_string = '_semantic.npy'
    feat_string = '_semantic_features.npy'

    ready = os.path.join(data_path, 'ready')
    for input_file in os.listdir(ready):
        full_path = os.path.join(ready, input_file)        
        try:
            prefix = input_file.split("_")[0]
            number = int(prefix)
        except ValueError as e:            
            raise e
        if input_file.endswith(sem_string):
            data_y[number] = numpy.load(full_path)
        elif input_file.endswith(feat_string):
            data_x[number] = numpy.load(full_path)
    
    model_training.prepare_training()
    data_size = max(len(data_x), len(data_y))
    train_size = int(0.9 * data_size)

    # Print length of data_x and data_y
    print(f'Length of data_x: {len(data_x)} / data_y: {len(data_y)}')

    keys = sorted(data_x.keys())

    numpy.random.seed(42)
    numpy.random.shuffle(keys)
    print(f'Keys: {keys[:20]}')
    
    data_x_train = {k: data_x[k] for k in keys[:train_size]}
    data_y_train = {k: data_y[k] for k in keys[:train_size]}
    data_x_valid = {k: data_x[k] for k in keys[train_size:]}
    data_y_valid = {k: data_y[k] for k in keys[train_size:]}

    train_dataset = CustomDataset(data_x_train, data_y_train)
    train_loader = DataLoader(train_dataset, batch_size=model_training.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    valid_dataset = CustomDataset(data_x_valid, data_y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=model_training.batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn)

    print(f'Length of data_x_train: {len(data_x_train)} / data_y_train: {len(data_y_train)}')

    epoch = 1
    while 1:
        losses_for_epoch = []
        for x, y in train_loader:
            x, y = x.to('cuda'), y.to('cuda')
            loss = model_training.train_step(x, y)
            losses_for_epoch.append(loss.item())

        
        # Validation loop
        model_training.eval()
        valid_losses = []
        for x, y in valid_loader:
            x, y = x.to('cuda'), y.to('cuda')
            with torch.no_grad():
                y_pred = model_training(x)
                loss = masked_loss(model_training.lossfunc, y_pred, y)
                valid_losses.append(loss.item())
        model_training.train()

        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        model_training.scheduler.step(avg_valid_loss)

        save_p = save_path
        save_p_2 = f'{base_save_path}_epoch_{epoch}.pth'
        model_training.save(save_p)
        model_training.save(save_p_2)
        print(f"Epoch {epoch} completed.")
        print(f'Training loss avg {sum(losses_for_epoch) / len(losses_for_epoch)} / min {min(losses_for_epoch)}/ max {max(losses_for_epoch)}')
        print(f'Validation loss avg {sum(valid_losses) / len(valid_losses)} / min {min(valid_losses)}/ max {max(valid_losses)}')
        epoch += 1
