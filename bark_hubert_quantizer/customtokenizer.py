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


from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0), :x.size(1)]
        return self.dropout(x)

class CustomTokenizer(nn.Module):
    def __init__(self, hidden_size=1024, input_size=768, output_size=10000, batch_size=1, transformer=False):
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
            self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
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
        if self.transformer:
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
        else:
            x, _ = self.lstm(x)
            x = self.intermediate(x)
        
        x = self.fc(x)
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

    def train_step(self, x_train, y_train, step_number, num_training_samples):
        # y_train = y_train[:-1]
        # y_train = y_train[1:]

        optimizer = self.optimizer
        lossfunc = self.lossfunc

        # Forward pass
        y_pred = self(x_train)

        y_train_len = len(y_train)
        y_pred_len = y_pred.shape[0]

        if y_train_len > y_pred_len:
            diff = y_train_len - y_pred_len
            y_train = y_train[diff:]
        elif y_train_len < y_pred_len:
            diff = y_pred_len - y_train_len
            y_pred = y_pred[:-diff, :]


        # TODO: Continue this investigation
        # if step_number == 1:
        #     print(x_train.shape, y_train.shape)
        #     print(y_pred.shape, y_train_hot.shape)
        #     # print every non-zero value of y_train_hot
        #     print(f'Non-zero values of y_train: {y_train[y_train != 0]}')
        #     # print every non-zero value of y_pred
        #     print(f'Non-zero values of y_pred: {y_pred[y_pred != 0]}')

        # Calculate the loss
        sample_loss = lossfunc(y_pred, y_train)

        loss = sample_loss / self.batch_size

        # Backward pass
        loss.backward()

        # Update weights
        if step_number % self.batch_size == 0 or step_number == num_training_samples - 1:
            optimizer.step()
            optimizer.zero_grad()

        return sample_loss

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
    epoch = 1
    data_size = max(len(data_x), len(data_y))
    train_size = int(0.9 * data_size)
    valid_size = data_size - train_size

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

    print(f'Length of data_x_train: {len(data_x_train)} / data_y_train: {len(data_y_train)}')

    while 1:
        losses_for_epoch = []
        for train_step, key in enumerate(data_x_train):
            x = data_x_train.get(key)
            y = data_y_train.get(key)
            if x is None or y is None:
                print(f'The training data does not match. key={train_step}')
                continue
            loss = model_training.train_step(torch.tensor(x).to('cuda'), torch.tensor(y).to('cuda'), train_step, train_size)
            losses_for_epoch.append(loss.item())
        
        # Validation loop
        model_training.eval()
        valid_losses = []
        for _, key in enumerate(data_x_valid):
            x = data_x_valid.get(key)
            y = data_y_valid.get(key)

            if x is None or y is None:
                continue
            with torch.no_grad():
                y_pred = model_training(torch.tensor(x).to('cuda'))

                y_len = len(y)
                y_pred_len = y_pred.shape[0]

                if y_len > y_pred_len:
                    diff = y_len - y_pred_len
                    y = y[diff:]
                elif y_len < y_pred_len:
                    diff = y_pred_len - y_len
                    y_pred = y_pred[:-diff, :]

                    
                loss = model_training.lossfunc(y_pred, torch.tensor(y).to('cuda'))
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
