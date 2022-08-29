#!/usr/bin/env python3
import torch
import pandas as pd
import os
from absl import app, flags
from os import cpu_count
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from sklearn.model_selection import train_test_split


flags.DEFINE_integer('epochs', 10, 'Number of epochs')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('gpus', 1, 'Number of GPUs')
flags.DEFINE_integer('model_dim', 64, 'Model hidden state dimension')
flags.DEFINE_integer('feedforward_dim', 128, 'Feedforward dimension')
flags.DEFINE_integer('num_heads', 8, 'Number of attention heads')
flags.DEFINE_integer('num_layers', 3, 'Number of encoder layers')
flags.DEFINE_integer('chunk_length', 10, 'Sequence length during training')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_float('mask_prob', 0.1,
                   'Bernoulli probability of masking a token in pretraining')
flags.DEFINE_enum('mode', 'pretrain', ['pretrain', 'regression'],
                  'Whether to run pretraining or regression')
flags.DEFINE_string('ckpt_path', '',
                    'Pretrained checkpoint to use in the regressor')
FLAGS = flags.FLAGS


'''
This model takes inspiration from cs/2010.02803, where the model uses only
the encoder of the full transformer (encoder-decoder) model. Pretraining task
is a type of denoising, where input is masked randomly and the encoder output
is trained to reconstruct the masked elements. The masking process is also
unconventional: complete sequence positions are not masked, instead, we mask
random features at random sequence positions (time steps) independently.
The reconstruction of these masked features is then rewarded during
pretraining. Other tasks such as regression use the same encoder with
a task-specific head on top.
'''


class StockEncoder(LightningModule):
    def __init__(self):
        super().__init__()
        # Layers which are independent of the number of features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FLAGS.model_dim,
            nhead=FLAGS.num_heads,
            dim_feedforward=FLAGS.feedforward_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=FLAGS.num_layers
        )
        self.train_data, self.val_data, max_time = prepare_data(
            'data/closes_1d.csv')
        # The embedding and output layer dimensions depend on the
        # number of features in the dataset
        # The feature embedding maps features to the model dimension:
        # analogous to word embeddings in NLP
        num_features = self.train_data[0]['features'].shape[-1]
        self.feature_embedding = nn.Linear(
            num_features,
            FLAGS.model_dim,
            bias=True
        )
        # The position embedding is fully learnable: there are
        # `max_time + 1` possibilities for the time coordinate
        # which is analogous to the max sequence length in NLP
        # Difference is that now the time coordinate can have jumps in it
        self.position_embedding = nn.Embedding(
            max_time + 1,
            FLAGS.model_dim
        )
        # Used in pretraining to map final encoded representations
        # back to predicted feature vectors
        self.output = nn.Linear(
            FLAGS.model_dim,
            num_features,
            bias=True
        )

    def train_dataloader(self):
        # Return training dataloader
        train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=FLAGS.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=cpu_count()
        )
        return train_loader

    def val_dataloader(self):
        # Return validation dataloader
        val_loader = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=FLAGS.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=cpu_count()
        )
        return val_loader

    def configure_optimizers(self):
        # Return optimizer
        opt = torch.optim.Adam(self.parameters(), lr=FLAGS.lr)
        return opt

    def forward(self, time_ids, features, feature_mask):
        '''
        Here we compute the forward pass of the model without the final
        output layer. This is because the output layer is not needed when
        we use the model as an encoder: forward encodes the input and returs
        the final hidden states.
        '''
        X_feature = self.feature_embedding(features * feature_mask)
        X_position = self.position_embedding(time_ids)
        X = X_feature + X_position
        hidden_states = self.encoder(X)
        return hidden_states

    def training_step(self, batch, batch_idx):
        # Compute and return loss
        feature_mask = torch.bernoulli(
            (1 - FLAGS.mask_prob) * torch.ones_like(batch['features'])
        )
        hidden_states = self.forward(
            batch['time'], batch['features'], feature_mask
        )
        pred_features = self.output(hidden_states)
        # Loss is the mean squared error of original vs predicted features
        # only taking into account masked elements
        loss = ((batch['features'] - pred_features)
                * (1 - feature_mask)).pow(2).mean()
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # Compute loss on validation set
        feature_mask = torch.bernoulli(
            (1 - FLAGS.mask_prob) * torch.ones_like(batch['features'])
        )
        hidden_states = self.forward(
            batch['time'], batch['features'], feature_mask
        )
        pred_features = self.output(hidden_states)
        # Loss is the mean squared error of original vs predicted features
        # only taking into account masked elements
        loss = ((batch['features'] - pred_features)
                * (1 - feature_mask)).pow(2).mean()
        self.log('val_loss', loss)
        return {'loss': loss}


class StockRegressor(LightningModule):
    def __init__(self, ckpt_path):
        super().__init__()
        self.encoder = StockEncoder.load_from_checkpoint(ckpt_path)
        self.train_data = self.encoder.train_data
        self.val_data = self.encoder.val_data
        num_features = self.train_data[0]['features'].shape[-1]
        # The output is a prediction for the price move of each feature
        # at the next step. In the input data, the last time step is considered
        # the regression target (masked when passed to the encoder)
        self.regressor = nn.Linear(
            FLAGS.model_dim,
            num_features,
            bias=True
        )

    def train_dataloader(self):
        # Return training dataloader
        train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=FLAGS.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=cpu_count()
        )
        return train_loader

    def val_dataloader(self):
        # Return validation dataloader
        val_loader = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=FLAGS.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=cpu_count()
        )
        return val_loader

    def configure_optimizers(self):
        # Return optimizer
        opt = torch.optim.Adam(self.parameters(), lr=FLAGS.lr)
        return opt

    def forward(self, time_ids, features):
        # The last time step is the regression target, so we mask it
        feature_mask = torch.ones_like(features)
        feature_mask[:, -1] = 0.0
        hidden_states = self.encoder(time_ids, features, feature_mask)
        # Flatten the time and feature dimensions into one
        # hidden_states = hidden_states.view(hidden_states.shape[0], -1)
        # pred_features = self.regressor(hidden_states)
        pred_features = self.regressor(hidden_states[:, -1])
        return pred_features

    def training_step(self, batch, batch_idx):
        # Compute and return loss
        pred_features = self.forward(batch['time'], batch['features'])
        target_features = batch['features'][:, -1]
        # Loss is the mean squared error of original vs predicted features
        # only taking into account masked elements
        loss = (target_features - pred_features).pow(2).mean()
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # Compute loss on validation set
        pred_features = self.forward(batch['time'], batch['features'])
        target_features = batch['features'][:, -1]
        # Loss is the mean squared error of original vs predicted features
        # only taking into account masked elements
        loss = (target_features - pred_features).pow(2).mean()
        self.log('val_loss', loss)
        return {'loss': loss}


def prepare_data(closes_path):
    # Prepare datasets
    # Load data
    closes = pd.read_csv(closes_path, parse_dates=['Date'], index_col=0)
    closes = closes.loc['1990':]
    # Drop features which are all nan and backfill remaining nans
    closes = closes.loc[:, closes.apply(lambda x: not x.isna().all())]
    closes = closes.fillna(method='backfill')
    # Actual training data is the relative differences between
    # subsequent close prices
    # Features have mean slightly over zero (small profits on average)
    # and standard deviation in the range 0.01-0.02. Therefore, we'll
    # multiply by 100 to have roughly O(1) values features
    ns = (closes.index[1:] - closes.index[1]).astype(int).to_numpy()
    time = torch.from_numpy(ns // ns[1]).int()
    X = torch.from_numpy(closes.to_numpy()).float()
    X = 100 * (X[1:] - X[:-1]) / X[:-1]
    # We chunk the price data and record the relative time
    # of each vector in the sequence
    num_chunks = X.shape[0] // FLAGS.chunk_length
    data = []
    max_time = 0
    time_chunks = time[:num_chunks * FLAGS.chunk_length].chunk(num_chunks)
    X_chunks = X[:num_chunks * FLAGS.chunk_length].chunk(num_chunks)
    for t, x in zip(time_chunks, X_chunks):
        max_time = max(max_time, t[-1] - t[0])
        data.append({'time': t - t[0], 'features': x})
    # Split train-validation
    train_data, val_data = train_test_split(data, test_size=0.1)
    print('Training data has', len(train_data), 'examples')
    print('Validation data has', len(val_data), 'examples')
    return train_data, val_data, max_time


def main(_):
    logger = TensorBoardLogger(save_dir=os.getcwd(),
                               name='logs_' + FLAGS.mode)
    if FLAGS.mode == 'regression':
        if not FLAGS.ckpt_path:
            print('Provide a pretrained checkpoint to use in regression.')
            exit(-1)
        model = StockRegressor(FLAGS.ckpt_path)
    else:
        model = StockEncoder()
    trainer = Trainer(
        log_every_n_steps=1,
        logger=logger,
        max_epochs=FLAGS.epochs,
        gpus=FLAGS.gpus
    )
    trainer.fit(model)


if __name__ == '__main__':
    app.run(main)
