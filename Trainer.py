import torch

from torch import FloatTensor, LongTensor, BoolTensor
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader, TensorDataset

from Utils import *
from Models import *
from QuestionAnswerer import *

import ipdb

import wandb

class Trainer:
    def __init__(self, model: Model):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr = 1e-5)

    def train(self, data, attn, target) -> float:
        dataset = TensorDataset(data, attn, target)
        dataloader = DataLoader(dataset, batch_size = 11, shuffle = True)

        loss = 0.
        for e in range(50):
            logging.info(f'Starting epoch {e}')
            loss = self.trainEpoch(dataloader)

            logging.info(f'Epoch {e} has loss {loss}')
            wandb.log(dict(loss = loss))

        return loss

    def trainEpoch(self, loader) -> float:
        epoch_loss = 0.
        for e, (data, attn, target) in enumerate(loader):
            logging.info(f'Starting minibatch {e}.')
            data = data.to(self.model.device)
            attn = attn.to(self.model.device)
            target = target.to(self.model.device)

            self.optimizer.zero_grad()

            output = self.model.model(input_ids = data, attention_mask = attn, labels = target)

            epoch_loss += output.loss.cpu().item() / data.shape[0]

            if output.loss.isnan():
                raise ValueError('Loss is nan')

            output.loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1)
            self.optimizer.step()

            logging.info(f'{e}: loss = {output.loss.cpu().item()}')

        return epoch_loss
