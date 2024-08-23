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
    def __init__(
        self,
        model: Model,
        data: LongTensor,
        data_attn: BoolTensor,
        target: LongTensor,
        target_attn: BoolTensor
    ):
        self.model = model
        self.data = data
        self.data_attn = data_attn
        self.target = target
        self.target_attn = target_attn

        self.optimizer = Adam(model.parameters(), lr = 1e-5)

    def train(self) -> float:
        dataset = TensorDataset(self.data, self.data_attn, self.target)
        dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

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

            # ipdb.set_trace()
            outputs = self.model.model(input_ids = data, attention_mask = attn, labels = target)
            loss = outputs.loss

            epoch_loss += loss.cpu().item() / data.shape[0]

            if loss.isnan():
                raise ValueError('Loss is nan')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1)
            self.optimizer.step()

            logging.info(f'{e}: loss = {loss.cpu().item()}')

        return epoch_loss
