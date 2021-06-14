import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from pytorch_lightning.metrics import Accuracy
from torch.optim.lr_scheduler import OneCycleLR


class KDDTrainer(pl.LightningModule):
    def __init__(self, model, learning_rate, steps_per_epoch):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.valid_acc = Accuracy()
        self.train_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, batch):
        return self.model(batch['input'].float())

    def training_step(self, batch, batch_idx):
        output = self(batch)
        train_loss = F.cross_entropy(output, batch['target'].long())
        train_acc = self.train_acc(output.softmax(dim=-1), batch['target'].long())
        return train_loss

    def training_epoch_end(self, losses):
        print(f"Training loss: {np.mean([i['loss'].item() for i in losses])}")
        print(f'Training accuracy: {self.train_acc.compute().item()}')
        self.train_acc.reset()
    
    def validation_step(self, batch, batch_idx: int):
        output = self(batch)
        val_loss = F.cross_entropy(output, batch['target'].long())
        val_acc = self.valid_acc(output.softmax(dim=-1), batch['target'].long())
        return val_loss

    def validation_epoch_end(self, out):
        print(f'Validation accuracy: {self.valid_acc.compute().item()}')
        self.valid_acc.reset()
    

    def test_step(self, batch, batch_idx: int):
        output = self(batch)
        val_loss = F.cross_entropy(output, batch['target'].long())
        val_acc = self.test_acc(output.softmax(dim=-1), batch['target'].long())
        return val_loss

    def test_epoch_end(self, out):
        print(f"Test loss: {np.mean([i.item() for i in out])}")
        print(f'Test accuracy: {self.test_acc.compute().item()}')
        self.test_acc.reset()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': OneCycleLR(optimizer,  max_lr=self.learning_rate, steps_per_epoch=self.steps_per_epoch, epochs=1, div_factor=2, final_div_factor=1.2, verbose=False),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]