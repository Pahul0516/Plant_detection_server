from torchmetrics import Accuracy
import torch
import pytorch_lightning as pl

class ClassificationLitModuleDenseNet(pl.LightningModule):
    def __init__(self, model, lr=1e-3, num_classes=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes or 30

        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=self.num_classes)

    def forward(self, x):
        outputs = self.model(x)
        # Some models (or wrappers) return an object with a `.logits` attribute (e.g., torchvision/Lightning wrappers).
        if hasattr(outputs, 'logits'):
            return outputs.logits
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", self.train_acc(preds, y), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", self.val_acc(preds, y), on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
