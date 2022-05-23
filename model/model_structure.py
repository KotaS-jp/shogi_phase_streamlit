import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

# 作成済みモデルetl8g3のネットワークを宣言
class Net1(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(2304, 956)#256 = 4*8*8?(3, 48, 48画像がpadding後に, , になる)

    def forward(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = self.bn1(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.conv2(h)
        h = F.relu(h)
        h = self.bn2(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.conv3(h)
        h = F.relu(h)
        h = self.bn3(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.conv4(h)
        h = F.relu(h)
        h = self.bn4(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(-1, 2304)#view関数は(-1, output)とすると自動でベクトルにしてくれる
        h = self.fc1(h)
        return h


    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy(y.softmax(dim=-1), t), on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


class Net2(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature_extractor = Net1()#特徴抽出機を訓練済みモデルのクラスからインスタンス化。net1()ではなくNet1()
        self.fc1 = nn.Linear(956, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 29)
        self.bn1 = nn.BatchNorm1d(300)
        self.bn2 = nn.BatchNorm1d(100)
    
    def forward(self, x):
        h = self.feature_extractor(x)
        h = self.fc1(h)

        h = F.relu(h)
        h = self.bn1(h)
        h = self.fc2(h)

        h = F.relu(h)
        h = self.bn2(h)
        h = self.fc3(h)

        return h
    
    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer