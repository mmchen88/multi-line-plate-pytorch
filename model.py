<<<<<<< HEAD
import torch
from torch import nn


class PlateNet(nn.Module):
    def __init__(self, batch_size, n_class=66):
        super(PlateNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1))
        )
        self.gru_1 = nn.GRU(512, 128, batch_first=True)
        # self.gru_1_h0 = torch.randn(1, batch_size, 128)
        self.gru_1b = nn.GRU(512, 128, batch_first=True)
        # self.gru_1b_h0 = torch.randn(batch_size, 1, 128)

        self.gru_2 = nn.GRU(128, 256, batch_first=True)
        self.gru_2b = nn.GRU(128, 256, batch_first=True)

        self.Dropout = nn.Dropout(p=0.25)
        self.Dense = nn.Linear(512, n_class)

    def forward(self, x, batch_size):
        x = self.backbone(x)
        top = x[:, :, :, 0]
        top = torch.reshape(top, shape=(batch_size, 512, 12, 1))
        bottom = x[:, :, :, 1]
        bottom = torch.reshape(bottom, shape=(batch_size, 512, 12, 1))

        x = torch.cat((top, bottom), 2)
        x = torch.reshape(x, shape=(batch_size, 24, 512))

        gru_1, _ = self.gru_1(x)
        x_b = torch.flip(x, [2])
        gru_1b, _ = self.gru_1b(x_b)

        gru1_merged = torch.add(gru_1, gru_1b)

        gru_2, _ = self.gru_2(gru1_merged)
        gru1_merged_b = torch.flip(gru1_merged, [2])
        gru_2b, _ = self.gru_2b(gru1_merged_b)

        x = torch.cat((gru_2, gru_2b), 2)
        x = self.Dropout(x)
        y_predict = self.Dense(x)
        return y_predict

    def get_optim_policy(self):
        params = [
            {'params': self.backbone.parameters()},
            {'params': self.gru_1.parameters()},
            {'params': self.gru_1b.parameters()},
            {'params': self.gru_2.parameters()},
            {'params': self.gru_2b.parameters()},
            {'params': self.Dropout.parameters()},
            {'params': self.Dense.parameters()},
        ]
        return params


=======
import torch
from torch import nn


class PlateNet(nn.Module):
    def __init__(self, batch_size, n_class=66):
        super(PlateNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1))
        )
        self.gru_1 = nn.GRU(512, 128, batch_first=True)
        # self.gru_1_h0 = torch.randn(1, batch_size, 128)
        self.gru_1b = nn.GRU(512, 128, batch_first=True)
        # self.gru_1b_h0 = torch.randn(batch_size, 1, 128)

        self.gru_2 = nn.GRU(128, 256, batch_first=True)
        self.gru_2b = nn.GRU(128, 256, batch_first=True)

        self.Dropout = nn.Dropout(p=0.25)
        self.Dense = nn.Linear(512, n_class)

    def forward(self, x, batch_size):
        x = self.backbone(x)
        top = x[:, :, :, 0]
        top = torch.reshape(top, shape=(batch_size, 512, 12, 1))
        bottom = x[:, :, :, 1]
        bottom = torch.reshape(bottom, shape=(batch_size, 512, 12, 1))

        x = torch.cat((top, bottom), 2)
        x = torch.reshape(x, shape=(batch_size, 24, 512))

        gru_1, _ = self.gru_1(x)
        x_b = torch.flip(x, [2])
        gru_1b, _ = self.gru_1b(x_b)

        gru1_merged = torch.add(gru_1, gru_1b)

        gru_2, _ = self.gru_2(gru1_merged)
        gru1_merged_b = torch.flip(gru1_merged, [2])
        gru_2b, _ = self.gru_2b(gru1_merged_b)

        x = torch.cat((gru_2, gru_2b), 2)
        x = self.Dropout(x)
        y_predict = self.Dense(x)
        return y_predict

    def get_optim_policy(self):
        params = [
            {'params': self.backbone.parameters()},
            {'params': self.gru_1.parameters()},
            {'params': self.gru_1b.parameters()},
            {'params': self.gru_2.parameters()},
            {'params': self.gru_2b.parameters()},
            {'params': self.Dropout.parameters()},
            {'params': self.Dense.parameters()},
        ]
        return params


>>>>>>> 95dea5d4413c3dc744d4c2f82ba515b9f4bc53f1
