# 必要なモジュールのインポート
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
#学習時に使ったのと同じ学習済みモデルをインポート
from torchvision.models import densenet121
feature = densenet121(weights='DEFAULT')

# 1. リサイズ変換を定義
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        # DenseNet121の最後の全結合層を変更して300次元の出力を持つようにする
        dnet = densenet121(pretrained=True)
        num_ftrs = dnet.classifier.in_features
        dnet.classifier = nn.Linear(num_ftrs, 300)
        self.feature = dnet

        self.fc1 = nn.Linear(300, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 3)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        print("Input shape:", x.shape)
        h = self.conv(x)
        print("After Conv2d shape:", h.shape)
        h = self.bn(h)
        h = F.relu(h)
        print("After ReLU shape:", h.shape)
        h = self.feature(h)
        print("After densenet121 shape:", h.shape)
        h = self.fc1(h)
        print("After fc1 shape:", h.shape)
        h = self.fc2(h)
        print("After fc2 shape:", h.shape)
        h = self.fc3(h)
        print("After fc3 shape:", h.shape)
        return h