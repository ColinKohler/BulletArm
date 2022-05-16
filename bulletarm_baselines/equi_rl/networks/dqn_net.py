import torch
import torch.nn as nn
import torch.nn.functional as F

# similar amount of parameters
class CNNCom(nn.Module):
    def __init__(self, input_shape=(2, 128, 128), n_p=2, n_theta=1):
        super().__init__()
        self.n_inv = 3 * n_theta * n_p
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 8x8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 6x6
            nn.MaxPool2d(2),
            # 3x3
            nn.Conv2d(512, 18, kernel_size=1, padding=0),
        )

        self.n_p = n_p
        self.n_theta = n_theta

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, x):
        batch_size = x.shape[0]
        q = self.conv(x)
        q = q.reshape(batch_size, self.n_inv, 9).permute(0, 2, 1)
        return q

class DQNComCURL(nn.Module):
    def __init__(self, input_shape=(2, 128, 128), n_p=2, n_theta=1, curl_z=128):
        super().__init__()
        conv_out_size = ((((input_shape[1]//2)//2)//2)//2)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 8x8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Flatten(),
            torch.nn.Linear(512 * conv_out_size * conv_out_size, 1024),
            nn.ReLU(inplace=True),
        )

        self.W_h = nn.Parameter(torch.rand(1024, 256))
        self.layer_norm_1 = nn.LayerNorm(256)
        self.W_c = nn.Parameter(torch.rand(256, 128))
        self.b_h = nn.Parameter(torch.zeros(256))
        self.b_c = nn.Parameter(torch.zeros(128))
        self.W = nn.Parameter(torch.rand(128, 128))
        self.layer_norm_2 = nn.LayerNorm(128)

        self.n_p = n_p
        self.n_theta = n_theta
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1024, 9 * 3 * n_theta * n_p),
        )

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        q = self.fc(x)

        h = torch.matmul(x, self.W_h) + self.b_h  # Contrastive head
        h = self.layer_norm_1(h)
        h = F.relu(h)
        h = torch.matmul(h, self.W_c) + self.b_c  # Contrastive head
        h = self.layer_norm_2(h)
        return q, h

class DQNComCURLOri(nn.Module):
    def __init__(self, input_shape=(2, 128, 128), n_p=2, n_theta=1):
        super().__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 5, stride=5, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=5, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        x = torch.randn([1] + list(input_shape))
        conv_out_dim = self.conv(x).reshape(-1).shape[-1]

        self.W_h = nn.Parameter(torch.rand(conv_out_dim, 256))
        self.layer_norm_1 = nn.LayerNorm(256)
        self.W_c = nn.Parameter(torch.rand(256, 128))
        self.b_h = nn.Parameter(torch.zeros(256))
        self.b_c = nn.Parameter(torch.zeros(128))
        self.W = nn.Parameter(torch.rand(128, 128))
        self.layer_norm_2 = nn.LayerNorm(128)

        self.n_p = n_p
        self.n_theta = n_theta
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_out_dim, 256),
            nn.ReLU(),
            torch.nn.Linear(256, 9 * 3 * n_theta * n_p),
        )

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        q = self.fc(x)

        h = torch.matmul(x, self.W_h) + self.b_h  # Contrastive head
        h = self.layer_norm_1(h)
        h = F.relu(h)
        h = torch.matmul(h, self.W_c) + self.b_c  # Contrastive head
        h = self.layer_norm_2(h)
        return q, h