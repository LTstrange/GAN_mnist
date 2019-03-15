import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# 超参数
BATCH_SIZE = 60
LR_G = 0.01
LR_D = 0.0003
N_IDEAS = 5
EPOCH = 100
DOWNLOAD_MNIST = False


train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,                        # download it if you don't have it
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.generator0 = nn.Sequential(
            nn.Linear(N_IDEAS, 28 * 28),
            nn.Sigmoid()
        )
        self.generator1 = nn.Sequential(
            nn.Linear(N_IDEAS, 28 * 28),
            nn.Sigmoid()
        )
        self.generator2 = nn.Sequential(
            nn.Linear(N_IDEAS, 28 * 28),
            nn.Sigmoid()
        )
        self.generator3 = nn.Sequential(
            nn.Linear(N_IDEAS, 28 * 28),
            nn.Sigmoid()
        )
        self.generator4 = nn.Sequential(
            nn.Linear(N_IDEAS, 28 * 28),
            nn.Sigmoid()
        )
        self.generator5 = nn.Sequential(
            nn.Linear(N_IDEAS, 28 * 28),
            nn.Sigmoid()
        )
        self.generator6 = nn.Sequential(
            nn.Linear(N_IDEAS, 28 * 28),
            nn.Sigmoid()
        )
        self.generator7 = nn.Sequential(
            nn.Linear(N_IDEAS, 28 * 28),
            nn.Sigmoid()
        )
        self.generator8 = nn.Sequential(
            nn.Linear(N_IDEAS, 28 * 28),
            nn.Sigmoid()
        )
        self.generator9 = nn.Sequential(
            nn.Linear(N_IDEAS, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        g0 = self.generator0(x).unsqueeze(1)
        g1 = self.generator1(x).unsqueeze(1)
        g2 = self.generator2(x).unsqueeze(1)
        g3 = self.generator3(x).unsqueeze(1)
        g4 = self.generator4(x).unsqueeze(1)
        g5 = self.generator5(x).unsqueeze(1)
        g6 = self.generator6(x).unsqueeze(1)
        g7 = self.generator7(x).unsqueeze(1)
        g8 = self.generator8(x).unsqueeze(1)
        g9 = self.generator9(x).unsqueeze(1)

        y = torch.cat([g0, g1, g2, g3, g4, g5, g6, g7, g8, g9], dim=1)
        label = torch.unsqueeze(label, 1)
        output = torch.bmm(label.float(), y)
        output = output.view((-1, 1, 28, 28))
        return output


class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()

        self.detector0 = nn.Sequential(
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.detector1 = nn.Sequential(
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.detector2 = nn.Sequential(
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.detector3 = nn.Sequential(
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.detector4 = nn.Sequential(
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.detector5 = nn.Sequential(
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.detector6 = nn.Sequential(
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.detector7 = nn.Sequential(
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.detector8 = nn.Sequential(
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.detector9 = nn.Sequential(
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x, label):
        x = x.view(x.size(0), -1)

        d0 = self.detector0(x)
        d1 = self.detector1(x)
        d2 = self.detector2(x)
        d3 = self.detector3(x)
        d4 = self.detector4(x)
        d5 = self.detector5(x)
        d6 = self.detector6(x)
        d7 = self.detector7(x)
        d8 = self.detector8(x)
        d9 = self.detector9(x)

        x = torch.cat([d0, d1, d2, d3, d4, d5, d6, d7, d8, d9], dim=1).unsqueeze(1)

        label = torch.unsqueeze(label, 2)
        predict = torch.bmm(x, label)
        return predict


D = D_Net()
G = G_Net()

print("D_net:", D)
print("G_net:", G)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)


for epoch in range(EPOCH):
    for step, (data, target) in enumerate(train_loader):
        tar = torch.zeros((BATCH_SIZE, 10))
        target = torch.unsqueeze(target, 1)
        onehot_tar = tar.scatter_(1, target, 1)
        ideas = torch.randn((BATCH_SIZE, N_IDEAS))

        Fpaint = G(ideas, onehot_tar)

        prob0 = D(Fpaint, onehot_tar)
        prob1 = D(data, onehot_tar)

        D_loss = - torch.mean(torch.log(prob1) + torch.log(1. - prob0))
        G_loss = torch.mean(torch.log(1. - prob0))
        if step < 500 and torch.mean(prob0) > 0.1:
            opt_D.zero_grad()
            D_loss.backward(retain_graph=True)
            opt_D.step()
        else:
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

        if step % 250 == 0:
            print("Epoch:", epoch, "Step:", step, "D_loss: % .4f" % D_loss.data.numpy(), "G_loss: %.4f" % G_loss, "prob0", torch.mean(prob0).data.numpy())
        if step == 0:
            plt.imsave("image/image%d|%d(%d)" % (epoch, step, target[0]), (Fpaint[0]*255).view((28, 28)).int().numpy(), cmap="gray")
    torch.save(G, "net.pkl")

#
# plt.figure()
# plt.imshow((Fpaint[0]*255).view((28, 28)).int().numpy(), cmap="gray")
# plt.ylabel("%d" % target[0])
# plt.show()
