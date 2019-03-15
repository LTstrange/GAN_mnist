import torch
import torch.nn as nn
import matplotlib.pyplot as plt

N_IDEAS = 5


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


G_net = torch.load('net.pkl')

ideas = torch.randn((100, 5))

all_tar = torch.zeros((10, 1))
for i in range(1, 10):
    tar = torch.ones((10, 1))*i
    all_tar = torch.cat([all_tar, tar])

all_tar = all_tar.long()

onehot_tar = torch.zeros((100, 10))
onehot_tar = onehot_tar.scatter_(1, all_tar, 1)


result = G_net(ideas, onehot_tar)

result = result.view(10, 10, 28, 28)

result = torch.cat(list(result), 1)
print(result.shape)
result = torch.cat(list(result), 1)
print(result.shape)

plt.imsave('show', (result*255).int().numpy(), cmap='gray')





