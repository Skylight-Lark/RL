import torch
import torch.nn.functional as F


class Qnet(torch.nn.Module):
    ''' 定义Q网络 '''
    def __init__(self, state_channel, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Conv2d(state_channel, 32, 8, stride=4)
        self.fc2 = torch.nn.Conv2d(32, 64, 4, stride=2)
        self.fc3 = torch.nn.Conv2d(64, 64, 3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        self.fc5 = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x