import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l_output = nn.Linear(hidden_size, num_classes)
        self.projection = nn.Linear(input_dim, hidden_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)

    def forward(self, x_input):
        """
        Feed forward network with residual connections.
        """
        x_proj = self.projection(x_input)
        x_ = self.bn1(F.leaky_relu(self.l1(x_input)))
        x = self.bn2(F.leaky_relu(self.l2(x_) + x_proj))
        x = self.bn3(F.leaky_relu(self.l3(x) + x_proj))
        x = self.l_output(self.bn4(F.leaky_relu(self.l4(x) + x_)))
        return x