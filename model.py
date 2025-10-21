import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, action_dim, hidden_dim=512, observation_shape=(4, 84, 84)):
        super(Model, self).__init__()

        # CNN Layers
        # Original: in_channels=1, out_channels=8, kernel_size=4, stride=2
        self.conv1 = nn.Conv2d(in_channels=observation_shape[0], out_channels=32, kernel_size=8, stride=4)
        # Original: in_channels=8, out_channels=16, kernel_size=4, stride=2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # Original: in_channels=16, out_channels=32, kernel_size=3, stride=2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Compute conv output size automatically
        conv_output_size = self.calculate_conv_output(observation_shape)
        print("conv_output_size: ", conv_output_size)

        # Fully connected layers
        # Original had 3 FC layers; now only 2 FC layers before output
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Optional: can remove this if you want exactly 2 FC

        self.output = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self.apply(self.weights_init)

    def calculate_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(o.numel())

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # Optional if keeping 3 FC layers
        return self.output(x)


    def save_the_model(self, filename='models/latest.pt'):
        torch.save(self.state_dict(), filename)
    
    def load_the_model(self, filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(filename))
            print(f"Loaded weights from {filename}")
        except FileNotFoundError:
            print(f"No weights file found at at {filename}")


def soft_update(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# CNN - Recognize Image
# FC layers