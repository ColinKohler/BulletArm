import torch
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Softmax
from torch import flatten
import torchvision.transforms as transforms
import cv2

class Q_value(Module):
    def __init__(self, state_size, action_size, hidden_size1=32, hidden_size2=64):
        super(Q_value, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.learning_rate = 0.001

        # First convolutional layer that consists of a conv, relu and maxpool
        self.conv1 = Conv2d(self.state_size, self.hidden_size1, (3,3))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d((3,3), stride=2)

        # Second convolutional layer that consists of a conv, relu and maxpool
        self.conv2 = Conv2d(self.hidden_size1, self.hidden_size2, (5,5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d((5,5), stride=(2, 2))

        # Initialize the FC layer
        self.fc1 = Linear(in_features=222784, out_features = 500)
        self.relu3 = ReLU()

        # Final layer that outputs the q_values
        self.fc2 = Linear(in_features = 500, out_features = self.action_size)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        # batch_size = x.size(1)
        # hidden = self.init_hidden(batch_size)

        # pass the input to our first layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # pass the output of the previous layer as input to the second layer
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # pass the conv output to the linear layers after flattening
        x = flatten(x,1)
        x = self.fc1(x)
        x = self.relu3(x)

        # pass through the softmax layer to get the qvalues
        x = self.fc2(x)
        q_values = self.softmax(x)

        return q_values



if __name__ == '__main__':
    img = cv2.imread('batman-image-zack-snyder.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dim = (256, 256)
    img = cv2.resize(img, dim)
    transform = transforms.Compose([transforms.ToTensor()])
    input = transform(img)
    input = torch.unsqueeze(input, 0)
    input_size = input.shape[1]
    output_size = 2
    model = Q_value(input_size, output_size)
    q_value = model(input)
    print(q_value)