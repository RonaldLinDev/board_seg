from torch import nn
class edgefinder(nn.Module):
    def __init__(self):
        super.__init__() 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d()
        
        self.linear1 = nn.Linear(32 * 8 * 8, 10, out_features = 16)
        self.linear2 = nn.Linear(16 * 4 * 4, 4)
            
    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x))) 
        x = self.maxpool(self.relu(self.conv2(x))) 
    #   x = self.maxpool(self.relu(self.conv3(x))) 
    #   x = self.maxpool(self.relu(self.conv4(x))) 
    
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
    
        return x
    