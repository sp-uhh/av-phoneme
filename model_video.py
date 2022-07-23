import torch
import torch.nn as nn

class LipNet(torch.nn.Module):
    def __init__(self, viseme=12, dropout_p=0.5):
        super(LipNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))     
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.gru1  = nn.GRU(1536, 256, 1, bidirectional=True)
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)
        
        self.FC    = nn.Linear(512, viseme)
        self.dropout_p  = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)        
        self.dropout3d = nn.Dropout3d(self.dropout_p)  
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, x):
        x = x.view(-1, 1, x.size(1), x.size(2), x.size(3))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool3(x)

        x = x.permute(2, 0, 1, 3, 4).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
    
        x, h = self.gru1(x)        
        x = self.dropout(x)
        x, h = self.gru2(x)   
        x = self.dropout(x)

        # comment the following line for AV-Model
        x = self.FC(x)
        x = x.permute(1, 0, 2).contiguous()
        # comment the following line for AV-Model
        x = self.softmax(x)
        return x