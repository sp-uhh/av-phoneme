# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# audio model resnet
class BasicBlockSpeech(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockSpeech, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNetSpeech(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetSpeech, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=8, padding=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        return x

# used for audio and mfcc model
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):#, lengths):
        #total_length = max(lengths)
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).to('cuda')
        x, _ = self.gru(x, h0)

        # comment the following two lines for AV-Model
        #x = self.fc(x)
        #x = self.softmax(x)
        return x

class SpeechModelAV(nn.Module):
    def __init__(self, inputDim=256, hiddenDim=512, nClasses=38):
        super(SpeechModelAV, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.nLayers = 2
        # frontend1D
        self.fronted1D = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(True)
                )
        # resnet
        self.resnet18 = ResNetSpeech(BasicBlockSpeech, [2, 2, 2, 2], num_classes=self.inputDim)
        # backend_gru
        self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses)
        # initialize
        self._initialize_weights()

    def forward(self, x):
        x = x.view(-1, 1, x.size(1))
        x = self.fronted1D(x)
        x = self.resnet18(x)
        x = self.gru(x)#, lengths)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class LipNetAV(torch.nn.Module):
    def __init__(self, viseme=12, dropout_p=0.5):
        super(LipNetAV, self).__init__()
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
        #x = self.FC(x)
        x = x.permute(1, 0, 2).contiguous()
        # comment the following line for AV-Model
        #x = self.softmax(x)
        return x

# AV Fusion with Squeeze + Excitation
class AVGRUSE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, reduction=16):
        super(AVGRUSE, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.se = SEBlock(input_size, hidden_size, reduction=reduction)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True) # - hidden_size (for input)
        self.fc = nn.Linear(2*hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = self.se(x, y)        
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).to('cuda')
        x, _ = self.gru(x, h0)
        x = self.fc(x)
        x = self.softmax(x)
        return x

# AV Fusion with Concatenation
class AVGRUBase(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AVGRUBase, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True) # - hidden_size (for input)
        self.fc = nn.Linear(2*hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=2)
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).to('cuda')
        x, _ = self.gru(x, h0)
        x = self.fc(x)
        x = self.softmax(x)
        return x

# key_channels=912/value_channels=456 or key_channels=456/value_channels=228, head_count=38 for each phoneme class
class AVEffAttGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, key_channels = 912, value_channels=912):
        super(AVEffAttGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.att = FactorizedAttention(in_channels = input_size, key_channels = key_channels, head_count = 38, value_channels=value_channels)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = self.att(x,y)
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).to('cuda')
        x, _ = self.gru(x, h0)
        x = self.fc(x)
        x = self.softmax(x)
        return x

# squeeze and excitation block
class SEBlock(nn.Module):
    def __init__(self, channels, channels_out, reduction=8):
        super(SEBlock, self).__init__()
        mid_cannels = channels // reduction
        
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)    
        self.conv1 = nn.Conv1d(in_channels=channels//2 +1, out_channels=mid_cannels, kernel_size=1)
        self.activ = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=mid_cannels,out_channels=1024, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, y):
        x1 = self.pool(x)
        x1 = torch.cat((x1, y), dim=2)
        x1 = x1.moveaxis(1,2)
        w = self.conv1(x1)

        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        w = w.moveaxis(1,2)
        x = x * w
        return x

class FactorizedAttention(nn.Module):   
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv1d(in_channels, key_channels, 1)
        self.queries = nn.Conv1d(in_channels, key_channels, 1)
        self.values = nn.Conv1d(in_channels//2, value_channels, 1)
        self.reprojection = nn.Conv1d(value_channels, in_channels, 1)

    def forward(self, input_, video):
        keys = self.keys(input_.transpose(1,2))
        queries = self.queries(input_.transpose(1,2))
        values = self.values(video.transpose(1,2))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:,i * head_key_channels: (i + 1) * head_key_channels,:], dim=1)
            value = values[:,i * head_value_channels: (i + 1) * head_value_channels,:]
            
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)

        reprojected_value = self.reprojection(aggregated_values).transpose(1,2)
        attention = reprojected_value

        return attention

# full example of AV networks to create experiments
class AVGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AVGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.att = FactorizedAttention(in_channels = 1024, key_channels = 912, head_count = 38, value_channels=456)
        self.se = SEBlock(input_size, hidden_size)       
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True) # - hidden_size (for input)
        self.fc = nn.Linear(2*hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = self.segru(x,y)
        #h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).to('cuda')
        #x, _ = self.gru(x, h0)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def speech_modelAV(input_size=1024, hidden_size=1024, num_classes=38):
    model = SpeechModelAV(input_size, hidden_size, num_classes)
    return model

def mfcc_model(input_size=39, hidden_size=512, num_Layers=2, num_classes=38):
    model = GRU(input_size, hidden_size, num_Layers, num_classes)
    return model