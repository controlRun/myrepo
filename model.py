import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

class VGG_SMALL(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.nonlinear0 = nn.ReLU(inplace=True)
        
        self.bn1 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.nonlinear1 = nn.ReLU(inplace=True)
        
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        #self.dropout2 = nn.Dropout(p=0.5)  # dropout训练
        self.nonlinear2 = nn.ReLU(inplace=True)
        #self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.nonlinear3 = nn.ReLU(inplace=True)
        
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        #self.dropout4 = nn.Dropout(p=0.5)  # dropout训练
        self.nonlinear4 = nn.ReLU(inplace=True)
        #self.pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bn5 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.pooling5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.nonlinear5 = nn.ReLU(inplace=True)
        
        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear0(x)
        
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.pooling1(x)
        x = self.nonlinear1(x)
        
        x = self.bn2(x)
        x = self.conv2(x)
        #x = self.dropout2(x)
        #x = self.pooling2(x)
        x = self.nonlinear2(x)
        
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.pooling3(x)
        x = self.nonlinear3(x)
        
        x = self.bn4(x)
        x = self.conv4(x)
        #x = self.dropout4(x)
        #x = self.pooling4(x)
        x = self.nonlinear4(x)
        
        x = self.bn5(x)
        x = self.conv5(x)
        x = self.pooling5(x)
        x = self.nonlinear5(x)
        
        # x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
