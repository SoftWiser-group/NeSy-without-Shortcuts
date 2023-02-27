import torch
import torch.nn as nn

class VGG(nn.Module):

    def __init__(self, cfgs, num_classes=10, init_weights=False):
        super(VGG, self).__init__()
        channels = 3
        self.layer1, channels = self._make_layers(cfgs[0], channels)
        self.layer2, channels = self._make_layers(cfgs[1], channels)
        self.layer3, channels = self._make_layers(cfgs[2], channels)
        self.layer4, channels = self._make_layers(cfgs[3], channels)
        self.layer5, channels = self._make_layers(cfgs[4], channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout())
        self.fc2 =  nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout())
        self.classifier = nn.Linear(4096, num_classes)  
        if init_weights:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = x
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.classifier(out)
        return out    

    def penultimate_forward(self, x):
        out = x
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out            

    def feature_list(self, x):
        out_list = []
        out = self.layer1(x)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = self.layer5(out)
        out_list.append(out)  
        out = self.avgpool(out)      
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out_list.append(out)       
        out = self.fc2(out)
        out_list.append(out)       
        y = self.classifier(out)
        return y, out_list  

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = self.layer1(x)
        if layer_index == 1:
            out = self.layer2(out)
        elif layer_index == 2:
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 3:
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)   
        elif layer_index == 4:
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)                         
            out = self.layer5(out)                         
        elif layer_index == 5:
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)                         
            out = self.layer5(out)    
            out = self.avgpool(out)         
            out = torch.flatten(out, 1)
            out = self.fc1(out)    
        elif layer_index == 6: 
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)                         
            out = self.layer5(out)  
            out = self.avgpool(out)           
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = self.fc2(out)   
        return out              


    def _make_layers(self, cfg, channels, batch_norm=True):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                channels = v
        return nn.Sequential(*layers), channels

cfgs = {
    'A': [[64, 'M'], [128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'B': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'D': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 'M'], [512, 512, 512, 'M']],
    'E': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M']],
}        



def _vgg(cfg, num_classes):
    model = VGG(cfgs[cfg], num_classes)
    return model


def vgg11(num_classes, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('A', num_classes)

def vgg13(num_classes, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('B', num_classes)

def vgg16(num_classes, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('D', num_classes)

def vgg19(num_classes, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('E', num_classes)
