import torch.nn as nn
import math
from .fusion import REFusion
from .utils import Bottleneck

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class PyramidFeatures(nn.Module):
    def __init__(self, dataset_name, C2_size, C3_size, C4_size, C5_size, feature_size=256): 
        super(PyramidFeatures, self).__init__()
        
        self.dataset_name = dataset_name

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0) 
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1) 

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0) 
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)


        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        
        if self.dataset_name == 'dsec':
            self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
            self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
            self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        elif self.dataset_name == 'ddd17':
            self.P5_upsampled = nn.Upsample(size=(17, 22), mode='bilinear', align_corners=False)
            self.P4_upsampled = nn.Upsample(size=(33, 44), mode='bilinear', align_corners=False)
            self.P3_upsampled = nn.Upsample(size=(65, 87), mode='bilinear', align_corners=False)
        else:
            raise ValueError("Unsupported dataset name") 
        

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs  

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x) 
        P5_x = self.P5_2(P5_x) 

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x 
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)

        P6_x = self.P6(C5)

        return [P2_x,P3_x, P4_x, P5_x, P6_x]




class ResNet_fpn_fusion(nn.Module):

    def __init__(self, dataset_name, event_block, rgb_block, event_layers, rgb_layers):
        #rgb branch
        self.inplanes = 64
        super(ResNet_fpn_fusion, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.dropout = nn.Dropout(0.2) 
        self.bn1 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(rgb_block, 64, rgb_layers[0])
        self.layer2 = self._make_layer(rgb_block, 128, rgb_layers[1], stride=2)
        self.layer3 = self._make_layer(rgb_block, 256, rgb_layers[2], stride=2)
        self.layer4 = self._make_layer(rgb_block, 512, rgb_layers[3], stride=2)
        rgb_fpn_sizes = [
            self.layer1[rgb_layers[0] - 1].conv3.out_channels,
            self.layer2[rgb_layers[1] - 1].conv3.out_channels,
            self.layer3[rgb_layers[2] - 1].conv3.out_channels,
            self.layer4[rgb_layers[3] - 1].conv3.out_channels
        ]

    


        
        #if rgb_block == BasicBlock:
            #rgb_fpn_sizes = [self.layer1[rgb_layers[0] - 1].conv2.out_channels, self.layer2[rgb_layers[1] - 1].conv2.out_channels, self.layer3[rgb_layers[2] - 1].conv2.out_channels,
                         #self.layer4[rgb_layers[3] - 1].conv2.out_channels]
        #elif rgb_block == Bottleneck:
            #rgb_fpn_sizes = [self.layer1[rgb_layers[0] - 1].conv3.out_channels, self.layer2[rgb_layers[1] - 1].conv3.out_channels, self.layer3[rgb_layers[2] - 1].conv3.out_channels,
                         #self.layer4[rgb_layers[3] - 1].conv3.out_channels]
        #else:
            #raise ValueError(f"Block type {rgb_block} not understood")

        #Event branch
        self.inplanes = 64
        self.conv1_event = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_event = nn.BatchNorm2d(64)
        self.relu_event = nn.ReLU(inplace=True)
        self.maxpool_event = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_event = self._make_layer(event_block, 64, event_layers[0])
        self.layer2_event = self._make_layer(event_block, 128, event_layers[1], stride=2)
        self.layer3_event = self._make_layer(event_block, 256, event_layers[2], stride=2)
        self.layer4_event = self._make_layer(event_block, 512, event_layers[3], stride=2)
        event_fpn_sizes = [

            self.layer1_event[event_layers[0] - 1].conv3.out_channels,

            self.layer2_event[event_layers[1] - 1].conv3.out_channels,

            self.layer3_event[event_layers[2] - 1].conv3.out_channels,

            self.layer4_event[event_layers[3] - 1].conv3.out_channels

        ]
        self.fpn = PyramidFeatures(dataset_name, rgb_fpn_sizes[0]+event_fpn_sizes[0], rgb_fpn_sizes[1]+event_fpn_sizes[1], rgb_fpn_sizes[2]+event_fpn_sizes[2], rgb_fpn_sizes[3]+event_fpn_sizes[3])

        #if event_block == BasicBlock:
            #event_fpn_sizes = [self.layer1[event_layers[0] - 1].conv2.out_channels, self.layer2[event_layers[1] - 1].conv2.out_channels, self.layer3[event_layers[2] - 1].conv2.out_channels,
                         #self.layer4[event_layers[3] - 1].conv2.out_channels]
        #elif event_block == Bottleneck:
            #event_fpn_sizes = [self.layer1[event_layers[0] - 1].conv3.out_channels, self.layer2[event_layers[1] - 1].conv3.out_channels, self.layer3[event_layers[2] - 1].conv3.out_channels,
                         #self.layer4[event_layers[3] - 1].conv3.out_channels]
        #else:
            #raise ValueError(f"Block type {event_block} not understood")

        

        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


        
        self.fus = nn.ModuleList([
            REFusion(256,256),
            REFusion(512, 512),
            REFusion(1024, 1024),
            REFusion(2048, 2048)
        ])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    

    def forward(self, inputs):
        img_batch_rgb, img_batch_event = inputs
          
    
        #if self.training:
            #img_batch_rgb, img_batch_event,annotations = inputs
            #if random.uniform(0, 1) < 0.15: #the inputfrom the RGB image is made completely blank (zeros) witha probability of 0.15
                #img_batch_rgb.zero_()
        #else:
            #img_batch_rgb, img_batch_event = inputs
        #rgb stream
        x_rgb = self.conv1(img_batch_rgb)
        x_rgb = self.bn1(x_rgb)
        x_rgb = self.relu(x_rgb)
        x_rgb = self.maxpool(x_rgb)

        x1_rgb = self.layer1(x_rgb)
        x2_rgb = self.layer2(x1_rgb)
        x3_rgb = self.layer3(x2_rgb)
        x4_rgb = self.layer4(x3_rgb)

        #Event stream
        x_event = self.conv1_event(img_batch_event)
        x_event = self.bn1_event(x_event)
        x_event = self.relu_event(x_event)
        x_event = self.maxpool_event(x_event)

        x1_event = self.layer1_event(x_event)
        x2_event = self.layer2_event(x1_event)
        x3_event = self.layer3_event(x2_event)
        x4_event = self.layer4_event(x3_event)

        x1 = self.fus[0](x1_event, x1_rgb)
        x2 = self.fus[1](x2_event, x2_rgb)
        x3 = self.fus[2](x3_event, x3_rgb)
        x4 = self.fus[3](x4_event, x4_rgb)
        
        # x1 = self.fus[0](x1_rgb, x1_event)
        # x2 = self.fus[1](x2_rgb, x2_event)
        # x3 = self.fus[2](x3_rgb, x3_event)
        # x4 = self.fus[3](x4_rgb, x4_event)
    
        features = self.fpn([x1, x2, x3, x4])      

        
        return features

        
