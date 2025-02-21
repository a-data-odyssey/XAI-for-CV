import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class CNN(nn.Module):
    """Convolutional Neural Network for image classification."""
    
    def __init__(self, num_classes=4, input_channels=3, conv_filters=[16, 32, 64], kernel_size=3, dropout_rate=0,input_dim=256):
        super(CNN, self).__init__() 

        # Dynamic Convolutional layers
        layers = []
        in_channels = input_channels

        for out_channels in conv_filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Compute the size after convolutional layers for the fully connected layers
        self.flattened_size = conv_filters[-1] * input_dim//(2**len(conv_filters)) * input_dim//(2**len(conv_filters))  

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.flattened_size, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = self.fc_layers(x)
        return x

class CNNWithGAP(nn.Module):
    """Convolutional Neural Network with Global Average Pooling for image classification."""
    
    def __init__(self, num_classes=4, input_channels=3, conv_filters=[16, 32, 64], kernel_size=3, dropout_rate=0):
        super(CNNWithGAP, self).__init__() 

        # Convolutional layers without max pooling
        layers = []
        in_channels = input_channels

        for out_channels in conv_filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Global Average Pooling layer
        self.gap = nn.AdaptiveAvgPool2d(1)  # GAP to a 1x1 spatial dimension

        # Fully connected layer (output layer only)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(conv_filters[-1], num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x)  # Apply Global Average Pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layer
        x = self.fc(x)
        return x


#================================================================================================
# U-Net
#================================================================================================

def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class U_Net(nn.Module):
    def __init__(self, img_ch=7, output_ch=2):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1