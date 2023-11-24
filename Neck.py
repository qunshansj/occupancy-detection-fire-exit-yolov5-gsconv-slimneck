python
# GSConv层定义
class GSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GSConv, self).__init__()
        self.main = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.side = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.main(x) + self.side(x)


# GSbottleneck定义
class GSbottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GSbottleneck, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = GSConv(in_channels, mid_channels, 1)
        self.conv2 = GSConv(mid_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


# VoV-GSCSP定义
class VoV_GSCSP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VoV_GSCSP, self).__init__()
        self.conv = GSbottleneck(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)


# Slim-Neck
class SlimNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SlimNeck, self).__init__()
        self.layer = VoV_GSCSP(in_channels, out_channels)

    def forward(self, x):
        return self.layer(x)
