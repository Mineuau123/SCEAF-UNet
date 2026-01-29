def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    assert num_channels % groups == 0, "channels must be divisible by groups"
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class CSU(nn.Module):
    def __init__(self, in_ch, out_ch, shuffle_groups=2):
        super().__init__()
        self.dwconv = nn.Conv2d(in_ch, in_ch, 9, padding=4, groups=in_ch, bias=False)
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.shuffle_groups = shuffle_groups

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        
        # depthwise separable conv
        feat = self.bn(self.pwconv(self.dwconv(x_up)))
        
        # channel shuffle
        feat = channel_shuffle(feat, self.shuffle_groups)
        
        return F.relu(feat, inplace=True)
