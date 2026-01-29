class MSAG(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int = 16, groups: int = 1):
        super().__init__()
        def ms_branch(in_ch, out_ch):
            def dw(in_ch, k, dil=1):
                pad = ((k - 1) // 2) * dil
                return nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, k, padding=pad, dilation=dil, groups=in_ch, bias=False),
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch)
                )
            return nn.ModuleList([
                nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)), # 1x1
                dw(in_ch, 3, 1),
                dw(in_ch, 5, 2),
                dw(in_ch, 7, 3),
            ])

        self.g_branches = ms_branch(F_g, F_int)
        self.x_branches = ms_branch(F_l, F_int)
        self.act = nn.GELU()

        self.fuse_proj = nn.Sequential(
            nn.Conv2d(F_int, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
            nn.GELU()
        )
        self.gate_a = nn.Conv2d(F_int, F_int, 1)  # sigmoid
        self.gate_m = nn.Conv2d(F_int, F_int, 1)  # tanh 

        nn.init.constant_(self.gate_a.bias, 1.0)  # sigmoid(1)≈0.73
        nn.init.zeros_(self.gate_m.bias)          # tanh(0)=0

        self.lateral_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(F_int, F_l, 1, bias=False),
            nn.BatchNorm2d(F_l)
        )

    def _agg(self, branches, inp):
        feats = [b(inp) for b in branches]
        return self.act(sum(feats))

    def forward(self, g, x):
        g_ms = self._agg(self.g_branches, g)
        x_ms = self._agg(self.x_branches, x)
        fused = self.fuse_proj(g_ms + x_ms)

        a = torch.sigmoid(self.gate_a(fused))   # (B, F_int, H, W)
        m = torch.tanh(self.gate_m(fused))      # (B, F_int, H, W)

        x_lat = self.lateral_x(x)               # (B, F_int, H, W)
        y = x_lat * a + x_lat * m              
        y = self.out_proj(y)                  
        return y


 

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor
    
class SE_gelu(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // r, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(ch // r, ch, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w

class CAM(nn.Module):
    def __init__(self, dim, drop_path=0.0, layerscale_init=1e-6, use_bn=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim) if use_bn else nn.GroupNorm(1, dim)  # LN 近似：GN(1,·)
        self.pw1   = nn.Conv2d(dim, 4*dim, 1, bias=True)
        self.act   = nn.GELU()
        self.pw2   = nn.Conv2d(4*dim, dim, 1, bias=True)
        self.se    = SE_gelu(dim, r=16)
        self.gamma = nn.Parameter(layerscale_init * torch.ones(dim), requires_grad=True)
        self.drop  = DropPath(drop_path)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.se(x)
        x = x * self.gamma.view(1, -1, 1, 1)
        x = residual + self.drop(x)
        return x

class SCEAF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SCEAF, self).__init__()
        self.msag = MSAG(F_g=in_channels, F_l=in_channels, F_int=in_channels // 2)
        self.cam = CAM(dim=in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        msag_out = self.msag(g, x)  
        cam_out = self.cam(x)      
        alpha = self.sigmoid(msag_out) 
        out = alpha * msag_out + (1 - alpha) * cam_out  
        return out
