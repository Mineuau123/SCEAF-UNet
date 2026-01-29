class EAF(nn.Module):
    def __init__(self, in_ch, reduction=4, edge_kernel_size=3):
        super().__init__()
        self.edge_conv = nn.Conv2d(
            in_ch, in_ch, kernel_size=edge_kernel_size,
            padding=edge_kernel_size // 2, groups=in_ch, bias=False
        )

        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_ch // reduction, in_ch, 1, bias=False),
            nn.Sigmoid()
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU()
        )

    def forward(self, x_skip):
        e = self.edge_conv(x_skip)  # [B, C, H, W]
        attn = self.attn_conv(e)    # [B, C, H, W]

        # 3) Concat 
        fused = torch.cat([x_skip, x_skip * attn], dim=1)  
        fused = self.fuse_conv(fused)
        out = self.refine(fused)
        return out
