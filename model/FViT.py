import math
import torch
import torch.nn as nn
from functools import partial
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


class DPFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        conv_features = int(hidden_features/2)
        self.dwconv_1 = nn.Conv2d(conv_features, conv_features, 3, 1, 1, bias=True, groups=conv_features)
        self.dwconv_2 = nn.Conv2d(conv_features, conv_features, 3, 1, 1, bias=True, groups=conv_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act_layer = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.drop(x)
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dwconv_1(x1)
        x1 = self.act_layer(x1)
        x2 = x1 + x2
        x2 = self.dwconv_2(x2)
        x2 = self.act_layer(x2)
        x = torch.cat([x1, x2], dim=1)
        out = x.flatten(2).permute(0, 2, 1)
        out = self.fc2(out)
        out = self.drop(out)
        return out


class LearnableGaborFilter(nn.Module):
    def __init__(self, in_channels, stride, padding, kernels, out_channels, kernel_size):
        super(LearnableGaborFilter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernels = kernels
        self.total_kernels = self.kernels
        self.responses = self.kernels
        self.kernel_size = kernel_size
        self.gelu = nn.ReLU()

        self.Lambdas = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(dim=1).unsqueeze(dim=2))
        self.psis = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(dim=1).unsqueeze(dim=2))
        self.sigmas = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(dim=1).unsqueeze(dim=2))
        self.gammas_y = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(dim=1).unsqueeze(dim=2))
        self.trans_x = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(dim=1).unsqueeze(dim=2))
        self.trans_y = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(dim=1).unsqueeze(dim=2))
        self.alphas = nn.Parameter(torch.randn(self.responses).unsqueeze(dim=1).unsqueeze(dim=2))
        self.bias = nn.Parameter(torch.zeros(self.responses))
        self.thetas = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(dim=1).unsqueeze(dim=2))

        xmin, xmax = -1, 1
        ymin, ymax = -1, 1
        x_space = torch.linspace(xmin, xmax, kernel_size)
        y_space = torch.linspace(ymin, ymax, kernel_size)
        (y, x) = torch.meshgrid(y_space, x_space, indexing='ij')
        self.x = nn.Parameter(x.unsqueeze(dim=0), requires_grad=False)
        self.y = nn.Parameter(y.unsqueeze(dim=0), requires_grad=False)
        self.channels1x1 = self.responses * self.in_channels

    def forward(self, x):
        device = x.device
        bs, C_, H, W = x.size()
        x = x.reshape(bs * self.in_channels, H, W).unsqueeze(dim=1).to(device)
        gabor_kernels = self.generate_gabor_kernels()
        out = F.conv2d(input=x, weight=gabor_kernels.to(device),
                       bias=self.bias.to(device),
                       groups=1, stride=self.stride, padding=self.padding)
        _, _, newH, newW = out.size()
        out = self.gelu(out)
        out = out.view(bs, self.in_channels, self.responses, newH, newW)
        out = out.view(bs, self.channels1x1, newH, newW)
        return out

    def generate_gabor_kernels(self):
        device = self.thetas.device
        sines = torch.sin(self.thetas).to(device)
        cosines = torch.cos(self.thetas).to(device)
        x = (self.x * cosines - self.y * sines).to(device)
        y = (self.x * sines + self.y * cosines).to(device)
        x = x + torch.tanh(self.trans_x).to(device)
        y = y + torch.tanh(self.trans_y).to(device)
        ori_y_term = (self.gammas_y * y) ** 2
        exponent_ori = (x ** 2 + ori_y_term) * self.sigmas ** 2
        gaussian_term_ori = torch.exp(-exponent_ori)
        cosine_term_ori = torch.cos(x * self.Lambdas + self.psis).to(device)
        ori_gb = gaussian_term_ori * cosine_term_ori
        ori_gb = self.alphas * ori_gb
        return ori_gb.unsqueeze(dim=1)


class GaborLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.GaborLayer = LearnableGaborFilter(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, kernels=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.GaborLayer(x)
        x = x.transpose(1, 2).reshape(B, N, C)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, img_size=256, patch_size=16, kernel_size=3, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = torch.div(H, self.patch_size[0], rounding_mode='floor'), torch.div(W, self.patch_size[1], rounding_mode='floor')
        return x, (H, W)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GaborLayer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DPFFN(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        cnn_feat = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class FViT(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=1000, embed_dims=[46, 92, 184, 368],
                 stem_channel=16, fc_dim=1280, mlp_ratios=[3.6, 3.6, 3.6, 3.6], drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, depths=[2, 2, 10, 2], dp=0.1):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.stem_conv1 = nn.Conv2d(in_chans, stem_channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.patch_embed_a = PatchEmbed(
            img_size=img_size // 2, patch_size=2, kernel_size=2, in_chans=stem_channel, embed_dim=embed_dims[0])

        self.patch_embed_b = PatchEmbed(
            img_size=img_size // 4, patch_size=2, kernel_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.patch_embed_c = PatchEmbed(
            img_size=img_size // 8, patch_size=2, kernel_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.patch_embed_d = PatchEmbed(
            img_size=img_size // 16, patch_size=2, kernel_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.blocks_stage1 = nn.ModuleList([
            Block(
                dim=embed_dims[0], mlp_ratio=mlp_ratios[0], drop=drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer)
            for i in range(depths[0])])
        cur += depths[0]
        self.blocks_stage2 = nn.ModuleList([
            Block(
                dim=embed_dims[1], mlp_ratio=mlp_ratios[1], drop=drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer)
            for i in range(depths[1])])
        cur += depths[1]
        self.blocks_stage3 = nn.ModuleList([
            Block(
                dim=embed_dims[2], mlp_ratio=mlp_ratios[2], drop=drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer)
            for i in range(depths[2])])
        cur += depths[2]
        self.blocks_stage4 = nn.ModuleList([
            Block(
                dim=embed_dims[3], mlp_ratio=mlp_ratios[3], drop=drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer)
            for i in range(depths[3])])

        self.head_norm = norm_layer(embed_dims[3])
        self.head_avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, GaborLayer):
                m.update_temperature()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)
        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)
        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)

        x, (H, W) = self.patch_embed_a(x)
        for i, blk in enumerate(self.blocks_stage1):
            x = blk(x, H, W)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_b(x)
        for i, blk in enumerate(self.blocks_stage2):
            x = blk(x, H, W)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_c(x)
        for i, blk in enumerate(self.blocks_stage3):
            x = blk(x, H, W)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_d(x)
        for i, blk in enumerate(self.blocks_stage4):
            x = blk(x, H, W)

        x = self.head_norm(x)
        x = self.head_avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def build_backbone(backbone_name, num_classes):
    if backbone_name == 'FViT_Tiny':
        backbone = FViT(img_size=224, in_chans=3, num_classes=num_classes, embed_dims=[64, 128, 256, 512],
                        stem_channel=32, fc_dim=1280, mlp_ratios=[3, 3, 3, 3], drop_rate=0.05,
                        drop_path_rate=0.0, dp=0.0, norm_layer=None, depths=[3, 3, 15, 3])
    elif backbone_name == 'FViT_Small':
        backbone = FViT(img_size=224, in_chans=3, num_classes=num_classes, embed_dims=[72, 144, 288, 576],
                        stem_channel=36, fc_dim=1280, mlp_ratios=[3.5, 3.5, 3.5, 3.5], drop_rate=0.05,
                        drop_path_rate=0.0, dp=0.0, norm_layer=None, depths=[4, 4, 24, 4])
    elif backbone_name == 'FViT_Base':
        backbone = FViT(img_size=224, in_chans=3, num_classes=num_classes, embed_dims=[80, 160, 320, 640],
                        stem_channel=40, fc_dim=1280, mlp_ratios=[4, 4, 4, 4], drop_rate=0.05,
                        drop_path_rate=0.0, dp=0.0, norm_layer=None, depths=[5, 5, 30, 5])
    elif backbone_name == 'FViT_Large':
        backbone = FViT(img_size=224, in_chans=3, num_classes=num_classes, embed_dims=[88, 176, 352, 704],
                        stem_channel=44, fc_dim=1280, mlp_ratios=[4, 4, 4, 4], drop_rate=0.05,
                        drop_path_rate=0.0, dp=0.0, norm_layer=None, depths=[6, 6, 36, 6])
    else:
        raise NotImplementedError
    return backbone


if __name__ == '__main__':
    _backbone_name = 'FViT_Tiny'
    num_classes = 1000

    model = build_backbone(_backbone_name, num_classes)

    model.eval()
    inputs = torch.randn(1, 3, 224, 224)
    model(inputs)

    flops = FlopCountAnalysis(model, inputs)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, inputs)

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")
