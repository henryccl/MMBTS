import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class mini_backbone(nn.Module):
    def __init__(self, in_channels=1, out_dim=1, n_groups=4):
        super(mini_backbone, self).__init__()

        mid_dim = int(in_channels*4)
        self.conv1 = nn.Conv3d(in_channels, mid_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn1 = nn.GroupNorm(n_groups, mid_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(mid_dim, mid_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.gn2 = nn.GroupNorm(n_groups, mid_dim)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(mid_dim, mid_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn3 = nn.GroupNorm(n_groups, mid_dim)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x):
        x_list = x.chunk(4, 1)
        x_masks = []
        for m in x_list:
            x = self.relu1(self.gn1(self.conv1(m)))
            x = torch.unsqueeze(torch.squeeze(F.adaptive_avg_pool3d(x, (5, 6, 4))), dim=0)
            res = x
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.relu3(self.gn3(x + res))
            x = self.conv4(x)

            x_mask = torch.unsqueeze(torch.squeeze(F.adaptive_avg_pool3d(x, (5, 6, 4))), dim=0)     #　[1, 32, 5, 6, 4]
            x_masks.append(x_mask)

        return x_masks


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x


class UNet3D(nn.Module):
    """3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape=0, in_channels=4, out_channels=3, init_channels=16, p=0.2):
        super(UNet3D, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_encoder()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p)

    def make_encoder(self):
        init_channels = self.init_channels
        self.conv1a = nn.Conv3d(self.in_channels, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b = BasicBlock(init_channels, init_channels)  # 32

        self.ds1 = nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                             padding=(1, 1, 1))  # down sampling and add channels

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds2 = nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds3 = nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4c = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4d = BasicBlock(init_channels * 8, init_channels * 8)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels, init_channels)

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.convc = nn.Conv3d(init_channels * 20, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.convco = nn.Conv3d(init_channels * 16, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.up1conv = nn.Conv3d(init_channels, self.out_channels, (1, 1, 1))

    def forward(self, x):
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)

        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)
        c2d_p = self.pool(c2d)

        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)

        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4)

        style = self.convc(torch.cat([c2d_p, c3d, c4d], dim=1))
        content = c4d

        c4d = self.convco(torch.cat([style, content], dim=1))

        c4d = self.dropout(c4d)

        u4 = self.up4conva(c4d)
        u4 = self.up4(u4)
        u4 = u4 + c3
        u4 = self.up4convb(u4)

        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = u3 + c2
        u3 = self.up3convb(u3)

        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = u2 + c1
        u2 = self.up2convb(u2)

        uout = self.up1conv(u2)
        uout = F.sigmoid(uout)

        return uout, style, u2, u3, u4


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Cross_Attention(dim, heads=heads,
                                dim_head=dim_head, dropout=dropout, softmax=softmax),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, m, mask=None):
        """target(query), memory"""

        for attn, ff in self.layers:
            x = attn(x, m, mask=mask)
        return x


class Mul_aware_module(nn.Module):
    def __init__(self, dim=128):
        super(Mul_aware_module, self).__init__()
        self.dim = dim
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=1,
                                                 heads=16, dim_head=32, mlp_dim=128, dropout=0,
                                                 softmax=False)
        self.position_embedding = False

    def forward(self, x, xq):

        m = F.adaptive_avg_pool3d(xq, (5, 6, 4)).view([1, -1, self.dim])
        x = rearrange(x, 'b c h w a -> b (h w a) c')
        x = self.transformer_decoder(x, m)

        return x


class modality_token(nn.Module):
    def __init__(self, dim=32):
        super(modality_token, self).__init__()
        self.dim = dim
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=1, heads=16, dim_head=32, mlp_dim=128, dropout=0,
                                                 softmax=False)
    def forward(self, x, xq_list):
        modality_token_list = []
        x = rearrange(x, 'b c h w a -> b (h w a) c')
        for xq in xq_list:
            m = xq.view([1, -1, self.dim])
            modality_token = self.transformer_decoder(x, m)
            modality_token = rearrange(modality_token, 'b (h w g) c -> b c h w g', h=20, w=24, g=16)
            modality_token_list.append(modality_token)

        return modality_token_list


class modality_muaware_block(nn.Module):
    def __init__(self):
        super(modality_muaware_block, self).__init__()
        self.miss_encode = Mul_aware_module(dim=32)
        self.nonmiss_encode = Mul_aware_module(dim=32)

        self.extra_fea = mini_backbone(1, 32)
        self.get_modality_tokens = modality_token(dim=32)
        self.conv_down = nn.Conv3d(128, 32, kernel_size=1)

    def forward(self, full_style, miss_style, x_full, randmiss):

        fea_s_masks = self.extra_fea(x_full*randmiss)
        fea_t_masks = self.extra_fea(x_full)
        del x_full
        full_style = self.conv_down(full_style)
        miss_style = self.conv_down(miss_style)

        full_style_m_block = self.get_modality_tokens(full_style, fea_t_masks)
        miss_style_m_block = self.get_modality_tokens(miss_style, fea_s_masks)
        del full_style, miss_style, fea_t_masks, fea_s_masks

        alpha = 0.6
        miss_loss = 0
        nonmiss_loss = 0
        for i, m in enumerate(torch.squeeze(randmiss)):
            if m == 1:
                nonmiss_t = self.nonmiss_encode(full_style_m_block[i], miss_style_m_block[i])
                nonmiss_s = self.nonmiss_encode(miss_style_m_block[i], full_style_m_block[i])
                nonmiss_loss += F.mse_loss(nonmiss_t[:, :], nonmiss_s[:, :], reduction='mean')
            else:
                miss_t = self.miss_encode(full_style_m_block[i], miss_style_m_block[i])
                miss_s = self.miss_encode(miss_style_m_block[i], full_style_m_block[i])
                miss_loss += F.mse_loss(miss_t[:, :], miss_s[:, :], reduction='mean')

        nonmiss_loss = nonmiss_loss / torch.sum(randmiss)
        miss_loss = miss_loss / (4 - torch.sum(randmiss))

        loss = alpha * nonmiss_loss + (1 - alpha) * miss_loss

        return loss


def mutiaxis_norm(x, gamma=0.7):

    gx = torch.norm(x, p=2, dim=1, keepdim=True)
    nx = gx/(gx.mean(dim=-1, keepdim=True)+1e-6)

    return gamma*(x*nx) + (1-gamma)*x


class Cat_conv(nn.Module):
    def __init__(self, in_dim=0, out_dim=16):
        super().__init__()
        self.net = nn.Conv1d(in_dim, int((in_dim / 4) * out_dim), kernel_size=3)

    def forward(self, x):
        x = x.permute([0, 2, 1])
        out = self.net(x)
        return out


class get_Mutiaxis_decoder_shuffle_fea(nn.Module):
    """"""
    def __init__(self, in_dim, out_dim=256, dowm_scale=1):

        super(get_Mutiaxis_decoder_shuffle_fea, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.resolution = int(480 / dowm_scale)
        self.dowm_scale = dowm_scale
        # self.cat_conv = nn.Linear(int(in_dim * self.resolution), int((in_dim / 4) * out_dim))
        self.cat_conv = Cat_conv(in_dim)

    def forward(self, f_decode_fea, m_decode_fea):
        f_axis2 = torch.unsqueeze(
            torch.squeeze(F.adaptive_avg_pool3d(f_decode_fea, (int(160 / self.dowm_scale), 1, 1))), dim=0).permute(
            [0, 2, 1])
        m_axis2 = torch.unsqueeze(
            torch.squeeze(F.adaptive_avg_pool3d(m_decode_fea, (int(160 / self.dowm_scale), 1, 1))), dim=0).permute(
            [0, 2, 1])

        f_axis3 = torch.unsqueeze(
            torch.squeeze(F.adaptive_avg_pool3d(f_decode_fea, (1, int(192 / self.dowm_scale), 1))), dim=0).permute(
            [0, 2, 1])
        m_axis3 = torch.unsqueeze(
            torch.squeeze(F.adaptive_avg_pool3d(m_decode_fea, (1, int(192 / self.dowm_scale), 1))), dim=0).permute(
            [0, 2, 1])

        f_axis4 = torch.unsqueeze(
            torch.squeeze(F.adaptive_avg_pool3d(f_decode_fea, (1, 1, int(128 / self.dowm_scale)))), dim=0).permute(
            [0, 2, 1])
        m_axis4 = torch.unsqueeze(
            torch.squeeze(F.adaptive_avg_pool3d(m_decode_fea, (1, 1, int(128 / self.dowm_scale)))), dim=0).permute(
            [0, 2, 1])

        f_cat = torch.cat([f_axis2, f_axis3, f_axis4], dim=1)
        m_cat = torch.cat([m_axis2, m_axis3, m_axis4], dim=1)
        f_cat = mutiaxis_norm(f_cat)
        m_cat = mutiaxis_norm(m_cat)

        f_cat = self.cat_conv(f_cat)
        m_cat = self.cat_conv(m_cat)

        return f_cat, m_cat



