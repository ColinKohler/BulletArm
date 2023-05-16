import torch
import torch.nn as nn
import warnings
import math
from torch.nn import functional as F
from torch.distributions import Normal

class VTT(nn.Module):
  def __init__(self, img_size=[64], img_patch_size=8, tactile_patches=2, in_chans=4, embed_dim=384, depth=6,
               num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
               drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
    super().__init__()
    self.patch_embed = PatchEmbed(
        img_size=img_size[0], img_patch_size=img_patch_size, tactile_patch=tactile_patches,
        in_chan=in_chans, embeded_dim=embed_dim
    )
    img_patches = self.patch_embed.img_patches

    # contact embedding, alignment embedding and position embedding
    self.contact_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.align_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1, img_patches + self.patch_embed.tactile_patch + 2, embed_dim))

    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
    self.blocks = nn.ModuleList([
        Block(
            dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
        for i in range(depth)])

    self.norm = norm_layer(embed_dim)
    self.compress_patches = nn.Sequential(nn.Linear(embed_dim, embed_dim//4),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Linear(embed_dim//4, embed_dim//12))

    self.compress_layer = nn.Sequential(nn.Linear((img_patches + self.patch_embed.tactile_patch + 2)*embed_dim//12, 640),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Linear(640, 288))

    self.align_recognition = nn.Sequential(nn.Linear(embed_dim, 1),
                                           nn.Sigmoid())

    self.contact_recognition = nn.Sequential(nn.Linear(embed_dim, 1),
                                             nn.Sigmoid())

    trunc_normal_(self.pos_embed, std=.02)
    trunc_normal_(self.align_embed, std=.02)
    trunc_normal_(self.contact_embed, std=.02)

  def interpolate_pos_encoding(self, x, w: int, h: int):
    npatch = x.shape[2] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
      return self.pos_embed
    else:
      raise ValueError('Position Encoder does not match dimension')

  def prepare_tokens(self, x, tactile):
    B, S, nc, w, h = x.shape
    x, patched_tactile = self.patch_embed(x, tactile)
    x = torch.cat((x, patched_tactile),dim=2)
    alignment_embed = self.align_embed.expand(B, S, -1, -1)
    contact_embed = self.contact_embed.expand(B, S, -1, -1)
    # introduce contact embedding & alignment embedding
    x = torch.cat((contact_embed, x), dim=2)
    x = torch.cat((alignment_embed, x), dim=2)
    x = x + self.interpolate_pos_encoding(x, w, h)
    return x

  def forward(self, x, tactile):
    x = self.prepare_tokens(x, tactile)
    for blk in self.blocks:
      x = blk(x)
    x = self.norm(x)
    img_tactile = self.compress_patches(x)
    B, S, patches, dim = img_tactile.size()
    img_tactile = img_tactile.view(B, S, -1)
    img_tactile = self.compress_layer(img_tactile)
    return img_tactile, self.align_recognition(x[:, :, 0]), self.contact_recognition(x[:, :, 1])

class Attention(nn.Module):
  def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
    super().__init__()
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = qk_scale or head_dim ** -0.5

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x):
    B, S, N, C = x.shape
    qkv = self.qkv(x).reshape(B*S, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, S, N, C)
    attn = attn.view(B, S, -1, N, N)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, attn

class PatchEmbed(nn.Module):
  def __init__(self, img_size=84, tactile_dim = 6, img_patch_size=14, tactile_patch=2, in_chan=3, embeded_dim=384):
  # def __init__(self, img_size=64, tactile_dim = 384, img_patch_size=8, tactile_patch=2, in_chan=4, embeded_dim=384):
    super().__init__()
    self.img_patches = int((img_size/img_patch_size)*(img_size/img_patch_size))
    self.img_size = img_size
    self.embeded_dim = embeded_dim
    self.proj = nn.Conv2d(in_chan, embeded_dim, kernel_size=img_patch_size, stride=img_patch_size)
    self.tactile_patch = tactile_patch
    self.decode_tactile = nn.Sequential(nn.Linear(tactile_dim, self.tactile_patch*embeded_dim))

  def forward(self, image, tactile):
    # Input shape batch, Sequence, in_Channels, H, W
    # Output shape batch, Sequence, patches & out_Channels
    B, S, C, H, W = image.shape
    image = image.view(B * S, C, H, W)
    pached_image = self.proj(image).flatten(2).transpose(1, 2).view(B, S, -1, self.embeded_dim)
    tactile = tactile.view(B*S, -1)
    decoded_tactile = self.decode_tactile(tactile).view(B, S, self.tactile_patch, -1)
    return pached_image, decoded_tactile

class Block(nn.Module):
  def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    super().__init__()
    self.norm1 = norm_layer(dim)
    self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                          proj_drop=drop)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    self.norm2 = norm_layer(dim)
    mlp_hidden_dim = int(dim*mlp_ratio)
    self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

  def forward(self, x, return_attention: bool = False):
    y, attn = self.attn(self.norm1(x))
    if return_attention:
      return attn
    x = x + self.drop_path(y)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
  if drop_prob == 0. or not training:
    return x
  keep_prob = 1 - drop_prob
  shape = (x.shape[0],) + (1,)*(x.ndim - 1)
  random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
  random_tensor.floor_()
  output = x.div(keep_prob) * random_tensor
  return output

class DropPath(nn.Module):
  def __init__(self, drop_prob=None):
    super(DropPath, self).__init__()
    self.drop_prob = drop_prob

  def forward(self, x):
    return drop_path(x, self.drop_prob, self.training)

class MLP(nn.Module):
  def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.MLP = nn.Sequential(nn.Linear(in_features, hidden_features),
                        act_layer(),
                        nn.Linear(hidden_features, out_features))

  def forward(self, x):
    x = self.MLP(x)
    return x

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
  # type: (Tensor, float, float, float, float) -> Tensor
  return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
  # Cut & paste from PyTorch official master until it's in a few official releases - RW
  # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
  def norm_cdf(x):
    # Computes standard normal cumulative distribution function
    return (1. + math.erf(x / math.sqrt(2.))) / 2.

  if (mean < a - 2 * std) or (mean > b + 2 * std):
    warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                  "The distribution of values may be incorrect.",
                  stacklevel=2)

  with torch.no_grad():
    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

class Concatenation_Encoder(nn.Module):
  """
  Concatenation
  """
  def __init__(self, input_dim=4, tactile_dim=6, img_dim=256, tactile_latent_dim=32):
    super(Concatenation_Encoder, self).__init__()

    self.img_net = nn.Sequential(
        # (4, 64, 64) -> (16, 32, 32)
        nn.Conv2d(input_dim, 16, 3, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        # (16, 32, 32) -> (32, 16, 16)
        nn.Conv2d(16, 32, 3, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        # (32, 16, 16) -> (64, 8, 8)
        nn.Conv2d(32, 64, 3, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        # (64, 8, 8) -> (256, 6,)
        nn.Conv2d(128, img_dim, 4, 2, 1), # TODO: Fix this
        nn.LeakyReLU(0.2, inplace=True),
    )

    self.tactile_net = nn.Sequential(
        nn.Tanh(),
        nn.Linear(tactile_dim, tactile_latent_dim),
        nn.LayerNorm(tactile_latent_dim),
        nn.LeakyReLU(0.2, inplace=True))

    self.tactile_recognize = nn.Sequential(nn.Linear(tactile_latent_dim + img_dim, 1),
                                           nn.Sigmoid())

    self.alignment_recognize = nn.Sequential(nn.Linear(tactile_latent_dim + img_dim, 1),
                                             nn.Sigmoid())

    self.bottle_neck = nn.Sequential(nn.Linear(tactile_latent_dim + img_dim, tactile_latent_dim + img_dim))
    self.img_norm = nn.LayerNorm(img_dim)
    self.tactile_norm = nn.LayerNorm(tactile_latent_dim)
    self.layer_norm = nn.LayerNorm(tactile_latent_dim + img_dim)

  def forward(self, img, tactile):
    B, S, C, H, W = img.size()
    img = img.view(B * S, C, H, W)
    img_x = self.img_norm(self.img_net(img).view(B * S, -1))
    tactile = tactile.view(B * S, -1)
    tactile_x = self.tactile_norm(self.tactile_net(tactile))
    x = torch.cat((img_x, tactile_x), dim=1)
    x = self.layer_norm(x)
    x = x.view(B, S, -1)
    return self.bottle_neck(x), self.tactile_recognize(x), self.alignment_recognize(x)

class ImageEncoder(nn.Module):
  def __init__(self, input_dim=4, img_dim=256):
    super(ImageEncoder, self).__init__()
    self.img_net = nn.Sequential(
        # (4, 64, 64) -> (16, 32, 32)
        nn.Conv2d(input_dim, 16, 3, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        # (16, 32, 32) -> (32, 16, 16)
        nn.Conv2d(16, 32, 3, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        # (32, 16, 16) -> (64, 8, 8)
        nn.Conv2d(32, 64, 3, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        # (64 8, 8) -> (128, 4, 4)
        nn.Conv2d(64, 128, 3, 2, 1), # TODO: Fix this
        nn.LeakyReLU(0.2, inplace=True),
        # (128, 4, 4) -> (256, 1,)
        nn.Conv2d(128, img_dim, 4, 2),
        nn.LeakyReLU(0.2, inplace=True),
    )
    self.img_norm = nn.LayerNorm(img_dim)

  def forward(self, x):
    B, S, C, H, W = x.size()
    x = x.view(B * S, C, H, W)
    img_x = self.img_norm(self.img_net(x).view(B * S, -1))
    return img_x

class TactileEncoder(nn.Module):
  def __init__(self, tactile_dim=6, tactile_latent_dim=32):
    super(TactileEncoder, self).__init__()
    self.tactile_net = nn.Sequential(
        nn.Tanh(),
        nn.Linear(tactile_dim, tactile_latent_dim),
        nn.LayerNorm(tactile_latent_dim),
        nn.LeakyReLU(0.2, inplace=True))
    self.tactile_norm = nn.LayerNorm(tactile_latent_dim)

  def forward(self, tactile):
    B, S, D = tactile.size()
    tactile = tactile.view(B * S, -1)
    tactile_x = self.tactile_norm(self.tactile_net(tactile))
    return tactile_x

class PoE_Encoder(nn.Module):
  def __init__(self, input_dim=4, tactile_dim=6, z_dim=288):
    super(PoE_Encoder, self).__init__()

    self.z_dim = z_dim
    self.img_encoder = ImageEncoder(input_dim, img_dim=z_dim * 2)
    self.tac_encoder = TactileEncoder(tactile_dim, tactile_latent_dim=z_dim * 2)

    self.z_prior_m = torch.nn.Parameter(
        torch.zeros(1, self.z_dim), requires_grad=False
    )

    self.z_prior_v = torch.nn.Parameter(
        torch.ones(1, self.z_dim), requires_grad=False
    )

    self.z_prior = (self.z_prior_m, self.z_prior_v)
    self.tactile_recognize = nn.Sequential(nn.Linear(z_dim, 1), nn.Sigmoid())
    self.alignment_recognize = nn.Sequential(nn.Linear(z_dim, 1), nn.Sigmoid())
    self.layer_norm = nn.LayerNorm(z_dim)

  def sample_gaussian(self, m, v, device):
    epsilon = Normal(0, 1).sample(m.size())
    z = m + torch.sqrt(v) * epsilon.to(device)
    return z

  def gaussian_parameters(self, h, dim: int = -1):
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

  def product_of_experts(self, m_vect, v_vect):
    T_vect = 1.0 / v_vect

    mu = (m_vect * T_vect).sum(2) * (1 / T_vect.sum(2))
    var = 1 / T_vect.sum(2)

    return mu, var

  def duplicate(self, x, rep):
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])

  def forward(self, img, tac):
    batch_dim = img.size()[0]
    sequence_dim = img.size()[1]

    temp_dim = batch_dim
    temp_dim *= sequence_dim

    img_out = self.img_encoder(img).unsqueeze(2)
    tac_out = self.tac_encoder(tac).unsqueeze(2)

    # multimodal fusion
    mu_z_img, var_z_img = self.gaussian_parameters(img_out, dim=1)  # B*S, 128
    mu_z_frc, var_z_frc = self.gaussian_parameters(tac_out, dim=1)  # B*S, 128

    mu_prior, var_prior = self.z_prior  # 1, 128 for both

    # B*S, 128, 1
    mu_prior_resized = mu_prior.expand(temp_dim, *mu_prior.shape).reshape(-1, *mu_prior.shape[1:]).unsqueeze(2)
    var_prior_resized = var_prior.expand(temp_dim, *var_prior.shape).reshape(-1, *var_prior.shape[1:]).unsqueeze(2)

    m_vect = torch.cat([mu_z_img, mu_z_frc, mu_prior_resized], dim=2)
    var_vect = torch.cat([var_z_img, var_z_frc, var_prior_resized], dim=2)
    mu_z, var_z = self.product_of_experts(m_vect, var_vect)
    # Sample Gaussian to get latent
    z = self.sample_gaussian(mu_z, var_z, img.device)
    z = self.layer_norm(z)
    # z at this point has shape B*S, z_dim
    z = z.reshape(batch_dim, sequence_dim, -1)
    contact_binary, align_binary = self.tactile_recognize(z), self.alignment_recognize(z)
    return z, contact_binary, align_binary

class Decoder(nn.Module):
  """
  Decoder.
  """
  def __init__(self, input_dim=288, output_dim=4, std=1.0):
    super(Decoder, self).__init__()

    self.net = nn.Sequential(
        # (32+256, 1, 1) -> (256, 8, 8)
        nn.ConvTranspose2d(input_dim, 256, 8),
        nn.LeakyReLU(0.2, inplace=True),
        # (256, 8, 8) -> (128, 16, 16)
        nn.ConvTranspose2d(256, 128, 2, stride=2),
        nn.LeakyReLU(0.2, inplace=True),
        # (128, 16, 16) -> (64, 32, 32)
        nn.ConvTranspose2d(128, 64, 2, stride=2),
        nn.LeakyReLU(0.2, inplace=True),
        # (64, 32, 32) -> (4, 64, 64)
        nn.ConvTranspose2d(64, output_dim, 2, stride=2),
        nn.LeakyReLU(0.2, inplace=True)
    )
    self.std = std

  def forward(self, x):
    B, S, latent_dim = x.size()
    x = x.view(B * S, latent_dim, 1, 1)
    x = self.net(x)
    _, C, W, H = x.size()
    x = x.view(B, S, C, W, H)
    return x, torch.ones_like(x).mul_(self.std)
