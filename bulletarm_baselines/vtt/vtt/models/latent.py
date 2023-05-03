import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
from bulletarm_baselines.vtt.vtt.models.initializer import initialize_weight
from bulletarm_baselines.vtt.vtt.models.encoder import VTT, Concatenation_Encoder, PoE_Encoder, Decoder
from bulletarm_baselines.vtt.vtt.utils import build_mlp, calculate_kl_divergence

class FixedGaussian(nn.Module):
  """
  Fixed diagonal gaussian distribution.
  """
  def __init__(self, output_dim, std):
    super(FixedGaussian, self).__init__()
    self.output_dim = output_dim
    self.std = std

  def forward(self, x):
    mean = torch.zeros(x.size(0), self.output_dim, device=x.device)
    std = torch.ones(x.size(0), self.output_dim, device=x.device).mul_(self.std)
    return mean, std

class Gaussian(nn.Module):
  """
  Diagonal gaussian distribution with state dependent variances.
  """

  def __init__(self, input_dim, output_dim, hidden_units=(256, 256)):
    super(Gaussian, self).__init__()
    self.net = build_mlp(
        input_dim=input_dim,
        output_dim=2 * output_dim,
        hidden_units=hidden_units,
        hidden_activation=nn.LeakyReLU(0.2),
    ).apply(initialize_weight)

  def forward(self, x):
    x = self.net(x)
    mean, std = torch.chunk(x, 2, dim=-1)
    std = F.softplus(std) + 1e-5
    return mean, std

class LatentModel(nn.Module):
  """
  Stochastic latent variable model to estimate latent dynamics and the reward.
  """

  def __init__(
      self,
      state_shape,
      action_shape,
      img_feature_dim=288,
      z1_dim=32,
      z2_dim=256,
      hidden_units=(256, 256),
      encoder='VTT'
  ):
    super(LatentModel, self).__init__()

    # p(z1(0)) = N(0, I)
    self.z1_prior_init = FixedGaussian(z1_dim, 1.0)
    # p(z2(0) | z1(0))
    self.z2_prior_init = Gaussian(z1_dim, z2_dim, hidden_units)
    # p(z1(t+1) | z2(t), a(t))
    self.z1_prior = Gaussian(
        z2_dim + action_shape[0],
        z1_dim,
        hidden_units,
    )
    # p(z2(t+1) | z1(t+1), z2(t), a(t))
    self.z2_prior = Gaussian(
        z1_dim + z2_dim + action_shape[0],
        z2_dim,
        hidden_units,
    )

    # q(z1(0) | feat(0), tactile(0))
    self.z1_posterior_init = Gaussian(img_feature_dim, z1_dim, hidden_units)
    # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
    self.z2_posterior_init = self.z2_prior_init
    # q(z1(t+1) | feat(t+1), tactile(t+1), z2(t), a(t))
    self.z1_posterior = Gaussian(
        img_feature_dim  + z2_dim + action_shape[0],
        z1_dim,
        hidden_units,
    )
    # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
    self.z2_posterior = self.z2_prior

    # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
    self.reward = Gaussian(
        2 * z1_dim + 2 * z2_dim + action_shape[0],
        1,
        hidden_units,
    )

    # feat(t) = Encoder(x(t))
    if encoder == 'VTT':
      self.encoder = VTT()
    elif encoder == "POE":
      self.encoder = PoE_Encoder()
    else:
      self.encoder = Concatenation_Encoder()
    # p(x(t) | z1(t), z2(t))
    self.decoder = Decoder(
        z1_dim + z2_dim,
        state_shape[0],
        np.sqrt(0.1),
    )

    self.BCE_loss = nn.BCELoss()
    torch.manual_seed(0)
    self.apply(initialize_weight)

  def sample_prior(self, actions_):
    z1_mean_ = []
    z1_std_ = []

    # p(z1(0)) = N(0, I)
    z1_mean, z1_std = self.z1_prior_init(actions_[:, 0])
    z1 = z1_mean + torch.randn_like(z1_std) * z1_std
    # p(z2(0) | z1(0))
    z2_mean, z2_std = self.z2_prior_init(z1)
    z2 = z2_mean + torch.randn_like(z2_std) * z2_std

    z1_mean_.append(z1_mean)
    z1_std_.append(z1_std)
    for t in range(1, actions_.size(1) + 1):
    # for t in range(1, actions_.size(1)):
      # p(z1(t) | z2(t-1), a(t-1))
      z1_mean, z1_std = self.z1_prior(torch.cat([z2, actions_[:, t - 1]], dim=1))
      z1 = z1_mean + torch.randn_like(z1_std) * z1_std
      # p(z2(t) | z1(t), z2(t-1), a(t-1))
      z2_mean, z2_std = self.z2_prior(torch.cat([z1, z2, actions_[:, t - 1]], dim=1))
      z2 = z2_mean + torch.randn_like(z2_std) * z2_std

      z1_mean_.append(z1_mean)
      z1_std_.append(z1_std)

    z1_mean_ = torch.stack(z1_mean_, dim=1)
    z1_std_ = torch.stack(z1_std_, dim=1)

    return (z1_mean_, z1_std_)

  def sample_posterior(self, features_, actions_):
    z1_mean_ = []
    z1_std_ = []
    z1_ = []
    z2_ = []

    # p(z1(0)) = N(0, I)
    z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
    z1 = z1_mean + torch.randn_like(z1_std) * z1_std
    # p(z2(0) | z1(0))
    z2_mean, z2_std = self.z2_posterior_init(z1)
    z2 = z2_mean + torch.randn_like(z2_std) * z2_std

    z1_mean_.append(z1_mean)
    z1_std_.append(z1_std)
    z1_.append(z1)
    z2_.append(z2)

    for t in range(1, actions_.size(1) + 1):
    # for t in range(1, actions_.size(1)):
      # q(z1(t) | feat(t), z2(t-1), a(t-1))
      z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, t], z2, actions_[:, t - 1]], dim=1))
      z1 = z1_mean + torch.randn_like(z1_std) * z1_std
      # q(z2(t) | z1(t), z2(t-1), a(t-1))
      z2_mean, z2_std = self.z2_posterior(torch.cat([z1, z2, actions_[:, t - 1]], dim=1))
      z2 = z2_mean + torch.randn_like(z2_std) * z2_std

      z1_mean_.append(z1_mean)
      z1_std_.append(z1_std)
      z1_.append(z1)
      z2_.append(z2)
    z1_mean_ = torch.stack(z1_mean_, dim=1)
    z1_std_ = torch.stack(z1_std_, dim=1)
    z1_ = torch.stack(z1_, dim=1)
    z2_ = torch.stack(z2_, dim=1)
    return (z1_mean_, z1_std_, z1_, z2_)

  def calculate_loss(self, state_, tactile_, action_, reward_, done_, force_norm):
    # Calculate the sequence of features.
    feature_, aligment_check, contact_check = self.encoder(state_, tactile_)
    # Sample from latent variable model.
    z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, action_)
    z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_)

    # Calculate KL divergence loss.
    loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum()

    # Prediction loss of images.
    z_ = torch.cat([z1_, z2_], dim=-1)
    state_mean_, state_std_ = self.decoder(z_)
    state_noise_ = (state_ - state_mean_) / (state_std_ + 1e-8)
    
    # save the reconstructed image
    reconstructed_image = state_mean_[4][4][0:3, :, :].permute(1, 2, 0)
    reconstructed_image = reconstructed_image.detach().cpu().numpy()
    fig = plt.figure(figsize=(3, 3))
    reconstructed_image = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())
    plt.imshow(reconstructed_image)
    fig.savefig('reconstructed_image.png')
    # save the original image
    original_image = state_[4][4][0:3, :, :].permute(1, 2, 0)
    original_image = original_image.detach().cpu().numpy()
    plt.imshow(original_image)
    fig.savefig('original_image.png')

    log_likelihood_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)
    loss_image = -log_likelihood_.mean(dim=0).sum()

    # alignment check
    aligned = torch.ones_like(aligment_check)
    alignment_loss = self.BCE_loss(aligment_check, aligned)

    # contact check
    contact_mag = torch.linalg.norm(tactile_, dim=2).unsqueeze(dim=2)
    non_contact = torch.zeros_like(contact_mag)
    contact = torch.ones_like(contact_mag)
    contact_sign = torch.where(contact_mag > (15.0/force_norm), contact, non_contact)
    contact_loss = self.BCE_loss(contact_check, contact_sign)
    # Prediction loss of rewards.
    x = torch.cat([z_[:, :-1], action_, z_[:, 1:]], dim=-1)
    B, S, X = x.shape
    reward_mean_, reward_std_ = self.reward(x.view(B * S, X))
    reward_mean_ = reward_mean_.view(B, S, 1)
    reward_std_ = reward_std_.view(B, S, 1)
    # unsqueeze to make the same shape as reward_mean_
    reward_ = reward_.unsqueeze(dim=2)
    reward_noise_ = (reward_ - reward_mean_) / (reward_std_ + 1e-8)
    log_likelihood_reward_ = (-0.5 * reward_noise_.pow(2) - reward_std_.log()) - 0.5 * math.log(2 * math.pi)
    # unsqueeze to make the same shape as reward_noise_
    done_ = done_.unsqueeze(dim=2)
    loss_reward = -log_likelihood_reward_.mul_(1 - done_).mean(dim=0).sum()
    return loss_kld, loss_image, loss_reward, alignment_loss, contact_loss

  def calculate_alignment_loss(self, state_, tactile_):
    feature_, aligment_check, contact_check = self.encoder(state_, tactile_)
    aligned = torch.zeros_like(aligment_check)
    alignment_loss = self.BCE_loss(aligment_check, aligned)
    return alignment_loss
