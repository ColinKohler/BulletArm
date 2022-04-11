from helping_hands_rl_baselines.equi_rl.utils.parameters import *
from helping_hands_rl_baselines.equi_rl.agents.dqn_agent_com import DQNAgentCom
from helping_hands_rl_baselines.equi_rl.agents.dqn_agent_com_drq import DQNAgentComDrQ
from helping_hands_rl_baselines.equi_rl.agents.curl_dqn_com import CURLDQNCom
from helping_hands_rl_baselines.equi_rl.networks.dqn_net import CNNCom
from helping_hands_rl_baselines.equi_rl.networks.equivariant_dqn_net import EquivariantCNNCom

from helping_hands_rl_baselines.equi_rl.agents.sac import SAC
from helping_hands_rl_baselines.equi_rl.agents.sacfd import SACfD
from helping_hands_rl_baselines.equi_rl.agents.curl_sac import CURLSAC
from helping_hands_rl_baselines.equi_rl.agents.curl_sacfd import CURLSACfD
from helping_hands_rl_baselines.equi_rl.agents.sac_drq import SACDrQ
from helping_hands_rl_baselines.equi_rl.agents.sacfd_drq import SACfDDrQ
from helping_hands_rl_baselines.equi_rl.networks.sac_net import SACCritic, SACGaussianPolicy
from helping_hands_rl_baselines.equi_rl.networks.equivariant_sac_net import EquivariantSACActor, EquivariantSACCritic, EquivariantSACActorDihedral, EquivariantSACCriticDihedral
from helping_hands_rl_baselines.equi_rl.networks.curl_sac_net import CURLSACCritic, CURLSACGaussianPolicy, CURLSACEncoderOri, CURLSACEncoder
from helping_hands_rl_baselines.equi_rl.networks.dqn_net import DQNComCURL, DQNComCURLOri

def createAgent(test=False):
    print('initializing agent')
    obs_channel = 2
    if load_sub is not None or load_model_pre is not None or test:
        initialize = False
    else:
        initialize = True
    n_p = 2
    if not random_orientation:
        n_theta = 1
    else:
        n_theta = 3

    # setup agent
    if alg in ['dqn_com', 'dqn_com_drq']:
        if alg == 'dqn_com':
            agent = DQNAgentCom(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p, n_theta=n_theta)
        elif alg == 'dqn_com_drq':
            agent = DQNAgentComDrQ(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p,
                                   n_theta=n_theta)
        else:
            raise NotImplementedError
        if model == 'cnn':
            net = CNNCom((obs_channel, crop_size, crop_size), n_p=n_p, n_theta=n_theta).to(device)
        elif model == 'equi':
            net = EquivariantCNNCom(n_p=n_p, n_theta=n_theta, initialize=initialize).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(net, initialize_target=not test)

    elif alg in ['curl_dqn_com']:
        if alg == 'curl_dqn_com':
            agent = CURLDQNCom(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p,
                               n_theta=n_theta, crop_size=crop_size)
        else:
            raise NotImplementedError
        if model == 'cnn':
            net = DQNComCURL((obs_channel, crop_size, crop_size), n_p, n_theta).to(device)
        # network from curl paper
        elif model == 'cnn_curl':
            net = DQNComCURLOri((obs_channel, crop_size, crop_size), n_p, n_theta).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(net)

    elif alg in ['sac', 'sacfd', 'sacfd_mean', 'sac_drq', 'sacfd_drq']:
        sac_lr = (actor_lr, critic_lr)
        if alg == 'sac':
            agent = SAC(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                        n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                        target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type)
        elif alg == 'sacfd':
            agent = SACfD(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                          n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                          target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                          demon_w=demon_w)
        elif alg == 'sacfd_mean':
            agent = SACfD(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                          n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                          target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                          demon_w=demon_w, demon_l='mean')
        elif alg == 'sac_drq':
            agent = SACDrQ(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                           n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                           target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type)
        elif alg == 'sacfd_drq':
            agent = SACfDDrQ(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                             n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                             target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                             demon_w=demon_w)
        else:
            raise NotImplementedError
        # pixel observation
        if obs_type == 'pixel':
            if model == 'cnn':
                actor = SACGaussianPolicy((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                critic = SACCritic((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
            elif model == 'equi_actor':
                actor = EquivariantSACActor((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                critic = SACCritic((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
            elif model == 'equi_critic':
                actor = SACGaussianPolicy((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                critic = EquivariantSACCritic((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            elif model == 'equi_both':
                actor = EquivariantSACActor((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                critic = EquivariantSACCritic((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            elif model == 'equi_both_d':
              actor = EquivariantSACActorDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
              critic = EquivariantSACCriticDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        agent.initNetwork(actor, critic, not test)

    elif alg in ['curl_sac', 'curl_sacfd', 'curl_sacfd_mean']:
        curl_sac_lr = [actor_lr, critic_lr, lr, lr]
        if alg == 'curl_sac':
            agent = CURLSAC(lr=curl_sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_a=len(action_sequence),
                            tau=tau, alpha=init_temp, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=True,
                            crop_size=crop_size)
        elif alg == 'curl_sacfd':
            agent = CURLSACfD(lr=curl_sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                              n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                              target_update_interval=1, automatic_entropy_tuning=True, crop_size=crop_size,
                              demon_w=demon_w, demon_l='pi')
        elif alg == 'curl_sacfd_mean':
            agent = CURLSACfD(lr=curl_sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                              n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                              target_update_interval=1, automatic_entropy_tuning=True, crop_size=crop_size,
                              demon_w=demon_w, demon_l='mean')
        else:
            raise NotImplementedError
        if model == 'cnn':
            actor = CURLSACGaussianPolicy(CURLSACEncoder((obs_channel, crop_size, crop_size)).to(device), action_dim=len(action_sequence)).to(device)
            critic = CURLSACCritic(CURLSACEncoder((obs_channel, crop_size, crop_size)).to(device), action_dim=len(action_sequence)).to(device)
        # ferm paper network
        elif model == 'cnn_ferm':
            actor = CURLSACGaussianPolicy(CURLSACEncoderOri((obs_channel, crop_size, crop_size)).to(device),
                                          action_dim=len(action_sequence)).to(device)
            critic = CURLSACCritic(CURLSACEncoderOri((obs_channel, crop_size, crop_size)).to(device),
                                   action_dim=len(action_sequence)).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(actor, critic)
    else:
        raise NotImplementedError
    agent.aug = aug
    agent.aug_type = aug_type
    print('initialized agent')
    return agent