from baselines.agents.agents_2d.dqn_2d_fcn import DQN2DFCN
from baselines.agents.agents_2d.margin_2d_fcn import Margin2DFCN
from baselines.agents.agents_3d.dqn_3d_fcn import DQN3DFCN
from baselines.agents.agents_3d.margin_3d_fcn import Margin3DFCN
from baselines.agents.agents_3d.dqn_3d_asr import DQN3DASR
from baselines.agents.agents_3d.margin_3d_asr import Margin3DASR
from baselines.agents.agents_6d.dqn_6d_asr_5l import DQN6DASR5L
from baselines.agents.agents_6d.margin_6d_asr_5l import Margin6DASR5L

from baselines.utils.parameters import *
from baselines.networks.models import ResUCatShared, CNNShared

def createAgent():
    if half_rotation:
        rz_range = (0, (num_rotations - 1) * np.pi / num_rotations)
    else:
        rz_range = (0, (num_rotations - 1) * 2 * np.pi / num_rotations)
    num_rx = 7
    min_rx = -np.pi / 6
    max_rx = np.pi / 6

    diag_length = float(heightmap_size) * np.sqrt(2)
    diag_length = int(np.ceil(diag_length / 32) * 32)
    if in_hand_mode == 'proj':
        patch_channel = 3
    else:
        patch_channel = 1
    patch_shape = (patch_channel, patch_size, patch_size)

    if model == 'resucat':
        fcn = ResUCatShared(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(
            device)
    else:
        raise NotImplementedError

    if action_sequence == 'xyp':
        if alg == 'dqn_fcn':
            agent = DQN2DFCN(workspace, heightmap_size, device, lr, gamma, num_primitives, patch_size)
        elif alg == 'margin_fcn':
            agent = Margin2DFCN(workspace, heightmap_size, device, lr, gamma, num_primitives, patch_size, margin, margin_l, margin_weight, margin_beta)
        else:
            raise NotImplementedError
        agent.initNetwork(fcn)

    elif action_sequence == 'xyrp':
        if alg.find('asr') > -1:
            if alg.find('deictic') > -1:
                q2_output_size = num_primitives
            else:
                q2_output_size = num_primitives * num_rotations
            q2_input_shape = (patch_channel + 1, patch_size, patch_size)
            if q2_model == 'cnn':
                q2 = CNNShared(q2_input_shape, q2_output_size).to(device)
            else:
                raise NotImplementedError
            if alg == 'dqn_asr':
                agent = DQN3DASR(workspace, heightmap_size, device, lr, gamma, num_primitives, patch_size,
                                 num_rotations, rz_range)
                agent.initNetwork(fcn, q2)
            elif alg == 'margin_asr':
                agent = Margin3DASR(workspace, heightmap_size, device, lr, gamma, num_primitives, patch_size,
                                    num_rotations, rz_range, margin, margin_l, margin_weight, margin_beta)
                agent.initNetwork(fcn, q2)
            else:
                raise NotImplementedError

        elif alg.find('fcn') > -1:
            if alg == 'dqn_fcn':
                agent = DQN3DFCN(workspace, heightmap_size, device, lr, gamma, num_primitives, patch_size, num_rotations, rz_range)
            elif alg == 'margin_fcn':
                agent = Margin3DFCN(workspace, heightmap_size, device, lr, gamma, num_primitives, patch_size, num_rotations, rz_range, margin, margin_l, margin_weight, margin_beta)
            else:
                raise NotImplementedError
            agent.initNetwork(fcn)

    elif action_sequence == 'xyzrrrp':
        if alg.find('asr') > -1:
            if alg.find('5l') > -1:
                q2_input_shape = (patch_channel + 1, patch_size, patch_size)
                q3_input_shape = (patch_channel + 1, patch_size, patch_size)
                q4_input_shape = (patch_channel + 3, patch_size, patch_size)
                q5_input_shape = (patch_channel + 3, patch_size, patch_size)
                if alg.find('deictic') > -1:
                    q2_output_size = num_primitives
                    q3_output_size = num_primitives
                    q4_output_size = num_primitives
                    q5_output_size = num_primitives
                    q3_input_shape = (patch_channel + 3, patch_size, patch_size)
                else:
                    q2_output_size = num_primitives * num_rotations
                    q3_output_size = num_primitives * num_zs
                    q4_output_size = num_primitives * num_rx
                    q5_output_size = num_primitives * num_rx
                q2 = CNNShared(q2_input_shape, q2_output_size).to(device)
                q3 = CNNShared(q3_input_shape, q3_output_size).to(device)
                q4 = CNNShared(q4_input_shape, q4_output_size).to(device)
                q5 = CNNShared(q5_input_shape, q5_output_size).to(device)
                if alg == 'dqn_asr_5l':
                    agent = DQN6DASR5L(workspace, heightmap_size, device, lr, gamma, num_primitives, patch_size,
                                       num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx, (min_rx, max_rx), num_zs,
                                       (min_z, max_z))
                elif alg == 'margin_asr_5l':
                    agent = Margin6DASR5L(workspace, heightmap_size, device, lr, gamma, num_primitives, patch_size,
                                       num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx, (min_rx, max_rx),
                                       num_zs, (min_z, max_z), margin, margin_l, margin_weight, margin_beta)
            agent.initNetwork(fcn, q2, q3, q4, q5)

    return agent

