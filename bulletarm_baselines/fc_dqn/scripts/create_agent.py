from bulletarm_baselines.fc_dqn.agents.agents_2d.dqn_2d_fcn import DQN2DFCN
from bulletarm_baselines.fc_dqn.agents.agents_2d.margin_2d_fcn import Margin2DFCN
from bulletarm_baselines.fc_dqn.agents.agents_3d.dqn_3d_fcn import DQN3DFCN
from bulletarm_baselines.fc_dqn.agents.agents_3d.margin_3d_fcn import Margin3DFCN
from bulletarm_baselines.fc_dqn.agents.agents_3d.dqn_3d_fcn_si import DQN3DFCNSingleIn
from bulletarm_baselines.fc_dqn.agents.agents_3d.margin_3d_fcn_si import Margin3DFCNSingleIn
from bulletarm_baselines.fc_dqn.agents.agents_3d.dqn_3d_asr import DQN3DASR
from bulletarm_baselines.fc_dqn.agents.agents_3d.margin_3d_asr import Margin3DASR

from bulletarm_baselines.fc_dqn.utils.parameters import *
from bulletarm_baselines.fc_dqn.networks.models import ResUCatShared, CNNShared, UCat, CNNSepEnc, CNNPatchOnly, CNNShared5l
from bulletarm_baselines.fc_dqn.networks.equivariant_models import EquResUExpand, EquResUDFReg, EquResUDFRegNOut, EquShiftQ2DF3, EquShiftQ2DF3P40, EquResUExpandRegNOut
from bulletarm_baselines.fc_dqn.networks.models import ResURot, ResUTransport, ResUTransportRegress

def createAgent(test=False):
    if half_rotation:
        rz_range = (0, (num_rotations - 1) * np.pi / num_rotations)
    else:
        rz_range = (0, (num_rotations - 1) * 2 * np.pi / num_rotations)
    num_rx = 7
    min_rx = -np.pi / 8
    max_rx = np.pi / 8

    diag_length = float(heightmap_size) * np.sqrt(2)
    diag_length = int(np.ceil(diag_length / 32) * 32)
    if in_hand_mode == 'proj':
        patch_channel = 3
    else:
        patch_channel = 1
    patch_shape = (patch_channel, patch_size, patch_size)

    if alg.find('fcn_si') > -1:
        fcn_out = num_rotations * num_primitives
    else:
        fcn_out = num_primitives

    if load_sub is not None or load_model_pre is not None or test:
        initialize = False
    else:
        initialize = True

    # conventional cnn
    if model == 'resucat':
        fcn = ResUCatShared(1, fcn_out, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)

    ########################### Equivariant FCN and ASR Q1 ############################

    # equivariant asr q1 with lift expansion using cyclic group
    elif model == 'equ_resu_exp':
        fcn = EquResUExpand(1, fcn_out, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape, N=equi_n, initialize=initialize).to(device)
    # equivariant asr q1 with lift expansion using dihedral group
    elif model == 'equ_resu_exp_flip':
        fcn = EquResUExpand(1, fcn_out, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape, N=equi_n, flip=True, initialize=initialize).to(device)
    # equivariant asr q1 with dynamic filter using cyclic group
    elif model == 'equ_resu_df':
        fcn = EquResUDFReg(1, fcn_out, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape, N=equi_n, initialize=initialize).to(device)
    # equivariant asr q1 with dynamic filter using dihedral group
    elif model == 'equ_resu_df_flip':
        fcn = EquResUDFReg(1, fcn_out, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape, N=equi_n, flip=True, initialize=initialize).to(device)
    # equivariant fcn with dynamic filter
    elif model == 'equ_resu_df_nout':
        assert half_rotation
        fcn = EquResUDFRegNOut(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape, N=num_rotations, n_middle_channels=(16, 32, 64, 128), kernel_size=3, quotient=False, last_quotient=True, initialize=initialize).to(device)
    # equivariant fcn with lift expansion
    elif model == 'equ_resu_exp_nout':
        assert half_rotation
        fcn = EquResUExpandRegNOut(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape, N=num_rotations, n_middle_channels=(16, 32, 64, 128), kernel_size=3, quotient=False, last_quotient=True, initialize=initialize).to(device)

    ###################################################################################

    # transporter network baselines
    elif alg.find('tp'):
        # pick = Attention(1, num_rotations, half_rotation).to(device)
        # place = Transport(1, num_rotations, half_rotation).to(device)
        pick = ResURot(1, num_rotations, half_rotation).to(device)
        if model == 'tp':
            place = ResUTransport(1, num_rotations, half_rotation).to(device)
        elif model == 'tp_regress':
            place = ResUTransportRegress(1, num_rotations, half_rotation).to(device)

    else:
        raise NotImplementedError

    if alg.find('asr') > -1:
        if alg.find('deictic') > -1:
            q2_output_size = num_primitives
        else:
            q2_output_size = num_primitives * num_rotations
        q2_input_shape = (patch_channel + 1, patch_size, patch_size)
        if q2_model == 'cnn':
            if alg.find('5l') > -1:
                q2 = CNNShared5l(q2_input_shape, q2_output_size).to(device)
            else:
                q2 = CNNShared(q2_input_shape, q2_output_size).to(device)

        ################################### Equivariant ASR Q2 ###########################
        # equivariant asr q2 with dynamic filter
        elif q2_model == 'equ_shift_df':
            if patch_size == 40:
                q2 = EquShiftQ2DF3P40(q2_input_shape, num_rotations, num_primitives, kernel_size=7, n_hidden=32, quotient=False,
                                   last_quotient=True, initialize=initialize).to(device)
            else:
                q2 = EquShiftQ2DF3(q2_input_shape, num_rotations, num_primitives, kernel_size=7, n_hidden=32, quotient=False,
                                    last_quotient=True, initialize=initialize).to(device)
        ###################################################################################

    # 2d agents (x, y)
    if action_sequence == 'xyp':
        if alg == 'dqn_fcn':
            agent = DQN2DFCN(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size)
        elif alg == 'margin_fcn':
            agent = Margin2DFCN(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, margin, margin_l, margin_weight, margin_beta)
        else:
            raise NotImplementedError
        agent.initNetwork(fcn)

    # 3d agents (x, y, theta)
    elif action_sequence == 'xyrp':
        # ASR agents
        if alg.find('asr') > -1:
            # ASR
            if alg == 'dqn_asr':
                agent = DQN3DASR(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                 num_rotations, rz_range)
                agent.initNetwork(fcn, q2)
            elif alg == 'margin_asr':
                agent = Margin3DASR(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                    num_rotations, rz_range, margin, margin_l, margin_weight, margin_beta)
                agent.initNetwork(fcn, q2)
            else:
                raise NotImplementedError
        # FCN agents
        elif alg.find('fcn_si') > -1:
            # FCN
            if alg == 'dqn_fcn_si':
                agent = DQN3DFCNSingleIn(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range)
            elif alg == 'margin_fcn_si':
                agent = Margin3DFCNSingleIn(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range, margin, margin_l, margin_weight, margin_beta)
            else:
                raise NotImplementedError
            agent.initNetwork(fcn)

        # Rot FCN agents
        elif alg.find('fcn') > -1:
            if alg == 'dqn_fcn':
                agent = DQN3DFCN(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range)
            elif alg == 'margin_fcn':
                agent = Margin3DFCN(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range, margin, margin_l, margin_weight, margin_beta)
            else:
                raise NotImplementedError
            agent.initNetwork(fcn)

    # 6d agent (x, y, z, theta, phi, psi)
    elif action_sequence == 'xyzrrrp':
        from bulletarm_baselines.fc_dqn.agents.agents_6d.dqn_6d_asr_5l import DQN6DASR5L
        from bulletarm_baselines.fc_dqn.agents.agents_6d.margin_6d_asr_5l import Margin6DASR5L
        from bulletarm_baselines.fc_dqn.agents.agents_6d.dqn_6d_asr_5l_deictic import DQN6DASR5LDeictic
        from bulletarm_baselines.fc_dqn.agents.agents_6d.dqn_6d_asr_5l_deictic_35 import DQN6DASR5LDeictic35
        from bulletarm_baselines.fc_dqn.agents.agents_6d.margin_6d_asr_5l_deictic import Margin6DASR5LDeictic
        from bulletarm_baselines.fc_dqn.agents.agents_6d.margin_6d_asr_5l_deictic_35 import Margin6DASR5LDeictic35
        # ASR agents
        if alg.find('asr') > -1:
            if alg.find('5l') > -1:
                q3_input_shape = (patch_channel + 1, patch_size, patch_size)
                q4_input_shape = (patch_channel + 3, patch_size, patch_size)
                q5_input_shape = (patch_channel + 3, patch_size, patch_size)
                if alg.find('deictic') > -1:
                    q3_output_size = num_primitives
                    q4_output_size = num_primitives
                    q5_output_size = num_primitives
                    q3_input_shape = (patch_channel + 3, patch_size, patch_size)
                else:
                    q3_output_size = num_primitives * num_zs
                    q4_output_size = num_primitives * num_rx
                    q5_output_size = num_primitives * num_rx
                q3 = CNNShared5l(q3_input_shape, q3_output_size).to(device)
                q4 = CNNShared5l(q4_input_shape, q4_output_size).to(device)
                q5 = CNNShared5l(q5_input_shape, q5_output_size).to(device)
                # cnn q2-q5
                if alg == 'dqn_asr_5l':
                    agent = DQN6DASR5L(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                       num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx, (min_rx, max_rx), num_zs,
                                       (min_z, max_z))
                elif alg == 'margin_asr_5l':
                    agent = Margin6DASR5L(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                          num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx, (min_rx, max_rx),
                                          num_zs, (min_z, max_z), margin, margin_l, margin_weight, margin_beta)
                # deictic q2-q5
                elif alg == 'dqn_asr_5l_deictic':
                    agent = DQN6DASR5LDeictic(workspace, heightmap_size, device, lr, gamma, sl, num_primitives,
                                              patch_size,
                                              num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx,
                                              (min_rx, max_rx),
                                              num_zs, (min_z, max_z))
                elif alg == 'margin_asr_5l_deictic':
                    agent = Margin6DASR5LDeictic(workspace, heightmap_size, device, lr, gamma, sl, num_primitives,
                                                 patch_size,
                                                 num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx,
                                                 (min_rx, max_rx),
                                                 num_zs, (min_z, max_z), margin, margin_l, margin_weight,
                                                 margin_beta)
                # deictic q3-q5 (specifically for using equivariant q2)
                elif alg == 'dqn_asr_5l_deictic35':
                    agent = DQN6DASR5LDeictic35(workspace, heightmap_size, device, lr, gamma, sl, num_primitives,
                                                patch_size,
                                                num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx,
                                                (min_rx, max_rx),
                                                num_zs, (min_z, max_z))
                elif alg == 'margin_asr_5l_deictic35':
                    agent = Margin6DASR5LDeictic35(workspace, heightmap_size, device, lr, gamma, sl, num_primitives,
                                                   patch_size,
                                                   num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx,
                                                   (min_rx, max_rx),
                                                   num_zs, (min_z, max_z), margin, margin_l, margin_weight,
                                                   margin_beta)

            agent.initNetwork(fcn, q2, q3, q4, q5)

    agent.aug = aug
    agent.aug_type = aug_type
    return agent

