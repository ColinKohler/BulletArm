import sys
import time
import copy
import collections
import torch
from tqdm import tqdm
from bulletarm_baselines.fc_dqn.utils.parameters import *
import matplotlib.pyplot as plt

sys.path.append('./')
sys.path.append('..')
from bulletarm_baselines.fc_dqn.utils.env_wrapper import EnvWrapper

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')

def getCurrentObs(in_hand, obs):
    obss = []
    for i, o in enumerate(obs):
        obss.append((o.squeeze(), in_hand[i].squeeze()))
    return obss

def fillDeconstruct(agent, replay_buffer):
    def states_valid(states_list):
        if len(states_list) < 2:
            return False
        for i in range(1, len(states_list)):
            if states_list[i] != 1 - states_list[i-1]:
                return False
        return True

    def rewards_valid(reward_list):
        if reward_list[0] != 1:
            return False
        for i in range(1, len(reward_list)):
            if reward_list[i] != 0:
                return False
        return True

    if env in ['block_stacking',
               'house_building_1',
               'house_building_2',
               'house_building_3',
               'house_building_4',
               'improvise_house_building_2',
               'improvise_house_building_3',
               'improvise_house_building_discrete',
               'improvise_house_building_random',
               'ramp_block_stacking',
               'ramp_house_building_1',
               'ramp_house_building_2',
               'ramp_house_building_3',
               'ramp_house_building_4',
               'ramp_improvise_house_building_2',
               'ramp_improvise_house_building_3']:
        deconstruct_env = env + '_deconstruct'
    else:
        raise NotImplementedError('deconstruct env not supported for env: {}'.format(env))

    plt.style.use('default')
    envs = EnvWrapper(num_processes, deconstruct_env, env_config, planner_config)

    states, in_hands, obs = envs.reset()
    total = 0
    s = 0
    step_times = []
    steps = [0 for i in range(num_processes)]
    local_state = [[] for i in range(num_processes)]
    local_obs = [[] for i in range(num_processes)]
    local_action = [[] for i in range(num_processes)]
    local_reward = [[] for i in range(num_processes)]

    pbar = tqdm(total=planner_episode)
    while total < planner_episode:
        # buffer_obs = agent.getCurrentObs(in_hands, obs)
        plan_actions = envs.getNextAction()
        actions_star_idx, actions_star = agent.getActionFromPlan(plan_actions)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        t0 = time.time()
        states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=False)
        state_id = action_sequence.find('p')
        dones[actions_star[:, state_id] + states_ != 1] = 1
        t = time.time()-t0
        step_times.append(t)

        buffer_obs = getCurrentObs(in_hands_, obs)
        for i in range(num_processes):
            local_state[i].append(states[i])
            local_obs[i].append(buffer_obs[i])
            local_action[i].append(actions_star_idx[i])
            local_reward[i].append(rewards[i])

        steps = list(map(lambda x: x + 1, steps))

        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            empty_in_hands = envs.getEmptyInHand()

            buffer_obs_ = getCurrentObs(empty_in_hands, copy.deepcopy(obs_))
            reset_states_, reset_in_hands_, reset_obs_ = envs.reset_envs(done_idxes)
            for i, idx in enumerate(done_idxes):
                local_obs[idx].append(buffer_obs_[idx])
                local_state[idx].append(copy.deepcopy(states_[idx]))
                if (num_objects-2)*2 <= steps[idx] <= num_objects*2 and states_valid(local_state[idx]) and rewards_valid(local_reward[idx]):
                    s += 1
                    for j in range(len(local_reward[idx])):
                        obs = local_obs[idx][j+1]
                        next_obs = local_obs[idx][j]

                        replay_buffer.add(ExpertTransition(local_state[idx][j+1],
                                                           obs,
                                                           local_action[idx][j],
                                                           local_reward[idx][j],
                                                           local_state[idx][j],
                                                           next_obs,
                                                           torch.tensor(float(j == 0)),
                                                           torch.tensor(float(j)),
                                                           torch.tensor(1)))

                states_[idx] = reset_states_[i]
                obs_[idx] = reset_obs_[i]

                total += 1
                steps[idx] = 0
                local_state[idx] = []
                local_obs[idx] = []
                local_action[idx] = []
                local_reward[idx] = []

        pbar.set_description(
            '{}/{}, SR: {:.3f}, step time: {:.2f}; avg step time: {:.2f}'
            .format(s, total, float(s)/total if total !=0 else 0, t, np.mean(step_times))
        )
        pbar.update(done_idxes.shape[0])

        states = copy.copy(states_)
        obs = copy.copy(obs_)
    pbar.close()
    envs.close()

def fillDeconstructUsingRunner(agent, replay_buffer):
  if env in ['block_stacking',
             'house_building_1',
             'house_building_2',
             'house_building_3',
             'house_building_4',
             'improvise_house_building_2',
             'improvise_house_building_3',
             'improvise_house_building_discrete',
             'improvise_house_building_random',
             'ramp_block_stacking',
             'ramp_house_building_1',
             'ramp_house_building_2',
             'ramp_house_building_3',
             'ramp_house_building_4',
             'ramp_improvise_house_building_2',
             'ramp_improvise_house_building_3']:
    deconstruct_env = env + '_deconstruct'
  else:
    raise NotImplementedError('deconstruct env not supported for env: {}'.format(env))
  decon_envs = EnvWrapper(num_processes, deconstruct_env, env_config, planner_config)

  transitions = decon_envs.gatherDeconstructTransitions(planner_episode)
  for i, transition in enumerate(transitions):
    (state, in_hand, obs), action, reward, done, (next_state, next_in_hand, next_obs) = transition
    actions_star_idx, actions_star = agent.getActionFromPlan(torch.tensor(np.expand_dims(action, 0)))
    replay_buffer.add(ExpertTransition(
      torch.tensor(state).float(),
      (torch.tensor(obs).float(), torch.tensor(in_hand).float()),
      actions_star_idx[0],
      torch.tensor(reward).float(),
      torch.tensor(next_state).float(),
      (torch.tensor(next_obs).float(), torch.tensor(next_in_hand).float()),
      torch.tensor(float(done)),
      torch.tensor(float(0)),
      torch.tensor(1))
    )
  decon_envs.close()

if __name__ == '__main__':
    fillDeconstruct()