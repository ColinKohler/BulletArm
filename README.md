# BulletArm
**[Documentation](https://colinkohler.github.io/BulletArm/html/index.html)** | **[Paper](https://arxiv.org/abs/2205.14292)** | **[License](https://github.com/ColinKohler/BulletArm/blob/main/LICENSE)**

**BulletArm** is a benchmark and learning-environment for robotic manipulation. BulletArm provides a set of standardized 
benchmark tasks in simulation alongside a collection of baseline algorithms.

## Table of Contents
1. [Dependencies](#dependencies)
2. [Installation](#install)
1. [Benchmarks](#benchmarks)
5. [Cite](#cite)

<a name="dependencies"></a>
## Dependencies

The library is based on Python>=3.7

```
attrdict
matplotlib
GitPython
numpy>=1.19.5
pybullet>=2.7.1
scikit-image>=0.14.2
scipy>=1.2.1
tqdm
```

<a name="install"></a>
## Install
1. Clone this repo
    ```
    git clone https://github.com/ColinKohler/helping_hands_rl_envs.git
    cd helping_hands_rl_envs
    ```
2. Install dependencies
    ```
    pip install -r requirements.txt 
    ```
3. Install this package
    ```
    pip install .
    ```
4. Run the block stacking demo
    ```
    python tutorials/block_stacking_demo.py
    ```
<a name="benchmarks"></a>
## Benchmarks
### Open-Loop Benchmarks
#### Prerequisite
1. Install [PyTorch](https://pytorch.org/) (Recommended: pytorch==1.7.0, torchvision==0.8.1)
1. (Optional, required for 6D benchmark) Install [CuPy](https://github.com/cupy/cupy)
1. Install other required packages
    ```
    pip install -r baseline_requirement.txt
    ```
1. Goto the baseline directory
    ```
    cd helping_hands_rl_envs/helping_hands_rl_baselines/fc_dqn/scripts
    ```
#### Open-Loop 3D Benchmark
```
python main.py --algorithm=[algorithm] --architecture=[architecture] --env=[env]
```
- Select `[algorithm]` from: `sdqfd` (recommended), `dqfd`, `adet`, `dqn`
- Select `[architecture]` from: `equi_asr` (recommended), `cnn_asr`, `equi_fcn`, `cnn_fcn`, `rot_fcn`
- Add `--fill_buffer_deconstruct` to use deconstruction planner for gathering expert data.
#### Open-Loop 2D Benchmark
```
python main.py  --algorithm=[algorithm] --architecture=[architecture] --action_sequence=xyp --random_orientation=f --env=[env]
```
- Select `[algorithm]` from: `sdqfd` (recommended), `dqfd`, `adet`, `dqn`
- Select `[architecture]` from: `equi_fcn` (recommended), `cnn_fcn`
- Add `--fill_buffer_deconstruct` to use deconstruction planner for gathering expert data.
#### Open-Loop 6D Benchmark
```
python main.py  --algorithm=[algorithm] --architecture=[architecture] --action_sequence=xyzrrrp --patch_size=[patch_size] --env=[env]
```
- Select `[algorithm]` from: `sdqfd` (recommended), `dqfd`, `adet`, `dqn`
- Select `[architecture]` from: `equi_deictic_asr` (recommended), `cnn_asr`
- Set `[patch_size]` to be `40` (required for `bumpy_box_palletizing` environment) or `24`
- Add `--fill_buffer_deconstruct` to use deconstruction planner for gathering expert data.

#### Additional Training Arguments
See [bulletarm_baselines/fc_dqn/utils/parameters.py](bulletarm_baselines/fc_dqn/utils/parameters.py)

### Close-Loop Benchmarks
#### Prerequisite
1. Install [PyTorch](https://pytorch.org/) (Recommended: pytorch==1.7.0, torchvision==0.8.1)
1. (Optional, required for 6D benchmark) Install [CuPy](https://github.com/cupy/cupy)
1. Install other required packages
    ```
    pip install -r baseline_requirement.txt
    ```
1. Goto the baseline directory
    ```
    cd helping_hands_rl_envs/helping_hands_rl_baselines/equi_rl/scripts
    ```
#### Close-Loop 4D Benchmark
```
python main.py --algorithm=[algorithm] --env=[env]
```
- Select `[algorithm]` from: `sac`, `sacfd`, `equi_sac`, `equi_sacfd`, `ferm_sac`, `ferm_sacfd`, `rad_sac`, `rad_sacfd`, `drq_sac`, `drq_sacfd`
- To use PER and data augmentation buffer, add `--buffer=per_expert_aug`
#### Close-Loop 3D Benchmark
```
python main.py --algorithm=[algorithm] --action_sequence=pxyz --random_orientation=f --env=[env]
```
- Select `[algorithm]` from: `sac`, `sacfd`, `equi_sac`, `equi_sacfd`, `ferm_sac`, `ferm_sacfd`, `rad_sac`, `rad_sacfd`, `drq_sac`, `drq_sacfd`

#### Additional Training Arguments
See [bulletarm_baselines/equi_rl/utils/parameters.py](bulletarm_baselines/equi_rl/utils/parameters.py)

<a name="cite"></a>
## Cite
The development of this package was part of the work done for our paper: [BulletArm: An Open-Source Robotic Manipulation
Benchmark and Learning Framework](https://arxiv.org/abs/2205.14292). Please cite this work if you use our code:
```
@misc{https://doi.org/10.48550/arxiv.2205.14292,
      doi = {10.48550/ARXIV.2205.14292},
      url = {https://arxiv.org/abs/2205.14292},
      author = {Wang, Dian and Kohler, Colin and Zhu, Xupeng and Jia, Mingxi and Platt, Robert},
      keywords = {Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {BulletArm: An Open-Source Robotic Manipulation Benchmark and Learning Framework},
      publisher = {arXiv},
      year = {2022},
      copyright = {arXiv.org perpetual, non-exclusive license}
} 
```
Feel free to [contact us](mailto:kohler.c@northeastern.edu).

## License
*BulletArm* is distributed under GNU General Public License. See [License](https://github.com/ColinKohler/BulletArm/blob/main/LICENSE).
