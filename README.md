README last updated on: 07/25/2022

# railrl

## Installation

### Create Conda Env

Install and use the included anaconda environment
```
$ conda env create -f docker/railrl-ptp/railrl-ptp.yml
$ source activate railrl-ptp
(railrl-ptp) $
```
Or if you want you can use the docker image included.

### Download Simulation Env Code
- [bullet-manipulation](https://github.com/patrickhaoy/bullet-manipulation) (contains environments): ```git clone https://github.com/patrickhaoy/bullet-manipulation.git```
- [multiworld](https://github.com/vitchyr/multiworld) (contains environments): ```git clone https://github.com/vitchyr/multiworld```

### Download Visualization Code
- [rllab](https://github.com/rll/rllab) (contains visualization code):  ```git clone https://github.com/rll/rllab```

### (Optional) Install doodad
I recommend installing [doodad](https://github.com/justinjfu/doodad) to
launch jobs. Some of its nice features include:
 - Easily switch between running code locally, on a remote compute with
 Docker, on EC2 with Docker
 - Easily add your dependencies that can't be installed via pip (e.g. you
 borrowed someone's code)

If you install doodad, also modify `CODE_DIRS_TO_MOUNT` in `config.py` to
include:
- Path to rllab directory
- Path to railrl directory
- Path to other code you want to juse

You'll probably also need to update the other variables besides the docker
images/instance stuff.

### Setup Config File

You must setup the config file for launching experiments, providing paths to your code and data directories. Inside `railrl/config/launcher_config.py`, fill in the appropriate paths. You can use `railrl/config/launcher_config_template.py` as an example reference.

```cp railrl/launchers/config-template.py railrl/launchers/config.py```

### Add paths
```
export PYTHONPATH=$PYTHONPATH:/path/to/multiworld
export PYTHONPATH=$PYTHONPATH:/path/to/doodad
export PYTHONPATH=$PYTHONPATH:/path/to/bullet-manipulation
export PYTHONPATH=$PYTHONPATH:/path/to/bullet-manipulation/bullet-manipulation/roboverse/envs/assets/bullet-objects
export PYTHONPATH=$PYTHONPATH:/path/to/railrl-private
```

## Training and Visualization
Below we assume the data is stored at `/hdd/data` and the trained models are stored at `/hdd/ckpts`.

### Dataset and Goals
Download the simulation data and goals from [here](https://drive.google.com/file/d/1o-jSgxibTH4FL6emFzUEQNkSfn7jdRus/view?usp=sharing). Alternatively, you can recollect a new dataset by running
```
python shapenet_scripts/4dof_rotate_td_pnp_push_demo_collector_parallel.py --save_path /hdd/data/ --name env6_td_pnp_push --downsample --num_threads 4
```
and resample new goals by running
```
python shapenet_scripts/presample_goals_with_plan.py --output_dir /hdd/data/env6_td_pnp_push/ --downsample --test_env_seeds 0 1 2 --timeout_k_steps_after_done 5 --mix_timeout_k
```
from the `bullet-manipulation` repository.

### Training and Visualizing VQ-VAE
To pretrain a VQ-VAE on the simulation dataset, run
```
python experiments/train_eval_vqvae.py --data_dir /hdd/data/env6_td_pnp_push --root_dir /hdd/ckpts/ptp/vqvae
```
To encode the existing data with the pretrained VQ-VAE, run
```
python experiments/encode_dataset.py --data_dir /hdd/data/env6_td_pnp_push --root_dir /hdd/ckpts/ptp/vqvae
```
To visualize loss curves and reconstructions of images with VQ-VAE, open the tensorboard log file with `tensorboard --logdir /hdd/ckpts/ptp/vqvae`.

### Training and Visualizing Affordance Model
Next, we train the affordance models of multiple time scales with the pretrained VQ-VAE (Note that for `dt=60`, we use `train_eval.dataset_type='final'` to enable goal prediction beyond the horizon of prior data):
```
python experiments/train_eval_affordance.py --data_dir /hdd/data/env6_td_pnp_push/ --vqvae /hdd/ckpts/ptp/vqvae/ --gin_param train_eval.z_dim=8 --gin_param train_eval.affordance_pred_weight=1000 --gin_param train_eval.affordance_beta=0.1 --dt 15 --dt_tolerance 5 --max_steps 3 --root_dir /hdd/ckpts/ptp/affordance_zdim8_weight1000_beta0.1_run0/dt15

python experiments/train_eval_affordance.py --data_dir /hdd/data/env6_td_pnp_push/ --vqvae /hdd/ckpts/ptp/vqvae/ --gin_param train_eval.z_dim=8 --gin_param train_eval.affordance_pred_weight=1000 --gin_param train_eval.affordance_beta=0.1 --dt 30 --dt_tolerance 10 --max_steps 2 --root_dir /hdd/ckpts/ptp/affordance_zdim8_weight1000_beta0.1_run0/dt30

python3 experiments/train_eval_affordance.py --data_dir /hdd/data/env6_td_pnp_push/ --vqvae /hdd/ckpts/ptp/vqvae/ --dataset_type final --gin_param train_eval.z_dim=8 --gin_param train_eval.affordance_pred_weight=1000 --gin_param train_eval.affordance_beta=0.1 --dt 60 --dt_tolerance 10 --max_steps 1 --root_dir /hdd/ckpts/ptp/affordance_zdim8_weight1000_beta0.1_run0/dt60
```
After training has completed, we compile the hierarchical affordance model:
```
python experiments/compile_hierarchical_affordance.py --input /hdd/ckpts/ptp/affordance_zdim8_weight1000_beta0.1_run0
```
Lastly, we copy the affordance model to the data folder:
```
cp -r /hdd/ckpts/ptp/affordance_zdim8_weight1000_beta0.1_run0 /hdd/data/env6_td_pnp_push/pretrained/
```
To visualize loss curves and samples from the affordance model, open the tensorboard log files with `tensorboard --logdir /hdd/ckpts/ptp`.

### Training RL
To train offline RL and finetune online in our simulated environment with our hierarchical planner, run
```
python experiments/train_eval_ptp.py 
--data_dir /hdd/data --local --gpu --save_pretrained 
--name exp01
--arg_binding num_demos=20
--arg_binding eval_seeds=2
--arg_binding algo_kwargs.batch_size=256
--arg_binding trainer_type=iql
--arg_binding expl_planner_type=hierarchical
--arg_binding eval_planner_type=hierarchical
--arg_binding use_image=0
--arg_binding use_gripper_observation=0
--arg_binding policy_kwargs.std=0.15
--arg_binding algo_kwargs.num_online_trains_per_train_loop=2000
--arg_binding reward_kwargs.reward_type=sparse
--arg_binding reward_kwargs.epsilon=2.0
--arg_binding expl_contextual_env_kwargs.num_planning_steps=8
--arg_binding eval_contextual_env_kwargs.num_planning_steps=8
--arg_binding expl_contextual_env_kwargs.subgoal_reaching_thresh=2.0
--arg_binding eval_contextual_env_kwargs.subgoal_reaching_thresh=2.0
--arg_binding expl_planner_kwargs.prior_weight=0.01
--arg_binding eval_planner_kwargs.prior_weight=0.01
--arg_binding expl_planner_kwargs.values_weight=0.001
--arg_binding eval_planner_kwargs.values_weight=0.001
--arg_binding method_name=ptp
```
For our target tasks A, B, and C that we showed in the paper, the corresponding `eval_seeds` are 0, 1, and 2 respectively.

### Visualizing RL
During training, the results will be saved to a file called under
```
LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```
 - `LOCAL_LOG_DIR` is the directory set by `railrl.launchers.config.LOCAL_LOG_DIR`
 - `<exp_prefix>` is given either to `setup_logger`.
 - `<foldername>` is auto-generated and based off of `exp_prefix`.
 - inside this folder, you should see a file called `params.pkl`. To visualize a policy, run

You can visualize the results by running `jupyter notebook`, opening `ptp_reproduce.ipynb`, and setting `dirs = [LOCAL_LOG_DIR/<exp_prefix>/<foldername>]`.

## Credit
This repo contains the code for

[Planning to Practice: Efficient Online Fine-Tuning by Composing Goals in Latent Space](https://arxiv.org/abs/2106.00671)
Kuan Fang*, Patrick Yin*, Ashvin Nair, Sergey Levine. International Conference on Intelligent Robots and Systems (IROS), 2022.

Videos of these experiments can seen on our website: https://sites.google.com/view/planning-to-practice.

This repository was developed by [Kuan Fang](https://github.com/kuanfang), [Patrick Yin](https://github.com/patrickhaoy), and [Ashvin Nair](https://github.com/anair13). It extends [rlkit](https://github.com/anair13/rlkit), which was developed by [Vitchyr Pong](https://github.com/vitchyr), [Murtaza Dalal](https://github.com/mdalal2020), [Steven Lin](https://github.com/stevenlin1111), and [Ashvin Nair](https://github.com/anair13).
