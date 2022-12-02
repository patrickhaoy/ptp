
# Planning-to-Practice (PTP)

This repo contains the code for:

[Planning to Practice: Efficient Online Fine-Tuning by Composing Goals in Latent Space](https://arxiv.org/abs/2106.00671)
Kuan Fang*, Patrick Yin*, Ashvin Nair, Sergey Levine. 
International Conference on Intelligent Robots and Systems (IROS), 2022.

Project Page: https://sites.google.com/view/planning-to-practice.

BibTex:
```
@article{fang2022ptp,
      title={Planning to Practice: Efficient Online Fine-Tuning by Composing Goals in Latent Space}, 
      author={Kuan Fang and Patrick Yin and Ashvin Nair and Sergey Levine},
      journal={International Conference on Intelligent Robots and Systems (IROS)}, 
      year={2022},
}
```

## Installation

### Create Conda Env

Install and use the included anaconda environment.
```
$ conda env create -f docker/ptp.yml
$ source activate ptp
(ptp) $
```

### Dependencies
Download the dependency repos.
- [bullet-manipulation](https://github.com/patrickhaoy/bullet-manipulation) (contains environments): ```git clone https://github.com/patrickhaoy/bullet-manipulation.git```
- [multiworld](https://github.com/vitchyr/multiworld) (contains environments): ```git clone https://github.com/vitchyr/multiworld```
- [rllab](https://github.com/rll/rllab) (contains visualization code):  ```git clone https://github.com/rll/rllab```

Add paths.
```
export PYTHONPATH=$PYTHONPATH:/path/to/multiworld
export PYTHONPATH=$PYTHONPATH:/path/to/doodad
export PYTHONPATH=$PYTHONPATH:/path/to/bullet-manipulation
export PYTHONPATH=$PYTHONPATH:/path/to/bullet-manipulation/bullet-manipulation/roboverse/envs/assets/bullet-objects
export PYTHONPATH=$PYTHONPATH:/path/to/railrl-private
```

### Setup Config File

You must setup the config file for launching experiments, providing paths to your code and data directories. Inside `railrl/config/launcher_config.py`, fill in the appropriate paths. You can use `railrl/config/launcher_config_template.py` as an example reference.

```cp railrl/launchers/config-template.py railrl/launchers/config.py```

## Running Experiments
Below we assume the data is stored at `DATA_PATH` and the trained models are stored at `DATA_CKPT`.

### Offline Dataset and Goals
Download the simulation data and goals from [here](https://drive.google.com/file/d/1o-jSgxibTH4FL6emFzUEQNkSfn7jdRus/view?usp=sharing). Alternatively, you can recollect a new dataset by running
```
python shapenet_scripts/4dof_rotate_td_pnp_push_demo_collector_parallel.py --save_path DATA_PATH/ --name env6_td_pnp_push --downsample --num_threads 4
```
and resample new goals by running
```
python shapenet_scripts/presample_goal_with_plan.py --output_dir DATA_PATH/env6_td_pnp_push/ --downsample --test_env_seeds 0 1 2 --timeout_k_steps_after_done 5 --mix_timeout_k
```
from the `bullet-manipulation` repository.

### Training VQ-VAE
To pretrain a VQ-VAE on the simulation dataset, run
```
python experiments/train_eval_vqvae.py --data_dir DATA_PATH/env6_td_pnp_push --root_dir DATA_CKPT/ptp/vqvae
```
To encode the existing data with the pretrained VQ-VAE, run
```
python experiments/encode_dataset.py --data_dir DATA_PATH/env6_td_pnp_push --root_dir DATA_CKPT/ptp/vqvae
```
To visualize loss curves and reconstructions of images with VQ-VAE, open the tensorboard log file with `tensorboard --logdir DATA_CKPT/ptp/vqvae`.

### Training Affordance Model
Next, we train the affordance models of multiple time scales with the pretrained VQ-VAE (Note that for `dt=60`, we use `train_eval.dataset_type='final'` to enable goal prediction beyond the horizon of prior data):
```
python experiments/train_eval_affordance.py --data_dir DATA_PATH/env6_td_pnp_push/ --vqvae DATA_CKPT/ptp/vqvae/ --gin_param train_eval.z_dim=8 --gin_param train_eval.affordance_pred_weight=1000 --gin_param train_eval.affordance_beta=0.1 --dt 15 --dt_tolerance 5 --max_steps 3 --root_dir DATA_CKPT/ptp/affordance_zdim8_weight1000_beta0.1_run0/dt15

python experiments/train_eval_affordance.py --data_dir DATA_PATH/env6_td_pnp_push/ --vqvae DATA_CKPT/ptp/vqvae/ --gin_param train_eval.z_dim=8 --gin_param train_eval.affordance_pred_weight=1000 --gin_param train_eval.affordance_beta=0.1 --dt 30 --dt_tolerance 10 --max_steps 2 --root_dir DATA_CKPT/ptp/affordance_zdim8_weight1000_beta0.1_run0/dt30

python experiments/train_eval_affordance.py --data_dir DATA_PATH/env6_td_pnp_push/ --vqvae DATA_CKPT/ptp/vqvae/ --dataset_type final --gin_param train_eval.z_dim=8 --gin_param train_eval.affordance_pred_weight=1000 --gin_param train_eval.affordance_beta=0.1 --dt 60 --dt_tolerance 10 --max_steps 1 --root_dir DATA_CKPT/ptp/affordance_zdim8_weight1000_beta0.1_run0/dt60
```
After training has completed, we compile the hierarchical affordance model:
```
python experiments/compile_hierarchical_affordance.py --input DATA_CKPT/ptp/affordance_zdim8_weight1000_beta0.1_run0
```
Lastly, we copy the affordance model to the data folder:
```
cp -r DATA_CKPT/ptp/affordance_zdim8_weight1000_beta0.1_run0 DATA_PATH/env6_td_pnp_push/pretrained/
```
To visualize loss curves and samples from the affordance model, open the tensorboard log files with `tensorboard --logdir DATA_CKPT/ptp`.

### Training PTP
To train offline RL and finetune online in our simulated environment with our hierarchical planner, run
```
python experiments/train_eval_ptp.py 
--data_dir DATA_PATH --local --gpu --save_pretrained 
--name exp_task0
--arg_binding eval_seeds=0
```
For our target tasks A, B, and C that we showed in the paper, the corresponding `eval_seeds` are 0, 1, and 2 respectively.

### Visualizing PTP
During training, the results will be saved to a file called under
```
LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```
 - `LOCAL_LOG_DIR` is the directory set by `railrl.launchers.config.LOCAL_LOG_DIR`
 - `<exp_prefix>` is given either to `setup_logger`.
 - `<foldername>` is auto-generated and based off of `exp_prefix`.
 - inside this folder, you should see a file called `params.pkl`. To visualize a policy, run

You can visualize the results by running `jupyter notebook`, opening `ptp_reproduce.ipynb`, and setting `dirs = [LOCAL_LOG_DIR/<exp_prefix>/<foldername>]`.
