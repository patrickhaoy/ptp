import os
from absl import app
from absl import flags

from roboverse.envs.sawyer_drawer_pnp_push import SawyerDrawerPnpPush

import rlkit.util.hyperparameter as hyp
from rlkit.demos.source.encoder_dict_to_mdp_path_loader import EncoderDictToMDPPathLoader  # NOQA
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.sac.policies import GaussianTwoChannelCNNPolicy
from rlkit.torch.networks.cnn import TwoChannelCNN
from rlkit.torch.networks.cnn import ConcatTwoChannelCNN

from rlkit.envs.drawer_pnp_push_commands import drawer_pnp_push_commands
from rlkit.learning.ptp import ptp_experiment
from rlkit.learning.ptp import process_args
from rlkit.utils import arg_util
from rlkit.utils.logging import logger as logging


flags.DEFINE_string('data_dir', './data', '')
flags.DEFINE_string('name', None, '')
flags.DEFINE_string('base_log_dir', None, '')
flags.DEFINE_bool('local', True, '')
flags.DEFINE_bool('gpu', True, '')
flags.DEFINE_bool('save_pretrained', True, '')
flags.DEFINE_bool('debug', False, '')
flags.DEFINE_bool('script', False, '')
flags.DEFINE_multi_string(
    'arg_binding', None, 'Variant binding to pass through.')

FLAGS = flags.FLAGS


dataset = 'env6_td_pnp_push'


def get_paths(data_dir):
    if dataset == 'env6_td_pnp_push':
        data_path = 'env6_td_pnp_push/'
        data_path = os.path.join(data_dir, data_path)
        demo_paths = [
            dict(path=data_path +
                 'env6_td_pnp_push_demos_{}.pkl'.format(str(i)),
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for i in range(40)]
    else:
        raise ValueError

    logging.info('data_path: %s', data_path)

    return data_path, demo_paths


def get_default_variant(data_path, demo_paths):
    vqvae = os.path.join(data_path, 'pretrained')

    default_variant = dict(
        imsize=48,
        env_kwargs=dict(
            test_env=True,
        ),
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            std=0.15,
            max_log_std=-1,
            min_log_std=-2,
            std_architecture='shared',
            output_activation=None,
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        vf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),

        trainer_kwargs=dict(
            discount=0.995,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,

            policy_weight_decay=1e-4,
            q_weight_decay=0,

            reward_transform_kwargs=dict(m=1, b=0),
            terminal_transform_kwargs=None,

            beta=0.1,
            quantile=0.9,
            clip_score=100,

            min_value=None,
            max_value=None,
        ),

        max_path_length=400,
        algo_kwargs=dict(
            batch_size=256,
            start_epoch=-100,  # offline epochs
            num_epochs=151,  # online epochs

            num_eval_steps_per_epoch=2000,
            num_expl_steps_per_train_loop=2000,
            num_trains_per_train_loop=1000,
            num_online_trains_per_train_loop=2000,
            min_num_steps_before_training=4000,

            eval_epoch_freq=5,
            offline_expl_epoch_freq=5,
        ),
        replay_buffer_kwargs=dict(
            fraction_next_context=0.1,
            fraction_future_context=0.6,
            fraction_foresight_context=0.0,
            fraction_perturbed_context=0.0,
            fraction_distribution_context=0.0,
            max_future_dt=None,
            max_size=int(1E6),
        ),
        online_offline_split=True,
        reward_kwargs=dict(
            reward_type='sparse',
            epsilon=2.0,
        ),
        online_offline_split_replay_buffer_kwargs=dict(
            online_replay_buffer_kwargs=dict(
                fraction_next_context=0.1,
                fraction_future_context=0.6,
                fraction_foresight_context=0.0,
                fraction_perturbed_context=0.0,
                fraction_distribution_context=0.0,
                max_future_dt=None,
                max_size=int(4E5),
            ),
            offline_replay_buffer_kwargs=dict(
                fraction_next_context=0.1,
                fraction_future_context=0.9,
                fraction_foresight_context=0.0,
                fraction_perturbed_context=0.0,
                fraction_distribution_context=0.0,
                max_future_dt=None,
                max_size=int(6E5),
            ),
            sample_online_fraction=0.6
        ),

        observation_key='latent_observation',
        observation_keys=['latent_observation'],
        goal_key='latent_desired_goal',
        save_video=True,
        expl_save_video_kwargs=dict(
            save_video_period=25,
            pad_color=0,
        ),
        eval_save_video_kwargs=dict(
            save_video_period=25,
            pad_color=0,
        ),

        reset_keys_map=dict(
            image_observation='initial_latent_state'
        ),
        pretrained_vae_path=vqvae,

        path_loader_class=EncoderDictToMDPPathLoader,
        path_loader_kwargs=dict(
            delete_after_loading=True,
            recompute_reward=True,
            demo_paths=demo_paths,
            split_max_steps=None,
        ),

        renderer_kwargs=dict(
            create_image_format='HWC',
            output_image_format='CWH',
            flatten_image=True,
            width=48,
            height=48,
        ),

        add_env_demos=False,
        add_env_offpolicy_data=False,

        load_demos=True,

        evaluation_goal_sampling_mode='presampled_images',
        exploration_goal_sampling_mode='conditional_vae_prior',
        training_goal_sampling_mode='presample_latents',

        presampled_goal_kwargs=dict(
            eval_goals='',  # HERE
            eval_goals_kwargs={},
            expl_goals='',
            expl_goals_kwargs={},
            training_goals='',
            training_goals_kwargs={},
        ),

        use_expl_planner=True,
        expl_planner_type='hierarchical',
        expl_planner_kwargs=dict(
            cost_mode='l2_vf_ptp',
            prior_weight=0.01,
            values_weight=0.001,
            buffer_size=1000,
        ),
        expl_planner_scripted_goals=None,
        expl_contextual_env_kwargs=dict(
            num_planning_steps=8,
            fraction_planning=1.0,
            subgoal_timeout=30,
            subgoal_reaching_thresh=2.0,
            mode='o',
        ),

        use_eval_planner=True,
        eval_planner_type='hierarchical',
        eval_planner_kwargs=dict(
            cost_mode='l2_vf_ptp',
            prior_weight=0.01,
            values_weight=0.001,
            buffer_size=1000,
        ),
        eval_planner_scripted_goals=None,
        eval_contextual_env_kwargs=dict(
            num_planning_steps=8,
            fraction_planning=1.0,
            subgoal_timeout=30,
            subgoal_reaching_thresh=2.0,
            mode='o',
        ),

        scripted_goals=None,

        expl_reset_interval=0,

        launcher_config=dict(
            unpack_variant=True,
            region='us-west-1',
        ),
        logger_config=dict(
            snapshot_mode='gap',
            snapshot_gap=50,
        ),

        trainer_type='iql',
        network_version=None,

        use_image=False,
        pretrained_rl_path='',
        eval_seeds=14,
        num_demos=20,
        num_video_columns=5,
        save_paths=False,

        method_name='ptp',
    )

    return default_variant


def get_search_space():
    ########################################
    # Search Space
    ########################################
    search_space = {
        'env_type': ['td_pnp_push'],

        # Training Parameters
        # Use first 'num_demos' demos for offline data
        'num_demos': [20],

        # Load up existing policy/q-network/value network vs train a new one
        'use_pretrained_rl_path': [False],
        # Negative epochs are pretraining.
        # For only finetuning, set start_epoch=0.
        'algo_kwargs.start_epoch': [-100],

        'trainer_kwargs.bc': [False],  # Run BC experiment
        # Reset environment every 'reset_interval' episodes
        'reset_interval': [1],

        # Training Hyperparameters
        'trainer_kwargs.beta': [0.01],

        # Overrides currently beta with beta_online during finetuning
        'trainer_kwargs.use_online_beta': [False],
        'trainer_kwargs.beta_online': [0.01],

        # Anneal beta every 'anneal_beta_every' by 'anneal_beta_by until
        # 'anneal_beta_stop_at'
        'trainer_kwargs.use_anneal_beta': [False],
        'trainer_kwargs.anneal_beta_every': [20],
        'trainer_kwargs.anneal_beta_by': [.05],
        'trainer_kwargs.anneal_beta_stop_at': [.0001],

        # If True, use pretrained reward classifier. If False, use epsilon.
        'reward_kwargs.use_pretrained_reward_classifier_path': [False],

        'trainer_kwargs.quantile': [0.9],

        'trainer_kwargs.use_online_quantile': [False],
        'trainer_kwargs.quantile_online': [0.99],

        # Network Parameters
        # Concatenate gripper position and rotation into network input
        'use_gripper_observation': [False],

        # Goals
        'use_both_ground_truth_and_affordance_expl_goals': [False],
        'affordance_sampling_prob': [1],
        'ground_truth_expl_goals': [True],
        'only_not_done_goals': [False],
    }

    return search_space


def process_variant(variant, data_path):  # NOQA
    # Error checking
    assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['eval_epoch_freq'] == 0  # NOQA
    if variant['algo_kwargs']['start_epoch'] < 0:
        assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['offline_expl_epoch_freq'] == 0  # NOQA
    if variant['use_pretrained_rl_path']:
        assert variant['algo_kwargs']['start_epoch'] == 0
    if variant['trainer_kwargs']['use_online_beta']:
        assert variant['trainer_kwargs']['use_anneal_beta'] is False
    env_type = variant['env_type']
    if dataset != 'val' and env_type == 'pnp':
        env_type = 'obj'

    ########################################
    # Set the eval_goals.
    ########################################
    full_open_close_str = ''
    if 'eval_seeds' in variant.keys():
        eval_seed_str = f"_seed{variant['eval_seeds']}"
    else:
        eval_seed_str = ''

    if variant['expl_planner_type'] == 'scripted':
        eval_goals = os.path.join(
            data_path,
            f'{full_open_close_str}{env_type}_scripted_goals{eval_seed_str}.pkl')  # NOQA
    else:
        eval_goals = os.path.join(
            data_path,
            f'{full_open_close_str}{env_type}_goals{eval_seed_str}.pkl')

    if variant['expl_planner_scripted_goals'] is not None:
        variant['expl_planner_scripted_goals'] = os.path.join(
            data_path, variant['expl_planner_scripted_goals'])

    if variant['eval_planner_scripted_goals'] is not None:
        variant['eval_planner_scripted_goals'] = os.path.join(
            data_path, variant['eval_planner_scripted_goals'])

    ########################################
    # Goal sampling modes.
    ########################################
    variant['presampled_goal_kwargs']['eval_goals'] = eval_goals
    variant['path_loader_kwargs']['demo_paths'] = (
        variant['path_loader_kwargs']['demo_paths'][:variant['num_demos']])
    variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = min(  # NOQA
        int(6E5), int(500*75*variant['num_demos']))
    variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = min(  # NOQA
        int(4/6 * 500*75*variant['num_demos']),
        int(1E6 - variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size']))  # NOQA

    if variant['use_both_ground_truth_and_affordance_expl_goals']:
        variant['exploration_goal_sampling_mode'] = (
            'conditional_vae_prior_and_not_done_presampled_images')
        variant['training_goal_sampling_mode'] = 'presample_latents'
        variant['presampled_goal_kwargs']['expl_goals'] = eval_goals
        variant['presampled_goal_kwargs']['expl_goals_kwargs']['affordance_sampling_prob'] = variant['affordance_sampling_prob']  # NOQA
    elif variant['ground_truth_expl_goals']:
        variant['exploration_goal_sampling_mode'] = 'presampled_images'
        variant['training_goal_sampling_mode'] = 'presampled_images'
        variant['presampled_goal_kwargs']['expl_goals'] = eval_goals
        variant['presampled_goal_kwargs']['training_goals'] = eval_goals

    if variant['only_not_done_goals']:
        _old_mode = 'presampled_images'
        _new_mode = 'not_done_presampled_images'

        if variant['training_goal_sampling_mode'] == _old_mode:
            variant['training_goal_sampling_mode'] = _new_mode
        if variant['exploration_goal_sampling_mode'] == _old_mode:
            variant['exploration_goal_sampling_mode'] = _new_mode
        if variant['evaluation_goal_sampling_mode'] == _old_mode:
            variant['evaluation_goal_sampling_mode'] = _new_mode

    ########################################
    # Environments.
    ########################################
    variant['env_class'] = SawyerDrawerPnpPush
    variant['env_kwargs']['downsample'] = True
    variant['env_kwargs']['env_obs_img_dim'] = 196
    variant['env_kwargs']['test_env_command'] = (
        drawer_pnp_push_commands[variant['eval_seeds']])

    ########################################
    # Gripper Observation.
    ########################################
    if variant['use_gripper_observation']:
        assert not variant['use_image'], (
            'image-based + gripper obs not implemented yet')

        variant['observation_keys'] = [
            'latent_observation',
            'gripper_state_observation']
        for demo_path in variant['path_loader_kwargs']['demo_paths']:
            demo_path['use_gripper_obs'] = True

    ########################################
    # Image.
    ########################################
    if variant['use_image']:
        variant['policy_class'] = GaussianTwoChannelCNNPolicy
        variant['qf_class'] = ConcatTwoChannelCNN
        variant['vf_class'] = TwoChannelCNN

        if variant['network_version'] == 0:
            c = 8
            h = 256
        elif variant['network_version'] == 1:
            c = 4
            h = 256
        elif variant['network_version'] == 2:
            c = 4
            h = 128
        else:
            raise ValueError

        variant['policy_kwargs'] = dict(
            # CNN params
            input_width=48,
            input_height=48,
            input_channels=3,
            kernel_sizes=[3, 3, 3],
            n_channels=[c, c, c],
            strides=[1, 1, 1],
            hidden_sizes=[h, h, h],
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
            # Gaussian params
            std=variant['policy_kwargs']['std'],
            max_log_std=variant['policy_kwargs']['max_log_std'],
            min_log_std=variant['policy_kwargs']['min_log_std'],
            std_architecture=variant['policy_kwargs']['std_architecture'],
        )
        variant['qf_kwargs'] = dict(
            input_width=48,
            input_height=48,
            input_channels=3,
            kernel_sizes=[3, 3, 3],
            n_channels=[c, c, c],
            strides=[1, 1, 1],
            hidden_sizes=[h, h, h],
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
        )
        variant['vf_kwargs'] = dict(
            input_width=48,
            input_height=48,
            input_channels=3,
            kernel_sizes=[3, 3, 3],
            n_channels=[c, c, c],
            strides=[1, 1, 1],
            hidden_sizes=[h, h, h],
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
        )

        # Keys used by reward function for reward calculation
        variant['observation_key_reward_fn'] = 'latent_observation'
        variant['goal_key_reward_fn'] = 'latent_desired_goal'

        # Keys used by policy/q-networks
        variant['observation_key'] = 'image_observation'
        variant['observation_keys'] = ['image_observation']
        variant['goal_key'] = 'image_desired_goal'

        for demo_path in variant['path_loader_kwargs']['demo_paths']:
            demo_path['use_latents'] = False

    ########################################
    # Misc.
    ########################################
    if variant['reward_kwargs']['reward_type'] in ['sparse']:
        variant['trainer_kwargs']['max_value'] = 0.0
        variant['trainer_kwargs']['min_value'] = -1. / (
            1. - variant['trainer_kwargs']['discount'])

    if variant['expl_planner_type'] != 'hierarchical':
        variant['pretrained_vae_path'] = os.path.join(
            variant['pretrained_vae_path'], 'dt15')

    if 'std' in variant['policy_kwargs']:
        if variant['policy_kwargs']['std'] <= 0:
            variant['policy_kwargs']['std'] = None


def main(_):
    data_path, demo_paths = get_paths(data_dir=FLAGS.data_dir)
    default_variant = get_default_variant(data_path, demo_paths)
    search_space = get_search_space()

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=default_variant,
    )

    logging.info('arg_binding: ')
    logging.info(FLAGS.arg_binding)

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variant = arg_util.update_bindings(variant,
                                           FLAGS.arg_binding,
                                           check_exist=True)
        process_variant(variant, data_path)
        variants.append(variant)

    run_variants(ptp_experiment,
                 variants,
                 run_id=0,
                 process_args_fn=process_args)


if __name__ == '__main__':
    app.run(main)
