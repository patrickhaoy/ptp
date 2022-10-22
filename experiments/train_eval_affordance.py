import os
import random
from absl import app
from absl import flags
from absl import logging  

import gin
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger

from rlkit.vae import vae_datasets
from rlkit.vae import affordance_networks
from rlkit.vae.vqvae import VqVae
from rlkit.vae.affordance_trainer import AffordanceTrainer  
from rlkit.utils import io_util
from rlkit.utils import device_util

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('data_dir', None, 'Path to the data.')
flags.DEFINE_string('vqvae', None, 'Path to the pretrained vqvae.')
flags.DEFINE_integer('dt', 1, 'Step size to sample goals.')
flags.DEFINE_integer('dt_tolerance', 0, 'Step size to sample goals.')
flags.DEFINE_integer('max_steps', 5, 'Number of goals.')
flags.DEFINE_string('dataset_type', 'multistep', '')
flags.DEFINE_integer('augment', 0, 'Whether to augment the data')
flags.DEFINE_integer('is_val_format', 1, 'Whether to load the VAL format.')
flags.DEFINE_multi_string(
    'gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')
flags.DEFINE_integer('seed', None, 'Random seed.')

FLAGS = flags.FLAGS


@gin.configurable  
def train_eval(
        root_dir,
        num_epochs=100001,
        batch_size=64,
        data_dir=None,
        pretrained_vqvae_dir=None,
        save_interval=100,
        trainer_kwargs=None,
        dt=None,
        dt_tolerance=None,
        max_steps=None,
        dataset_type='multistep',
        augment_image=False,
        is_val_format=True,
        num_train_batches_per_epoch=100,
        num_test_batches_per_epoch=1,
        dump_samples=False,
        # Model parameters.
        embedding_dim=5,
        z_dim=8,
        # Training parameters.
        affordance_pred_weight=1000.,
        affordance_beta=0.1,
        image_dist_thresh=None,
):
    device_util.set_device(True)

    logger.set_snapshot_dir(root_dir)
    logger.add_tensorboard_output(root_dir)

    vqvae_path = os.path.join(root_dir, 'vqvae.pt')
    affordance_path = os.path.join(root_dir, 'affordance.pt')
    classifier_path = os.path.join(root_dir, 'classifier.pt')

    trainer_kwargs = {} if trainer_kwargs is None else trainer_kwargs

    root_dir = os.path.expanduser(root_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    if pretrained_vqvae_dir is not None:
        pretrained_vqvae_path = os.path.join(pretrained_vqvae_dir, 'vqvae.pt')
        print('Loading the pretrained VQVAE from %s...'
              % (pretrained_vqvae_path))
        vqvae = torch.load(pretrained_vqvae_path).to(ptu.device)
        assert embedding_dim == vqvae.embedding_dim
        torch.save(vqvae, vqvae_path)
        use_pretrained_vqvae = True
    else:
        vqvae = VqVae(
            embedding_dim=embedding_dim,
        ).to(ptu.device)
        use_pretrained_vqvae = False

    affordance = affordance_networks.CcVae(
        data_channels=embedding_dim,
        z_dim=z_dim,
    ).to(ptu.device)

    classifier = None

    print('dataset_type: %r' % dataset_type)

    if dataset_type == 'multistep':
        dataset_ctor = vae_datasets.VaeMultistepDataset
    elif dataset_type == 'final':
        dataset_ctor = vae_datasets.VaeFinalGoalDataset
    elif dataset_type == 'anystep':
        dataset_ctor = vae_datasets.VaeAnyStepDataset
    else:
        raise ValueError

    datasets = io_util.load_datasets(
        data_dir=data_dir,
        encoding_dir=pretrained_vqvae_dir,
        dataset_ctor=dataset_ctor,
        dt=dt,
        dt_tolerance=dt_tolerance,
        num_goals=max_steps,
        is_val_format=is_val_format,
    )
    train_dataset = datasets['train']
    test_dataset = datasets['test']

    if pretrained_vqvae_dir is not None and not augment_image:
        assert train_dataset.encoding is not None
        assert test_dataset.encoding is not None

    train_loader, test_loader = io_util.data_loaders(
        train_dataset, test_dataset, batch_size)

    print('Finished loading data')

    trainer_ctor = AffordanceTrainer

    trainer = trainer_ctor(
        vqvae=vqvae,
        affordance=affordance,
        classifier=classifier,
        use_pretrained_vqvae=use_pretrained_vqvae,
        tf_logger=logger.tensorboard_logger,
        affordance_pred_weight=affordance_pred_weight,
        affordance_beta=affordance_beta,
        image_dist_thresh=image_dist_thresh,
        augment_image=augment_image,
        ** trainer_kwargs,
    )

    print('Start training')

    progress_filename = os.path.join(root_dir, 'vae_progress.csv')
    logger.add_tabular_output(progress_filename,
                              relative_to_snapshot_dir=False)

    for epoch in range(num_epochs):
        logging.info('epoch: %d' % epoch)
        should_save = (((epoch > 0) and (epoch % save_interval == 0)) or
                       epoch == num_epochs - 1)
        trainer.train_epoch(epoch, train_loader, num_train_batches_per_epoch)
        trainer.test_epoch(epoch, test_loader, num_test_batches_per_epoch)

        if should_save:
            logging.info('Saving the model to %s...' % (root_dir))

            if pretrained_vqvae_dir is None:
                torch.save(vqvae, vqvae_path)

            affordance_path = os.path.join(
                root_dir, 'affordance.pt')
            torch.save(affordance, affordance_path)

            if classifier is not None:
                torch.save(classifier, classifier_path)

        stats = trainer.get_diagnostics()

        for k, v in stats.items():
            logger.record_tabular(k, v)

        logger.dump_tabular()
        trainer.end_epoch(epoch)

    logger.add_tabular_output(progress_filename,
                              relative_to_snapshot_dir=False)

    return


def main(_):
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    train_eval(FLAGS.root_dir,
               data_dir=FLAGS.data_dir,
               pretrained_vqvae_dir=FLAGS.vqvae,
               dt=FLAGS.dt,
               dt_tolerance=FLAGS.dt_tolerance,
               max_steps=FLAGS.max_steps,
               dataset_type=FLAGS.dataset_type,
               augment_image=FLAGS.augment,
               is_val_format=FLAGS.is_val_format,
               )


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
