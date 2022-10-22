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
from rlkit.vae.vqvae import encode_dataset
from rlkit.vae.vqvae import VqVae
from rlkit.vae.vqvae_trainer import VqVaeTrainer  
from rlkit.utils import io_util
from rlkit.utils import device_util

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('data_dir', None, 'Path to the data.')
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
        save_interval=100,
        trainer_kwargs=None,
        augment_image=False,
        is_val_format=False,
        num_train_batches_per_epoch=100,
        num_test_batches_per_epoch=1,
        dump_samples=False,
        # Model parameters.
        embedding_dim=5,
):
    logger.set_snapshot_dir(root_dir)
    logger.add_tensorboard_output(root_dir)

    vqvae_path = os.path.join(root_dir, 'vqvae.pt')

    trainer_kwargs = {} if trainer_kwargs is None else trainer_kwargs

    root_dir = os.path.expanduser(root_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    vqvae = VqVae(
        embedding_dim=embedding_dim,
    ).to(ptu.device)

    trainer = VqVaeTrainer(
        vqvae=vqvae,
        tf_logger=logger.tensorboard_logger,
        augment_image=augment_image,
        ** trainer_kwargs,
    )

    datasets = io_util.load_datasets(
        data_dir=data_dir,
        dataset_ctor=vae_datasets.VaeDataset,
        augment_image=augment_image,
        is_val_format=is_val_format,
    )
    train_dataset = datasets['train']
    test_dataset = datasets['test']

    train_loader, test_loader = io_util.data_loaders(
        train_dataset, test_dataset, batch_size)

    print('Finished loading data')

    print('Start training...')

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
            vqvae_path = os.path.join(root_dir, 'vqvae.pt'.format(epoch))
            torch.save(vqvae, vqvae_path)

        stats = trainer.get_diagnostics()

        for k, v in stats.items():
            logger.record_tabular(k, v)

        logger.dump_tabular()
        trainer.end_epoch(epoch)

    logger.add_tabular_output(progress_filename,
                              relative_to_snapshot_dir=False)

    logging.info('Encoding the test dataset...')
    test_encoding = encode_dataset(vqvae, datasets['test'])
    output_path = os.path.join(FLAGS.root_dir, 'test_encoding.npy')
    np.save(output_path, test_encoding)
    logging.info('Saved the data to %s.', output_path)

    logging.info('Encoding the train dataset...')
    train_encoding = encode_dataset(vqvae, datasets['train'])
    output_path = os.path.join(FLAGS.root_dir, 'train_encoding.npy')
    np.save(output_path, train_encoding)
    logging.info('Saved the data to %s.', output_path)

    return


def main(_):
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    device_util.set_device(True)

    train_eval(FLAGS.root_dir,
               data_dir=FLAGS.data_dir,
               augment_image=FLAGS.augment,
               is_val_format=FLAGS.is_val_format,
               )


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
