import os
import random
from absl import app
from absl import flags
from absl import logging  

import numpy as np
import matplotlib.pyplot as plt
import pickle  
import joblib  
import torch

from rlkit.vae import vae_datasets
from rlkit.vae.vqvae import encode_dataset
from rlkit.utils import io_util
from rlkit.utils import device_util


flags.DEFINE_string('root_dir', None, 'Root directory.')
flags.DEFINE_string('data_dir', None, 'Path to the data.')
flags.DEFINE_integer('is_val_format', 1, 'Whether to load the VAL format.')
flags.DEFINE_integer('seed', None, 'Random seed.')

FLAGS = flags.FLAGS


def main(_):
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)

    device_util.set_device(True)

    logging.info('Loading the model from %s ...' % (FLAGS.root_dir))
    model = io_util.load_model(FLAGS.root_dir)
    vqvae = model['vqvae']

    logging.info('Loading the dataset from %s ...' % (FLAGS.data_dir))

    datasets = io_util.load_datasets(
        FLAGS.data_dir,
        dataset_ctor=vae_datasets.VaeDataset,
        augment_image=0,
        is_val_format=FLAGS.is_val_format,
    )

    logging.info('Encoding the test dataset...')
    test_encoding = encode_dataset(vqvae, datasets['test'])
    output_path = os.path.join(FLAGS.root_dir, 'test_encoding.npy')
    # np.save(output_path, test_encoding)
    joblib.dump(test_encoding, output_path)
    logging.info('Saved the data to %s.', output_path)

    logging.info('Encoding the train dataset...')
    train_encoding = encode_dataset(vqvae, datasets['train'])
    output_path = os.path.join(FLAGS.root_dir, 'train_encoding.npy')
    # np.save(output_path, train_encoding)
    joblib.dump(train_encoding, output_path)
    logging.info('Saved the data to %s.', output_path)

    plt.show()


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)