import os
from absl import app
from absl import flags
from absl import logging  

import torch

from rlkit.vae import affordance_networks
from rlkit.vae.affordance_trainer import AffordanceTrainer  
from rlkit.utils import device_util
from rlkit.util.io import load_local_or_remote_file

flags.DEFINE_string('input', None, '')
flags.DEFINE_string('output', None, '')
flags.DEFINE_integer('multiplier', 2, '')
flags.DEFINE_integer('num_levels', 3, '')
flags.DEFINE_integer('dt', 15, 'Step size to sample goals.')

FLAGS = flags.FLAGS


def main(_):
    device_util.set_device(True)

    load_from_trainer = False
    try:
        path_list = []
        for level in range(FLAGS.num_levels):
            dt = FLAGS.dt * FLAGS.multiplier ** level
            path = os.path.join(FLAGS.input, 'dt%d' % (dt), 'affordance.pt')
            path_list.append(path)

        path_list.reverse()

        affordance = affordance_networks.HierarchicalCcVae(
            multiplier=FLAGS.multiplier,
            num_levels=FLAGS.num_levels,
            min_dt=FLAGS.dt,
            path_list=path_list)

    except Exception:
        load_from_trainer = True

        path_list = []
        for level in range(FLAGS.num_levels):
            dt = FLAGS.dt * FLAGS.multiplier ** level
            path = os.path.join(
                FLAGS.input, 'dt%d' % (dt), 'run0', 'id0', 'itr_-750.pt')
            path_list.append(path)

        path_list.reverse()

        affordance = affordance_networks.HierarchicalCcVae(
            multiplier=FLAGS.multiplier,
            num_levels=FLAGS.num_levels,
            min_dt=FLAGS.dt,
            path_list=path_list,
            load_from_trainer=load_from_trainer)

    # Save the compiled affordance.
    if FLAGS.output is None:
        output_dir = FLAGS.input
    else:
        output_dir = FLAGS.output
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not load_from_trainer:
        output_path = os.path.join(output_dir, 'affordance.pt')
        print('Saving the compiled affordance to %s ...' % (output_path))
        torch.save(affordance, output_path)

        # Copy the pretrained VQVAE
        pretrained_vqvae_path = os.path.join(
            FLAGS.input, 'dt%d' % (FLAGS.dt), 'vqvae.pt')
        vqvae = torch.load(pretrained_vqvae_path)
        torch.save(vqvae, os.path.join(output_dir, 'vqvae.pt'))

    else:
        path = path_list[0]
        model_dict = load_local_or_remote_file(path)
        output_path = os.path.join(output_dir, 'model.pt')
        model_dict['trainer/affordance'] = affordance
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))


if __name__ == '__main__':
    app.run(main)