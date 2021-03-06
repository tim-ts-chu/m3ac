
import os
import sys
import time
import torch
import signal
import random
import logging
import argparse
from datetime import datetime
from replay.replay import BufferFields
from minibatch import MiniBatchRL

def main(args: argparse.Namespace):

    folder_path = '/mnt/brain1/scratch/timchu/data/'
    task_name = 'm3ac-'+datetime.now().strftime('%m%d-%H,%M,%S,%f-')+str(args.seed)
    multiprocess = False
    default_device_id = args.device
    random_seed = args.seed     # for reproducibility: default is 0
    # torch.set_default_tensor_type(torch.DoubleTensor) # set default dtype to torch.float64

    if multiprocess:
        device_count = torch.cuda.device_count()
        world_size = device_count if device_count > 1 else 1
    else:
        world_size = 1

    params = {
            'env': {
                # 'id': 'Hopper-v4'
                # 'id': 'HalfCheetah-v4'
                'id': 'Walker2d-v4'
                # 'id': 'Ant-v4'
                },
            'replay_buffer': {
                'buffer_size': int(1e6),
                },
            'sac_agent': {
                'model_path': '',
                'policy_hidden_size': [256, 256],
                'q_hidden_size': [256, 256]
                },
            'sac_algo': {
                'num_updates': int(10),
                'real_batch_size': int(128),
                'imag_batch_size': int(128),
                },
            'model_agent': {
                'predict_reward': True,
                'predict_done': False,
                # 'trans_hidden_size': [512, 512, 512, 512, 512, 512, 512, 512],
                # 'reward_hidden_size': [128, 128, 128, 128],
                'trans_hidden_size': [256, 256, 256, 256],
                'reward_hidden_size': [256, 256, 256, 256],
                'done_hidden_size': [256, 256], # useless if predict_done is set to false
                'model_activation': torch.nn.LeakyReLU,
                'use_batchnorm': False,
                'dropout_prob': None,
                },
            'model_algo': {
                'transition_reg_loss_weight': 1,
                'transition_gan_loss_weight': 0,
                'reward_reg_loss_weight': 1,
                'reward_gan_loss_weight': 0,
                'h_step_loss': 1,
                'trans_lr': 1e-4,
                'reward_lr': 1e-5,
                'num_updates': int(10),
                'model_batch_size': int(256)
                },
            'disc_agent': {
                'hidden_size': [256, 256],
                'activation': torch.nn.ReLU,
                },
            'disc_algo': {
                'num_updates': int(1),
                'disc_batch_size': int(256)
                },
            'minibatch': {
                'n_steps': int(1e6),
                'max_steps': int(1e3),
                'log_interval': int(1e3),
                'eval_interval': int(1e4),
                'eval_n_steps': int(1e3),
                'eval_max_steps': int(1e3),
                'dump_video': False
                },
            'other_info': {   # dumping some information for the record only, not really params
                'default_device_id': default_device_id,
                'BufferFields': BufferFields,
                'random_seed': random_seed,
                'task_name': task_name,
                'world_size': world_size,
                'multiprocesses': multiprocess,
                'other_comments': ''
                },
            }

    minibatch = MiniBatchRL(
            folder_path=folder_path,
            task_name=task_name,
            **params['minibatch'])

    if world_size == 1:
        # no need to spawn process
        minibatch.run(
                rank=0,
                world_size=1,
                port=0,
                default_device_id=default_device_id,
                random_seed=random_seed,
                params=params)
    else:
        port = 54000 + random.randint(100, 999)
        torch.multiprocessing.spawn(minibatch.run,
                 args=(world_size, port, default_device_id, random_seed, params),
                 nprocs=world_size,
                 join=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters for training')
    parser.add_argument('--seed', metavar='0', type=int,  default=0,
                        help='random seed base (different process might add different offset on it)')
    parser.add_argument('--device', metavar='0', type=int,  default=0,
                        help='cuda device id, [0, 1, 2, 3]')

    args = parser.parse_args()
    print('args:', args)
    main(args)

