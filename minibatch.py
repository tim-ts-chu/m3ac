
import os
import sys
import math
import time
import tqdm
import torch
import random
import signal
import logging
import numpy as np
import torch.distributed as dist
from typing import Dict, Tuple, Generator
from algo.m3ac_algo import M3ACAlgorithm
from agent.imagine_agent import ImagineAgent
from agent.discriminate_agent import DiscriminateAgent
from agent.policy_agent import PolicyAgent
from replay.replay import ReplayBuffer
#from env.env import Environment
from envs.gym import GymEnv
from summary_manager import SummaryManager

try:
    import moviepy.editor as mpy
    SUPPORT_MOVIEPY = True
except ImportError:
    logging.getLogger().info('import moviepy failed. Not support generating evaluation video')
    SUPPORT_MOVIEPY = False

class MiniBatchRL:
    '''
    This class mainly for handling minibatch training and evaluating process.
    Moreover, it also take care of prarllel training at the same time
    '''

    def __init__(self,
            folder_path: str,
            task_name: str,
            n_steps: int,
            max_steps: int,
            log_interval: int,
            eval_interval: int,
            eval_n_steps: int,
            eval_max_steps: int,
            batch_size: int):

        # reources here are shared cross processes
        self._folder_path = folder_path
        self._task_name = task_name
        self._n_steps = n_steps
        self._max_steps = max_steps
        self._log_interval = log_interval
        self._eval_interval = eval_interval
        self._eval_n_steps = eval_n_steps
        self._eval_max_steps = eval_max_steps
        self._batch_size = batch_size

        self._task_folder = os.path.join(folder_path, task_name)
        os.mkdir(self._task_folder)
        self._last_log_time = None

        # should be initialized in proc setup (process independent)
        self._env = None
        self._buffer = None
        self._algo = None
        self._agent = None
        self._summary_manager = None
        self._logger = None

    def run(self,
            rank: int,
            world_size: int,
            port: int,
            default_device_id: int,
            random_seed: int,
            params: Dict) -> None:
        '''
        Multiple processes have been forked after __init__ before run.
        The only shared reource is summary_manager, but only the master process
        should use it.
        '''
        self._rank = rank
        self._world_size = world_size
        self._set_logging(rank)
        self._set_random_seed(random_seed, rank)

        if world_size == 1:
            device_id = default_device_id
        else:
            # setup proc
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(port)

            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
            device_count = torch.cuda.device_count()
            device_id = rank % device_count
            self._logger.info(f'init proc rank: {rank}, world size: {world_size}, device: {device_id}, master port: {port}')

        self._env = GymEnv(**params['env'])
        self._algo = M3ACAlgorithm(self._batch_size)
        self._real_buffer = ReplayBuffer(**params['replay_buffer'])
        self._imag_buffer = ReplayBuffer(**params['replay_buffer'])
        self._imag_agent = ImagineAgent()
        self._disc_agent = DiscriminateAgent()
        self._policy_agent = PolicyAgent()

        if rank == 0:
            # only master process is responsible for summary wirtting
            self._summary_manager = SummaryManager(
                    folder_path=self._folder_path,
                    task_name=self._task_name)
            self._summary_manager.dump_params(params)

        # do the main job
        self._train(rank, world_size, params)

        # cleanup proc
        if world_size != 1:
            dist.destroy_process_group()

    def _tqdm_wrapper(self, start: int, stop: int) -> Generator[int, None, None]:
        '''
        If stdout is redirected to a file, do not use tqdm.
        '''
        if sys.stdout.isatty():
            for it in tqdm.trange(start, stop):
                yield it
        else:
            for it in range(start, stop):
                yield it

    def _set_random_seed(self, random_seed: int, rank: int) -> None:
        '''
        Set random seed for all library we use.
        But for different process, we use different offset
        '''
        random.seed(random_seed+rank)
        np.random.seed(random_seed+rank)
        torch.manual_seed(random_seed+rank)

    def _set_logging(self, rank: int) -> None:
        '''
        Set logging system
        TODO: dump to different files saperately for different processes.
        '''
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        self._logger = logger

    def _train(self, rank: int, world_size: int, params: Dict) -> None:
        '''
        Main training flow control. Based on the assigned total number of required steps
        and log interval, the evaluation timing is calculated and inserted in the training
        process.
        '''
        obs = self._env.reset()
        cumulative_reward = 0
        traj_len = 1

        # evaluate initial performance
        # self._logger.info('Initial evaluation!')
        # self._evaluation(0)
        if self._world_size > 1:
            dist.barrier() # sync before begin
        self._logger.info('Training is begin!')

        total_round = math.ceil(self._n_steps/self._eval_interval)
        train_step = 0
        for curr_round in range(1, total_round+1):
            if self._n_steps - train_step >= self._eval_interval:
                round_steps = self._eval_interval
            else:
                round_steps = self._n_steps - train_step
            self._logger.info(f'Round {curr_round}/{total_round}: start from step {train_step+1} to {train_step+round_steps}')
            self._last_log_time = time.time_ns()

            for round_step in self._tqdm_wrapper(1, round_steps+1):

                train_step += 1

                with torch.no_grad():
                    # collect samples
                    #mu, log_std, action, log_pi = self._agent.pi(obs)
                    #next_obs, r, d, info = self._env.step(action.detach().view(2))
                    action = torch.rand(*self._env.action_space.shape)*2-1
                    next_obs, r, d, info = self._env.step(action)
                    cumulative_reward += r*(self._algo.discount**traj_len)

                    self._real_buffer.push(
                            state=obs.detach(),
                            action=action.detach(),
                            reward=r.detach().view(1, -1),
                            done=d.detach().int())

                    #q1, q2 = self._agent.q(obs, action)
                    #if self._summary_manager: self._summary_manager.update_step_info(q1, q2, q1-q2, log_std[:, 0], log_std[:, 1], r)

                if d or traj_len >= self._max_steps:
                    # terminate trajectory
                    #if self._summary_manager: self._summary_manager.update_traj_info(cumulative_reward, traj_len, cum_reward_no_exp)
                    traj_len = 1
                    cumulative_reward = 0
                    obs = self._env.reset()
                else:
                    traj_len += 1
                    obs = next_obs

                if self._real_buffer.size < self._batch_size:
                    continue # haven't collected enough data yet, skip optimization

                # optimize agent
                # samples = self._buffer.sample(self._batch_size)
                self._algo.optimize_agent(self._imag_agent, self._disc_agent, self._policy_agent,
                        self._real_buffer, self._imag_buffer)
                # optim_info = self._algo.optimize_agent(samples, train_step)
                # if self._summary_manager: self._summary_manager.update(optim_info)

                if train_step % self._log_interval == 0:
                    pass
                    # if self._summary_manager:
                        # curr_time = time.time_ns()
                        # iters_per_sec = self._log_interval/((curr_time-self._last_log_time)/1e9)
                        # self._last_log_time = curr_time
                        # self._summary_manager.update({'ItersPerSec': iters_per_sec})
                        # self._summary_manager.flush(train_step)

            # evaluate performance for each round
            if self._world_size > 1:
                dist.barrier() # sync after each round and before evaluation
            self._evaluation(train_step)
            traj_len = 0
            cumulative_reward = 0
            obs = self._env.reset()

        self._logger.info('Training is done!')

    @torch.no_grad()
    def _evaluation(self, step: int) -> None:
        '''
        Evaluating current policy, generating qualitative result such as videos if support,
        logging some statistics, and saving current models.
        '''
        self._logger.info('Evaluating agent...')
        movie_images = []
        traj_len = 1
        total_trajs = 0
        success_trajs = 0
        cumulative_reward = 0
        obs = self._env.reset()
        movie_images.append(self._env.render())
        #self._agent.eval_mode(True)
        for eval_step in self._tqdm_wrapper(1, self._eval_n_steps+1):
            #_, _, action, _ = self._agent.pi(obs)
            #next_obs, r, d, info = self._env.step(action.detach().view(2))
            action = torch.rand(*self._env.action_space.shape)*2-1
            next_obs, r, d, info = self._env.step(action)
            cumulative_reward += r*(self._algo.discount**traj_len)
            movie_images.append(self._env.render(
                step_reward=r,
                cumulative_reward=cumulative_reward))

            if d or traj_len >= self._eval_max_steps:
                # terminate trajectory
                # if self._summary_manager: self._summary_manager.update_eval_traj_info(cumulative_reward, traj_len)
                traj_len = 1
                total_trajs += 1
                cumulative_reward = 0
                obs = self._env.reset()
                movie_images.append(self._env.render())
            else:
                if self._summary_manager: self._summary_manager.update_eval_step_info(r)
                traj_len += 1
                obs = next_obs

        #self._agent.eval_mode(False)
        # if self._summary_manager:
            # self._summary_manager.update({
                # 'EvalTotalTraj': total_trajs,
                # 'EvalSuccessTraj': success_trajs,
                # 'EvalSuccessRate': success_trajs/total_trajs})
            # self._summary_manager.flush(step)
        folder_path = os.path.join(self._task_folder, f'iter_{step}')
        try:
            os.mkdir(folder_path) # handle concurrency error
        except FileExistsError:
            pass

        # dump video
        if SUPPORT_MOVIEPY:
            clip = mpy.ImageSequenceClip(movie_images, fps=15)
            clip.write_videofile(os.path.join(folder_path, f'iter_{step}_rank{self._rank}.mp4'), logger=None)

        # dump model
        #model_path = os.path.join(folder_path, f'model_rank{self._rank}.pth')
        #self._agent.save_model(model_path)

