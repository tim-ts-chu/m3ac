
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
#from algo.m3ac_algo import M3ACAlgorithm
from agent.imagine_agent import ImagineAgent
from agent.discriminate_agent import DiscriminateAgent
#from agent.policy_agent import PolicyAgent
from agent.sac_agent import SACAgent
from algo.sac_algo import SACAlgorithm
from agent.model_agent import ModelAgent
from algo.model_algo import ModelAlgorithm
from agent.disc_agent import DiscriminateAgent
from algo.disc_algo import DiscriminateAlgorithm
from replay.replay import ReplayBuffer, BufferFields, set_buffer_dim
#from env.env import Environment
from envs.gym import GymEnv
from envs.fake_env import get_fake_env
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
            dump_video: bool):

        # reources here are shared cross processes
        self._folder_path = folder_path
        self._task_name = task_name
        self._n_steps = n_steps
        self._max_steps = max_steps
        self._log_interval = log_interval
        self._eval_interval = eval_interval
        self._eval_n_steps = eval_n_steps
        self._eval_max_steps = eval_max_steps
        self._dump_video = dump_video

        self._task_folder = os.path.join(folder_path, task_name)
        os.mkdir(self._task_folder)
        self._last_log_time = None

        # should be initialized in proc setup (process independent)
        self._env = None
        self._buffer = None
        self._sac_algo = None
        self._sac_agent = None
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
            self.device_id = default_device_id
        else:
            # setup proc
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(port)

            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
            device_count = torch.cuda.device_count()
            self.device_id = rank % device_count
            self._logger.info(f'init proc rank: {rank}, world size: {world_size}, device: {self.device_id}, master port: {port}')

        self._env = GymEnv(**params['env'])
        self._set_buffer_dim()
        self._real_buffer = ReplayBuffer(**params['replay_buffer'])
        self._imag_buffer = ReplayBuffer(**params['replay_buffer'])

        self._sac_agent = SACAgent(self.device_id, world_size, **params['sac_agent'])
        self._model_agent = ModelAgent(self.device_id, **params['model_agent'])
        self._disc_agent = DiscriminateAgent(self.device_id, **params['disc_agent'])

        self._fake_env = get_fake_env(params['env']['id'], self._model_agent, self._env)

        self._model_algo = ModelAlgorithm(self.device_id,
                self._real_buffer,
                self._imag_buffer,
                self._model_agent,
                self._sac_agent,
                self._disc_agent,
                self._fake_env,
                **params['model_algo'])
        self._disc_algo = DiscriminateAlgorithm(self.device_id,
                self._real_buffer,
                self._imag_buffer,
                self._disc_agent,
                self._model_agent,
                self._sac_agent,
                self._fake_env,
                **params['disc_algo'])
        self._sac_algo = SACAlgorithm(self.device_id,
                self._sac_agent,
                self._model_algo,
                self._real_buffer,
                **params['sac_algo'])

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

    def _set_buffer_dim(self):
        '''
        Set buffer dimension according to the environment object.
        '''
        state_dim = self._env.observation_space.shape[0]
        action_dim = self._env.action_space.shape[0]
        set_buffer_dim(state_dim, action_dim)
        self._logger.info('state dim:{}, action dim:{}'.format(state_dim, action_dim))

    def _train(self, rank: int, world_size: int, params: Dict) -> None:
        '''
        Main training flow control. Based on the assigned total number of required steps
        and log interval, the evaluation timing is calculated and inserted in the training
        process.
        '''
        obs = self._env.reset()
        cumulative_reward = 0
        cumulative_discounted_reward = 0
        traj_len = 1
        use_init_std = True

        # evaluate initial performance
        self._logger.info('Initial evaluation!')
        self._evaluation(0)
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
                    mu, log_std, action, log_pi = self._sac_agent.pi(obs, use_init_std)
                    next_obs, r, d, info = self._env.step(action.detach().to('cpu').view(BufferFields['action']))
                    cumulative_reward += r
                    cumulative_discounted_reward += r*(self._sac_algo.discount**traj_len)

                    if d or traj_len >= self._max_steps:
                        traj_end = True
                    else:
                        traj_end = False

                    self._real_buffer.push(
                            seq_end=traj_end,
                            state=obs.detach(),
                            action=action.detach(),
                            reward=r.detach().view(1, -1),
                            done=d.detach().int(),
                            next_state=next_obs.detach(),
                            end=traj_end)

                    q1, q2 = self._sac_agent.q(obs, action)
                    if self._summary_manager: self._summary_manager.update_step_info(
                            q1, q2, q1-q2, log_std[:, 0], log_std[:, 1], r)

                if traj_end:
                    if self._summary_manager: self._summary_manager.update_traj_info(
                            cumulative_reward, cumulative_discounted_reward, traj_len)
                    traj_len = 1
                    cumulative_reward = 0
                    cumulative_discounted_reward = 0
                    obs = self._env.reset()
                else:
                    traj_len += 1
                    obs = next_obs

                if self._real_buffer.size < 10000: # late start
                # if self._real_buffer.size < 1000: # late start
                   continue # haven't collected enough data yet, skip optimization
                else:
                    use_init_std = False

                # optimize model
                optim_info = self._model_algo.optimize_agent(train_step) #FIXME change to model_batch_size?
                if self._summary_manager: self._summary_manager.update(optim_info)

                # optimize policy agent
                optim_info = self._sac_algo.optimize_agent(train_step)
                if self._summary_manager: self._summary_manager.update(optim_info)

                # optimize discriminator
                optim_info = self._disc_algo.optimize_agent(train_step)
                if self._summary_manager: self._summary_manager.update(optim_info)

                if train_step % self._log_interval == 0:
                    if self._summary_manager:
                        curr_time = time.time_ns()
                        iters_per_sec = self._log_interval/((curr_time-self._last_log_time)/1e9)
                        self._last_log_time = curr_time
                        self._summary_manager.update({'ItersPerSec': iters_per_sec})
                        self._summary_manager.flush(train_step)

            # evaluate performance for each round
            if self._world_size > 1:
                dist.barrier() # sync after each round and before evaluation
            self._evaluation(train_step)
            traj_len = 0
            cumulative_reward = 0
            obs = self._env.reset()

        self._logger.info('Training is done!')

    @torch.no_grad()
    def _evaluation(self, train_step: int) -> None:
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
        cumulative_discounted_reward = 0

        # model validation
        model_val_steps = 5
        obs_valset = torch.empty((self._eval_max_steps, BufferFields['state']), device=self.device_id, requires_grad=False)
        act_valset = torch.empty((self._eval_max_steps, BufferFields['action']), device=self.device_id, requires_grad=False)
        rwd_valset = torch.empty((self._eval_max_steps, BufferFields['reward']), device=self.device_id, requires_grad=False)
        nobs_valset = torch.empty((self._eval_max_steps, BufferFields['next_state']), device=self.device_id, requires_grad=False)
        val_info = {
                'transitionError-1': [],
                'transitionError-3': [],
                'transitionError-5': [],
                'rewardError-1': [],
                'rewardError-3': [],
                'rewardError-5': []}

        obs = self._env.reset()
        movie_images.append(self._env.render())
        self._sac_agent.eval_mode(True)
        # set fake env to eval_mode?
        for eval_step in self._tqdm_wrapper(1, self._eval_n_steps+1):
            # env step
            _, _, action, _ = self._sac_agent.pi(obs.to(torch.float32).view(1, -1))
            next_obs, r, d, info = self._env.step(action.detach().to('cpu').view(BufferFields['action']))
            cumulative_reward += r
            cumulative_discounted_reward += r*(self._sac_algo.discount**traj_len)
            movie_images.append(self._env.render(
                step_reward=r,
                cumulative_reward=cumulative_reward))

            # model validation
            obs_valset[traj_len-1, :] = obs
            act_valset[traj_len-1, :] = action
            rwd_valset[traj_len-1, :] = r
            nobs_valset[traj_len-1, :] = next_obs

            if traj_len > model_val_steps:
                start_idx = traj_len - model_val_steps - 1
                curr_obs = obs_valset[start_idx, :].view(1, -1)
                for val_step in range(model_val_steps):
                    val_act = act_valset[start_idx + val_step, :].view(1, -1)
                    val_rwd = rwd_valset[start_idx + val_step, :].view(1, -1)
                    val_nobs = nobs_valset[start_idx + val_step, :].view(1, -1)
                    pred_nobs, pred_rwd, _, _ = self._fake_env.step(curr_obs, val_act)
                    if val_step in [0, 2, 4]:
                        val_info['transitionError-'+str(val_step+1)].append((val_nobs-pred_nobs).square().sum())
                        val_info['rewardError-'+str(val_step+1)].append((val_rwd-pred_rwd).square()) # scalar 
                    curr_obs = pred_nobs

            # check episode termination
            if d or traj_len >= self._eval_max_steps:
                # terminate trajectory
                if self._summary_manager: self._summary_manager.update_eval_traj_info(
                        cumulative_reward, cumulative_discounted_reward, traj_len)
                traj_len = 1
                total_trajs += 1
                cumulative_reward = 0
                cumulative_discounted_reward = 0
                obs = self._env.reset()
                movie_images.append(self._env.render())
            else:
                if self._summary_manager: self._summary_manager.update_eval_step_info(r)
                traj_len += 1
                obs = next_obs

        if self._summary_manager: self._summary_manager.update(val_info)
        if self._summary_manager: self._summary_manager.flush(train_step)
        self._sac_agent.eval_mode(False)
        folder_path = os.path.join(self._task_folder, f'iter_{train_step}')
        try:
            os.mkdir(folder_path) # handle concurrency error
        except FileExistsError:
            pass

        # dump video
        if SUPPORT_MOVIEPY and self._dump_video:
            clip = mpy.ImageSequenceClip(movie_images, fps=30)
            clip.write_videofile(os.path.join(folder_path, f'iter_{train_step}_rank{self._rank}.mp4'), logger=None)

        # dump model
        #model_path = os.path.join(folder_path, f'model_rank{self._rank}.pth')
        #self._sac_agent.save_model(model_path)

