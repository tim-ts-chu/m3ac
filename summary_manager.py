
import os
import yaml
import logging
import numpy as np
from typing import Dict
from torch.utils.tensorboard import SummaryWriter

class SummaryManager:
    '''
    SummaryManager maintains the summay structure, saves data for the training process,
    and dumps these information into tensorboad using SummaryWriter object.
    Besides, it computes basic statistics metrics such as average, min, max, etc.
    Also, it provides both generic interface and specific interface for more
    clear information flow and using timing.
    '''

    def __init__(self,
            folder_path: str,
            task_name: str):

        self._task_name = task_name
        self._task_folder = os.path.join(folder_path, task_name)
        self._writer = SummaryWriter(self._task_folder)
        self._logger = logging.getLogger()
        self._logger.info(f'Task folder: {self._task_folder}')

        # fields could be as an arguments to make the class more general purpose
        self._summary_fields = [
                # update each optimization step
                'alpha',
                'piLoss',
                'q1Loss',
                'q2Loss',
                'piGradNorm',
                'q1GradNorm',
                'q2GradNorm',
                # update each traj
                'CumReward',
                'TrajLength',
                # update each step
                'q1',
                'q2',
                'qDiff',
                'pi1LogStd',
                'pi2LogStd',
                'StepReturn',
                # update each evaluation traj
                'EvalCumReward',
                'EvalTrajLength',
                # update each evaluation step
                'EvalStepReturn',
                # update each evaluation
                'EvalTotalTraj',
                'EvalSuccessTraj',
                'EvalSuccessRate',
                # update each log interval
                'ItersPerSec',
                # new
                'transitionError',
                'transitionError-1',
                'transitionError-3',
                'transitionError-5',
                'transitionError-10',
                'transitionError-20',
                'rewardError',
                'discError',
                'doneError',
                'transitionRegLoss',
                'transitionGanLoss',
                'rewardRegLoss',
                'rewardGanLoss',
                'discLoss',
                'doneLoss',
                ]

        self._summary_info = {f:[] for f in self._summary_fields}

    def dump_params(self, params: Dict) -> None:
        '''
        Save all the params into a yaml file.
        '''
        with open(os.path.join(self._task_folder, 'params.yaml'), 'w') as f:
            yaml.dump(params, f, sort_keys=True)

    def load_params(self, path: str) -> Dict:
        '''
        Load params from a yaml file.
        '''
        with open(path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        return params

    def update(self, kwargs: Dict) -> None:
        '''
        Generic update function
        '''
        for k, v in kwargs.items():
            if isinstance(v, list):
                self._summary_info[k].extend([float(e) for e in v])
            else:
                self._summary_info[k].append(float(v))

    def flush(self, step: int) -> None:
        '''
        Dump collected data into tensorboard
        '''
        for k, v in self._summary_info.items():
            if v:
                vals = np.asarray(v)
                self._writer.add_scalar(k+'/Average', np.average(vals), step)
                self._writer.add_scalar(k+'/Std', np.std(vals), step)
                self._writer.add_scalar(k+'/Median', np.median(vals), step)
                self._writer.add_scalar(k+'/Min', np.min(vals), step)
                self._writer.add_scalar(k+'/Max', np.max(vals), step)
                self._writer.flush()
                self._summary_info[k].clear()

    # specific update interface
    def update_step_info(self,
            q1: float,
            q2: float,
            qDiff: float,
            pi1LogStd: float,
            pi2LogStd: float,
            StepReturn: float) -> None:
        '''
        Should be called at each step.
        '''
        self.update({
            'q1': q1,
            'q2': q2,
            'qDiff': qDiff,
            'pi1LogStd': pi1LogStd,
            'pi2LogStd': pi2LogStd,
            'StepReturn': StepReturn})

    def update_traj_info(self,
            CumReward: float,
            TrajLength: int) -> None:
        '''
        Should be called at each trajectory finished.
        '''
        self.update({
            'CumReward': CumReward,
            'TrajLength': TrajLength})

    def update_eval_step_info(self,
            EvalStepReturn: float) -> None:
        '''
        Should be called at each evaluation step.
        '''
        self.update({'EvalStepReturn': EvalStepReturn})

    def update_eval_traj_info(self,
            EvalCumReward: float,
            EvalTrajLength: int) -> None:
        '''
        Should be called at each evaluation trajectory finished.
        '''
        self.update({
            'EvalCumReward': EvalCumReward,
            'EvalTrajLength': EvalTrajLength})
