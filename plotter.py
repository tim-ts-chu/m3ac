
import torch
import math
import os
import matplotlib.pyplot as plt

class Plotter:

    def __init__(self, dim):

        self._dim = dim

        # key will be the n_step error
        # for each key, there will be a table saving data point in the form of (batch, dim)
        self._reward_gt = {}
        self._reward_pred = {}

        self._state_gt = {}
        self._state_pred = {}


    def add_next_state_dp(self, n_steps, ground_truth, prediction):
        """
        n_steps: int
        ground_truth: (1, dim)
        prediction: (1, dim)
        """
        if n_steps not in self._state_gt:
            self._state_gt[n_steps] = ground_truth
            self._state_pred[n_steps] = prediction
        else: 
            self._state_gt[n_steps] = torch.cat((self._state_gt[n_steps], ground_truth), dim = 0)
            self._state_pred[n_steps] = torch.cat((self._state_pred[n_steps], prediction), dim = 0)

    def add_reward_dp(self, n_steps, ground_truth, prediction):
        """
        n_steps: int
        ground_truth: (1, 1)
        prediction: (1, 1)
        """
        if n_steps not in self._reward_gt:
            self._reward_gt[n_steps] = ground_truth
            self._reward_pred[n_steps] = prediction
        else: 
            self._reward_gt[n_steps] = torch.cat((self._reward_gt[n_steps], ground_truth), dim = 0)
            self._reward_pred[n_steps] = torch.cat((self._reward_pred[n_steps], prediction), dim = 0)

    def dump_plots(self, folder_path, val_steps):
        for n_steps in val_steps:
            ncols = 3 # hardcode 3 as ncols
            nrows = math.ceil((self._dim + 1) / ncols)
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,30))
            for i in range(nrows):
                for j in range(ncols):
                    plot_count =  ncols * i + j
                    if plot_count == 0: 
                        # draw reward subplot
                        self._draw_plot(ax[i][j], 
                                "Reward Validation",
                                self._reward_gt[n_steps],
                                self._reward_pred[n_steps])
                    elif plot_count > self._dim:
                        # shouldn't happend
                        continue
                    else:
                        # draw dim subplot
                        dim = plot_count - 1
                        self._draw_plot(ax[i][j],
                                "Dim {} Prediction".format(dim),
                                self._state_gt[n_steps][:, dim],
                                self._state_pred[n_steps][:, dim])
            fig.patch.set_facecolor('white')
            plt.savefig(os.path.join(folder_path + "/{}_step_validation.png".format(n_steps)))

        # draw reward plot
        # self._draw_plot(folder_path, 
                # "Reward Prediction",
                # self._reward_gt,
                # self._reward_pred)

        # draw state plots
        # for d in range(self._dim):
            # self._draw_plot(folder_path,
                    # "Dim {} Prediction".format(d),
                    # self._state_gt[:, d],
                    # self._state_pred[:, d])

    def _draw_plot(self, ax, title, ground_truth, prediction):
        """    
        ax: subplot object
        title: string    
        ground_truth: (n, 1)    
        prediction: (n, 1)    
        """    
        gt = ground_truth.flatten()    
        pred = prediction.flatten()    

        sorted_indices = gt.view(-1).sort().indices
        ax.plot(pred[sorted_indices].detach().cpu().numpy(), 'ro', markersize=1, label='Prediction')
        ax.plot(gt[sorted_indices].detach().cpu().numpy(), 'bo', markersize=0.3, label='Ground Truth')
        ax.legend(loc='upper left')
        errors = pred - gt
        ax.text(0.65, 0.20, 'MSE: {:.2e}'.format(errors.square().mean()), transform=ax.transAxes)
        ax.text(0.65, 0.15, 'MSE STD: {:.2e}'.format(errors.std()), transform=ax.transAxes)
        ax.set_xlabel('Samples Sorted by Ground Truth Value')
        ax.set_ylabel('Values')
        ax.set_title(title)

