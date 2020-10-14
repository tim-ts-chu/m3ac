
from typing import Iterable
import torch
from torch.nn import Module

from agent.models.observation import ObservationEncoder, ObservationDecoder
from agent.models.rnns import RSSMRepresentation, RSSMTransition, RSSMRollout
from agent.models.rnns import get_feat, get_dist
from agent.models.dense import DenseModel

from replay.replay import BufferFields

from rlpyt.utils.buffer import buffer_to, buffer_method

def get_parameters(modules: Iterable[Module]):                                                                
    """                                          
    Given a list of torch modules, returns a list of their parameters.    
    :param modules: iterable of modules                                                                         
    :returns: a list of parameters                                  
    """                                                                                           
    model_parameters = []                                                
    for module in modules:                 
        model_parameters += list(module.parameters())    
    return model_parameters              
                                                                                                  
                                                            
class FreezeParameters:                                                     
    def __init__(self, modules: Iterable[Module]):                  
        """                                                          
        Context manager to locally freeze gradients.                                                       
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.                                        
        example:                                                                    
        ```                                                                                                
        with FreezeParameters([module]):                                                                               
            output_tensor = module(input_tensor)                                     
        ```                                                                                              
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.    
        """                                                                                              
        self.modules = modules                                          
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]                                       
                                                                                                    
    def __enter__(self):                                                                          
        for param in get_parameters(self.modules):                             
            param.requires_grad = False                                                                  
                                                                 
    def __exit__(self, exc_type, exc_val, exc_tb):                                                              
        for i, param in enumerate(get_parameters(self.modules)):                                      
            param.requires_grad = self.param_states[i]

class ImagineAgent:

    def __init__(self):

        # TODO
        self._obs_size = BufferFields['state']
        self._action_size = BufferFields['action']
        self._embed_size = 64
        self._stochastic_size = 30
        self._deterministic_size = 200
        self._hidden_size = 200
        self._reward_shape = (1,)
        self._reward_layers = 3
        self._reward_hidden = 300
        self._model_lr = 6e-4
        self._free_nats = 3 # FIXME not sure what is this
        self._kl_scale = 1
        self._grad_clip = 100.0
        self._horizon = 15

        feature_size = self._stochastic_size + self._deterministic_size

        # dreamer model
        encoder_embed_size = self._embed_size
        decoder_embed_size = self._stochastic_size + self._deterministic_size
        self.observation_encoder = ObservationEncoder(self._obs_size, encoder_embed_size)
        self.observation_decoder = ObservationDecoder(self._obs_size, decoder_embed_size)

        self.transition = RSSMTransition(self._action_size, self._stochastic_size, self._deterministic_size, self._hidden_size)
        self.representation = RSSMRepresentation(self.transition, self._embed_size, self._action_size, self._stochastic_size, self._hidden_size)
        self.rollout = RSSMRollout(self.representation, self.transition)
        self.reward_model = DenseModel(feature_size, self._reward_shape, self._reward_layers, self._reward_hidden)

        self.model_modules = [self.observation_encoder,
                              self.observation_decoder,
                              self.reward_model,
                              self.representation,
                              self.transition]

        self.model_optimizer = torch.optim.Adam(get_parameters(self.model_modules), lr=self._model_lr)
        #self.value_optimizer = torch.optim.Adam(get_parameters(self.value_modules), lr=self.value_lr,
        #                                        **self.optim_kwargs)

    def optimize_agent(self, real_samples):
        '''
        Optimize model
        '''

        model_loss, post = self._loss(real_samples)

        print('model_loss:', model_loss)

        self.model_optimizer.zero_grad()
        #self.value_optimizer.zero_grad()

        model_loss.backward()
        #value_loss.backward()

        grad_norm_model = torch.nn.utils.clip_grad_norm_(get_parameters(self.model_modules), self._grad_clip)
        #grad_norm_value = torch.nn.utils.clip_grad_norm_(get_parameters(self.value_modules), self.grad_clip)

        self.model_optimizer.step()
        #self.value_optimizer.step()

        return post

    def imagine(self, real_samples, post, policy_agent):

        batch_time, batch_size, _ = real_samples['state'].shape
        flat_post = buffer_method(post, 'reshape', batch_time*batch_size, -1)

        # Rollout the policy for self.horizon steps. Variable names with imag_ indicate this data is imagined not real.
        # imag_feat shape is [horizon, batch_t * batch_b, feature_size]
        with FreezeParameters(self.model_modules):
            imag_dist, actions = self.rollout.rollout_policy(self._horizon, policy_agent, flat_post)

        # Use state features (deterministic and stochastic) to predict the image and reward
        imag_feat = get_feat(imag_dist)  # [horizon, batch_t * batch_b, feature_size]
        # Assumes these are normal distributions. In the TF code it's be mode, but for a normal distribution mean = mode
        # If we want to use other distributions we'll have to fix this.
        # We calculate the target here so no grad necessary

        # freeze model parameters as only action model gradients needed
        with FreezeParameters(self.model_modules):
            imag_reward = self.reward_model(imag_feat).mean

        imag_pred = self.observation_decoder(imag_feat) # Gaussian Distribution

        # construct imag samples
        imaginary_samples = {}
        imaginary_samples['state'] = imag_pred.rsample()
        imaginary_samples['action'] = actions
        imaginary_samples['reward'] = imag_reward
        imaginary_samples['done'] = torch.full((self._horizon, batch_time*batch_size, 1), False, dtype=torch.bool) # assume imag samples never done?

        return imaginary_samples

    def _loss(self, real_samples):

        batch_time, batch_size, _ = real_samples['state'].shape

        observation = real_samples['state'] # (time, batch, obs)
        action = real_samples['action'] # (time, batch, action)
        reward = real_samples['reward'] # (time, batch, reward)
        done = real_samples['done'] # (time, batch, done)

        # learn embed
        embed = self.observation_encoder(observation)

        prev_state = self.representation.initial_state(batch_size, device=action.device, dtype=action.dtype) # prev_state is a type of RSSMState

        # Rollout model by taking the same series of actions as the real model
        prior, post = self.rollout.rollout_representation(batch_time, embed, action, prev_state) # (batch_time, batch_size, -1)

        # Model Loss
        feat = get_feat(post)
        image_pred = self.observation_decoder(feat) # Gaussian Distribution
        reward_pred = self.reward_model(feat) # Gaussian Distribution
        reward_loss = -torch.mean(reward_pred.log_prob(reward))
        image_loss = -torch.mean(image_pred.log_prob(observation))
        prior_dist = get_dist(prior)
        post_dist = get_dist(post)
        div = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
        div = torch.max(div, div.new_full(div.size(), self._free_nats))
        model_loss = self._kl_scale * div + reward_loss + image_loss

        # Value Loss ?
        value_loss = None

        return model_loss, post

