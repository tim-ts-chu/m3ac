from agent.imagine_agent import ImagineAgent
from agent.discriminate_agent import DiscriminateAgent
from agent.policy_agent import PolicyAgent
from replay.replay import ReplayBuffer, BufferFields

class M3ACAlgorithm:

    def __init__(self, batch_size):
        '''
        setup hyperparameter
        '''
        self.discount = 0.99
        self._batch_size = batch_size

    def optimize_agent(self,
            imag_agent: ImagineAgent,
            disc_agent: DiscriminateAgent,
            policy_agent: PolicyAgent,
            real_buffer: ReplayBuffer,
            imaginary_buffer: ReplayBuffer):

        # draw sample from real buffer
        real_samples = real_buffer.sample(self._batch_size)
        #for k in BufferFields.keys():
        #    print(real_samples[k].shape)

        # optimize imagine_agent
        post = imagine_agent.optimize_agent(real_samples)

        # imagine samples using imagine_agent
        imaginary_samples = imagine_agent.imagine(post, policy_agent)

        # discriminate imag_samples using discriminate_agent

        # optimize policy_agent using real_samples and discriminated imag_samples

        # (push imag_samples into imag_buffer)

        # optimize imagine_agent using discriminate gradient

        # optimize discriminate_agent



