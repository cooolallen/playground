'''IgnoreAgent'''
from . import SimpleAgent
from .. import constants

class IgnoreAgent(SimpleAgent):
    '''IgnoreAgent'''
    def __init__(self, *args, **kwargs):
        super(IgnoreAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        # modify the obs
        obs['board'][obs['board'] == obs['teammate'].value] = constants.Item.Flames.value

        # return the acts
        return super(IgnoreAgent, self).act(obs, action_space)