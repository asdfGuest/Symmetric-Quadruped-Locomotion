from skrl.models.torch import Model, DeterministicMixin
from src.skrl_model import GaussianMixin

from gymnasium.spaces import Space

import torch as th
import math
from typing import Type, List


class MLP(th.nn.Module) :
    def __init__(self, net_arch:List[int], activ_fn:Type[th.nn.Module], activ_output:bool=False) :
        th.nn.Module.__init__(self)

        self.net = th.nn.Sequential()
        for k in range(len(net_arch) - 1) :
            self.net.append(th.nn.Linear(net_arch[k], net_arch[k+1]))
            self.net.append(th.nn.Identity() if (k == len(net_arch) - 2 and not activ_output) else activ_fn())

    def forward(self, x:th.Tensor) :
        return self.net(x)


class MlpPolicy(GaussianMixin, Model) :
    def __init__(
            self,
            observation_space:Space,
            action_space:Space,
            device=None,
            min_std:float=0.1,
            max_std:float=1.2,
            init_std:float=1.0,
            net_arch:List[int]=[64,64],
            activ_fn:Type[th.nn.Module]=th.nn.SiLU
        ) :

        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, min_log_std=math.log(min_std), max_log_std=math.log(max_std))

        self.obs_num = observation_space.shape[0]
        self.act_num = action_space.shape[0]
        net_arch = [self.obs_num] + net_arch + [self.act_num]

        self.mean = MLP(net_arch, activ_fn)
        self.log_std = th.nn.Parameter(th.full(size=(1,self.act_num), fill_value=math.log(init_std)))
        #self.std = th.nn.Parameter(th.full(size=(1,self.act_num), fill_value=init_std))

    def compute(self, inputs, role=''):
        obs = inputs['states']
        return self.mean(obs), self.log_std, {}
        #return self.mean(obs), self.std.log(), {}


class MlpValue(DeterministicMixin, Model) :
    def __init__(
            self,
            observation_space:Space,
            device=None,
            net_arch:List[int]=[64,64],
            activ_fn:Type[th.nn.Module]=th.nn.SiLU
        ) :

        Model.__init__(self, observation_space, 1, device)
        DeterministicMixin.__init__(self)

        self.obs_num = observation_space.shape[0]
        net_arch = [self.obs_num] + net_arch + [1]

        self.mlp = MLP(net_arch, activ_fn)

    def compute(self, inputs, role=''):
        obs = inputs['states']
        return self.mlp(obs), {}
