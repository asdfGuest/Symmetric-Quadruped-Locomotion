

def apply_cfg_mapping(cfg:dict|list, mapping:dict) :
    '''
    - key of mapping must not be the dictionary or list
    - structure of cfg must not have cycle
    '''

    def dfs(x:dict|list) -> bool:

        if isinstance(x, (dict, list)) :
            keys = []
            for key in (x if isinstance(x, dict) else range(len(x))) :
                if dfs(x[key]) :
                    keys.append(key)
            for key in keys :
                if x[key] in mapping :
                    x[key] = mapping[x[key]]
            return False
        else :
            return True
    
    dfs(cfg)


def load_cfg(cfg_path:str) :

    from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG

    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveRL

    from src.model import MlpPolicy, MlpValue

    from torch.nn import Identity, Sigmoid, Tanh, ReLU, LeakyReLU, SELU, SiLU, ELU


    _mapping = {
        'None' : None,

        'Identity' : Identity,
        'Sigmoid' : Sigmoid,
        'Tanh' : Tanh,
        'ReLU' : ReLU,
        'LeakyReLU' : LeakyReLU,
        'SELU' : SELU,
        'SiLU' : SiLU,
        'ELU' : ELU,

        'RunningStandardScaler' : RunningStandardScaler,
        'KLAdaptiveRL' : KLAdaptiveRL,

        'MlpPolicy' : MlpPolicy,
        'MlpValue' : MlpValue
    }


    import yaml
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    apply_cfg_mapping(cfg, _mapping)
    cfg['ppo'] = PPO_DEFAULT_CONFIG|cfg['ppo']

    return cfg
