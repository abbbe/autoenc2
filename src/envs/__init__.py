from src.envs.myenv import TwoBallsEnv
from src.envs.gymenv import ArmRobotEnv

def get_env_names():
    return ['twoballs', 'gymarm']
    
def load_env(env_name):
    if env_name == 'twoballs':
        env = TwoBallsEnv()
    elif env_name == 'gymarm':
        env = ArmRobotEnv()
    else:
        raise NotImplementedError()
    return env
