from gym.envs.registration import register

# Yumi_mpc
register(
    id='yumi-v0',
    entry_point='yumi_gym.envs:YumiEnv',
)

# Yumi_pid
register(
    id='yumi-v1',
    entry_point='yumi_gym.envs:YumiEnv1',
)