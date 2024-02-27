from gymnasium.envs.registration import register

register(
     id="TT2",
     entry_point="tt2_env.envs:TT2Env",
)