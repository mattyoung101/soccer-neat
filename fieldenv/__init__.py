from gym.envs.registration import register

register(
    id='Soccer-v0',
    entry_point='fieldenv.fieldenv:FieldEnv',
)