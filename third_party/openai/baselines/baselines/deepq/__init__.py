# from third_party.openai.baselines.baselines.deepq import models  # noqa
# from third_party.openai.baselines.baselines.deepq.build_graph import build_act, build_train  # noqa
# from third_party.openai.baselines.baselines.deepq.simple import learn, load  # noqa
from third_party.openai.baselines.baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

# def wrap_atari_dqn(env):
#     from third_party.openai.baselines.baselines.common.atari_wrappers import wrap_deepmind
#     return wrap_deepmind(env, frame_stack=True, scale=True)