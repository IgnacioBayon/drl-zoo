import gymnasium as gym
from gymnasium.wrappers import HumanRendering

from observation_space import ImgObsWrapper


def run_gym_simulation():

    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    env = ImgObsWrapper(env, width=84, height=84)
    env = HumanRendering(env)

    observation, info = env.reset(seed=42)

    try:
        while True:
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("\nSimulation closed.")


if __name__ == "__main__":
    run_gym_simulation()
