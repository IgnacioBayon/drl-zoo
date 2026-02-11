import gymnasium as gym
from pixel_observation import PixelObsWrapper


def run_gym_simulation():

    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    env = PixelObsWrapper(env, width=84, height=84)

    observation, info = env.reset(seed=42)

    try:
        while True:
            # Sample a random action
            action = env.action_space.sample()

            # Step the environment
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
