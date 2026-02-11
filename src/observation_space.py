import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ImgObsWrapper(gym.ObservationWrapper):
    """
    Replaces the observation with an RGB image from env.render().
    """

    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )

    def observation(self, obs):
        frame = self.env.render()
        frame = self._resize(frame, (self.height, self.width))
        return frame

    def _resize(self, frame, shape_hw):
        H, W = shape_hw
        h0, w0 = frame.shape[:2]
        ys = (np.linspace(0, h0 - 1, H)).astype(np.int32)
        xs = (np.linspace(0, w0 - 1, W)).astype(np.int32)
        return frame[ys][:, xs]
