from PIL import Image
import numpy as np
from typing import Any, Dict, List
from collections import deque
from contextlib import contextmanager
from .client import WMClient, WMPredictionOutput


class WMEnv:
    def __init__(
            self,
            env,
            empty_env,
            client: WMClient=None,
            panel_cams: List[str]=["agentview", "birdview", "robot0_eye_in_hand", "sideview"]
    ):
        self.env = env
        self.empty_env = empty_env
        self.client = client if client is not None else WMClient()
        self.panel_cams = panel_cams

    def reset(self):
        # reset envs
        self.empty_env.reset()
        obs = self.env.reset()
        self.empty_env.copy_robot_state(self.env)

        # reset buffers
        self.history_frames = deque(maxlen=160)
        self.history_conds = deque(maxlen=160)

        # # add first frames
        # for _ in range(80):
        #     self.update_frame_buffer(obs)
        #     self.update_cond_buffer()

        return obs

    def update_frame_buffer(self, obs):
        frames = []
        for cam in self.panel_cams:
            frame = obs[f"{cam}_image"][::-1, :, :]
            frames.append(frame)
        frame = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])])
        self.history_frames.append(frame)

    def get_cond_frame(self):
        cond_frames = []
        cam_infos = self.empty_env.get_camera_info()
        for cam in self.panel_cams:
            cam_info = cam_infos[cam]
            height = cam_info['camera_height']
            width = cam_info['camera_width']
            cam_transform = cam_info['camera_transform']
            if "robot" in cam:
                pose_image = self.empty_env.plot_wrist_pose(cam_transform, height=height, width=width)
            else:
                pose_image = self.empty_env.plot_pose(cam_transform, height=height, width=width)
            cond_frames.append(pose_image)  
        cond_frame = np.vstack([np.hstack(cond_frames[:2]), np.hstack(cond_frames[2:])])
        return cond_frame

    def update_cond_buffer(self):
        self.empty_env.copy_robot_state(self.env)
        cond_frame = self.get_cond_frame()
        self.history_conds.append(cond_frame)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.update_frame_buffer(obs)
        self.update_cond_buffer()
        return obs, reward, done, info
    
    @contextmanager
    def simulation(self):
        """Context manager that initializes simulation buffers on entry and cleans up on exit."""
        # Initialize simulation buffers
        self.sim_frame_buffer = list(self.history_frames)  # Start with current history frames
        self.sim_cond_buffer = list(self.history_conds)    # Start with current history conditions
        
        # Copy robot state on entry
        self.empty_env.copy_robot_state(self.env)
        
        try:
            yield self
        finally:
            # Restore robot state on exit
            self.empty_env.copy_robot_state(self.env)
            
            # Clean up simulation buffers
            self.sim_frame_buffer = None
            self.sim_cond_buffer = None
    

    def simulate(self, actions):
        future_conds = []
        future_obs = []
        for i in range(len(actions)):
            obs, _, _, _ = self.empty_env.step(actions[i])
            cond_frame = self.get_cond_frame()
            future_conds.append(cond_frame)
            future_obs.append(obs)

        result = self.client.predict(
            history_frames=self.sim_frame_buffer,
            history_conds=self.sim_cond_buffer,
            future_conds=future_conds
        )
        if not isinstance(result, WMPredictionOutput):
            raise RuntimeError(f"WM prediction failed: {result}")

        future_panels = result.pred_panels
        future_indices = np.linspace(0, len(future_conds)-1, num=len(future_panels), dtype=int)
        for i, idx in enumerate(future_indices):
            for cam_id, cam in enumerate(self.panel_cams):
                future_obs[idx][f"{cam}_image"] = future_panels[i][cam_id][::-1, :, :]
        
        output = {
            "WMPredictionOutput": result,
            "future_obs": [future_obs[i] for i in future_indices],
        }

        # update simulation buffer
        self.sim_cond_buffer.extend(future_conds)
        frame_indices = np.linspace(0, len(result.pred_frames)-1, num=len(future_conds), dtype=int)
        future_frames = np.array(result.pred_frames)[frame_indices]
        self.sim_frame_buffer.extend(future_frames)
        assert len(self.sim_frame_buffer) == len(self.sim_cond_buffer), "Frame and condition buffers must be the same length."

        return output