"""Pick-and-place variant of gym_hil's Franka pick-cube task.

The scene is gym_hil's pick-cube scene plus a flat yellow target pad on the floor. Each reset
places the cube and the pad at random positions (from numpy's global RNG, like gym_hil), at
least MIN_SEPARATION apart. The robot must carry the cube onto the pad and let go.

Success: the cube's center is within PAD_HALF of the pad center on both axes, the cube rests on
the floor, the gripper command is open and the gripper has backed off at least CLEARANCE above
the cube, all for SUCCESS_HOLD_STEPS consecutive steps.
The episode then ends with info["succeed"] = True.

Usage:
  from pick_place_env import make_pick_place_env
  env = make_pick_place_env(image_obs=True)
"""
from pathlib import Path
from typing import Any, Dict, Tuple

import gym_hil
import mujoco
import numpy as np
from gymnasium import spaces
from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv, _SAMPLING_BOUNDS
from gym_hil.mujoco_gym_env import MAX_GRIPPER_COMMAND
from gym_hil.wrappers.factory import wrap_env

SCENE_XML = Path(gym_hil.__file__).parent / "assets" / "scene.xml"

PAD_SIZE = 0.07                  # pad side length (m); the cube is 0.04 m
PAD_HALF = PAD_SIZE / 2
PAD_THICKNESS = 0.002
PAD_RGBA = (0.95, 0.80, 0.10, 1.0)  # yellow: stands out from the blue floor and the purple cube
MIN_SEPARATION = 0.10            # minimum cube-to-pad center distance at reset (m)
REST_TOLERANCE = 0.005           # cube center may be at most this far above its resting height (m)
GRIPPER_OPEN_MAX = 0.1           # gripper command (0 open .. 1 closed) that still counts as open
CLEARANCE = 0.04                 # fingertips must be this far above the cube center (m), i.e. let go
SUCCESS_HOLD_STEPS = 3


def build_model(width: int, height: int) -> mujoco.MjModel:
    """gym_hil's pick-cube scene with the target pad added as a non-colliding static body."""
    spec = mujoco.MjSpec.from_file(SCENE_XML.as_posix())
    pad = spec.worldbody.add_body(name="pad", pos=[0.4, 0.0, PAD_THICKNESS / 2])
    pad.add_geom(
        name="pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[PAD_HALF, PAD_HALF, PAD_THICKNESS / 2],
        rgba=list(PAD_RGBA),
        contype=0,
        conaffinity=0,
    )
    model = spec.compile()
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    return model


class PandaPickPlaceGymEnv(PandaPickCubeGymEnv):
    """Carry the cube onto the yellow pad and release it."""

    def __init__(self, **kwargs):
        kwargs.setdefault("random_block_position", True)
        super().__init__(**kwargs)
        self._swap_in_model(build_model(self._render_specs.width, self._render_specs.height))
        self._pad_body_id = self._model.body("pad").id
        self._success_streak = 0
        if not self.image_obs:
            # block position + pad position
            self.observation_space["environment_state"] = spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)

    def _swap_in_model(self, model: mujoco.MjModel) -> None:
        """Replace the model built by gym_hil with one that also contains the pad.

        The pad is appended after every existing element and has no joints, actuators or
        sensors, so the ids gym_hil cached during __init__ stay valid.
        """
        physics_dt = self._model.opt.timestep
        self._model = model
        self._model.opt.timestep = physics_dt
        self._data = mujoco.MjData(self._model)
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
        self._viewer = mujoco.Renderer(self._model, height=self._render_specs.height, width=self._render_specs.width)

    @property
    def pad_pos(self) -> np.ndarray:
        return self._data.xpos[self._pad_body_id].copy()

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        # The parent samples the cube with np.random; the pad is sampled right after it.
        super().reset(seed=seed, **kwargs)
        block_xy = self._data.jnt("block").qpos[:2].copy()
        while True:
            pad_xy = np.random.uniform(*_SAMPLING_BOUNDS)
            if np.linalg.norm(pad_xy - block_xy) >= MIN_SEPARATION:
                break
        self._model.body_pos[self._pad_body_id] = (*pad_xy, PAD_THICKNESS / 2)
        mujoco.mj_forward(self._model, self._data)
        self._success_streak = 0
        return self._compute_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.apply_action(action)
        obs = self._compute_observation()

        placed = self._cube_on_pad()
        self._success_streak = self._success_streak + 1 if placed else 0
        success = self._success_streak >= SUCCESS_HOLD_STEPS
        rew = float(success) if self.reward_type == "sparse" else self._dense_reward()

        block_pos = self._data.sensor("block_pos").data
        exceeded_bounds = np.any(block_pos[:2] < (_SAMPLING_BOUNDS[0] - 0.05)) or np.any(
            block_pos[:2] > (_SAMPLING_BOUNDS[1] + 0.05)
        )
        terminated = bool(success or exceeded_bounds)
        return obs, rew, terminated, False, {"succeed": success, "placed": placed}

    def _cube_on_pad(self) -> bool:
        block = self._data.sensor("block_pos").data
        offset = np.abs(block[:2] - self.pad_pos[:2])
        resting = block[2] < self._z_init + REST_TOLERANCE
        gripper_open = self._data.ctrl[self._gripper_ctrl_id] / MAX_GRIPPER_COMMAND <= GRIPPER_OPEN_MAX
        clear = self._data.sensor("2f85/pinch_pos").data[2] - block[2] > CLEARANCE
        return bool(np.all(offset < PAD_HALF) and resting and gripper_open and clear)

    def _dense_reward(self) -> float:
        block = self._data.sensor("block_pos").data
        tcp = self._data.sensor("2f85/pinch_pos").data
        r_reach = np.exp(-20 * np.linalg.norm(block - tcp))
        r_place = np.exp(-20 * np.linalg.norm(block[:2] - self.pad_pos[:2]))
        return float(0.3 * r_reach + 0.7 * r_place)

    def _compute_observation(self) -> dict:
        obs = super()._compute_observation()
        if not self.image_obs:
            block = self._data.sensor("block_pos").data.astype(np.float32)
            obs["environment_state"] = np.concatenate([block, self.pad_pos.astype(np.float32)])
        return obs


def make_pick_place_env(image_obs: bool = True, **kwargs):
    """Same wrappers as gym_hil.make_env, so actions are [dx, dy, dz, gripper] in [-1,1]^3 x [0,2]."""
    env = PandaPickPlaceGymEnv(image_obs=image_obs, render_mode="rgb_array", **kwargs)
    return wrap_env(env, use_viewer=False, use_inputs_control=False, reset_delay_seconds=0.0, gripper_penalty=-0.02)
