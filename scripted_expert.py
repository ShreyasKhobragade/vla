"""Scripted expert for gym_hil Franka pick-cube.

Produces the same 4-D action the keyboard teleop produces:
  [dx, dy, dz, gripper] with xyz in [-1, 1] (1.0 = 0.025 m of end-effector target travel)
  and gripper in [0, 2] where 1.0 means "no change".
It uses privileged simulator state (cube position), which is fine for generating demos:
the recorded dataset only contains images, robot state and actions.
"""
import numpy as np

EE_STEP = 0.025          # DEFAULT_EE_STEP_SIZE in gym_hil
GRIP_HOLD = 1.0          # gripper action meaning "do not change"
GRIP_CLOSE = 2.0         # verified empirically in test_expert.py (ctrl 0 -> 255 closes the Robotiq)
GRIP_OPEN = 0.0


class PickCubeExpert:
    def __init__(self, hover_z=0.12, grasp_z=0.022, lift_z=0.30, xy_tol=0.006,
                 z_tol=0.006, close_steps=6, noise=0.0, rng=None):
        self.hover_z, self.grasp_z, self.lift_z = hover_z, grasp_z, lift_z
        self.xy_tol, self.z_tol, self.close_steps = xy_tol, z_tol, close_steps
        self.noise = noise
        self.rng = rng or np.random.default_rng(0)
        self.reset()

    def reset(self):
        self.phase = "hover"
        self.counter = 0

    def _goto(self, mocap, target):
        delta = np.clip(target - mocap, -EE_STEP, EE_STEP) / EE_STEP
        return delta

    def act(self, env):
        u = env.unwrapped
        block = np.array(u._data.sensor("block_pos").data)
        tcp = np.array(u._data.sensor("2f85/pinch_pos").data)
        mocap = np.array(u._data.mocap_pos[0])
        grip = GRIP_HOLD
        xy_err = np.linalg.norm(tcp[:2] - block[:2])

        if self.phase == "hover":
            target = np.array([block[0], block[1], self.hover_z])
            if xy_err < self.xy_tol and abs(tcp[2] - self.hover_z) < self.z_tol * 2:
                self.phase = "descend"
        if self.phase == "descend":
            target = np.array([block[0], block[1], self.grasp_z])
            if abs(tcp[2] - self.grasp_z) < self.z_tol and xy_err < self.xy_tol * 2:
                self.phase = "close"
                self.counter = 0
        if self.phase == "close":
            target = np.array([block[0], block[1], self.grasp_z])
            grip = GRIP_CLOSE
            self.counter += 1
            if self.counter >= self.close_steps:
                self.phase = "lift"
        if self.phase == "lift":
            target = np.array([block[0], block[1], self.lift_z])
            grip = GRIP_CLOSE

        a = self._goto(mocap, target)
        if self.noise > 0 and self.phase in ("hover", "descend"):
            a = np.clip(a + self.rng.normal(0, self.noise, 3), -1, 1)
        return np.array([*a, grip], dtype=np.float32)


class PickPlaceExpert(PickCubeExpert):
    """Pick the cube, carry it over the target pad, lower it, open the gripper and back away.

    Used with pick_place_env.PandaPickPlaceGymEnv. The first phases (hover, descend, close)
    are the pick-cube expert's; after the grasp it lifts only to carry_z instead of lift_z.
    """

    def __init__(self, carry_z=0.10, place_z=0.026, open_steps=4, retreat_z=0.12, **kwargs):
        super().__init__(**kwargs)
        self.carry_z, self.place_z = carry_z, place_z
        self.open_steps, self.retreat_z = open_steps, retreat_z

    def act(self, env):
        if self.phase in ("hover", "descend", "close"):
            a = super().act(env)
            if self.phase == "lift":  # the parent just finished closing
                self.phase = "carry_up"
            return a

        u = env.unwrapped
        pad = u.pad_pos
        tcp = np.array(u._data.sensor("2f85/pinch_pos").data)
        mocap = np.array(u._data.mocap_pos[0])
        grip = GRIP_HOLD

        if self.phase == "carry_up":
            target = np.array([tcp[0], tcp[1], self.carry_z])
            if abs(tcp[2] - self.carry_z) < self.z_tol * 2:
                self.phase = "carry"
        if self.phase == "carry":
            target = np.array([pad[0], pad[1], self.carry_z])
            if np.linalg.norm(tcp[:2] - pad[:2]) < self.xy_tol:
                self.phase = "lower"
        if self.phase == "lower":
            target = np.array([pad[0], pad[1], self.place_z])
            if abs(tcp[2] - self.place_z) < self.z_tol:
                self.phase = "open"
                self.counter = 0
        if self.phase == "open":
            target = np.array([pad[0], pad[1], self.place_z])
            grip = GRIP_OPEN
            self.counter += 1
            if self.counter >= self.open_steps:
                self.phase = "retreat"
        if self.phase == "retreat":
            target = np.array([pad[0], pad[1], self.retreat_z])

        return np.array([*self._goto(mocap, target), grip], dtype=np.float32)
