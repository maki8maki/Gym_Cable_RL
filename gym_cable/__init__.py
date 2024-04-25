from gymnasium.envs.registration import register


def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""

    register(
        id="MZ04CableGrasp-v0",
        entry_point="gym_cable.envs.mz04.cable_grasp:MujocoMZ04CableGraspEnv",
        max_episode_steps=100,
    )
