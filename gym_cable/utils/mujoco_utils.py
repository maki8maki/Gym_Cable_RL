from typing import Dict, Tuple, Union

import numpy as np
from gymnasium import error
import collections
from absl import logging
from copy import deepcopy

from . import rotations

try:
    import mujoco
    from mujoco import MjData, MjModel, mjtObj
except ImportError as e:
    raise error.DependencyNotInstalled(f"{e}. (HINT: you need to install mujoco")

MJ_OBJ_TYPES = [
    "mjOBJ_BODY",
    "mjOBJ_JOINT",
    "mjOBJ_GEOM",
    "mjOBJ_SITE",
    "mjOBJ_CAMERA",
    "mjOBJ_ACTUATOR",
    "mjOBJ_SENSOR",
]

_INVALID_JOINT_NAMES_TYPE = ('`joint_names` must be either None, a list, a tuple, or a numpy array; got {}.')
_REQUIRE_TARGET_POS_OR_QUAT = ('At least one of `target_pos` or `target_quat` must be specified.')

IKResult = collections.namedtuple('IKResult', ['qpos', 'err_norm', 'steps', 'success'])

DOWN_QUATERNION = np.array([0., 0.70710678118, 0.70710678118, 0.])

def qpos_from_site_pose(model, data, site_name, target_pos=None, target_quat=None, joint_names=None,
                        tol=1e-14, rot_weight=1.0, regularization_threshold=0.1, regularization_strength=3e-2,
                        max_update_norm=2.0, progress_thresh=20.0, max_steps=100, inplace=False):
    """Find joint positions that satisfy a target site position and/or rotation.
    Borrowed from google-deepmind/dm_control (https://github.com/google-deepmind/dm_control.git).

    Args:
        model: A `mujoco.MjModel` instance.
        data: A `mujoco.MjData` instance.
        site_name: A string specifying the name of the target site.
        target_pos: A (3,) numpy array specifying the desired Cartesian position of
            the site, or None if the position should be unconstrained (default).
            One or both of `target_pos` or `target_quat` must be specified.
        target_quat: A (4,) numpy array specifying the desired orientation of the
            site as a quaternion, or None if the orientation should be unconstrained
            (default). One or both of `target_pos` or `target_quat` must be specified.
        joint_names: (optional) A list, tuple or numpy array specifying the names of
            one or more joints that can be manipulated in order to achieve the target
            site pose. If None (default), all joints may be manipulated.
        tol: (optional) Precision goal for `qpos` (the maximum value of `err_norm`
            in the stopping criterion).
        rot_weight: (optional) Determines the weight given to rotational error
            relative to translational error.
        regularization_threshold: (optional) L2 regularization will be used when
            inverting the Jacobian whilst `err_norm` is greater than this value.
            regularization_strength: (optional) Coefficient of the quadratic penalty
            on joint movements.
        max_update_norm: (optional) The maximum L2 norm of the update applied to
            the joint positions on each iteration. The update vector will be scaled
            such that its magnitude never exceeds this value.
            progress_thresh: (optional) If `err_norm` divided by the magnitude of the
            joint position update is greater than this value then the optimization
            will terminate prematurely. This is a useful heuristic to avoid getting
            stuck in local minima.
        max_steps: (optional) The maximum number of iterations to perform.
        inplace: (optional) If True, `physics.data` will be modified in place.
            Default value is False, i.e. a copy of `physics.data` will be made.

    Returns:
        An `IKResult` namedtuple with the following fields:
        qpos: An (nq,) numpy array of joint positions.
        err_norm: A float, the weighted sum of L2 norms for the residual
            translational and rotational errors.
        steps: An int, the number of iterations that were performed.
        success: Boolean, True if we converged on a solution within `max_steps`,
            False otherwise.

    Raises:
        ValueError: If both `target_pos` and `target_quat` are None, or if
            `joint_names` has an invalid type.
    """

    dtype = data.qpos.dtype

    if target_pos is not None and target_quat is not None:
        jac = np.empty((6, model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, model.nv), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if target_pos is not None:
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif target_quat is not None:
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError(_REQUIRE_TARGET_POS_OR_QUAT)

    update_nv = np.zeros(model.nv, dtype=dtype)

    if target_quat is not None:
        site_xquat = np.empty(4, dtype=dtype)
        neg_site_xquat = np.empty(4, dtype=dtype)
        err_rot_quat = np.empty(4, dtype=dtype)
    
    if not inplace:
        data = deepcopy(data)

    # Ensure that the Cartesian position of the site is up to date.
    mujoco.mj_fwdPosition(model, data)

    # Convert site name to index.
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    # These are views onto the underlying MuJoCo buffers. mj_fwdPosition will
    # update them in place, so we can avoid indexing overhead in the main loop.
    site_xpos = data.site(site_name).xpos
    site_xmat = data.site(site_name).xmat

    # This is an index into the rows of `update` and the columns of `jac`
    # that selects DOFs associated with joints that we are allowed to manipulate.
    if joint_names is None:
        dof_indices = slice(None)  # Update all DOFs.
    elif isinstance(joint_names, (list, np.ndarray, tuple)):
        if isinstance(joint_names, tuple):
            joint_names = list(joint_names)
        # Find the indices of the DOFs belonging to each named joint. Note that
        # these are not necessarily the same as the joint IDs, since a single joint
        # may have >1 DOF (e.g. ball joints).
        indexer = model.dof_jntid
        # `dof_jntid` is an `(nv,)` array indexed by joint name. We use its row
        # indexer to map each joint name to the indices of its corresponding DOFs.
        ids = []
        for joint_name in joint_names:
            ids.append(model.joint(joint_name).id)
        dof_indices = np.isin(indexer, ids)
    else:
        raise ValueError(_INVALID_JOINT_NAMES_TYPE.format(type(joint_names)))

    steps = 0
    success = False

    for steps in range(max_steps):
        err_norm = 0.0
        if target_pos is not None:
            # Translational error.
            err_pos[:] = target_pos - site_xpos
            err_norm += np.linalg.norm(err_pos)
        if target_quat is not None:
            # Rotational error.
            mujoco.mju_mat2Quat(site_xquat, site_xmat)
            mujoco.mju_negQuat(neg_site_xquat, site_xquat)
            mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
            mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1)
            err_norm += np.linalg.norm(err_rot) * rot_weight

        if err_norm < tol:
            logging.debug('Converged after %i steps: err_norm=%3g', steps, err_norm)
            success = True
            break
        else:
            # TODO(b/112141670): Generalize this to other entities besides sites.
            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
            jac_joints = jac[:, dof_indices]

            # TODO(b/112141592): This does not take joint limits into consideration.
            reg_strength = (regularization_strength if err_norm > regularization_threshold else 0.0)
            update_joints = nullspace_method(jac_joints, err, regularization_strength=reg_strength)

            update_norm = np.linalg.norm(update_joints)

            # Check whether we are still making enough progress, and halt if not.
            progress_criterion = err_norm / update_norm
            if progress_criterion > progress_thresh:
                logging.debug('Step %2i: err_norm / update_norm (%3g) > tolerance (%3g). Halting due to insufficient progress',
                            steps, progress_criterion, progress_thresh)
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            # Write the entries for the specified joints into the full `update_nv` vector.
            update_nv[dof_indices] = update_joints

            # Update `data.qpos`, taking quaternions into account.
            mujoco.mj_integratePos(model, data.qpos, update_nv, 1)

            # Compute the new Cartesian position of the site.
            mujoco.mj_fwdPosition(model, data)

            logging.debug('Step %2i: err_norm=%-10.3g update_norm=%-10.3g', steps, err_norm, update_norm)

    if not success and steps == max_steps - 1:
        logging.warning('Failed to converge after %i steps: err_norm=%3g', steps, err_norm)

    if not inplace:
        # Our temporary copy of data is about to go out of scope, and when
        # it does the underlying mjData pointer will be freed and data.qpos
        # will be a view onto a block of deallocated memory. We therefore need to
        # make a copy of data.qpos while data is still alive.
        qpos = deepcopy(data.qpos)
    else:
        # If we're modifying data in place then it's fine to return a view.
        qpos = data.qpos

    return IKResult(qpos=qpos, err_norm=err_norm, steps=steps, success=success)


def nullspace_method(jac_joints, delta, regularization_strength=0.0):
    """Calculates the joint velocities to achieve a specified end effector delta.

    Args:
        jac_joints: The Jacobian of the end effector with respect to the joints. A
            numpy array of shape `(ndelta, nv)`, where `ndelta` is the size of `delta`
            and `nv` is the number of degrees of freedom.
        delta: The desired end-effector delta. A numpy array of shape `(3,)` or
            `(6,)` containing either position deltas, rotation deltas, or both.
        regularization_strength: (optional) Coefficient of the quadratic penalty
            on joint movements. Default is zero, i.e. no regularization.

    Returns:
        An `(nv,)` numpy array of joint velocities.

    Reference:
        Buss, S. R. S. (2004). Introduction to inverse kinematics with jacobian
        transpose, pseudoinverse and damped least squares methods.
        https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
    """
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if regularization_strength > 0:
        # L2 regularization
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]


def set_site_to_xpos(model, data, site, joint_names, target_pos, target_quat=None, max_ik_attempts=10,
                     random_state=np.random.RandomState(0)):
    """Moves the arm so that a site occurs at the specified location.

    This function runs the inverse kinematics solver to find a configuration
    arm joints for which the pinch site occurs at the specified location in
    Cartesian coordinates.

    Args:
        model: A `mujoco.MjModel` instance.
        data: A `mujoco.MjData` instance.
        site: A string specifying the full name of the site whose position is being set.
        joint_names: A list, tuple or numpy array specifying the names of one or more 
            joints that can be manipulated in order to achieve the target site pose.
        target_pos: The desired Cartesian location of the site.
        target_quat: (optional) The desired orientation of the site, expressed
            as a quaternion. If `None`, the default orientation is to point
            vertically downwards.
        max_ik_attempts: (optional) Maximum number of attempts to make at finding
            a solution satisfying `target_pos` and `target_quat`. The joint
            positions will be randomized after each unsuccessful attempt.
        random_state: An `np.random.RandomState` instance.

    Returns:
        A boolean indicating whether the desired configuration is obtained.

    Raises:
        ValueError: If site is not a string.
    """
    if isinstance(site, str):
        site_name = site
    else:
        raise ValueError('site should either be a string: got {}'.format(site))
    
    if target_quat is None:
        target_quat = DOWN_QUATERNION
    
    joint_ranges = np.array([model.joint(joint_name).range for joint_name in joint_names])

    for _ in range(max_ik_attempts):
        result = qpos_from_site_pose(model=model, data=data, site_name=site_name, target_pos=target_pos, target_quat=target_quat,
                                     joint_names=joint_names, rot_weight=2, inplace=True)
        success = result.success

        # Canonicalise the angle to [-pi, pi]
        if success:
            joints_qpos, _, _ = robot_get_obs(model, data, joint_names)
            normalize_joints_qpos = rotations.normalize_angles(joints_qpos)
            if (normalize_joints_qpos < joint_ranges[:, 0]).any() or (normalize_joints_qpos > joint_ranges[:, 1]).any():
                success = False
            else:
                for joint_name, qpos in zip(joint_names, normalize_joints_qpos):
                    set_joint_qpos(model, data, joint_name, qpos)

        # If succeeded or only one attempt, break and do not randomize joints.
        if success or max_ik_attempts <= 1:
            break
        else:
            for joint_name, joint_range in zip(joint_names, joint_ranges):
                set_joint_qpos(model, data, joint_name, random_state.uniform(joint_range[0], joint_range[1]))
        
        mujoco.mj_fwdPosition(model, data)

    return success


def ik_set_action(model, data, action, site_name, joint_names):
    target_pos = get_site_xpos(model, data, site_name) + action[:3]
    target_quat = rotations.mat2quat(get_site_xmat(model, data, site_name)) + action[3:7]
    success = set_site_to_xpos(model, data, site_name, joint_names, target_pos, target_quat)
    return success


def robot_get_obs(model, data, joint_names):
    """Returns all joint positions and velocities associated with a robot."""
    if data.qpos is not None and joint_names is not None:
        names = [n for n in joint_names if n.startswith("robot")]
        return (
            np.squeeze(np.array([get_joint_qpos(model, data, name) for name in names])),
            np.squeeze(np.array([get_joint_qvel(model, data, name) for name in names])),
            np.squeeze(np.array([get_joint_qacc(model, data, name) for name in names])),
        )
    return np.zeros(0), np.zeros(0), np.zeros(0)


def ctrl_set_action(model, data, action):
    """For torque actuators it copies the action into mujoco ctrl field.

    For position actuators it sets the target relative to the current qpos.
    """
    if model.nmocap > 0:
        _, action = np.split(action, (model.nmocap * 7,))

    if len(data.ctrl) > 0:
        for i in range(action.shape[0]):
            if model.actuator_biastype[i] == 0:
                data.ctrl[i] = action[i]
            else:
                idx = model.jnt_qposadr[model.actuator_trnid[i, 0]]
                data.ctrl[i] = data.qpos[idx] + action[i]


def mocap_set_action(model, data, action):
    """Update the position of the mocap body with the desired action.

    The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if model.nmocap > 0:
        action, _ = np.split(action, (model.nmocap * 7,))
        action = action.reshape(model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        # reset_mocap2body_xpos(model, data)
        data.mocap_pos[:] = data.mocap_pos + pos_delta
        data.mocap_quat[:] = data.mocap_quat + quat_delta


def reset_mocap_welds(model, data):
    """Resets the mocap welds that we use for actuation."""
    if model.nmocap > 0 and model.eq_data is not None:
        for i in range(model.eq_data.shape[0]):
            if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                model.eq_data[i, :7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    mujoco.mj_forward(model, data)


def reset_mocap2body_xpos(model, data):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if model.eq_type is None or model.eq_obj1id is None or model.eq_obj2id is None:
        return
    for eq_type, obj1_id, obj2_id in zip(
        model.eq_type, model.eq_obj1id, model.eq_obj2id
    ):
        if eq_type != mujoco.mjtEq.mjEQ_WELD:
            continue

        mocap_id = model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert mocap_id != -1
        data.mocap_pos[mocap_id][:] = data.xpos[body_idx]
        data.mocap_quat[mocap_id][:] = data.xquat[body_idx]


def get_site_jacp(model, data, site_id):
    """Return the Jacobian' translational component of the end-effector of
    the corresponding site id.
    """
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)

    return jacp


def get_site_jacr(model, data, site_id):
    """Return the Jacobian' rotational component of the end-effector of
    the corresponding site id.
    """
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, None, jacr, site_id)

    return jacr


def set_joint_qpos(model, data, name, value):
    """Set the joint positions (qpos) of the model."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_qposadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 7
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim
    value = np.array(value)
    if ndim > 1:
        assert value.shape == (
            end_idx - start_idx
        ), f"Value has incorrect shape {name}: {value}"
    data.qpos[start_idx:end_idx] = value


def set_joint_qvel(model, data, name, value):
    """Set the joints linear and angular velocities (qvel) of the model."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_dofadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 3
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim
    value = np.array(value)
    if ndim > 1:
        assert value.shape == (
            end_idx - start_idx
        ), f"Value has incorrect shape {name}: {value}"
    data.qvel[start_idx:end_idx] = value


def set_joint_qacc(model, data, name, value):
    """Set the joints linear and angular acceleration (qacc) of the model."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_dofadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 3
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim
    value = np.array(value)
    if ndim > 1:
        assert value.shape == (
            end_idx - start_idx
        ), f"Value has incorrect shape {name}: {value}"
    data.qacc[start_idx:end_idx] = value


def get_joint_qpos(model, data, name):
    """Return the joints position and orientation (qpos) of the model."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_qposadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 7
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim

    return data.qpos[start_idx:end_idx].copy()


def get_joint_qvel(model, data, name):
    """Return the joints linear and angular velocities (qvel) of the model."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_dofadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim

    return data.qvel[start_idx:end_idx].copy()


def get_joint_qacc(model, data, name):
    """Return the joints linear and angular acceleration (qacc) of the model."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_dofadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim

    return data.qacc[start_idx:end_idx].copy()


def get_site_xpos(model, data, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xpos[site_id]


def get_site_xvelp(model, data, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    jacp = get_site_jacp(model, data, site_id)
    xvelp = jacp @ data.qvel
    return xvelp


def get_site_xvelr(model, data, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    jacp = get_site_jacr(model, data, site_id)
    xvelp = jacp @ data.qvel
    return xvelp


def set_mocap_pos(model, data, name, value):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    mocap_id = model.body_mocapid[body_id]
    data.mocap_pos[mocap_id] = value


def set_mocap_quat(model: MjModel, data: MjData, name: str, value):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    mocap_id = model.body_mocapid[body_id]
    data.mocap_quat[mocap_id] = value


def get_site_xmat(model: MjModel, data: MjData, name: str):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xmat[site_id].reshape(3, 3)


def extract_mj_names(
    model: MjModel, obj_type: mjtObj
) -> Tuple[Union[Tuple[str, ...], Tuple[()]], Dict[str, int], Dict[int, str]]:

    if obj_type == mujoco.mjtObj.mjOBJ_BODY:
        name_addr = model.name_bodyadr
        n_obj = model.nbody

    elif obj_type == mujoco.mjtObj.mjOBJ_JOINT:
        name_addr = model.name_jntadr
        n_obj = model.njnt

    elif obj_type == mujoco.mjtObj.mjOBJ_GEOM:
        name_addr = model.name_geomadr
        n_obj = model.ngeom

    elif obj_type == mujoco.mjtObj.mjOBJ_SITE:
        name_addr = model.name_siteadr
        n_obj = model.nsite

    elif obj_type == mujoco.mjtObj.mjOBJ_LIGHT:
        name_addr = model.name_lightadr
        n_obj = model.nlight

    elif obj_type == mujoco.mjtObj.mjOBJ_CAMERA:
        name_addr = model.name_camadr
        n_obj = model.ncam

    elif obj_type == mujoco.mjtObj.mjOBJ_ACTUATOR:
        name_addr = model.name_actuatoradr
        n_obj = model.nu

    elif obj_type == mujoco.mjtObj.mjOBJ_SENSOR:
        name_addr = model.name_sensoradr
        n_obj = model.nsensor

    elif obj_type == mujoco.mjtObj.mjOBJ_TENDON:
        name_addr = model.name_tendonadr
        n_obj = model.ntendon

    elif obj_type == mujoco.mjtObj.mjOBJ_MESH:
        name_addr = model.name_meshadr
        n_obj = model.nmesh
    else:
        raise ValueError(
            "`{}` was passed as the MuJoCo model object type. The MuJoCo model object type can only be of the following mjtObj enum types: {}.".format(
                obj_type, MJ_OBJ_TYPES
            )
        )

    id2name = {i: None for i in range(n_obj)}
    name2id = {}
    for addr in name_addr:
        name = model.names[addr:].split(b"\x00")[0].decode()
        if name:
            obj_id = mujoco.mj_name2id(model, obj_type, name)
            assert 0 <= obj_id < n_obj and id2name[obj_id] is None
            name2id[name] = obj_id
            id2name[obj_id] = name

    return tuple(id2name[id] for id in sorted(name2id.values())), name2id, id2name


class MujocoModelNames:
    """Access mjtObj object names and ids of the current MuJoCo model.

    This class supports access to the names and ids of the following mjObj types:
        mjOBJ_BODY
        mjOBJ_JOINT
        mjOBJ_GEOM
        mjOBJ_SITE
        mjOBJ_CAMERA
        mjOBJ_ACTUATOR
        mjOBJ_SENSOR

    The properties provided for each ``mjObj`` are:
        ``mjObj``_names: list of the mjObj names in the model of type mjOBJ_FOO.
        ``mjObj``_name2id: dictionary with name of the mjObj as keys and id of the mjObj as values.
        ``mjObj``_id2name: dictionary with id of the mjObj as keys and name of the mjObj as values.
    """

    def __init__(self, model: MjModel):
        """Access mjtObj object names and ids of the current MuJoCo model.

        Args:
            model: mjModel of the MuJoCo environment.
        """
        (
            self._body_names,
            self._body_name2id,
            self._body_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_BODY)
        (
            self._joint_names,
            self._joint_name2id,
            self._joint_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_JOINT)
        (
            self._geom_names,
            self._geom_name2id,
            self._geom_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_GEOM)
        (
            self._site_names,
            self._site_name2id,
            self._site_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_SITE)
        (
            self._camera_names,
            self._camera_name2id,
            self._camera_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_CAMERA)
        (
            self._actuator_names,
            self._actuator_name2id,
            self._actuator_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR)
        (
            self._sensor_names,
            self._sensor_name2id,
            self._sensor_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_SENSOR)

    @property
    def body_names(self):
        return self._body_names

    @property
    def body_name2id(self):
        return self._body_name2id

    @property
    def body_id2name(self):
        return self._body_id2name

    @property
    def joint_names(self):
        return self._joint_names

    @property
    def joint_name2id(self):
        return self._joint_name2id

    @property
    def joint_id2name(self):
        return self._joint_id2name

    @property
    def geom_names(self):
        return self._geom_names

    @property
    def geom_name2id(self):
        return self._geom_name2id

    @property
    def geom_id2name(self):
        return self._geom_id2name

    @property
    def site_names(self):
        return self._site_names

    @property
    def site_name2id(self):
        return self._site_name2id

    @property
    def site_id2name(self):
        return self._site_id2name

    @property
    def camera_names(self):
        return self._camera_names

    @property
    def camera_name2id(self):
        return self._camera_name2id

    @property
    def camera_id2name(self):
        return self._camera_id2name

    @property
    def actuator_names(self):
        return self._actuator_names

    @property
    def actuator_name2id(self):
        return self._actuator_name2id

    @property
    def actuator_id2name(self):
        return self._actuator_id2name

    @property
    def sensor_names(self):
        return self._sensor_names

    @property
    def sensor_name2id(self):
        return self._sensor_name2id

    @property
    def sensor_id2name(self):
        return self._sensor_id2name