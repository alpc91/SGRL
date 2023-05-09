import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from utils import *
import os
import math


class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml):
        self.xml = xml
        self.target = np.zeros(2)
        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        torso_quat = self.sim.data.qpos[3:7]
        torso_rotmat = quat2mat(torso_quat)
        heading = np.arctan2(torso_rotmat[1,0], torso_rotmat[0,0])
        pitch = np.arctan2(-torso_rotmat[2,0], np.sqrt(torso_rotmat[2,1]**2 + torso_rotmat[2,2]**2))
        roll = np.arctan2(torso_rotmat[2,1], torso_rotmat[2,2])
        heading  = [np.cos(heading), np.sin(heading)]
        pos_before = self.data.get_body_xpos("torso")[:2].copy()
        dist_before = np.linalg.norm(self.target-pos_before)
        self.do_simulation(a, self.frame_skip)
        pos_after = self.data.get_body_xpos("torso")[:2].copy()
        dist_after = np.linalg.norm(self.target-pos_after)
        torso_height = self.sim.data.qpos[2]
        alive_bonus = 1.0
        reward = (dist_before - dist_after) / self.dt
        reward += np.dot(pos_after -  pos_before, heading) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        # print(torso_height,torso_ang)
        done = not (torso_height > 1.0-0.165375 and torso_height < 2.0-0.165375 and pitch > -1.0 and pitch < 1.0 and roll > -1.0 and roll < 1.0)
        ob = self._get_obs()

        if dist_after < 1.0 and np.linalg.norm(self.target) > 1:
            rad = self.np_random.uniform(low=-np.pi, high=np.pi)
            len = self.np_random.uniform(low=3, high=5)
            self.target = pos_after+np.array([np.cos(rad),np.sin(rad)])*len
        return ob, reward, done, {"dist":dist_after}
    
    def _get_obs(self):
        torso_x_pos = self.data.get_body_xpos("torso")
        dir = self.target-torso_x_pos[:2]
        dir = dir / np.linalg.norm(dir)

        def _get_obs_per_limb(b):
            if b == "torso":
                limb_type_vec = np.array((1, 0, 0, 0))
            elif "thigh" in b:
                limb_type_vec = np.array((0, 1, 0, 0))
            elif "shin" in b:
                limb_type_vec = np.array((0, 0, 1, 0))
            elif "foot" in b:
                limb_type_vec = np.array((0, 0, 0, 1))
            else:
                limb_type_vec = np.array((0, 0, 0, 0))

            xpos = self.data.get_body_xpos(b) - torso_x_pos
            torso_quat = self.data.get_body_xquat(b)
            torso_rotmat = quat2mat(torso_quat)
            heading = np.arctan2(torso_rotmat[1,0], torso_rotmat[0,0])
            pitch = np.arctan2(-torso_rotmat[2,0], np.sqrt(torso_rotmat[2,1]**2 + torso_rotmat[2,2]**2))
            roll = np.arctan2(torso_rotmat[2,1], torso_rotmat[2,2])
            heading  = [np.cos(heading), np.sin(heading)]
            pitch  = [np.cos(pitch), np.sin(pitch)]
            roll  = [np.cos(roll), np.sin(roll)]
            
            # include current joint angle and joint range as input
            if b == "torso":
                angle0_x = 0
                angle0_y = 0
                angle0_z = 0
                
                angle_x = 0.5
                angle_y = 0.5
                angle_z = 0.5
                joint_range_x = [0.5, 0.5]
                joint_range_y = [0.5, 0.5]
                joint_range_z = [0.5, 0.5]
                axis_x = [0,0,0]
                axis_y = [0,0,0]
                axis_z = [0,0,0]
            else:
                def get_angle(jnt_adr):
                    qpos_adr = self.sim.model.jnt_qposadr[
                        jnt_adr
                    ]  
                    angle0 = self.data.qpos[qpos_adr]
                    angle = np.degrees(
                        angle0
                    )  # angle of current joint, scalar
                    joint_range = np.degrees(
                        self.sim.model.jnt_range[jnt_adr]
                    )  # range of current joint, (2,)
                    # normalize
                    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
                    joint_range[0] = (180.0 + joint_range[0]) / 360.0
                    joint_range[1] = (180.0 + joint_range[1]) / 360.0
                    return  angle0,angle,joint_range

                # print(b)
                body_id = self.sim.model.body_name2id(b)
                jnt_adr = self.sim.model.body_jntadr[body_id]
                angle0_x,angle_x,joint_range_x = get_angle(jnt_adr)
                angle0_y,angle_y,joint_range_y = get_angle(jnt_adr+1)
                angle0_z,angle_z,joint_range_z = get_angle(jnt_adr+2)
                axis_x = self.data.get_joint_xaxis(b+'_joint_x')
                axis_y = self.data.get_joint_xaxis(b+'_joint_y')
                axis_z = self.data.get_joint_xaxis(b+'_joint_z')
            
            obs = np.zeros(37 + len(limb_type_vec))
            obs[0:3] = xpos
            obs[3] = 0.0
            obs[4] = 0.0
            obs[5] = -9.81
            obs[6:8] = dir
            obs[9:12] = np.clip(self.data.get_body_xvelp(b), -10, 10)
            obs[12:15] = self.data.get_body_xvelr(b)
            # obs[15:18] = [heading[0],heading[1],0]
            # obs[18:21] = [pitch[1],0,pitch[0]]
            # obs[21:24] = [0,roll[1],roll[0]]
            obs[15:18] = axis_x
            obs[18:21] = axis_y
            obs[21:24] = axis_z
            obs[24] = angle0_x
            obs[25] = angle0_y
            obs[26] = angle0_z
            obs[27] = angle_x
            obs[27+1:27+3] = joint_range_x
            obs[30] = angle_y
            obs[30+1:30+3] = joint_range_y
            obs[33] = angle_z
            obs[33+1:33+3] = joint_range_z
            obs[36: 36 + len(limb_type_vec)] = limb_type_vec
            obs[36 + len(limb_type_vec)] = self.data.get_body_xpos(b)[2]

            return obs

        full_obs = np.concatenate(
            [_get_obs_per_limb(b) for b in self.model.body_names[1:]]
        )

        return full_obs.ravel()

    def reset_model(self):
        qpos = self.init_qpos
        rad = self.np_random.uniform(low=-np.pi, high=np.pi)
        rad = rad/2 
        qpos[3] = np.cos(rad)
        qpos[6] = np.sin(rad)
        self.set_state(
            qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        rad = self.np_random.uniform(low=-np.pi, high=np.pi)
        len = self.np_random.uniform(low=3, high=5)
        self.target = np.array([np.cos(rad),np.sin(rad)])*len
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
