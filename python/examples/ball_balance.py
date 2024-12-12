# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi

# from isaacgymenvs.utils.torch_jit_utils import to_torch, torch_rand_float, tensor_clamp, torch_random_dir_2
from .base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import *



class BallBalance(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.action_speed_scale = self.cfg["env"]["actionSpeedScale"]
        self.debug_viz = True #self.cfg["env"]["enableDebugVis"]

        sensors_per_env = 1
        actors_per_env = 2
        dofs_per_env = 21
        bodies_per_env = 1 + 1  #7+1

        # Observations:
        # 0:3 - activated DOF positions
        # 3:6 - activated DOF velocities
        # 6:9 - ball position
        # 9:12 - ball linear velocity
        # 12:15 - sensor force (same for each sensor)
        # 15:18 - sensor torque 1
        # 18:21 - sensor torque 2
        # 21:24 - sensor torque 3
        self.cfg["env"]["numObservations"] = 20

        # Actions: target velocities for the 3 actuated DOFs
        self.cfg["env"]["numActions"] = 7

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, actors_per_env, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, dofs_per_env, 2)
        vec_rigid_body_tensor = gymtorch.wrap_tensor(self.rigid_body_tensor).view(self.num_envs, self.num_body + 1, 13)
        # vec_sensor_tensor = gymtorch.wrap_tensor(self.sensor_tensor).view(self.num_envs, sensors_per_env, 6)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        self.root_states = self.root_states
        self.tray_positions = vec_rigid_body_tensor[..., self.bbot_tray_idx, 0:3]
        self.tray_orientations = vec_rigid_body_tensor[..., self.bbot_tray_idx, 3:7]
        self.ball_positions = self.root_states[..., 1, 0:3]
        self.ball_orientations = self.root_states[..., 1, 3:7]
        self.ball_linvels = self.root_states[..., 1, 7:10]
        self.ball_angvels = self.root_states[..., 1, 10:13]

        self.dof_states = vec_dof_tensor
        self.dof_positions = vec_dof_tensor[..., 0]


        self.dof_velocities = vec_dof_tensor[..., 1]

        # self.sensor_forces = vec_sensor_tensor[..., 0:3]
        # self.sensor_torques = vec_sensor_tensor[..., 3:6]

        self.initial_tray_positions = self.tray_positions.clone()
        self.initial_tray_orientations = self.tray_orientations.clone()
        self.initial_dof_states = self.dof_states.clone()
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_positions = self.dof_positions.clone()

        self.dof_position_targets = torch.zeros((self.num_envs, dofs_per_env), dtype=torch.float32, device=self.device, requires_grad=False)
        # print (self.dof_position_targets )
        self.all_actor_indices = torch.arange(actors_per_env * self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, actors_per_env)
        self.all_bbot_indices = actors_per_env * torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        # vis
        self.axes_geom = gymutil.AxesGeometry(0.5)

        self.reset_idx(torch.arange(0,self.num_envs,device=self.device))

    def create_sim(self):
            

            
            
            
            self.dt = self.sim_params.dt
            self.sim_params.up_axis = gymapi.UP_AXIS_Z
            self.sim_params.gravity.x = 0
            self.sim_params.gravity.y = 0
            self.sim_params.gravity.z = -9.81

            self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

            
            self._create_ground_plane()
            self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
            if self.randomize:
                self.apply_randomizations(self.randomization_params)



    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
     
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/upper_thormang_copy.urdf"
        # Load asset
        
        
        
        bbot_options = gymapi.AssetOptions()
        bbot_options.fix_base_link = False
        # bbot_options.slices_per_cylinder = 40
        
        bbot_options.flip_visual_attachments = False
        bbot_options.fix_base_link = True
        bbot_options.collapse_fixed_joints = False
        bbot_options.disable_gravity = True
        bbot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, bbot_options)
        

        # printed view of asset built
        # self.gym.debug_print_asset(bbot_asset)

        self.num_bbot_dofs = self.gym.get_asset_dof_count(bbot_asset)

        bbot_dof_props = self.gym.get_asset_dof_properties(bbot_asset)
        self.bbot_dof_lower_limits = []
        self.bbot_dof_upper_limits = []
        for i in range(self.num_bbot_dofs):
            self.bbot_dof_lower_limits.append(bbot_dof_props['lower'][i])
            self.bbot_dof_upper_limits.append(bbot_dof_props['upper'][i])

        self.bbot_dof_lower_limits = to_torch(self.bbot_dof_lower_limits, device=self.device)
        self.bbot_dof_upper_limits = to_torch(self.bbot_dof_upper_limits, device=self.device)

        bbot_pose = gymapi.Transform()
        bbot_pose.p.z = 1.0

        
        self.bbot_tray_idx = self.gym.find_asset_rigid_body_index(bbot_asset, "tray")
        self.num_body = self.gym.get_asset_rigid_body_count(bbot_asset)
        print(self.bbot_tray_idx)
    

        # create ball asset
        self.ball_radius = 0.05
        ball_options = gymapi.AssetOptions()
        ball_options.density = 200
        ball_asset = self.gym.create_sphere(self.sim, self.ball_radius, ball_options)

        self.envs = []
        self.bbot_handles = []
        self.obj_handles = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            
            # print(self.gym.get_asset_dof_names(bbot_asset))
            bbot_handle = self.gym.create_actor(env_ptr, bbot_asset, bbot_pose, "bbot", i, 0, 0)
            
            actuated_dofs = np.array([[11,12,13,14,15,16,17]])
            
            fixed_dof =  np.arange(21)
            fixed_dof = np.delete(fixed_dof, [11,12,13,14,15,16,17])
            # free_dofs = np.array([0, 2, 4])
            # actuated_dofs = self.gym.get_asset_actuator_count(bbot_asset)
            # print(self.gym.get_actor_rigid_body_names(env_ptr, bbot_handle))
            # free_dofs =self.gym.get_asset_dof_count(bbot_asset)
            dof_props = self.gym.get_actor_dof_properties(env_ptr, bbot_handle)
            
            dof_props['driveMode'][actuated_dofs] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][actuated_dofs] = 100.0
            dof_props['damping'][actuated_dofs] = 20.0
            # dof_props['driveMode'][free_dofs] = gymapi.DOF_MODE_NONE
            # dof_props['stiffness'][free_dofs] = 0
            # dof_props['damping'][free_dofs] = 0
            dof_props["driveMode"][fixed_dof] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][fixed_dof] = 10000.0
            dof_props["damping"][fixed_dof] = 10.0

            # dof_props["driveMode"][fixed_dof1] = gymapi.DOF_MODE_NONE
            # dof_props["stiffness"][fixed_dof1] = 0.0
            # dof_props["damping"][fixed_dof1] = 0.0

            # dof_props["driveMode"][fixed_dof2] = gymapi.DOF_MODE_NONE
            # dof_props["stiffness"][fixed_dof2] = 0.0
            # dof_props["damping"][fixed_dof2] = 0.0

            # dof_props["driveMode"][fixed_dof3] = gymapi.DOF_MODE_NONE
            # dof_props["stiffness"][fixed_dof3] = 0.0
            # dof_props["damping"][fixed_dof3] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, bbot_handle, dof_props)

            # lower_leg_handles = []
            # lower_leg_handles.append(self.gym.find_actor_rigid_body_handle(env_ptr, bbot_handle, "Ring_Proximal_to_tray_joint"))
            # lower_leg_handles.append(self.gym.find_actor_rigid_body_handle(env_ptr, bbot_handle, "lower_leg1"))
            # lower_leg_handles.append(self.gym.find_actor_rigid_body_handle(env_ptr, bbot_handle, "lower_leg2"))

            # # create attractors to hold the feet in place
            # attractor_props = gymapi.AttractorProperties()
            # attractor_props.stiffness = 5e7
            # attractor_props.damping = 5e3
            # attractor_props.axes = gymapi.AXIS_TRANSLATION
            # for j in range(3):
            #     angle = self.leg_angles[j]
            #     attractor_props.rigid_handle = lower_leg_handles[j]
            #     # attractor world pose to keep the feet in place
            #     attractor_props.target.p.x = self.leg_outer_offset * math.cos(angle)
            #     attractor_props.target.p.z = self.leg_radius
            #     attractor_props.target.p.y = self.leg_outer_offset * math.sin(angle)
            #     # attractor local pose in lower leg body
            #     attractor_props.offset.p.z = 0.5 * self.leg_length
            #     self.gym.create_rigid_body_attractor(env_ptr, attractor_props)

            ball_pose = gymapi.Transform()
            ball_pose.p.x = 0.0
            ball_pose.p.y = 0.0
            ball_pose.p.z = 0.0
            ball_handle = self.gym.create_actor(env_ptr, ball_asset, ball_pose, "ball", i, 0, 0)
            self.obj_handles.append(ball_handle)

            # # pretty colors
            # self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.99, 0.66, 0.25))
            # self.gym.set_rigid_body_color(env_ptr, bbot_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.48, 0.65, 0.8))
            # for j in range(1, 7):
            #     self.gym.set_rigid_body_color(env_ptr, bbot_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.15, 0.2, 0.3))

            self.envs.append(env_ptr)
            self.bbot_handles.append(bbot_handle)
        

    def compute_observations(self):
        #print("~!~!~!~! Computing obs")

        actuated_dof_indices = torch.tensor([11,12,13,14,15,16,17], device=self.device)
        #print(self.dof_states[:, actuated_dof_indices, :])

        self.obs_buf[..., 0:7] = self.dof_positions[..., actuated_dof_indices]
        self.obs_buf[..., 7:14] = self.dof_velocities[..., actuated_dof_indices]
        self.obs_buf[..., 14:17] = self.ball_positions
        self.obs_buf[..., 17:20] = self.ball_linvels
        # self.obs_buf[..., 20:23] = self.tray_positions[:]
        # self.obs_buf = torch.cat((self.dof_positions[..., actuated_dof_indices], self.dof_velocities[..., actuated_dof_indices]), dim=1)
        # self.obs_buf[..., 12:15] = self.sensor_forces[..., 0] / 20  # !!! lousy normalization
        # self.obs_buf[..., 15:18] = self.sensor_torques[..., 0] / 20  # !!! lousy normalization
        # self.obs_buf[..., 18:21] = self.sensor_torques[..., 1] / 20  # !!! lousy normalization
        # self.obs_buf[..., 21:24] = self.sensor_torques[..., 2] / 20  # !!! lousy normalization

        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_bbot_reward(
            self.tray_positions,
            self.tray_orientations,
            self.ball_positions,
            self.ball_linvels,
            self.ball_radius,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # num_resets = len(env_ids)

        # reset bbot and ball root states
        self.root_states[env_ids] = self.initial_root_states[env_ids]

        # min_d = 0.001  # min horizontal dist from origin
        # max_d = 0.5  # max horizontal dist from origin
        # min_height = 0.5
        # max_height = 0.5
        # min_horizontal_speed = 0
        # max_horizontal_speed = 5

        # dists = torch_rand_float(min_d, max_d, (num_resets, 1), self.device)
        # dirs = torch_random_dir_2((num_resets, 1), self.device)
        # hpos = dists * dirs

        # speedscales = (dists - min_d) / (max_d - min_d)
        # hspeeds = torch_rand_float(min_horizontal_speed, max_horizontal_speed, (num_resets, 1), self.device)
        # hvels = -speedscales * hspeeds * dirs
        # vspeeds = -torch_rand_float(5.0, 5.0, (num_resets, 1), self.device).squeeze()

        # self.ball_positions[env_ids, 0] = hpos[..., 0]
        # self.ball_positions[env_ids, 2] = torch_rand_float(min_height, max_height, (num_resets, 1), self.device).squeeze()
        # self.ball_positions[env_ids, 1] = hpos[..., 1]

        # self.ball_orientations[env_ids, 0:3] = 0
        # self.ball_orientations[env_ids, 3] = 1
        # self.ball_linvels[env_ids, 0] = hvels[..., 0]
        # self.ball_linvels[env_ids, 2] = vspeeds
        # self.ball_linvels[env_ids, 1] = hvels[..., 1]
        # self.ball_angvels[env_ids] = 0
        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_bbot_dofs), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_bbot_dofs), device=self.device)

        self.dof_positions[env_ids] = tensor_clamp(self.initial_dof_positions[env_ids] + positions, self.bbot_dof_lower_limits, self.bbot_dof_upper_limits)
        self.dof_velocities[env_ids] = velocities

        # self.root_states[env_ids] = self.initial_root_states[env_ids].clone()

        corner_to_center_vector = torch.zeros_like(self.tray_positions)
        corner_to_center_vector[:, 0] = -0.015
        corner_to_center_vector[:, 1] = -0.5/2
        corner_to_center_vector[:, 2] = -0.5/2
        

        # print(self.tray_orientations[0])
        # print(corner_to_center_vector[0])
        # print()

        rotated_offset = quat_rotate(self.initial_tray_orientations, corner_to_center_vector)
        center_pos = self.initial_tray_positions + rotated_offset

        # self.root_states[env_ids, 1, 0] = center_pos[env_ids,0]
        # self.root_states[env_ids, 1, 2] = center_pos[env_ids,2]+0.3
        # self.root_states[env_ids, 1, 1] = center_pos[env_ids,1]

        self.ball_positions[env_ids, 0] = center_pos[env_ids,0]
        self.ball_positions[env_ids, 2] = center_pos[env_ids,2]
        self.ball_positions[env_ids, 1] = center_pos[env_ids,1]

        # print(self.ball_positions[0])
        # print(center_pos[0])



        self.ball_orientations[env_ids, 0:3] = 0
        self.ball_orientations[env_ids, 3] = 1
        self.ball_linvels[env_ids, 0] = 0
        self.ball_linvels[env_ids, 2] = 0
        self.ball_linvels[env_ids, 1] = 0
        self.ball_angvels[env_ids] = 0


        # reset root state for bbots and balls in selected envs
        actor_indices = self.all_actor_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(actor_indices), len(actor_indices))

        # reset DOF states for bbots in selected envs
        bbot_indices = self.all_bbot_indices[env_ids].flatten()
        self.dof_states[env_ids] = self.initial_dof_states[env_ids]
        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(bbot_indices), len(bbot_indices))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, _actions):

        # resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = _actions.to(self.device)
        # excluded_indices = torch.LongTensor(range(26, 33))
        # mapped_indices = torch.LongTensor([i for i in range(49) if i not in excluded_indices])
        # result = mapped_indices[torch.LongTensor([])]
        
        actuated_indices = torch.LongTensor([11,12,13,14,15,16,17])

        # update position targets from actions
        # print(self.dof_position_targets[..., actuated_indices])
        # print(self.dt * self.action_speed_scale * actions)
        self.dof_position_targets[..., actuated_indices] += self.dt * self.action_speed_scale * actions
        
        # self.dof_position_targets[..., result] = 0
        self.dof_position_targets[:] = tensor_clamp(self.dof_position_targets, self.bbot_dof_lower_limits, self.bbot_dof_upper_limits)

        # reset position targets for reset envs
        self.dof_position_targets[reset_env_ids] = 0

        # self.dof_position_targets[result] = 0.0

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_position_targets))

    def post_physics_step(self):

        self.progress_buf += 1
        self.randomize_buf += 1
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()
        # print(self.tray_positions)
        # vis
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            points = []
            colors = []
            corner_to_center_vector = torch.zeros_like(self.tray_positions)
            corner_to_center_vector[:, 0] = -0.015
            corner_to_center_vector[:, 1] = -0.5/2
            corner_to_center_vector[:, 2] = -0.5/2
            
            rotated_offset = quat_rotate(self.tray_orientations, corner_to_center_vector)
            center_pos = self.tray_positions + rotated_offset
            for i in range(self.num_envs):

                env = self.envs[i]
                bbot_handle = self.bbot_handles[i]
                body_handles = []
                body_handles.append(self.gym.find_actor_rigid_body_handle(env, bbot_handle, "tray"))
                # body_handles.append(self.gym.find_actor_rigid_body_handle(env, bbot_handle, "upper_leg1"))
                # body_handles.append(self.gym.find_actor_rigid_body_handle(env, bbot_handle, "upper_leg2"))

                

                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(*center_pos[i])
                pose.r = gymapi.Quat(*self.tray_orientations[i])
                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, pose)

                corner = gymapi.Transform(p=gymapi.Vec3(*self.tray_positions[i]), r=gymapi.Quat(*self.tray_orientations[i]))
                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, corner)

                # for lhandle in body_handles:
                #     lpose = self.gym.get_rigid_transform(env, lhandle)
                #     gymutil.draw_lines(self.axes_geom, self.gym, sself.gym, env, lpose)
                


#####################################################################
###=========================jit functions=========================###
#####################################################################


#@torch.jit.script
def compute_bbot_reward(tray_positions, tray_orientations, ball_positions, ball_velocities, ball_radius, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    # calculating the norm for ball distance to desired height above the ground plane (i.e. 0.7)
    
    corner_to_center_vector = torch.zeros_like(tray_positions)
    corner_to_center_vector[:, 0] = -0.015
    corner_to_center_vector[:, 1] = -0.5/2
    corner_to_center_vector[:, 2] = -0.5/2
    
    # print(tray_orientations[0])
    # print(corner_to_center_vector[0])
    # print()
    
    rotated_offset = quat_rotate(tray_orientations, corner_to_center_vector)
    center_pos = tray_positions + rotated_offset

    ball_pos = ball_positions
    to_target = ball_pos - center_pos


    # print(ball_positions[0])
    # print(center_pos[0])

    # dist_to_target = torch.norm(to_target, p=2, dim=-1)

    dist_to_target= torch.sqrt(to_target[..., 0]  * to_target[..., 0] +
                           to_target[..., 2]  * to_target[..., 2]  +
                           to_target[..., 1]  * to_target[..., 1] )
    ball_speed = torch.sqrt(ball_velocities[..., 0] * ball_velocities[..., 0] +
                            ball_velocities[..., 1] * ball_velocities[..., 1] +
                            ball_velocities[..., 2] * ball_velocities[..., 2])
   
    pos_reward = 1.0 / (1.0 + dist_to_target)
    speed_reward = 1.0 / (1.0 + ball_speed)
    
    tray_deviation = torch.sqrt(tray_positions[..., 0] * tray_positions[..., 0] +
                                 tray_positions[..., 2] * tray_positions[..., 2] +
                                 tray_positions[..., 1] * tray_positions[..., 1])
    tray_reward = 1.0 / (1.0 + tray_deviation)

    
 
    reward = pos_reward * speed_reward * tray_reward
    
    # # 计算球与板子中心的欧氏距离
    # board_center_dist = torch.sqrt(ball_positions[..., 0] ** 2 + ball_positions[..., 1] ** 2 + ball_positions[..., 2] ** 2)

    # # 检查球是否超出了 3x3 的范围
    # out_of_range = board_center_dist > 0.3
    #print(ball_positions[..., 1])
    # reset_condition = (torch.sqrt(ball_positions[..., 1] * ball_positions[..., 1]) <= 0.3) | (torch.sqrt(ball_positions[..., 0] * ball_positions[..., 0]) <= 0.3)
    # reset = torch.where(out_of_range, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    # reset = torch.where(ball_positions[..., 2] < ball_radius * 1.5, torch.ones_like(reset_buf), reset)
    # reset = torch.where(torch.sqrt(ball_positions[..., 1] * ball_positions[..., 1]) <= 0.3, torch.ones_like(reset_buf), reset_buf)
    # reset = torch.where(torch.sqrt(ball_positions[..., 1] * ball_positions[..., 1]) <= 0.3, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where( dist_to_target > 0.2, torch.ones_like(reset_buf), reset)
    # reset = torch.where(ball_positions[..., 2] < ball_radius * 1.5, torch.ones_like(reset_buf), reset)
    # reset = torch.where(torch.abs(ball_positions[..., 0]) >  0.3  , torch.ones_like(reset_buf), reset)
    
    return reward, reset