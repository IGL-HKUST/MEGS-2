import torch
import torch.nn as nn
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.sh_utils import RGB2SH, SH2RGB
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


class SphericalGaussianModel:
    
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, max_sg_degree=3, variable_sg_bands=True):
        self.active_sg_degree = 0  
        self.max_sg_degree = max_sg_degree
        self.variable_sg_bands = variable_sg_bands
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._rgb_base = torch.empty(0) 

        if variable_sg_bands:
            self._sg_directions = [torch.empty(0)] * (self.max_sg_degree + 1)
            self._sg_sharpness = [torch.empty(0)] * (self.max_sg_degree + 1)
            self._sg_rgb = [torch.empty(0)] * (self.max_sg_degree + 1)
        else:
            self._sg_directions = torch.empty(0)  
            self._sg_sharpness = torch.empty(0)   
            self._sg_rgb = torch.empty(0)         

        self._sg_axis_count = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    @property
    def per_band_count(self):
        result = list()
        if self.variable_sg_bands:
            for tensor_list in [self._sg_directions, self._sg_sharpness, self._sg_rgb]:
                if isinstance(tensor_list, list):
                    for tensor in tensor_list:
                        if tensor.numel() > 0:
                            result.append(tensor.shape[0])
                        else:
                            result.append(0)
                    break  # Only need to check one tensor list since they should have same structure
        return result

    def capture(self):
        captured_data = {
            "active_sg_degree": self.active_sg_degree,
            "variable_sg_bands": self.variable_sg_bands,
            "xyz": self._xyz,
            "rgb_base": self._rgb_base,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "optimizer": self.optimizer.state_dict(),
            "spatial_lr_scale": self.spatial_lr_scale
        }

        if self.max_sg_degree > 0:
            captured_data["sg_directions"] = self._sg_directions
            captured_data["sg_sharpness"] = self._sg_sharpness
            captured_data["sg_rgb"] = self._sg_rgb
            captured_data["sg_axis_count"] = self._sg_axis_count

        return captured_data

    def restore(self, model_args, training_args):
        self.active_sg_degree = model_args["active_sg_degree"]
        if "variable_sg_bands" in model_args:
            self.variable_sg_bands = model_args["variable_sg_bands"]
        self._xyz = model_args["xyz"]
        self._rgb_base = model_args["rgb_base"]
        self._scaling = model_args["scaling"]
        self._rotation = model_args["rotation"]
        self._opacity = model_args["opacity"]
        self.max_radii2D = model_args["max_radii2D"]
        self.xyz_gradient_accum = model_args["xyz_gradient_accum"]
        self.denom = model_args["denom"]
        self.spatial_lr_scale = model_args["spatial_lr_scale"]

        if self.max_sg_degree > 0 and "sg_directions" in model_args:
            self._sg_directions = model_args["sg_directions"]
            self._sg_sharpness = model_args["sg_sharpness"]
            self._sg_rgb = model_args["sg_rgb"]
            if "sg_axis_count" in model_args:
                self._sg_axis_count = model_args["sg_axis_count"]
            else:
                self._sg_axis_count = torch.full((self._xyz.shape[0],), self.max_sg_degree, device="cuda", dtype=torch.int)

        self.training_setup(training_args)
        self.optimizer.load_state_dict(model_args["optimizer"])
        
    @property
    def num_primitives(self):
        return self._xyz.shape[0]
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_rgb_base(self):
        return self._rgb_base
    
    @property
    def get_sg_directions(self):
        if self.variable_sg_bands and isinstance(self._sg_directions, list):
            return [torch.nn.functional.normalize(tensor, dim=2) if tensor.numel() > 0 else tensor for tensor in self._sg_directions]
        else:
            return torch.nn.functional.normalize(self._sg_directions, dim=2)

    @property
    def get_sg_sharpness(self):
        if self.variable_sg_bands and isinstance(self._sg_sharpness, list):
            return [torch.abs(tensor) if tensor.numel() > 0 else tensor for tensor in self._sg_sharpness]
        else:
            return torch.abs(self._sg_sharpness)

    @property
    def get_sg_rgb(self):
        return self._sg_rgb

    @property
    def get_sg_axis_count(self):
        return self._sg_axis_count

    def get_active_sg_degree(self):
        return self.active_sg_degree

    def get_covariance(self, scaling_modifier):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSGdegree(self):
        if self.active_sg_degree < self.max_sg_degree:
            self.active_sg_degree += 1

    def cull_low_sharpness_axes(self, sharpness_threshold=0.5):

        if self.max_sg_degree ==0 or self.num_primitives == 0:
            return

        with torch.no_grad():
            device = self._xyz.device
            _, max_axes, _ = self._sg_directions.shape
            
            arange_d = torch.arange(max_axes, device=device)
            active_axes_mask = arange_d[None, :] < self._sg_axis_count[:, None]  
            
            current_sharpness = self.get_sg_sharpness.squeeze(-1)  
            is_low_sharpness = current_sharpness < sharpness_threshold  
            
            prune_mask = active_axes_mask & is_low_sharpness  
            keep_mask = active_axes_mask & ~is_low_sharpness  
            
            rgb_from_pruned = self._sg_rgb * prune_mask.unsqueeze(-1)
            sharpness_from_pruned = self.get_sg_sharpness * prune_mask.unsqueeze(-1)
            
            eps = 1e-8
            safe_sharpness = torch.clamp(sharpness_from_pruned, min=eps)
            rgb_energy = rgb_from_pruned*((1 - torch.exp(-2*safe_sharpness))/(2*safe_sharpness))
            rgb_to_add = torch.sum(rgb_energy, dim=1)  
            self._rgb_base += rgb_to_add
            
            new_axis_count = keep_mask.sum(dim=1)  

            kept_directions = self._sg_directions[keep_mask]
            kept_sharpness = self._sg_sharpness[keep_mask]
            kept_rgb = self._sg_rgb[keep_mask]

            new_sg_directions = torch.zeros_like(self._sg_directions)
            new_sg_sharpness = torch.zeros_like(self._sg_sharpness)
            new_sg_rgb = torch.zeros_like(self._sg_rgb)

            unpack_mask = arange_d[None, :] < new_axis_count[:, None]  

            new_sg_directions[unpack_mask] = kept_directions
            new_sg_sharpness[unpack_mask] = kept_sharpness
            new_sg_rgb[unpack_mask] = kept_rgb

            self._sg_directions.data = new_sg_directions
            self._sg_sharpness.data = new_sg_sharpness
            self._sg_rgb.data = new_sg_rgb
            self._sg_axis_count = new_axis_count

    def compute_colors_precomp(self, viewpoint_camera, is_training=False):
        colors = self.get_rgb_base.clone()

        if self.max_sg_degree > 0 and self.active_sg_degree > 0:
            campos = viewpoint_camera.camera_center.to(self._xyz.device)
            active_bases = self.get_active_sg_degree()

            if is_training or not self.variable_sg_bands:
                view_dirs = campos.unsqueeze(0) - self._xyz  
                view_dirs = view_dirs / (torch.norm(view_dirs, dim=1, keepdim=True) + 1e-8)

                device = self._xyz.device
                arange_d = torch.arange(self.max_sg_degree, device=device)
                valid_axis_mask = (arange_d[None, :] < self._sg_axis_count[:, None]) & (arange_d[None, :] < active_bases) 

                if not valid_axis_mask.any():
                    color_rgb = SH2RGB(colors)
                    return torch.clamp(color_rgb, 0.0, 1.0)

                valid_directions_raw = self._sg_directions[valid_axis_mask]
                valid_sharpness_raw = self._sg_sharpness[valid_axis_mask]
                valid_rgb = self._sg_rgb[valid_axis_mask]
                gaussian_indices_for_valid_axes = torch.where(valid_axis_mask)[0]
                valid_view_dirs = view_dirs[gaussian_indices_for_valid_axes]

                cos_theta = torch.sum(torch.nn.functional.normalize(valid_directions_raw, dim=1) * valid_view_dirs, dim=1, keepdim=True)
                directional_scale = torch.exp(torch.abs(valid_sharpness_raw) * (cos_theta - 1.0))
                weighted_rgb = valid_rgb * directional_scale

                colors.scatter_add_(0, gaussian_indices_for_valid_axes.unsqueeze(1).expand(-1, 3), weighted_rgb)

            else:
                view_dirs = campos.unsqueeze(0) - self._xyz  
                view_dirs = view_dirs / (torch.norm(view_dirs, dim=1, keepdim=True) + 1e-8)

                degree_indices_dict = {}
                current = 0
                for sg_degree in range(self.max_sg_degree + 1):
                    num = (self._sg_axis_count == sg_degree).sum().item()
                    if num > 0:
                        degree_indices_dict[sg_degree] = (current, current + num)
                        current += num

                for sg_degree in range(1, min(active_bases + 1, len(self._sg_directions))):
                    if sg_degree not in degree_indices_dict:
                        continue
                    index_start, index_end = degree_indices_dict[sg_degree]

                    degree_view_dirs = view_dirs[index_start:index_end]
                    for axis_idx in range(sg_degree):
                        if axis_idx < self._sg_directions[sg_degree].shape[1]:
                            valid_directions_raw = self._sg_directions[sg_degree][:, axis_idx, :]
                            valid_sharpness_raw = self._sg_sharpness[sg_degree][:, axis_idx, :]
                            valid_rgb = self._sg_rgb[sg_degree][:, axis_idx, :]

                            cos_theta = torch.sum(torch.nn.functional.normalize(valid_directions_raw, dim=1) * degree_view_dirs, dim=1, keepdim=True)
                            directional_scale = torch.exp(torch.abs(valid_sharpness_raw) * (cos_theta - 1.0))
                            weighted_rgb = valid_rgb * directional_scale

                            colors[index_start:index_end].add_(weighted_rgb)

        color_rgb = SH2RGB(colors)

        if is_training:
            return torch.clamp(color_rgb, 0.0, 1.0)
        else:
            torch.clamp(color_rgb, 0.0, 1.0, out=color_rgb)
            return color_rgb
        
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))       
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._rgb_base = nn.Parameter(fused_color.requires_grad_(True))  
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self._initialize_spherical_gaussians_unified(fused_point_cloud.shape[0])

        self._sg_axis_count = torch.full((fused_point_cloud.shape[0],), self.max_sg_degree, device="cuda", dtype=torch.int)

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _initialize_spherical_gaussians_unified(self, num_points):
        if self.max_sg_degree > 0:
            directions = torch.randn((num_points, self.max_sg_degree, 3), device="cuda")
            directions = directions / (torch.norm(directions, dim=2, keepdim=True) + 1e-8)
            self._sg_directions = nn.Parameter(directions.requires_grad_(True))

            sharpness = torch.ones((num_points, self.max_sg_degree, 1), device="cuda") * 0.1
            self._sg_sharpness = nn.Parameter(sharpness.requires_grad_(True))

            rgb = torch.randn((num_points, self.max_sg_degree, 3), device="cuda") * 0.1
            self._sg_rgb = nn.Parameter(rgb.requires_grad_(True))
            # initial_rgb_for_sg = torch.randn((num_points * self.max_sg_degree, 3), device="cuda") * 0.1
            # sg_rgb_sh = RGB2SH(initial_rgb_for_sg).view(num_points, self.max_sg_degree, 3)
            # self._sg_rgb = nn.Parameter(sg_rgb_sh.requires_grad_(True))
        else:
            self._sg_directions = nn.Parameter(torch.empty((num_points, 0, 3), device="cuda").requires_grad_(True))
            self._sg_sharpness = nn.Parameter(torch.empty((num_points, 0, 1), device="cuda").requires_grad_(True))
            self._sg_rgb = nn.Parameter(torch.empty((num_points, 0, 3), device="cuda").requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._rgb_base], 'lr': training_args.feature_lr, "name": "rgb_base"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.max_sg_degree > 0:
            l.append({'params': [self._sg_directions], 'lr': training_args.feature_lr, "name": "sg_directions"})
            l.append({'params': [self._sg_sharpness], 'lr': training_args.feature_lr * 4.0, "name": "sg_sharpness"})
            l.append({'params': [self._sg_rgb], 'lr': training_args.feature_lr, "name": "sg_rgb"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._rgb_base = optimizable_tensors["rgb_base"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]

        if self.max_sg_degree > 0:
            self._sg_directions = optimizable_tensors["sg_directions"]
            self._sg_sharpness = optimizable_tensors["sg_sharpness"]
            self._sg_rgb = optimizable_tensors["sg_rgb"]

        self._sg_axis_count = self._sg_axis_count[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_rgb_base, new_opacities, new_scaling, new_rotation, new_sg_directions=None, new_sg_sharpness=None, new_sg_rgb=None, new_sg_axis_count=None):
        d = {
            "xyz": new_xyz,
            "rgb_base": new_rgb_base,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        if self.max_sg_degree > 0:
            d["sg_directions"] = new_sg_directions
            d["sg_sharpness"] = new_sg_sharpness
            d["sg_rgb"] = new_sg_rgb

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._rgb_base = optimizable_tensors["rgb_base"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]

        if self.max_sg_degree > 0:
            self._sg_directions = optimizable_tensors["sg_directions"]
            self._sg_sharpness = optimizable_tensors["sg_sharpness"]
            self._sg_rgb = optimizable_tensors["sg_rgb"]

        if new_sg_axis_count is not None:
            self._sg_axis_count = torch.cat((self._sg_axis_count, new_sg_axis_count), dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_rgb_base = self._rgb_base[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        new_sg_directions = None
        new_sg_sharpness = None
        new_sg_rgb = None
        new_sg_axis_count = None
        if self.max_sg_degree > 0:
            new_sg_directions = self._sg_directions[selected_pts_mask].repeat(N,1,1)
            new_sg_sharpness = self._sg_sharpness[selected_pts_mask].repeat(N,1,1)
            new_sg_rgb = self._sg_rgb[selected_pts_mask].repeat(N,1,1)
            new_sg_axis_count = self._sg_axis_count[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_rgb_base, new_opacity, new_scaling, new_rotation,
                                 new_sg_directions, new_sg_sharpness, new_sg_rgb, new_sg_axis_count)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_rgb_base = self._rgb_base[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_sg_directions = None
        new_sg_sharpness = None
        new_sg_rgb = None
        new_sg_axis_count = None
        if self.max_sg_degree > 0:
            new_sg_directions = self._sg_directions[selected_pts_mask]
            new_sg_sharpness = self._sg_sharpness[selected_pts_mask]
            new_sg_rgb = self._sg_rgb[selected_pts_mask]
            new_sg_axis_count = self._sg_axis_count[selected_pts_mask]

        self.densification_postfix(new_xyz, new_rgb_base, new_opacities, new_scaling, new_rotation,
                                 new_sg_directions, new_sg_sharpness, new_sg_rgb, new_sg_axis_count)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(3):
            l.append(f'rgb_base_{i}')

        l.append('sg_axis_count')

        for base_idx in range(self.max_sg_degree):
            for i in range(3):
                l.append(f'sg_dir_{base_idx}_{i}')
            l.append(f'sg_sharp_{base_idx}')
            for i in range(3):
                l.append(f'sg_rgb_{base_idx}_{i}')
        for i in range(3):
            l.append(f'scale_{i}')
        for i in range(4):
            l.append(f'rot_{i}')
        l.append('opacity')
        return l

    def construct_list_of_attributes_for_degree(self, sg_degree):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(3):
            l.append(f'rgb_base_{i}')

        l.append('sg_axis_count')

        for base_idx in range(sg_degree):
            for i in range(3):
                l.append(f'sg_dir_{base_idx}_{i}')
            l.append(f'sg_sharp_{base_idx}')
            for i in range(3):
                l.append(f'sg_rgb_{base_idx}_{i}')

        for i in range(3):
            l.append(f'scale_{i}')
        for i in range(4):
            l.append(f'rot_{i}')
        l.append('opacity')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.get_rgb_base.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        sg_directions = self._sg_directions.detach().cpu().numpy() if self.max_sg_degree > 0 else None
        sg_sharpness = self._sg_sharpness.detach().cpu().numpy() if self.max_sg_degree > 0 else None
        sg_rgb = self._sg_rgb.detach().cpu().numpy() if self.max_sg_degree > 0 else None
        sg_axis_count = self.get_sg_axis_count.detach().cpu().numpy() if self.max_sg_degree > 0 else None

        if self.variable_sg_bands:
            elements_list = []

            for sg_degree in range(self.max_sg_degree + 1):
                degrees_mask = (sg_axis_count == sg_degree).squeeze() if sg_axis_count is not None else np.ones(xyz.shape[0], dtype=bool)

                if not degrees_mask.any():
                    continue

                xyz_degree = xyz[degrees_mask]
                normals_degree = normals[degrees_mask]
                f_dc_degree = f_dc[degrees_mask]
                opacities_degree = opacities[degrees_mask]
                scale_degree = scale[degrees_mask]
                rotation_degree = rotation[degrees_mask]
                axis_count_degree = sg_axis_count[degrees_mask] if sg_axis_count is not None else np.zeros((xyz_degree.shape[0], 1))

                attribute_list = [xyz_degree, normals_degree, f_dc_degree, axis_count_degree.reshape(-1, 1)]

                if sg_degree > 0 and sg_directions is not None:
                    sg_directions_degree = sg_directions[degrees_mask, :sg_degree, :]  
                    sg_sharpness_degree = sg_sharpness[degrees_mask, :sg_degree, :]  
                    sg_rgb_degree = sg_rgb[degrees_mask, :sg_degree, :]               

                    for axis_idx in range(sg_degree):
                        attribute_list.append(sg_directions_degree[:, axis_idx, :])  
                        attribute_list.append(sg_sharpness_degree[:, axis_idx, :])   
                        attribute_list.append(sg_rgb_degree[:, axis_idx, :])         

                attribute_list.extend([scale_degree, rotation_degree, opacities_degree])
                attributes = np.concatenate(attribute_list, axis=1)

                dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_for_degree(sg_degree)]
                elements = np.empty(xyz_degree.shape[0], dtype=dtype_full)
                for i, attribute_name in enumerate(self.construct_list_of_attributes_for_degree(sg_degree)):
                    elements[attribute_name] = attributes[:, i]

                elements_list.append(PlyElement.describe(elements, f'vertex_{sg_degree}'))

            PlyData(elements_list).write(path)
        else:
            attribute_list = [xyz, normals, f_dc, sg_axis_count.reshape(-1, 1) if sg_axis_count is not None else np.zeros((xyz.shape[0], 1))]

            if self.max_sg_degree > 0 and sg_directions is not None:
                for axis_idx in range(self.max_sg_degree):
                    attribute_list.append(sg_directions[:, axis_idx, :])  
                    attribute_list.append(sg_sharpness[:, axis_idx, :])   
                    attribute_list.append(sg_rgb[:, axis_idx, :])         

            attribute_list.extend([scale, rotation, opacities])
            attributes = np.concatenate(attribute_list, axis=1)
            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            for i, attribute_name in enumerate(self.construct_list_of_attributes()):
                elements[attribute_name] = attributes[:, i]

            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz_list = []
        rgb_base_list = []
        axis_counts_list = []
        scales_list = []
        rots_list = []
        opacities_list = []
        sg_directions_list = []
        sg_sharpness_list = []
        sg_rgb_list = []

        for sg_degree in range(self.max_sg_degree + 1):
            element_name = f'vertex_{sg_degree}'
            if element_name not in [elem.name for elem in plydata.elements]:
                continue

            element = next(elem for elem in plydata.elements if elem.name == element_name)

            xyz = np.stack((np.asarray(element["x"]),
                           np.asarray(element["y"]),
                           np.asarray(element["z"])), axis=1)

            rgb_base = np.stack((np.asarray(element["rgb_base_0"]),
                               np.asarray(element["rgb_base_1"]),
                               np.asarray(element["rgb_base_2"])), axis=1)

            axis_counts = np.asarray(element["sg_axis_count"]).astype(int)

            scale_names = [f'scale_{i}' for i in range(3)]
            scales = np.stack([np.asarray(element[attr_name]) for attr_name in scale_names], axis=1)

            rot_names = [f'rot_{i}' for i in range(4)]
            rots = np.stack([np.asarray(element[attr_name]) for attr_name in rot_names], axis=1)

            opacities = np.asarray(element["opacity"])[..., None]

            xyz_list.append(xyz)
            rgb_base_list.append(rgb_base)
            axis_counts_list.append(axis_counts)
            scales_list.append(scales)
            rots_list.append(rots)
            opacities_list.append(opacities)

            if sg_degree > 0:
                num_points = xyz.shape[0]
                directions = np.zeros((num_points, sg_degree, 3))
                sharpness = np.zeros((num_points, sg_degree, 1))
                rgb = np.zeros((num_points, sg_degree, 3))

                for axis_idx in range(sg_degree):
                    directions[:, axis_idx, :] = np.stack((
                        np.asarray(element[f"sg_dir_{axis_idx}_0"]),
                        np.asarray(element[f"sg_dir_{axis_idx}_1"]),
                        np.asarray(element[f"sg_dir_{axis_idx}_2"])
                    ), axis=1)

                    sharpness[:, axis_idx, 0] = np.asarray(element[f"sg_sharp_{axis_idx}"])

                    rgb[:, axis_idx, :] = np.stack((
                        np.asarray(element[f"sg_rgb_{axis_idx}_0"]),
                        np.asarray(element[f"sg_rgb_{axis_idx}_1"]),
                        np.asarray(element[f"sg_rgb_{axis_idx}_2"])
                    ), axis=1)

                sg_directions_list.append(torch.tensor(directions, dtype=torch.float, device="cuda"))
                sg_sharpness_list.append(torch.tensor(sharpness, dtype=torch.float, device="cuda"))
                sg_rgb_list.append(torch.tensor(rgb, dtype=torch.float, device="cuda"))
            else:
                num_points = xyz.shape[0]
                sg_directions_list.append(torch.empty((num_points, 0, 3), dtype=torch.float, device="cuda"))
                sg_sharpness_list.append(torch.empty((num_points, 0, 1), dtype=torch.float, device="cuda"))
                sg_rgb_list.append(torch.empty((num_points, 0, 3), dtype=torch.float, device="cuda"))

        xyz = np.concatenate(xyz_list, axis=0)
        rgb_base = np.concatenate(rgb_base_list, axis=0)
        axis_counts = np.concatenate(axis_counts_list, axis=0)
        scales = np.concatenate(scales_list, axis=0)
        rots = np.concatenate(rots_list, axis=0)
        opacities = np.concatenate(opacities_list, axis=0)

        if not self.variable_sg_bands:
            sg_directions = torch.cat(sg_directions_list, dim=0) if sg_directions_list else torch.empty((0, self.max_sg_degree, 3), device="cuda")
            sg_sharpness = torch.cat(sg_sharpness_list, dim=0) if sg_sharpness_list else torch.empty((0, self.max_sg_degree, 1), device="cuda")
            sg_rgb = torch.cat(sg_rgb_list, dim=0) if sg_rgb_list else torch.empty((0, self.max_sg_degree, 3), device="cuda")
        else:
            sg_directions = sg_directions_list
            sg_sharpness = sg_sharpness_list
            sg_rgb = sg_rgb_list

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rgb_base = nn.Parameter(torch.tensor(rgb_base, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sg_axis_count = torch.tensor(axis_counts, dtype=torch.int, device="cuda")

        if not self.variable_sg_bands:
            self._sg_directions = nn.Parameter(sg_directions.requires_grad_(True))
            self._sg_sharpness = nn.Parameter(sg_sharpness.requires_grad_(True))
            self._sg_rgb = nn.Parameter(sg_rgb.requires_grad_(True))
        else:
            for tensor in sg_directions:
                tensor.requires_grad_(True)
            for tensor in sg_sharpness:
                tensor.requires_grad_(True)
            for tensor in sg_rgb:
                tensor.requires_grad_(True)
            self._sg_directions = sg_directions
            self._sg_sharpness = sg_sharpness
            self._sg_rgb = sg_rgb

        self.active_sg_degree = self.max_sg_degree

    def densify_and_prune_split(self, max_grad, min_opacity, extent, max_screen_size, mask):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split_mask(grads, max_grad, extent, mask)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify_and_split_mask(self, grads, grad_threshold, scene_extent, mask, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        padded_mask = torch.zeros((n_init_points), dtype=torch.bool, device='cuda')
        padded_mask[:grads.shape[0]] = mask
        selected_pts_mask = torch.logical_or(selected_pts_mask, padded_mask)
        

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_rgb_base = self._rgb_base[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        if self.max_sg_degree > 0:
            new_sg_directions = self._sg_directions[selected_pts_mask].repeat(N,1,1)
            new_sg_sharpness = self._sg_sharpness[selected_pts_mask].repeat(N,1,1)
            new_sg_rgb = self._sg_rgb[selected_pts_mask].repeat(N,1,1)
            new_sg_axis_count = self._sg_axis_count[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_rgb_base, new_opacity, new_scaling, new_rotation,
                                 new_sg_directions, new_sg_sharpness, new_sg_rgb, new_sg_axis_count)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def reinitial_pts(self, pts, rgb):

        fused_point_cloud = pts
        fused_color = RGB2SH(rgb)

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._rgb_base = nn.Parameter(fused_color.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._initialize_spherical_gaussians_unified(fused_point_cloud.shape[0])
        self._sg_axis_count = torch.full((fused_point_cloud.shape[0],), self.max_sg_degree, device="cuda", dtype=torch.int)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")