import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from spherical_gaussian_renderer import render_depth, render_imp, network_gui
import sys
from scene import Scene
from scene.spherical_gaussian_model import SphericalGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from lpipsPyTorch import lpips
from optimizing_spa import OptimizingSpa
from optimizing_spa_sg import OptimizingSpaSG
from utils.sh_utils import SH2RGB

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,args):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    print("sg_degree: ", dataset.sg_degree)
    gaussians = SphericalGaussianModel(dataset.sg_degree)
    scene = Scene(dataset, gaussians)
    imp_score = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    gaussians.training_setup(opt) 
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    device = "cuda"
    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_imp(custom_cam, gaussians, pipe, background, scaling_modifer,is_training=True)["render"]  
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
                
        iter_start.record()

        simp_iteration1=args.simp_iteration1 
        if iteration<simp_iteration1:
            gaussians.update_learning_rate(iteration)
        else:
            gaussians.update_learning_rate(iteration-simp_iteration1+5000)
        if iteration % 1000 == 0 and iteration>simp_iteration1:
            gaussians.oneupSGdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_imp(viewpoint_cam, gaussians, pipe, bg, is_training=True) 
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        Llag = loss
        if opt.optimizing_spa == True and iteration > args.optimizing_spa_start_iter and iteration % opt.optimizing_spa_interval == 0 and iteration < args.optimizing_spa_stop_iter:
            temp = loss 
            loss = optimizingSpa.append_spa_loss(loss)
            Llag = temp - loss
            
        if opt.optimizing_spa == True and iteration > args.optimizing_spa_sg_start_iter and iteration % opt.optimizing_spa_interval == 0 and iteration < args.optimizing_spa_sg_stop_iter:
            temp = loss 
            loss = optimizingSpaSg.append_spa_loss_sg(loss)
            Llag = temp - loss
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, opt,iteration, Ll1, loss, l1_loss, Llag, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_imp, (pipe, background), args)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                area_max = render_pkg["area_max"]
                mask_blur = torch.logical_or(mask_blur, area_max>(image.shape[1]*image.shape[2]/5000))
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and iteration % 5000!=0 and gaussians._xyz.shape[0]<args.num_max:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune_split(opt.densify_grad_threshold, 
                                                    0.005, scene.cameras_extent, 
                                                    size_threshold, mask_blur)
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                
                if iteration%5000==0:
                    out_pts_list=[]
                    gt_list=[]
                    views=scene.getTrainCameras()
                    for view in views:
                        gt = view.original_image[0:3, :, :]
                        render_depth_pkg = render_depth(view, gaussians, pipe, background)  
                        out_pts = render_depth_pkg["out_pts"]
                        accum_alpha = render_depth_pkg["accum_alpha"]
                        prob=1-accum_alpha
                        prob = prob/prob.sum()
                        prob = prob.reshape(-1).cpu().numpy()
                        factor=1/(image.shape[1]*image.shape[2]*len(views)/args.num_depth)
                        N_xyz=prob.shape[0]
                        num_sampled=int(N_xyz*factor)
                        indices = np.random.choice(N_xyz, size=num_sampled, 
                                                   p=prob,replace=False)
                        out_pts = out_pts.permute(1,2,0).reshape(-1,3)
                        gt = gt.permute(1,2,0).reshape(-1,3)
                        out_pts_list.append(out_pts[indices])
                        gt_list.append(gt[indices])
                               
                    out_pts_merged=torch.cat(out_pts_list)
                    gt_merged=torch.cat(gt_list)
                    gaussians.reinitial_pts(out_pts_merged, gt_merged)
                    gaussians.training_setup(opt)
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    viewpoint_stack = scene.getTrainCameras().copy()
            
            if opt.optimizing_spa == True:
                
                if iteration == args.optimizing_spa_start_iter:
                    imp_score = update_imp_score( scene, args, gaussians, pipe, background)
                    optimizingSpa = OptimizingSpa(gaussians, opt, device,imp_score_flag=True)
                    optimizingSpa.update(imp_score, update_u=False)
                    
                if iteration == args.optimizing_spa_sg_start_iter:
                    imp_sg_score = update_sg_color_diff(gaussians)
                    optimizingSpaSg = OptimizingSpaSG(gaussians, opt, device,imp_score_flag=True)
                    optimizingSpaSg.update(imp_sg_score, update_u=False)
                
            elif iteration % opt.optimizing_spa_interval == 0 and opt.optimizing_spa == True:
                
                if iteration > args.optimizing_spa_start_iter and iteration <= args.optimizing_spa_stop_iter:
                    imp_score = update_imp_score( scene, args, gaussians, pipe, background)
                    optimizingSpa.update(imp_score)
            
                if iteration > args.optimizing_spa_sg_start_iter and iteration <= args.optimizing_spa_sg_stop_iter:
                    imp_sg_score = update_sg_color_diff(gaussians)
                    optimizingSpaSg.update(imp_sg_score)
                
            if iteration == simp_iteration1:
                imp_score = update_imp_score( scene, args, gaussians, pipe, background)

                prob = (imp_score+1)/(imp_score+1).sum()
                prob = prob.cpu().numpy()

                N_xyz = gaussians._xyz.shape[0]
                num_sampled=int(N_xyz*(1-opt.prune_ratio1))
            
                indices = np.random.choice(N_xyz, size=num_sampled, p=prob, replace=False)
                mask = np.zeros(N_xyz, dtype=bool)
                mask[indices] = True

                gaussians.prune_points(mask==False) 
                gaussians.max_sg_degree=dataset.sg_degree  
                gaussians.reinitial_pts(gaussians._xyz,SH2RGB(gaussians._rgb_base))
                gaussians.training_setup(opt)
                torch.cuda.empty_cache()
                viewpoint_stack = scene.getTrainCameras().copy()
                
            if iteration == args.optimizing_spa_stop_iter:
                imp_score = update_imp_score( scene, args, gaussians, pipe, background)
                mask = np.zeros(imp_score.shape[0], dtype=bool)
                threshold = int(opt.prune_ratio2 * imp_score.shape[0])
                imp_score_sort = torch.zeros(imp_score.shape)
                imp_score_sort, _ = torch.sort(imp_score,0)
                imp_score_threshold = imp_score_sort[threshold-1]
                mask = (imp_score <= imp_score_threshold).squeeze()
                print("\nBefore sparsifyting pruning:",gaussians.get_opacity.shape[0]) 
                gaussians.prune_points(mask)
                print("\nAfter sparsifying pruning",gaussians.get_opacity.shape[0])
                torch.cuda.empty_cache()

            if iteration == args.optimizing_spa_sg_stop_iter:
                print(f"\n[ITER {iteration}] Performing sharpness-based axis culling")
                print(f"Before culling: Total axes = {gaussians.get_sg_axis_count.sum().item()}")
                gaussians.cull_low_sharpness_axes(sharpness_threshold=args.sharpness_threshold)
                print(f"After culling: Total axes = {gaussians.get_sg_axis_count.sum().item()}")
                torch.cuda.empty_cache()
                
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    print("Final gaussians number after pruning: ", gaussians.get_xyz.shape[0])         
    return    

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, opt, iteration, Ll1, loss, l1_loss, Llag, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, args):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        if opt.optimizing_spa == True and iteration > args.optimizing_spa_start_iter and iteration <= args.optimizing_spa_stop_iter:
            tb_writer.add_scalar('train_loss_patches/lagrange_loss', Llag.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims = []
                lpipss = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs,is_training=True)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        if iteration == testing_iterations[-1]:
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips(image, gt_image, net_type='vgg'))

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                ssims_test=torch.tensor(ssims).mean()
                lpipss_test=torch.tensor(lpipss).mean()  
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssims_test, lpipss_test))
                #print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssims_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpipss_test, iteration)

        if tb_writer and iteration == testing_iterations:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            #tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def update_imp_score( scene, args, gaussians, pipe, background):
    imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
    accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()
    views = scene.getTrainCameras()
    for view in views:
        render_pkg = render_imp(view, gaussians, pipe, background,is_training=True)
        accum_weights = render_pkg["accum_weights"]
        area_proj = render_pkg["area_proj"]
        area_max = render_pkg["area_max"]
        accum_area_max = accum_area_max+area_max

        if args.imp_metric=='outdoor':
            mask_t=area_max!=0
            temp = imp_score+accum_weights/area_proj
            imp_score[mask_t] = temp[mask_t]
        else:
            imp_score = imp_score + accum_weights
    
    imp_score[accum_area_max==0]=0
    return imp_score

def update_sg_imp_score(scene, args, gaussians, pipe, background):
    imp_score = torch.zeros((gaussians._xyz.shape[0], gaussians.max_sg_degree)).cuda()
    accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()
    views = scene.getTrainCameras()

    for view in views:
        render_pkg = render_imp(view, gaussians, pipe, background, is_training=True)
        accum_weights = render_pkg["accum_weights"]  
        area_proj = render_pkg["area_proj"]  
        area_max = render_pkg["area_max"]  
        accum_area_max = accum_area_max + area_max

        campos = view.camera_center.to(gaussians._xyz.device)
        view_dirs = campos.unsqueeze(0) - gaussians._xyz  
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=1, keepdim=True) + 1e-8)

        normalized_directions = gaussians.get_sg_directions

        view_dirs_expanded = view_dirs.unsqueeze(1)  
        cos_similarities = torch.sum(normalized_directions * view_dirs_expanded, dim=2)  

        axis_mask = torch.arange(gaussians.max_sg_degree, device=gaussians._xyz.device).unsqueeze(0) < gaussians.get_sg_axis_count.unsqueeze(1)  

        positive_mask = (cos_similarities > 0) & axis_mask  
        masked_cos = cos_similarities.masked_fill(~positive_mask, float('-inf'))  

        weights = torch.softmax(masked_cos, dim=1)  
        weights = weights * positive_mask.float()  

        importance_scores = accum_weights.clone()
        if args.imp_metric == 'outdoor':
            valid_area_mask = area_max != 0
            importance_scores[valid_area_mask] = importance_scores[valid_area_mask] / area_proj[valid_area_mask]

        imp_score += (importance_scores.unsqueeze(1) * weights)

    no_render_mask = accum_area_max == 0
    imp_score[no_render_mask, :] = 0

    return imp_score

def update_sg_color_diff(gaussians):
    sg_color = gaussians.get_sg_rgb  
    sg_sharpness = gaussians.get_sg_sharpness  

    sg_color_diff = torch.abs(sg_color) * (1 - torch.exp(-2 * sg_sharpness)) 
    sg_color_diff = torch.mean(torch.abs(sg_color_diff), dim=2, keepdim=True)

    return sg_color_diff
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--simp_iteration1", type=int, default = 15_000)
    parser.add_argument("--num_depth", type=int, default = 3_500_000)
    parser.add_argument("--num_max", type=int, default = 4_500_000)
    parser.add_argument("--imp_metric", required=True, type=str, default = None)
    parser.add_argument("--sharpness_threshold", type=float, default=1)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    
    print("Training complete")
