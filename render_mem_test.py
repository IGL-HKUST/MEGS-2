import torch
from scene import Scene
from scene.spherical_gaussian_model import SphericalGaussianModel
from spherical_gaussian_renderer_light import render_imp
import os
from tqdm import tqdm
from os import makedirs
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
import torchvision
import numpy as np
import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    mems = []
    times = []
    torch.cuda.reset_peak_memory_stats()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        view.load_cam_parm_to_device(torch.device("cuda"))
        torch.cuda.synchronize(); t_start = time.time()
        rendering = render_imp(view, gaussians, pipeline, background, is_training=False)["render"]
        torch.cuda.synchronize(); t_end = time.time()
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  
        mems.append(peak_memory)
        times.append(t_end - t_start)
        del view
        
    if mems != []:
        rendering_memory = np.array(mems).mean()
        
    
    if times != []:
        fps = 1. /  np.array(times[5:]).mean()
    
    return rendering_memory, fps

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):

    with torch.no_grad():
        dataset.data_device = "cpu"

        # Measure loading memory
        torch.cuda.reset_peak_memory_stats()
        gaussians = SphericalGaussianModel(dataset.sg_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        torch.cuda.synchronize()
        loading_peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        rendering_memory, fps = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

        return loading_peak_memory, rendering_memory, fps

        

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    l, m, f = render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    print(f"Loading peak memory: {l} MB")
    print(f"Rendering peak memory: {m} MB")
    print(f"fps: {f}")