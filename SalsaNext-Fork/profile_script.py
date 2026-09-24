import torch
import time
import os
import sys
import numpy as np

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.tasks.semantic.modules.user import User
import yaml

def monkey_patched_infer_subset(self, loader, to_orig_fn, cnn, knn):
    # switch to evaluate mode
    if not self.uncertainty:
        self.model.eval()
    
    if self.gpu:
        torch.cuda.empty_cache()
    
    h2d_times = []
    fwd_times = []
    med_times = []
    d2h_times = []
    remap_times = []
    total_times = []
    
    with torch.inference_mode():
        for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
            if i >= 100: # profile 100 frames
                break
                
            p_x = p_x[0, :npoints]
            p_y = p_y[0, :npoints]
            proj_range = proj_range[0, :npoints]
            unproj_range = unproj_range[0, :npoints]
            path_seq = path_seq[0]
            path_name = path_name[0]
            
            torch.cuda.synchronize()
            t0 = time.time()
            
            # H2D
            if self.gpu:
                proj_in = proj_in.to(self.device)
                p_x = p_x.to(self.device)
                p_y = p_y.to(self.device)
            
            torch.cuda.synchronize()
            t1 = time.time()
            
            # Forward
            proj_output = self.model(proj_in)
            proj_argmax = proj_output.argmax(dim=1)[0]
            
            torch.cuda.synchronize()
            t2 = time.time()
            
            # Median
            proj_argmax = self.median_filter_label_image(proj_argmax)
            unproj_argmax = proj_argmax[p_y, p_x]
            
            torch.cuda.synchronize()
            t3 = time.time()
            
            # D2H
            pred_np = unproj_argmax.cpu().numpy()
            
            torch.cuda.synchronize()
            t4 = time.time()
            
            # Remap
            pred_np = pred_np.reshape((-1)).astype(np.int32)
            pred_np = to_orig_fn(pred_np)
            pred_np = self.map_to_4_classes(pred_np)
            
            torch.cuda.synchronize()
            t5 = time.time()
            
            if i >= 10: # Skip 10 warmup frames
                h2d_times.append(t1 - t0)
                fwd_times.append(t2 - t1)
                med_times.append(t3 - t2)
                d2h_times.append(t4 - t3)
                remap_times.append(t5 - t4)
                total_times.append(t5 - t0)
                
    print("\n--- Profiling Results (Standard Branch) ---")
    print(f"H2D:     {np.mean(h2d_times)*1000:.2f} ms")
    print(f"Forward: {np.mean(fwd_times)*1000:.2f} ms")
    print(f"Median:  {np.mean(med_times)*1000:.2f} ms")
    print(f"D2H:     {np.mean(d2h_times)*1000:.2f} ms")
    print(f"Remap:   {np.mean(remap_times)*1000:.2f} ms")
    print(f"Total:   {np.mean(total_times)*1000:.2f} ms")
    print("-------------------------------------------\n")

def run_profiling():
    # Model and data directories
    modeldir = "pretrained/pretrained"
    datadir = "dataset_test"
    logdir = "predictions/test_profiling"
    
    ARCH = yaml.safe_load(open(modeldir + "/arch_cfg.yaml", 'r'))
    DATA = yaml.safe_load(open(modeldir + "/data_cfg.yaml", 'r'))
    
    # Patch the method
    User.infer_subset = monkey_patched_infer_subset
    
    # Instantiate and infer
    user = User(ARCH, DATA, datadir, logdir, modeldir, split='valid', uncertainty=False, mc=30)
    user.infer()

if __name__ == '__main__':
    run_profiling()
