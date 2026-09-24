# Semantic Segmentation Architecture Analysis

## 1. Executive Summary

The repository contains two related but distinct paths:

1. **Offline/live-model inference:** `SalsaNext-Fork/train/tasks/semantic/infer.py` loads the model configuration and checkpoint, constructs `User`, and calls `User.infer()`. `User.infer_subset()` loads each `.bin` scan through the KITTI parser, creates a 5-channel spherical range image, runs SalsaNext, projects the result back to the original points, optionally runs KNN refinement, maps the model classes back to SemanticKITTI IDs, then maps them to the project’s four classes and writes `.label` files. In uncertainty mode it also writes a per-point variance file.
2. **Current React/backend playback:** `src/fovmap/replay_pipeline.py` reads already-produced `.label` and uncertainty files, builds grid cells, runs dynamics/fusion and rating, and writes `.npz` snapshots. `backend/server.py` and the React frontend serve/read those snapshots. This path does not instantiate PyTorch or run SalsaNext.

The exact model implementation is the repository’s modified SalsaNext in `train/tasks/semantic/modules/SalsaNext.py`. It is a 5-channel, 2-D encoder/decoder with residual context blocks, dilated residual blocks, average pooling, PixelShuffle upsampling, skip connections, and a per-pixel classification head. The standard model has 20 output channels when used with the shipped SemanticKITTI configuration. The uncertainty variant is `SalsaNextUncertainty` in `SalsaNextAdf.py` and propagates mean/variance pairs through ADF layers.

The shipped pretrained configuration uses a 64 x 2048 range image, batch size 1, `max_points=150000`, and `post.KNN.use: false`. Therefore the shipped standard inference configuration uses direct projected-pixel lookup rather than KNN. `salsanext.yml` contains a separate default with KNN enabled and `knn=5`, but `infer.py` does not load that file; it loads `arch_cfg.yaml` and `data_cfg.yaml` from the selected model directory.

The available timing report measures the replay path, not the neural network. It reports prediction-file loading at 1.86 ms mean, preprocessing/ring binning at 2.87 ms mean, grid construction at 33.42 ms, fusion at 426.33 ms, rating at 397.54 ms, and snapshot serialization at 1053.06 ms. No persisted CNN or KNN timing samples were found. The inference source prints per-frame CNN and KNN timings, and synchronizes CUDA around those measurements, but the resulting console output is not stored in a report in this repository.

The evidence supports a **mixed / currently uncertain** classification for end-to-end segmentation: the CNN is the likely GPU-heavy portion when a CUDA device is available, while projection, tensor construction, host-device transfers, and post-processing are CPU/PyTorch-side overheads. The repository does not contain GPU utilization traces or a segmentation-only component profile, so a stronger claim would be speculative.

## 2. Entry Point

### Model inference entry point

| Item | Actual implementation |
|---|---|
| CLI entry file | `SalsaNext-Fork/train/tasks/semantic/infer.py` |
| CLI entry | `if __name__ == '__main__'` at `infer.py:45` |
| Model/config loading | `infer.py:130-145` loads `<model>/arch_cfg.yaml` and `<model>/data_cfg.yaml` |
| Runtime object | `User(ARCH, DATA, dataset, log, model, split, uncertainty, monte_carlo)` at `infer.py:189` |
| Model constructor/checkpoint | `User.__init__` at `modules/user.py:97-110` |
| Frame loop | `User.infer_subset(...)` at `modules/user.py:153-292` |
| Projection/input construction | `dataset/kitti/parser.py:155-247` and `common/laserscan.py:67-194` |
| Post-processing | median filter at `modules/user.py:293-324`; optional KNN at `postproc/KNN.py:36-142` |
| Label conversion | `Parser.to_original()` at `dataset/kitti/parser.py:413-415`, then `User.map_to_4_classes()` at `modules/user.py:327-345` |
| Outputs | `.label` predictions at `modules/user.py:287-290`; uncertainty files at `modules/user.py:225-236` in uncertainty mode |

### Important runtime distinction

The React application does not call `infer.py`. Its backend points at `SalsaNext-Fork/predictions/uncertainty_valid/maps/sequences/08`, and `replay_pipeline.py` points at prediction files under `SalsaNext-Fork/predictions/uncertainty_valid/sequences/08/predictions`. Consequently, the frontend’s runtime semantic input is the saved four-class per-point prediction, not a live model output.

## 3. Complete Call Graph

### Inference and optional map pipeline

```text
infer.py main
├── parse CLI arguments
├── load <model>/arch_cfg.yaml
├── load <model>/data_cfg.yaml
├── create output directory tree
├── User(...)
│   ├── dynamically load tasks.semantic.dataset.kitti.parser
│   ├── Parser(..., batch_size=1, workers=0, gt=False)
│   │   ├── SemanticKitti(... train dataset)
│   │   ├── SemanticKitti(... valid dataset)
│   │   └── SemanticKitti(... test dataset, if configured)
│   ├── choose device: cuda if available, otherwise cpu
│   ├── SalsaNext(nclasses) OR SalsaNextUncertainty(nclasses)
│   ├── load_checkpoint(modeldir/SalsaNext)
│   ├── load_model_weights(...)
│   ├── optional DataParallel when multiple CUDA devices exist
│   └── optional KNN(params, nclasses) when ARCH[post][KNN][use] is true
└── User.infer()
    └── User.infer_subset(loader, parser.to_original, cnn_times, knn_times)
        └── DataLoader -> SemanticKitti.__getitem__(index)
            ├── LaserScan/ SemLaserScan(..., project=True)
            ├── open_scan()
            │   └── np.fromfile(.bin, float32) -> (-1, 4)
            ├── set_points()
            ├── do_range_projection()
            │   ├── range/depth calculation
            │   ├── yaw/pitch calculation
            │   ├── spherical pixel quantization
            │   ├── far-to-near ordering
            │   └── projected range/XYZ/remission/index arrays
            ├── create padded point/range/remission tensors
            ├── create projection coordinate tensors
            ├── concatenate 5 input channels
            ├── normalize by configured means/stds
            └── return batch tuple
        ├── trim point-index/range tensors to npoints
        ├── optional CPU -> CUDA transfers
        ├── standard branch:
        │   ├── self.model(proj_in)
        │   │   └── SalsaNext.forward()
        │   │       ├── 3 ResContextBlock calls
        │   │       ├── 5 encoder ResBlock calls
        │   │       ├── 4 decoder UpBlock calls
        │   │       ├── 1x1 logits convolution
        │   │       └── softmax(dim=1)
        │   ├── argmax over class dimension
        │   ├── 3x3 median_filter_label_image()
        │   │   ├── float conversion
        │   │   ├── reflect padding
        │   │   ├── F.unfold()
        │   │   └── median()
        │   ├── optional CUDA synchronize
        │   ├── optional KNN(proj_range, unproj_range, labels, p_x, p_y)
        │   │   ├── F.unfold() range neighborhood
        │   │   ├── range-distance weighting
        │   │   ├── top-k search
        │   │   ├── F.unfold() label neighborhood
        │   │   ├── gather/vote/cutoff
        │   │   └── per-point output
        │   ├── optional CUDA synchronize
        │   ├── GPU -> CPU prediction copy
        │   ├── Parser.to_original()
        │   ├── User.map_to_4_classes()
        │   └── write prediction `.label`
        └── uncertainty branch:
            ├── self.model(proj_in) -> raw mean/raw variance
            ├── adf.Softmax() -> predictive mean/variance
            ├── argmax predictive mean
            ├── median_filter_label_image()
            ├── optional KNN or direct projection lookup
            ├── synchronize and time
            ├── GPU -> CPU labels
            ├── to_original() -> map_to_4_classes()
            ├── sample point variance at p_y,p_x
            └── write `.label` and uncertainty file

if --pipeline:
└── infer.py.run_map_pipeline()
    └── replay_prediction_sequence()
        └── for each frame:
            ├── load_semantickitti_scan(.bin)
            ├── load saved prediction(.label)
            ├── load saved uncertainty or use zero uncertainty
            ├── confidence = clip(1/(1+max(uncertainty,0)), .05, 1.0)
            ├── assign_rings()
            ├── build_cell_schema()
            ├── _relative_velodyne_pose()
            ├── process_frame() [rebin, dynamic detection, fusion]
            ├── rate_cells()
            └── np.savez_compressed()
```

### Current playback call graph

```text
backend/server.py
├── preload MAP_ROOT/*.npz
├── get_snapshot(frame_id)
│   └── np.load(..., allow_pickle=False)
├── get_frame_json()
│   └── expose z/semantic/rating/dynamic arrays
└── get_frame_binary()
    └── pack typed arrays for the React client

React FrameProvider
└── fetch saved snapshot -> typed arrays -> canvas rendering
```

## 4. All Associated Files

The following files actually exist and are associated with the segmentation implementation or its direct runtime handoff. Training-only, evaluation-only, visualization-only, and runtime-critical status are separated below.

### A. Core model/inference

| Relative path | Functions/classes | Why involved | Runtime-critical? |
|---|---|---|---|
| `SalsaNext-Fork/train/tasks/semantic/infer.py` | CLI main, `run_map_pipeline`, `str2bool` | Loads configs, creates `User`, starts inference, optionally starts map replay | Yes for model inference |
| `SalsaNext-Fork/train/tasks/semantic/modules/user.py` | `load_checkpoint`, `load_model_weights`, `User`, `infer`, `infer_subset`, `median_filter_label_image`, `map_to_4_classes` | Owns model creation, checkpoint loading, frame loop, post-processing, and output writing | Yes |
| `SalsaNext-Fork/train/tasks/semantic/modules/SalsaNext.py` | `ResContextBlock`, `ResBlock`, `UpBlock`, `SalsaNext` | Standard SalsaNext architecture and forward path | Yes for standard model |
| `SalsaNext-Fork/train/tasks/semantic/modules/SalsaNextAdf.py` | ADF `ResContextBlock`, `ResBlock`, `UpBlock`, `SalsaNextUncertainty` | Mean/variance uncertainty model | Yes only in uncertainty mode |
| `SalsaNext-Fork/train/tasks/semantic/modules/adf.py` | ADF layers including `Softmax`, `Dropout`, `Conv2d`, `BatchNorm2d`, pooling and activations | Implements uncertainty propagation used by `SalsaNextUncertainty` | Yes only in uncertainty mode |

### B/C. Configuration and checkpoint loading

| Relative path | Functions/classes | Why involved | Category / runtime status |
|---|---|---|---|
| `SalsaNext-Fork/pretrained/pretrained/arch_cfg.yaml` | YAML data | Shipped runtime architecture, sensor, normalization, point limit and KNN flag | B/N; runtime-critical when this model directory is selected |
| `SalsaNext-Fork/pretrained/pretrained/data_cfg.yaml` | YAML data | Labels, SemanticKITTI learning map, inverse map, splits | B/N; runtime-critical |
| `SalsaNext-Fork/pretrained/pretrained/SalsaNext` | PyTorch checkpoint data | Loaded by `load_checkpoint()` and applied by `load_model_weights()` | C; runtime-critical |
| `SalsaNext-Fork/salsanext.yml` | YAML data | Repository training/default post-processing configuration; KNN enabled here | B/N; not loaded by `infer.py` for the shipped pretrained run |
| `SalsaNext-Fork/salsanext_cuda09.yml` | YAML data | Alternate environment/training configuration | N; not shown on the inference call path |
| `SalsaNext-Fork/salsanext_cuda10.yml` | YAML data | Alternate environment/training configuration | N; not shown on the inference call path |

### D/E/J. Input, projection and dataset handling

| Relative path | Functions/classes | Why involved | Runtime-critical? |
|---|---|---|---|
| `SalsaNext-Fork/train/tasks/semantic/dataset/kitti/parser.py` | `SemanticKitti`, `Parser`, `my_collate`, `SemanticKitti.map`, `to_original`, `to_xentropy`, `to_color` | Discovers scans, loads each frame, creates tensors, normalizes five channels, and creates DataLoaders | Yes |
| `SalsaNext-Fork/train/common/laserscan.py` | `LaserScan`, `SemLaserScan`, `open_scan`, `set_points`, `do_range_projection`, label projection helpers | Parses `[x,y,z,intensity]` and performs spherical projection | Yes |
| `src/fovmap/data/loader.py` | `load_point_cloud`, `load_poses`, `load_calib_tr`, `compute_T_rel`, `SemanticKITTILoader` | Alternate project data/pose loader and direct raw scan reader; it is not called by SalsaNext `infer.py` | Direct handoff/support, not model-runtime-critical |
| `src/fovmap/data/remap.py` | `_REMAP_LUT`, `LEARNING_MAP_INV`, `map_to_4_classes` | Alternate vectorized 4-class remapping used by the FOVMAP data layer; not called by `User.infer_subset` | Not on the traced SalsaNext runtime path |
| `SalsaNext-Fork/train/tasks/semantic/config/labels/semantic-kitti.yaml` | label maps and split data | Canonical label configuration source used to generate/copy model `data_cfg.yaml` | Runtime-relevant when selected as data config; shipped inference reads model-local copy |
| `SalsaNext-Fork/train/tasks/semantic/config/labels/semantic-kitti-all.yaml` | label maps | Alternate label configuration | Not on the traced shipped path |

### F/G/H/I. Post-processing, decoding, remapping

| Relative path | Functions/classes | Why involved | Runtime-critical? |
|---|---|---|---|
| `SalsaNext-Fork/train/tasks/semantic/postproc/KNN.py` | `get_gaussian_kernel`, `KNN.forward` | Optional range-image KNN refinement and majority vote back to points | Conditional; shipped pretrained config disables it |
| `SalsaNext-Fork/train/tasks/semantic/modules/user.py` | `median_filter_label_image`, `map_to_4_classes` | Always-used median filtering in both branches and final four-class grouping | Yes |
| `SalsaNext-Fork/train/tasks/semantic/modules/adf.py` | `Softmax` | Converts ADF logits/uncertain features into predictive mean and variance | Conditional |
| `SalsaNext-Fork/train/tasks/semantic/config/labels/four_class_colors.yaml` | four-class color map | Visualization of saved four-class labels | No for inference; visualization-only |

### K/L/M. Output, evaluation and downstream handoff

| Relative path | Functions/classes | Why involved | Runtime-critical? |
|---|---|---|---|
| `src/fovmap/replay_pipeline.py` | `replay_prediction_sequence`, scan/pose/calibration helpers | Consumes saved segmentation outputs and uncertainty, creates map snapshots | Yes for `--pipeline` and current frontend data generation; not for neural inference itself |
| `src/fovmap/grid_engine.py` | `load_semantickitti_scan`, `build_cell_schema`, `assign_rings`, `remap_labels` | Converts per-point semantic outputs into four-class grid cells | Yes downstream |
| `src/fovmap/dynamics_fusion.py` | `detect_dynamic_cells`, `process_frame`, fusion helpers | Consumes cell semantic labels/confidences and tracks temporal changes | Yes downstream |
| `src/fovmap/rating.py` | `rate_cells`, `estimate_local_ground` | Consumes semantic labels/confidence for traversability | Yes downstream |
| `backend/server.py` | `get_snapshot`, `get_frame_json`, `get_frame_binary` | Serves saved semantic labels, ratings and dynamic masks to the frontend | Yes for playback, not inference |
| `SalsaNext-Fork/train/tasks/semantic/evaluate_iou.py` | `map_to_four_classes`, `eval` | Reads saved predictions and ground truth to compute four-class metrics | Evaluation-only |
| `SalsaNext-Fork/train/tasks/semantic/modules/ioueval.py` | `iouEval` | Confusion-matrix/IoU implementation used by evaluation | Evaluation-only |
| `SalsaNext-Fork/train/tasks/semantic/visualize.py` | visualization CLI | Displays saved labels using `SemLaserScan` | Visualization-only |
| `SalsaNext-Fork/train/common/visualization.py` | visualization helpers | Supporting SalsaNext visualization code | Visualization-only unless that CLI is used |

### Training-only or auxiliary model files

| Relative path | Why it exists | Category / runtime status |
|---|---|---|
| `SalsaNext-Fork/train/tasks/semantic/train.py` | Training entry point that constructs standard or uncertainty SalsaNext | Training-only |
| `SalsaNext-Fork/train/tasks/semantic/modules/trainer.py` | Training/validation loop, losses, checkpoint save/load | Training/evaluation support; not called by inference |
| `SalsaNext-Fork/train/tasks/semantic/modules/Lovasz_Softmax.py` | Training loss | Training-only |
| `SalsaNext-Fork/train/tasks/semantic/modules/ioueval.py` | IoU evaluator | Evaluation/training validation support |
| `SalsaNext-Fork/train/tasks/semantic/dataset/kitti/__init__.py` | Package marker | Import support |
| `SalsaNext-Fork/train/tasks/semantic/postproc/__init__.py` | Package marker | Import support |
| `SalsaNext-Fork/train/tasks/semantic/modules/__init__.py` | Package marker | Import support |
| `SalsaNext-Fork/train/tasks/semantic/__init__.py` | Package/import support | Import support |

The repository also contains generated prediction artifacts under `SalsaNext-Fork/predictions/uncertainty_valid/sequences/08/{predictions,uncertainty}` and map snapshots under `.../maps/sequences/08`. These are outputs, not source implementation files, but they are the actual input to the current replay/frontend path.

## 5. Raw LiDAR Input

The model-side raw input is a SemanticKITTI/KITTI `.bin` file. `LaserScan.open_scan()` calls `np.fromfile(filename, dtype=np.float32)` and reshapes the flat stream to `(-1, 4)`. The four fields are:

```text
[x, y, z, remission/intensity]
```

The project’s `src/fovmap/data/loader.py` documents the same representation. In the included sequence-08 test data, the first scans are approximately 124k points per frame; exact `N` varies by scan. The model loader supports up to `max_points=150000` and pads shorter per-point arrays to this fixed capacity.

Ring/channel information is not read as a separate field from the `.bin` file. The 64 vertical channels are implicit in the spherical projection and configured HDL64 field of view (`fov_up=3`, `fov_down=-25`).

At inference time `gt=False`, so the loader uses `LaserScan`, not `SemLaserScan`; ground-truth `.label` files are not required for the model input.

## 6. Preprocessing and Projection

### Per-frame transformation sequence

1. `LaserScan.reset()` allocates projection buffers:
   - `proj_range`: `(H,W)` float32 initialized to `-1`.
   - `proj_xyz`: `(H,W,3)` float32 initialized to `-1`.
   - `proj_remission`: `(H,W)` float32 initialized to `-1`.
   - `proj_idx`: `(H,W)` int32 initialized to `-1`.
   - `proj_mask`: `(H,W)` int32 initialized to zero.
   - per-point arrays are initially empty and later replaced.
2. `open_scan()` reads the `(N,4)` float32 scan and takes views for XYZ and remission.
3. `set_points()` stores points/remissions and optionally applies data augmentation. In inference, `transform=False`, so augmentation is disabled.
4. `do_range_projection()` computes Euclidean depth with `np.linalg.norm(points, 2, axis=1)`, yaw with `-arctan2(y,x)`, and pitch with `arcsin(z/depth)`.
5. Yaw/pitch are scaled to pixel coordinates, floored, clamped and converted to int32. `proj_x` and `proj_y` are copied in original point order.
6. Points are sorted in decreasing depth so nearer points overwrite farther points in the same range-image pixel.
7. Range, XYZ, remission and original point index are scattered into the `(H,W)` projection arrays. `proj_mask` is derived from `proj_idx > 0`; the code therefore treats index 0 as invalid in this mask.
8. The parser creates fixed-capacity unprojected tensors and clones the projected arrays.
9. Five channels are concatenated in this order:

```text
range, x, y, z, remission
```

10. The five channels are normalized using the configured `img_means` and `img_stds`, then multiplied by `proj_mask.float()`.
11. The DataLoader adds a batch dimension. With the shipped config, the model input is `(1, 5, 64, 2048)` float32 on CPU before the explicit device transfer.

### Shapes and copies

| Value | Shape before batch | Type | Notes |
|---|---:|---|---|
| Raw scan | `(N,4)` | NumPy float32 | `np.fromfile` allocation; XYZ/remission are views |
| Point XYZ | `(N,3)` | NumPy float32 | Stored by `LaserScan`; source scan view unless transformed |
| `unproj_range` | `(N,)` NumPy then `(max_points,)` tensor | float32 | `np.copy(depth)` followed by padded tensor allocation |
| Projection range | `(64,2048)` | NumPy/tensor float32 | Reset allocation, then tensor clone |
| Projection XYZ | `(64,2048,3)` | NumPy/tensor float32 | Reset allocation, then tensor clone and permutation |
| Projection remission | `(64,2048)` | NumPy/tensor float32 | Reset allocation, then tensor clone |
| Projection mask | `(64,2048)` | int32 tensor | No explicit `.clone()` in parser at creation |
| Point pixel X/Y | `(max_points,)` | int64 tensor | Padded with `-1`, then first `N` entries used |
| Model input | `(1,5,64,2048)` | float32 torch | Normalized and masked; CPU-to-device transfer occurs if CUDA |

The parser builds a number of full-size padded arrays even when `N < max_points`. The model itself only uses the fixed range-image resolution; the padded arrays are primarily for alignment with the DataLoader tuple and optional KNN/direct point projection.

## 7. Model Architecture and Loading

### Exact model

The standard path constructs `SalsaNext(self.parser.get_n_classes())`. With the shipped `learning_map_inv`, `nclasses=20` because the inverse map has keys 0 through 19. The model is therefore a 20-channel classifier at each range-image pixel.

`SalsaNext` contains:

- Three `ResContextBlock`s, mapping 5 input channels to 32 features.
- Five encoder `ResBlock`s, with channel progression 32 -> 64 -> 128 -> 256 -> 256.
- Four `UpBlock`s with PixelShuffle-based upsampling and skip connections.
- A final `Conv2d(32, nclasses, kernel_size=1)`.
- `F.softmax(logits, dim=1)` in `SalsaNext.forward()`.

This is the SalsaNext-style modified model in the repository, not a PointNet++ or RangeFormer implementation.

### Uncertainty model

When `--uncertainty` is true, `User.__init__` constructs `SalsaNextUncertainty`. Its ADF layers carry a `(mean, variance)` pair, starting with zero input variance plus `2e-7`. The final ADF convolution returns raw mean and variance. `user.py` then applies `adf.Softmax`, producing predictive mean and variance.

### Checkpoint/config behavior

`infer.py` reads the selected model directory’s `arch_cfg.yaml` and `data_cfg.yaml`; it does not read the repository-root `salsanext.yml`. The shipped model directory is:

```text
SalsaNext-Fork/pretrained/pretrained/
├── arch_cfg.yaml
├── data_cfg.yaml
└── SalsaNext
```

The checkpoint exists and is approximately 108 MB. `load_checkpoint()` loads it onto CPU, and `load_model_weights()` requires a strict state-dict match, with a fallback for `module.` keys from DataParallel checkpoints. The model is constructed and loaded once per `User` object, then moved to the selected device. It is not recreated per frame.

Device selection is `cuda` if `torch.cuda.is_available()` else `cpu`. `DataParallel` is used only when CUDA is available and more than one CUDA device is visible.

## 8. Model Inference

The standard branch executes one forward pass per frame, with batch size one. `User.infer_subset()` wraps the loop in `torch.inference_mode()`. The standard model is set to evaluation mode before the loop. No `torch.autocast`, half precision, BF16, TensorRT, ONNX Runtime, or `torch.compile` path is present.

The model receives `(1,5,64,2048)` float32 and returns `(1,20,64,2048)` float32 probabilities because the final softmax is inside `SalsaNext.forward()`. `argmax(dim=1)[0]` produces `(64,2048)` integer labels.

The standard path explicitly transfers `proj_in`, `p_x`, and `p_y` to CUDA. If KNN is enabled, it also transfers `proj_range` and `unproj_range`. The result is transferred back with `unproj_argmax.cpu().numpy()` before label conversion and file writing.

The code calls `torch.cuda.synchronize()` after the model and median filter, and again after the KNN/direct projection stage. These synchronizations make the printed timings include completed CUDA work, but they serialize the host with the device at each boundary.

The uncertainty branch performs one ADF forward pass per frame. Its mathematical operations are more expensive than the standard path because mean and variance propagate through most layers. It uses `torch.inference_mode()`, but `infer_subset()` does not call `self.model.eval()` when `self.uncertainty` is true. ADF dropout therefore depends on the model’s current training flag; the ADF `Dropout` implementation uses a stochastic mask only when `self.training` is true. The source does not invoke the configured `mc` count in the frame loop, so the CLI Monte Carlo value is not observed to create repeated forward passes in this path.

## 9. Post-Processing

After network output:

1. Standard SalsaNext returns probabilities and `argmax` selects one learning-class ID per projection pixel.
2. Both standard and uncertainty branches apply a 3x3 median filter. This is implemented with `F.pad` and `F.unfold`, followed by `windows.median(dim=1)`. It operates over the full 64 x 2048 image.
3. If KNN is disabled, `proj_argmax[p_y, p_x]` projects one label back to every original point.
4. If KNN is enabled, `KNN.forward()` refines the per-point labels using local range-image neighborhoods.
5. The result is copied to CPU, flattened, converted to NumPy int32, decoded from model learning IDs to canonical SemanticKITTI IDs via `to_original`, and reduced to four project classes with `map_to_4_classes`.
6. The final four-class point labels are written as raw int32 `.label` files.

Invalid or empty projection pixels are not written as original points because only the original point coordinate arrays are indexed. `proj_mask` is applied to model features, so invalid pixels contribute zeroed input channels after normalization/masking. The direct projection path assumes `p_x,p_y` are valid for the `N` original points.

## 10. KNN / Refinement

### Configuration

The shipped pretrained `arch_cfg.yaml` contains `post.KNN.use: false`. The repository-root `salsanext.yml` contains `post.KNN.use: true` with:

```text
knn = 5
search = 5
sigma = 1.0
cutoff = 1.0
```

Only the model-directory config controls `User`; therefore the shipped pretrained inference path does not instantiate KNN unless that config is changed outside this analysis.

### Algorithm when enabled

For a 64 x 2048 projected range image and `N` original points, KNN:

1. Uses `F.unfold` with a 5x5 window over `proj_range`.
2. Gathers the 25 range candidates for each original point using `py*W + px`.
3. Replaces invalid negative ranges with infinity and restores the center range from `unproj_range`.
4. Computes absolute range differences against each point’s range.
5. Multiplies distances by an inverse Gaussian spatial weight.
6. Uses `topk(knn=5, largest=False)` to choose five neighbors.
7. Unfolds `proj_argmax` with the same 5x5 window and gathers the selected labels.
8. Marks neighbors beyond the cutoff as class `nclasses`.
9. Performs a scatter-add one-hot vote and selects the winning class.

There is one vectorized neighborhood extraction and top-k operation over all original points, not a Python loop per point. The tensors are on the same device as `proj_range`; in the CUDA case this is GPU KNN implemented with PyTorch operations. There is no persistent neighbor cache. The working set scales as `O(HW*search^2 + N*search^2)` before reduction, and the top-k/vote work scales as `O(N*search^2 + N*K)`.

## 11. Semantic Label Mapping

The model’s learning space has 20 IDs (`0..19`). The shipped inverse map is:

| Model ID | Canonical SemanticKITTI ID | Name |
|---:|---:|---|
| 0 | 0 | unlabeled |
| 1 | 10 | car |
| 2 | 11 | bicycle |
| 3 | 15 | motorcycle |
| 4 | 18 | truck |
| 5 | 20 | other-vehicle |
| 6 | 30 | person |
| 7 | 31 | bicyclist |
| 8 | 32 | motorcyclist |
| 9 | 40 | road |
| 10 | 44 | parking |
| 11 | 48 | sidewalk |
| 12 | 49 | other-ground |
| 13 | 50 | building |
| 14 | 51 | fence |
| 15 | 70 | vegetation |
| 16 | 71 | trunk |
| 17 | 72 | terrain |
| 18 | 80 | pole |
| 19 | 81 | traffic-sign |

The inference-time four-class mapping is defined directly in `modules/user.py`:

| Final ID | Project name | Canonical IDs included by `User.map_to_4_classes()` |
|---:|---|---|
| 0 | Non-drivable / terrain fallback | Everything not explicitly assigned |
| 1 | Drivable | 40 road, 44 parking |
| 2 | Static obstacle | 50 building, 51 fence, 71 trunk, 80 pole, 81 traffic-sign |
| 3 | Dynamic object | 10 car, 11 bicycle, 15 motorcycle, 18 truck, 20 other-vehicle, 30 person, 31 bicyclist, 32 motorcyclist |

The remapping is performed **per point after projection back to the original point order**, after `to_original()` expands a model ID to one canonical raw ID. It does not happen during preprocessing, grid construction, fusion, or rating in the live inference path.

There are other mapping definitions in the repository and they are not identical. `src/fovmap/data/remap.py` includes lane marking, vegetation, other-structure and moving IDs in a different four-class LUT. `evaluate_iou.py` also has a broader four-class mapping. `grid_engine.py` has a `SEMANTIC_KITTI_REMAP` table, but `replay_prediction_sequence()` receives already-four-class prediction files and does not call it. These are important consistency risks, but this report does not change them.

## 12. Uncertainty and Confidence

The uncertainty path is produced by `SalsaNextUncertainty` and `adf.Softmax`:

```text
input variance = zeros_like(input) + 2e-7
ADF forward -> raw_mean, raw_variance
ADF Softmax -> predictive mean, predictive variance
point uncertainty = mean over class dimension of predictive variance at (p_y,p_x)
```

The saved uncertainty array is flattened to `(N,)`, converted to float32, and written next to the prediction `.label` file. Its mathematical cost is the additional variance propagation through ADF layers and the final class-variance reduction. No entropy or explicit class-confidence array is generated in `User.infer_subset()`.

The downstream replay converts variance to confidence as:

```text
confidence = clip(1 / (1 + max(uncertainty, 0)), 0.05, 1.0)
```

The confidence is then averaged into grid cells, used by dynamic detection thresholds, and consumed by rating. The frontend’s binary frame payload does not include per-point uncertainty; the backend serves cell semantic labels, z, ratings, and dynamic masks. The saved `.npz` does retain `point_uncertainty`.

If uncertainty files are absent, `replay_prediction_sequence()` substitutes zero uncertainty and confidence one for every point. That fallback is explicit in the source.

## 13. Data Flow

```text
SemanticKITTI .bin [N,4] float32
    |
    | np.fromfile + reshape
    v
LaserScan points/remission [N,3] + [N]
    |
    | spherical projection: depth, yaw, pitch, floor/clamp, depth sort
    v
Range image buffers [64,2048]
    |
    | concatenate and normalize
    v
Model tensor [1,5,64,2048] float32
    |
    | optional CPU -> CUDA, one forward pass
    v
Model output [1,20,64,2048]
    |
    | argmax + 3x3 median filter
    v
Projected labels [64,2048]
    |
    | direct p_y,p_x lookup or optional vectorized KNN
    v
Per-point learning labels [N]
    |
    | to_original + map_to_4_classes
    v
Per-point four-class labels [N] int32
    |
    | replay_prediction_sequence
    v
Grid cells: semantic_label uint8 + confidence float32
    |
    +--> rating -> float32 traversability
    +--> dynamics/fusion -> dynamic mask + fused cells
    +--> compressed .npz snapshots
             |
             v
      backend JSON/binary payload
             |
             v
          React visualization
```

## 14. Complexity Analysis

Let `N` be point count, `H=64`, `W=2048`, `P=max_points=150000`, `C=20`, `S=search=5`, and `K=knn=5`.

| Component | Approximate complexity | Notes |
|---|---|---|
| Scan loading | `O(N)` | Flat binary read and reshape |
| Projection trig/range calculation | `O(N)` | Norm, atan2, arcsin, coordinate quantization |
| Projection depth ordering | `O(N log N)` | `np.argsort(depth)` dominates point projection asymptotically |
| Projection scatter | `O(N)` | Four projected arrays and index writes |
| Tensor padding/concatenation/normalization | `O(P + HW)` | Several fixed-capacity and range-image allocations |
| Standard CNN | `O(sum convolution work)` | Approximately proportional to `C_in*C_out*kernel_area*feature_area` over encoder/decoder; exact FLOPs are not calculated in source |
| Standard argmax | `O(C*HW)` | One output pixel per class |
| Median filter | `O(9*HW)` | `F.unfold` materializes windows and median-reduces them |
| Direct point projection | `O(N)` | Indexing by `p_y,p_x` |
| KNN refinement | `O(HW*S^2 + N*S^2 + N*S^2 + N*K)` | Two unfolds, range distances, top-k, gather and vote |
| Learning-ID inverse map | `O(N)` | Lookup-table indexing in `SemanticKitti.map` |
| Four-class mapping in `User` | `O(N)` | Eight/three/five boolean comparisons and assignment passes |
| Grid cell construction | `O(N log N)` | `np.unique` over quantized cell keys; several bincount/min/max operations |
| Rating | roughly `O(M*R^2 log M)` | `M` cells and neighborhood radius `R=3`; each offset uses searchsorted on sorted keys |

## 15. Performance Bottlenecks

The classifications below distinguish evidence from inference.

| Status | File/function | Operation | Complexity/frequency | Why it may be expensive |
|---|---|---|---|---|
| CONFIRMED | `laserscan.py:134-194`, `LaserScan.do_range_projection` | Per-frame trigonometry, depth sort, and projected-buffer writes | `O(N log N)`, once per frame | Full point cloud is processed in Python/NumPy and sorted every frame |
| CONFIRMED | `parser.py:205-238`, `SemanticKitti.__getitem__` | Fixed `max_points` padding, clone, concatenate and normalize | `O(P+HW)`, once per frame | Allocates/copies several arrays even when actual `N` is lower than `P` |
| CONFIRMED | `user.py:175-181`, `infer_subset` | Host-to-device transfers | once per frame; multiple tensors | `proj_in`, `p_x`, `p_y`, and optional KNN ranges are transferred separately; no nonblocking transfer is used |
| CONFIRMED | `SalsaNext.forward` | 2-D encoder/decoder convolutions at 64x2048 input resolution | once per frame | This is the largest GPU computation in the standard branch by architecture |
| CONFIRMED | `user.py:293-324` | Full-resolution 3x3 median filter using `F.unfold` | `O(9HW)`, once per frame | Materializes sliding windows for all pixels after the neural network |
| CONFIRMED | `user.py:250-275` | Explicit CUDA synchronizations | twice per frame in standard path | Forces host/device synchronization and removes overlap at timing boundaries |
| INFERRED | `KNN.py:78-140` | Two 5x5 unfolds plus top-k and vote | conditional, once per frame if enabled | The working set scales with `N*S^2`; it can be substantial for ~124k-point scans, though shipped pretrained config disables it |
| CONFIRMED | `replay_pipeline.py:126-149` | Grid/fusion/rating/snapshot processing | once per frame in replay | Existing report measures these as much slower than prediction loading; this is downstream, not neural segmentation |
| CONFIRMED | `user.py:287-290` | File output of labels | once per frame | Synchronous per-frame write; no timing breakdown is persisted |
| INFERRED | `user.py:216-236` | Uncertainty branch variance extraction and CPU copy | once per frame in uncertainty mode | ADF variance propagation increases tensor arithmetic and memory traffic |

Existing measurements from `LATENCY_MIGRATION_REPORT.md` are not segmentation measurements. They show, for the replay path, mean scan loading 22.11 ms, prediction loading 1.86 ms, uncertainty loading 1.67 ms, preprocessing/ring binning 2.87 ms, grid construction 33.42 ms, fusion 426.33 ms, rating 397.54 ms, serialization 1053.06 ms, and raw pipeline total 1940.80 ms. The CNN forward and KNN timings are not present in that report.

## 16. GPU/CPU Analysis

### Confirmed behavior

- Device selection is dynamic: CUDA if available, CPU otherwise.
- The model is moved to the selected device once.
- Standard inference uses `torch.inference_mode()` and `model.eval()`.
- The input projection is built on CPU by NumPy/PyTorch parser code.
- The model input and selected auxiliary tensors are copied to CUDA per frame.
- Predictions are copied back to CPU before NumPy remapping and disk output.
- CUDA synchronization is explicitly called around the measured network and post-processing sections.
- No mixed precision or GPU-native projection path is present.
- Batch size is fixed at one; DataLoader workers are forced to zero in `User.__init__`.

### Assessment

The segmentation path is best classified as **mixed, with likely GPU-bound model arithmetic surrounded by CPU/memory-transfer overhead** when CUDA is available. It cannot be classified as definitively GPU-bound because no GPU utilization trace, kernel timeline, or segmentation-only wall-clock breakdown is stored. On CPU-only execution, the model and all post-processing are CPU-bound.

Potential under-utilization mechanisms visible in the code include small batch size, per-frame transfers, no pinned-memory/nonblocking transfer path, CPU projection and normalization, full-resolution post-processing, and synchronization boundaries. These are observations, not changes or optimization recommendations in this report.

## 17. Memory and Allocation Analysis

Per frame, the major memory objects are:

- Raw scan `(N,4)` float32, approximately 16N bytes.
- Point XYZ/remission views and stored arrays.
- Projection buffers: range, XYZ, remission, index, mask; the XYZ image alone is `64*2048*3*4` bytes.
- Padded unprojected tensors for up to 150,000 points: XYZ, range, remission, labels if ground truth is enabled, and X/Y coordinates.
- Five-channel normalized model input `(1,5,64,2048)` float32.
- Model activations and skip tensors at several encoder/decoder resolutions.
- Standard output `(1,20,64,2048)` float32 before argmax.
- Median-filter unfold workspace for 3x3 windows.
- Optional KNN unfold workspaces for a 5x5 range and label neighborhood indexed for all points.
- Optional uncertainty mean/variance tensors, which add a second tensor stream through much of the network.
- CPU NumPy output arrays and the file-write buffer.

The model is resident on the selected device across frames. Projection and parser buffers are recreated/reset for each item. The use of `.clone()`, `np.copy`, `torch.full`, `torch.cat`, `F.unfold`, and explicit CPU/GPU conversions indicates substantial per-frame allocation/copy traffic. Exact peak memory is not measured in the repository. The shipped checkpoint is approximately 108 MB on disk; device-resident parameter memory is not reported.

## 18. Accuracy-Critical Components

| Component | Classification | Reason |
|---|---|---|
| SalsaNext forward pass | A. Accuracy-critical | Produces the learned semantic evidence for each range-image pixel |
| Input normalization and 5-channel construction | A. Accuracy-critical | The checkpoint expects the configured order, means and standard deviations |
| Spherical projection and depth ordering | A. Accuracy-critical | Determines which point appears at each model pixel and how predictions map back |
| Direct point projection via `p_y,p_x` | A. Accuracy-sensitive | Preserves one-to-one point association; invalid/index behavior affects labels |
| Median filtering | B. Accuracy-sensitive | Changes boundary and isolated-pixel predictions after inference |
| KNN refinement | B. Accuracy-sensitive | Can correct range-image artifacts but can also alter object boundaries; optional |
| ADF variance/probability path | B. Accuracy-sensitive | Affects uncertainty and downstream confidence even where argmax is unchanged |
| `to_original` inverse learning map | A. Accuracy-critical | Decodes the learned class IDs into the intended SemanticKITTI IDs |
| `User.map_to_4_classes` | A. Accuracy-critical for project semantics | Defines the final taxonomy consumed by grid, fusion and rating |
| Invalid-point masking | B. Accuracy-sensitive | Controls which projection pixels are treated as actual observations |
| Four-class color YAML | C. Performance/visualization-only | Changes display colors, not labels |
| File serialization | A. Accuracy-critical for handoff | A mismatched count, dtype or order would corrupt downstream alignment |

## 19. Potential Optimization Areas

No optimization was implemented. The applicable categories, with tradeoffs, are:

| Component | Strategy category | Expected benefit type | Accuracy/architecture impact |
|---|---|---|---|
| Model input/forward | Mixed precision or autocast | Lower arithmetic and activation bandwidth on compatible CUDA hardware | Requires validation for semantic output and uncertainty numerical stability |
| Model forward | Torch compilation or inference graph capture | Reduce Python dispatch and improve repeated-shape execution | Adds compilation/warm-up behavior and compatibility constraints |
| Model forward | ONNX/TensorRT | Engine-level kernel fusion and deployment optimization | Requires a separate export/runtime path; ADF uncertainty may need special handling |
| Projection | Vectorized/JIT/GPU projection | Reduce CPU projection and sort overhead | Changes the preprocessing architecture; must preserve exact pixel collision/depth ordering |
| Transfers | Pinned memory/nonblocking transfers and fewer copies | Reduce host-device transfer overhead and enable overlap | Requires DataLoader/device pipeline changes; correctness risk around synchronization |
| Batch scheduling | Batch multiple frames | Better GPU occupancy | Adds latency, memory, ordering and temporal scheduling complexity; not naturally available in live single-scan handoff |
| Median filter | Fused or lower-overhead GPU implementation | Reduce post-network full-image workspace and dispatch | Must preserve filter semantics if accuracy is to remain unchanged |
| KNN | Fused PyTorch/CUDA, GPU-native implementation, or approximate NN | Reduce optional refinement cost | Approximate methods can change labels at boundaries; shipped config currently disables KNN |
| Label mapping | LUT-based four-class remap | Reduce repeated boolean passes | The exact mapping must remain the inference-defined mapping; repository mappings currently differ |
| Allocation | Reuse fixed-shape buffers | Reduce allocator and copy overhead | Requires lifetime/stream discipline and careful handling of variable `N` |
| Selective inference | Region/point/frame selection | Reduce model work | Directly changes coverage and potentially accuracy; no current temporal gating exists |

## 20. Incremental / Temporal Opportunities

Temporal reuse is technically plausible but not implemented in segmentation. The current inference loop treats every frame independently and has no previous prediction input. The downstream pipeline does preserve `stored_cells`, re-bins the previous fused map, compares current/prior semantic labels, and creates a dynamic mask.

Possible future reuse dependencies include stable point associations, ego-motion from `poses.txt` and calibration, range-image pixel correspondence, confidence/uncertainty thresholds, and a policy for moving objects. Static regions could theoretically reuse labels or skip inference, but this would require robust change detection and would be accuracy-sensitive near occlusions and object boundaries. KNN neighborhoods are recomputed every frame and are not cached. The existing downstream temporal fusion is not a substitute for temporal segmentation reuse because it consumes already aggregated cells rather than reusing CNN features or predictions.

## 21. Parallelization Opportunities

- Point-wise range, angle and coordinate computations are data-parallel and already use NumPy vector operations.
- Projection scattering has point-wise work but collision resolution depends on the depth ordering; preserving the current exact behavior requires ordered or equivalent reduction semantics.
- CNN layers are naturally GPU-parallel, but the current batch size is one.
- Semantic classes are computed together in the final convolution/softmax; splitting classes independently would generally be counterproductive and would alter the model graph.
- Range-image regions can be processed by convolution kernels, but the receptive-field and skip-connection dependencies prevent independent per-region inference without overlap/tiling logic.
- KNN neighborhoods are parallel over points and are already expressed using PyTorch tensor operations; they are suitable for CUDA execution when enabled.
- Frames are independent during pure model inference and could be batched or pipelined, but output ordering, memory, and live latency constrain that choice.
- NumPy/Numba can target CPU preprocessing; multiprocessing is less attractive with `workers=0` already selected to avoid Windows shared-memory pressure and because large tensors are copied through worker queues.
- TensorRT/ONNX are applicable to the standard CNN if operators and output conventions are supported; the ADF uncertainty graph is a separate compatibility problem.

## 22. Relationship With Grid, Fusion and Rating

The semantic handoff is a per-point four-class integer label array aligned with the raw scan point order. `replay_prediction_sequence()` verifies that prediction length equals scan length, computes per-point confidence from saved variance, and calls `build_cell_schema` with:

```text
scan[:, :2]      -> cell XY aggregation
scan[:, 2]       -> elevation aggregation
assign_rings()   -> foveated ring ID
labels           -> semantic_label
confidence       -> confidence
```

`build_cell_schema()` reduces multiple points to one row per unique cell. It stores `semantic_label` as uint8 and `confidence` as float32. `process_frame()` uses current/prior cell keys and semantic transitions to detect dynamic cells and updates log-odds fusion. `rate_cells()` uses semantic class, confidence, elevation and dynamic flag to produce a float32 rating in `[0,100]`.

The frontend/backend consumes the saved cell-level `semantic_label`, rating and dynamic mask. It does not consume raw per-point predicted probabilities. The `.npz` snapshots retain `point_uncertainty`, but `backend/server.py` does not expose that field in its JSON or binary frame payload. Thus the semantic information generated by inference is consumed downstream in aggregated form, while the full model probability tensor is discarded after argmax and uncertainty extraction.

## 23. Confirmed vs Inferred vs Speculative Findings

### Confirmed

- `infer.py` is the model inference CLI entry point.
- `User.infer_subset()` is the per-frame inference function.
- The shipped model is SalsaNext/SalsaNextUncertainty, not another architecture.
- Input is `[x,y,z,intensity]` float32 from `.bin` files.
- Model input is five normalized channels at 64 x 2048, batch size one.
- The checkpoint is loaded once per `User` object and the model is kept on the selected device.
- `torch.inference_mode()` is used.
- Standard `SalsaNext.forward()` includes softmax.
- Median filtering is applied after argmax.
- Shipped pretrained config disables KNN; the code supports KNN when configured.
- Inference output is mapped to four classes and written per point.
- Uncertainty mode writes per-point predictive variance; replay maps it to confidence with `1/(1+variance)` and clipping.
- React/backend playback consumes saved snapshots and does not run inference.
- No persisted CNN/KNN timing report was found.

### Inferred

- Standard CNN convolutional work is likely the dominant GPU arithmetic cost for live inference.
- Fixed-capacity tensor construction and repeated transfers likely contribute to latency around the CNN.
- Optional KNN would be a significant post-processing workload for ~124k-point scans because it materializes 5x5 neighborhoods per point.
- The uncertainty branch is computationally heavier than the standard branch because it propagates mean and variance.
- Reuse of temporal predictions would require ego-motion and change/occlusion handling.

### Speculative / not established by the repository

- Exact GPU utilization percentage.
- Exact CNN FLOPs or per-layer latency.
- Exact peak CPU/GPU memory.
- Whether a particular generated `uncertainty_valid` artifact was produced with KNN enabled; the artifact metadata does not record the KNN setting.
- Any specific speedup from mixed precision, TensorRT, batching, or temporal reuse.
- Whether the uncertainty model was intentionally left in train mode or whether that is an oversight; the source behavior is observable, intent is not.

## 24. Files That Should Be Inspected Next

For a segmentation-only performance investigation, the next source/runtime inspection targets are:

1. The exact command and model directory used to generate `predictions/uncertainty_valid`, including the selected `--uncertainty`, `--split`, and KNN config values.
2. A segmentation run log capturing `Mean CNN inference time`, `Mean KNN inference time`, per-frame `Network seq` and `KNN Infered seq` lines.
3. `torch.profiler` or Nsight traces around `self.model(proj_in)`, the median filter, transfers and optional KNN.
4. The actual PyTorch/CUDA versions and GPU model used for the checkpoint run.
5. A consistency audit comparing `User.map_to_4_classes`, `src/fovmap/data/remap.py`, `evaluate_iou.py`, and any label mapping used to create stored predictions.
6. A run-level check of prediction dtype, value range, point-count alignment and uncertainty alignment across all saved sequence-08 frames.
7. Peak RSS and CUDA allocator measurements around a representative frame, especially in uncertainty mode and with KNN enabled.

## 25. Recommended Profiling Experiments

These are measurement experiments only; none were executed or implemented as part of this analysis.

1. Run 20-50 warm-up and measured frames with the shipped pretrained config, separately recording scan load, projection, parser tensor construction, H2D transfer, forward, argmax/median, D2H transfer, remap and file write.
2. Repeat on CPU and CUDA, with CUDA events for device segments and `perf_counter` around host segments.
3. Compare standard and uncertainty modes with the same frames and record output shapes, memory peaks and timing distributions.
4. Run the same frames with KNN disabled and enabled; record KNN-only time and memory for `N`, `H`, `W`, `S=5`, and `K=5`.
5. Profile one full forward with `torch.profiler` and export a trace showing convolution, PixelShuffle, normalization, softmax and synchronization costs.
6. Measure the effect of the fixed `max_points=150000` padding by logging actual `N` and allocation sizes; do not infer this cost from model timing alone.
7. Measure mapping consistency and downstream sensitivity by comparing the actual inference mapping with the alternate LUTs, but keep this as an accuracy experiment separate from performance results.
8. For temporal reuse feasibility, log consecutive-frame point/range-image overlap after applying the existing relative pose and quantify disagreement on static versus dynamic regions.

No source code or dependency was modified for this report.
