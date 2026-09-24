# LiDAR React Frontend Startup

This project uses precomputed sequence-08 LiDAR predictions and map data. Start the backend first, then start the React/Vite frontend.

## One-click startup

From Windows Explorer or a terminal, run:

```powershell
.\start_lidar_app.cmd
```

The batch file opens two command windows:

1. Backend: `http://127.0.0.1:8000`
2. React frontend: `http://127.0.0.1:5173`

The browser opens the React frontend automatically.

## Manual commands, in order

Open PowerShell in:

```text
D:\Mainpro\Sihresources00
```

### 1. Confirm the GPU PyTorch environment

```powershell
& .\SalsaNext-Fork\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected result includes `+cu130`, `CUDA: True`, and `NVIDIA GeForce RTX 5060 Laptop GPU`.

### 2. Start the backend first

In Terminal 1:

```powershell
cd D:\Mainpro\Sihresources00
& .\SalsaNext-Fork\.venv\Scripts\python.exe .\backend\server.py
```

Wait until the backend reports that it is listening on port `8000`. It may also process the saved sequence-08 frames in a background worker during startup.

Verify it from another terminal:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health
```

### 3. Install frontend dependencies once

Only needed if `frontend\node_modules` does not exist:

```powershell
cd D:\Mainpro\Sihresources00\frontend
npm.cmd install
```

### 4. Start the React frontend

In Terminal 2:

```powershell
cd D:\Mainpro\Sihresources00\frontend
npm.cmd run dev -- --host 127.0.0.1
```

### 5. Open the application

```text
http://127.0.0.1:5173/
```

The Vite proxy forwards `/api` and `/ws` requests to the backend on port `8000`.

## Stopping the application

Close the backend and frontend command windows, or press `Ctrl+C` in each window.

## Optional: regenerate segmentation and map snapshots

This is not required for normal frontend startup. It is an expensive offline operation and should be run only when new predictions are needed:

```powershell
cd D:\Mainpro\Sihresources00\SalsaNext-Fork
& .\.venv\Scripts\python.exe .\train\tasks\semantic\infer.py `
  --dataset .\dataset_test `
  --model .\pretrained\pretrained `
  --log .\predictions\live_seq08 `
  --uncertainty true `
  --split valid `
  --pipeline `
  --pipeline-sequence 08
```

The normal launcher does not run this command; it serves the existing saved prediction/map data.

## Common issues

- `Failed to fetch /api/metadata`: start the backend first and confirm port `8000` is reachable.
- `npm` is blocked by PowerShell execution policy: use `npm.cmd`, as shown above.
- Port `8000` or `5173` is already in use: close the existing backend/frontend window before launching again.
- Do not use the root `.venv` path from older documentation; the working PyTorch environment is `SalsaNext-Fork\.venv`.
