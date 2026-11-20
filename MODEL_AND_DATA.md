# Models, data and local secrets

The Git repository contains the application source, DeepSeek control agent,
desktop console, tests, configuration and small robot assets.

The following large/licensed items are intentionally not committed:

| Local path | Approx. size | Reason |
| --- | ---: | --- |
| `G1/models/WHAM` | 7.8 GB | Downloaded model checkpoints |
| `G1/models/SMPL` | 3.0 GB | SMPL-X license-controlled models |
| `G1/G1_LLM_Robot/bin` | local venv | Platform-specific Python environment |
| `G1/G1_LLM_Robot/lib` | local venv | Platform-specific Python environment |
| `api_key.txt` | local secret | DeepSeek credential |
| `tavily_api_key.txt` | local secret | Tavily credential |

## WHAM

Use the WHAM checkout instructions and place checkpoints under:

```text
G1/models/WHAM/checkpoints/
```

Expected examples include:

```text
hmr2a.ckpt
vitpose-h-multi-coco.pth
wham_vit_w_3dpw.pth.tar
yolov8x.pt
```

## SMPL-X

Download SMPL-X from the official licensed distribution:

```text
https://smpl-x.is.tue.mpg.de/
```

Place the extracted model tree under:

```text
G1/models/SMPL/models_smplx_v1_1/
```

Do not redistribute licensed SMPL-X weights.

## DeepSeek and Tavily keys

Create these files locally and never commit them:

```text
G1/G1_LLM_Robot/G1_LLM_Robot/api_key.txt
G1/G1_LLM_Robot/G1_LLM_Robot/tavily_api_key.txt
```

Alternatively set:

```text
DEEPSEEK_API_KEY
TAVILY_API_KEY
```

## Python environment

Create a local environment from the project requirements:

```powershell
cd G1\G1_LLM_Robot
py -3 -m venv .
.\Scripts\python.exe -m pip install -r G1_LLM_Robot\requirements-video.txt
.\Scripts\python.exe -m pip install -r G1_LLM_Robot\requirements-fullbody.txt
```

## Data directories

The Windows Junctions under `G1_LLM_Robot/G1_LLM_Robot` are local runtime
links and are not stored by Git. Recreate them with:

```powershell
cd G1
powershell -ExecutionPolicy Bypass -File .\setup_data_root.ps1
```
