# G1 Project archive

This archive contains the complete working project tree except:

- DeepSeek and Tavily API key files
- Python virtual environments
- `__pycache__`, `.pytest_cache` and temporary files
- Git metadata

The uploaded GitHub source repository contains the maintainable source and
small assets. Large WHAM/SMPL-X model weights are documented in
`MODEL_AND_DATA.md` and remain local because of file-size and license limits.

## Extract on Windows

```powershell
tar -xzf G1_Project_Data_20261121.tar.gz
```

Then:

```powershell
cd G1_Project
powershell -ExecutionPolicy Bypass -File .\G1\setup_data_root.ps1
```

Create API key files locally according to `MODEL_AND_DATA.md`.
