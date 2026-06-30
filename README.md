# Reinforcement Learning Algorithms in GridWorld

This repository contains lab session materials (code) for the SNU Reinforcement Learning course, 2026 Spring.

---
## Materials
Download slides from here: [Google Drive Link](https://drive.google.com/drive/folders/1_5_LwYWTPU9xn7svZ7Xw8FTMxDwIJhhB?usp=sharing)

## Installation

```bash
# clone the repository
git clone https://github.com/SNU-IntelligentMotionLab/2026Spring_RL_Lab.git
```

### Windows

Open PowerShell in the repository folder and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
```

### Manual setup

```bash
# create and activate a virtual environment
python3 -m venv lab_env # or
python -m venv lab_env
# Windows
.\lab_env\Scripts\activate
# Linux/Mac
source lab_env/bin/activate

# install required packages
pip install -r requirements.txt
```
---

For each lab session, do `cd Lab[num]_[theme]` (e.g., `cd Lab1_DP`) and refer to README for instructions.
