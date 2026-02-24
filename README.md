# Thermal Intruder Detection AI

## Overview
AI-based thermal intruder detection system using YOLOv8 and transfer learning on infrared datasets.

This project detects:
- Humans at night
- Hidden thermal targets

## Tech Stack
- Python
- FastAPI (Backend)
- YOLOv8
- PyTorch
- HTML Frontend

## Dataset
- FLIR-ADAS-v2

## Project Structure
- backend/ → Detection API and model management
- frontend/ → Simple UI
- tests/ → Backend test cases

## Folder Structure
```text
thermal_intruder_ai/
│
├── backend/
│ ├── app/
│ │ ├── api/
│ │ ├── core/
│ │ ├── models/
│ │ ├── utils/
│ │ └── main.py
│ ├── tests/
│ └── requirements.txt
│
├── frontend/
│ └── thermal_intruder_ui.html
│
└── .gitignore
```
## How To Run

### 1️) Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2) Run Backend
```bash
uvicorn backend.app.main:app --reload
```

### 3) Open Frontend

Then push again:

```powershell
frontend/thermal_intruder_ui.html
