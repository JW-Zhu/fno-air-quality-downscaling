# FNO Air Quality Downscaling

Physics-informed FNO for PM2.5 air quality downscaling from WRF to LES scale using domain decomposition.

## Overview

This repository contains code and data for downscaling PM2.5 concentrations from coarse-resolution Weather Research and Forecasting (WRF) model outputs to fine-resolution Large Eddy Simulation (LES) scale using a Physics-Informed Fourier Neural Operator (FNO) approach with boundary-buffered domain decomposition.

**Dataset:** Data for: Seamless Microscale Air Quality Downscaling for Megacities: A Physics-Informed FNO Approach – PM2.5 Downscaling Dataset for Shanghai (March 1, 2024)

## Repository Structure

```
fno-air-quality-downscaling/
├── code/
│   ├── FNO.py              # FNO model definition
│   ├── data.py             # Data loading and normalization functions
│   ├── train.py            # Training script
│   ├── best_model.pth      # Trained FNO model (auto-generated)
│   ├── normalization.pkl   # Normalization parameters (auto-generated)
│   └── infer.ipynb         # Inference notebook
├── result/                 # Inference results (auto-generated)
├── plot/
│   └── plot_result.ipynb   # Visualization notebook
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Jupyter

## Usage

### 1. Training

Run the training script to train the FNO model:

```bash
cd code
python train.py
```

This will:
- Load and normalize WRF and LES data
- Train the FNO model
- Save the best model to `best_model.pth`
- Save normalization parameters to `normalization.pkl`

### 2. Inference

Use the inference notebook for downscaling predictions:

```bash
jupyter notebook code/infer.ipynb
```

Results will be saved to the `result/` folder.

### 3. Visualization

Generate plots of inference results:

```bash
jupyter notebook plot/plot_result.ipynb
```

## Key Features

- **Physics-Informed FNO**: Incorporates domain knowledge into the neural operator framework
- **Boundary-Buffered Domain Decomposition**: Handles domain boundaries efficiently for seamless downscaling
- **WRF to LES Downscaling**: Bridges scale gaps from weather model to microscale simulations
- **PM2.5 Focus**: Tailored for air quality applications in urban megacities

## Data

The PM2.5 downscaling dataset for Shanghai (March 1, 2024) includes:
- WRF model outputs at coarse resolution
- LES reference data at fine resolution
- Meteorological covariates
- Domain decomposition indices

## License

This work is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
See LICENSE file for details.
