# AudioCAN: Enhanced Few-Shot Audio Classification via Energy-Guided Temporal Cross Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This repository contains the official PyTorch implementation of **AudioCAN**, evaluated using the MetaAudio benchmark framework.

## Repository Structure

The repository is organized with the primary execution and logic scripts in the root directory:

* **`BaseLooperProto.py`**: The main entry point. Orchestrates sequential experiments over multiple models and seeds.
* **`proto_params.yaml`**: The primary configuration file governing hyperparameters, hardware allocation, and dataset paths.
* **`ProtoMain.py`**: Sets up datasets, N-shot task samplers, and orchestrates the training/evaluation pipelines.
* **`fit_proto.py`**: Contains the meta-learning training and validation loops.
* **`proto_steps.py`**: Implements the AudioCAN forward steps.
* **`dataset_/`**: Directory containing dataset setup classes and dataloaders.

## Installation

Ensure you have Python 3.8+ installed. It is recommended to use an isolated virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/zdsy/AudioCAN-MetaAudio.git](https://github.com/zdsy/AudioCAN-MetaAudio.git)
    cd AudioCAN-MetaAudio
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

All experimental settings are localized within `proto_params.yaml`. Before running any experiments, you must configure your dataset paths and desired hyperparameters.

### Dataset Configuration
Update the `data` section in `proto_params.yaml` to point to your local dataset directories:

```yaml
data:
  variable: True
  name: 'VoxCeleb_5s' 
  norm: 'global'
  type: 'variable_spec' 
  fixed: True

  # Update these paths to your local data directories
  fixed_path: 'dataset_/splits/VoxCeleb_norm_split.npy'
  data_path: '../../Datasets/VoxCeleb1_Mirror/features'
