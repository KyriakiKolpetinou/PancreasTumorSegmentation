# PancreasTumorSegmentation

Official code release for our method in the **PANTHER Challenge (MICCAI 2025, Task 1)**.  
This repository contains the model, training pipeline, and preprocessing description used to achieve **4th place**.

---

**🧠 Project Idea**

This work explores whether lightweight transformer-based models, without any pre-training, can achieve competitive performance in 3D medical image segmentation challenges.
Instead of relying on very large models, we design a compact SegFormer3D variant with carefully chosen modules (ASPP, FPN, scSE, Attention Gates).
The goal is to maintain efficiency (low parameter count, memory footprint) while reaching segmentation quality close to state-of-the-art large networks.
Tested on the PANTHER pancreas tumor segmentation challenge, our approach ranked 4th place overall, demonstrating that smaller, efficient models can remain competitive in real-world medical imaging benchmarks.

---
## 🔬 Model

- **Base architecture:** SegFormer3D (GPLv3).  
- **Modifications:**  
  - Hybrid convolutional stem  
  - ASPP + FPN decoder  
  - scSE on skip connections  
  - Attention Gates  
  - Depthwise-separable smoothing blocks  
  - Auxiliary deep supervision head  

Implementation is available in [`models/segformer3d_variant.py`](models/segformer3d_variant.py).

---

## 🧑‍💻 Training

Training pipeline implemented with **PyTorch Lightning** + **MONAI**.

- Patch-based training: `(32, 160, 208)`  
- Loss: DiceCELoss  
- Optimizer: AdamW  
- Scheduler: Warmup + polynomial decay  
- Sliding window inference for validation  

See [`training/`](training/) for full scripts.  
Key parameters are defined in [`training/config.py`](training/config.py).

---

## 🏗 Preprocessing

Preprocessing follows the **nnU-Net MRI strategy** (Isensee et al., 2021).    
- Spacing normalisation (median / anisotropy-aware).  
- Z-score normalisation on non-zero voxels.  
- Safe label resampling.  
- Output format: compressed `.npz` with `(1, D, H, W)` arrays.

---

✨ Installation:
  - "git clone https://github.com/KyriakiKolpetinou/PancreasTumorSegmentation.git"
  - "cd PancreasTumorSegmentation"
  - "pip install -r requirements.txt"

🚀 Usage:
  We followed the MRI preprocessing strategy of nnU-Net (Isensee et al., Nature Methods 2021).
  You need preprocessed data in Task07_Pancreas_Preprocessed/ before training.

---
  
## 🧪 Training

The training scripts use **fixed configuration** from `training/config.py`.

**1) Configure paths & hyper-params**
Open `training/config.py` and set:
- `DATA_DIR` → folder with your **preprocessed** dataset  
  Expected layout: DATA_DIR/
imagesTr/ case_000.npz, case_001.npz, ...
labelsTr/ case_000.npz, case_001.npz, ...

**Each `.npz` must contain `arr_0` with shapes:**
- image: `(1, D, H, W)` as float32 
- label: `(1, D, H, W)` as uint8 (0=bg, 1=tumor, 2=pancreas)
- `CHECKPOINT_DIR`, `LOG_NAME`, and the rest of the hyper-params if needed  
(batch size, patch size, LR, steps, etc.).

**2) Install requirements**
pip install -r requirements.txt

**3) Run training**
 python training/train.py
---
    
## 🚀 Inference

The inference script uses **fixed paths**.  
Please arrange your files as shown below before running.

### 📂 Expected paths
- **Checkpoint** → `./checkpoints/best_model.ckpt`  
- **Statistics CSV** → `./model/mri_statistics.csv`  
- **Input case** → `/input/images/<folder-with-"mri">/case.mha`  
  - Example: `/input/images/abdominal-t1-mri/case.mha`  
- **Output** → `./output/images/tumor_seg.mha`

### 📝 Steps

1. **Place your trained checkpoint**
   mkdir -p checkpoints
   cp /path/to/best_model.ckpt checkpoints/best_model.ckpt
2. **Place the spacing statistics**
   mkdir -p model
   cp /path/to/mri_statistics.csv model/mri_statistics.csv
3. **Put your input case under /input/images/ in a folder containing "mri" in its name:**
   sudo mkdir -p /input/images/abdominal-t1-mri
   sudo cp /path/to/case.mha /input/images/abdominal-t1-mri/
4. **run the inference**
   python inference/infer.py
---

**📊 Results (PANTHER Task 1):**
  placement: "4th place overall 🏅"

**📚 References**

- SegFormer3D: Perera et al., *SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation*, 2023.  
- nnU-Net preprocessing: Isensee et al., *nnU-Net: a self-adapting framework for U-Net-based medical image segmentation*, Nature Methods 2021.  
- scSE module: Roy et al., *Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks*, MICCAI 2018.  
- ASPP: Chen et al., *Rethinking Atrous Convolution for Semantic Image Segmentation (DeepLabV3)*, arXiv 2017.  
- Attention Gates: Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*, arXiv 2018.  
- FPN-style decoder: Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017.  
- Pancreatic tumor segmentation with ASPP + AG: Deng & Mou, *Pancreatic Tumor Segmentation Based on 3D U-Net with Densely Connected Atrous Spatial Pyramid Module and Attention Module*, ISAIM 2023.  


**🙏 Acknowledgements:**
  - PANTHER Challenge organizers for providing the dataset
  - Prof. George Matsopoulos (NTUA) for supervision
    

## 📄 License
This repository is released under the GPLv3 license, following the original SegFormer3D implementation.

---
