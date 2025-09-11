# PancreasTumorSegmentation

Official code release for our method in the **PANTHER Challenge (MICCAI 2025, Task 1)**.  
This repository contains the model, training pipeline, and preprocessing description used to achieve **4th place**.

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
See [`preprocessing/README.md`](preprocessing/README.md) for details.  
- Spacing normalisation (median / anisotropy-aware).  
- Z-score normalisation on non-zero voxels.  
- Safe label resampling.  
- Output format: compressed `.npz` with `(1, D, H, W)` arrays.

---

## 📦 Installation
Clone this repository and install dependencies:
```bash
git clone https://github.com/KyriakiKolpetinou/PancreasTumorSegmentation.git
cd PancreasTumorSegmentation
pip install -r requirements.txt

🚀 Usage
1. Preprocessing
Prepare the dataset with nnU-Net–style preprocessing:

python preprocessing/run_preprocessing.py --input /path/to/raw --output /path/to/preprocessed
(see preprocessing/README.md for detailed explanation).

2. Training
Train the model:
python training/train.py --config training/config.py

3. Inference
Run inference with a trained checkpoint

python training/infer.py --checkpoint ./checkpoints/best.ckpt --input /path/to/images --output ./preds

📊 Results (PANTHER Task 1)
4th place overall 🏅
Competitive Dice and Hausdorff95 metrics, close to 3rd place.

📜 License & Attribution
Code released under GPLv3, following SegFormer3D’s license.
Preprocessing adapted from nnU-Net (Isensee et al., Nature Methods, 2021).
Developed by Kyriaki Kolpetinou,
PhD candidate at NTUA, Biomedical Engineering Laboratory.

🙏 Acknowledgements
PANTHER Challenge organizers for providing the dataset.
Prof. George Matsopoulos (NTUA) for supervision.
SegFormer3D and nnU-Net authors for their original open-source work.
