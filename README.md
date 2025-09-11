# PancreasTumorSegmentation

## Preprocessing

For preprocessing we closely followed the strategy of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) (Isensee et al., *Nature Methods*, 2021).  
Specifically:

- **Target spacing**: estimated from the training set as the median voxel spacing; if strong anisotropy was detected (through-plane spacing ≥ 3× in-plane spacing), we adopted the 10th percentile for the out-of-plane axis.
- **Resampling**: images were resampled to the target spacing using 3rd-order spline interpolation, masks with nearest-neighbor interpolation.
- **Intensity normalization**: z-score normalization based on nonzero voxels per case (zero-mean, unit variance).
- **Storage**: the preprocessed images and labels were saved as compressed `.npz` files with spacing metadata for downstream training.

This is identical in spirit to the default nnU-Net preprocessing for MRI, but re-implemented here for reproducibility and compatibility with our pipeline.
