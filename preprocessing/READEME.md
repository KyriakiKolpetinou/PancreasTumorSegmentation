Preprocessing (nnU-Net–style for MRI)

For preprocessing we closely followed the strategy of nnU-Net (Isensee et al., Nature Methods, 2021): [https://github.com/MIC-DKFZ/nnUNet]
Specifically:


*Input layout*

DATASET_ROOT/

  ImagesTr/   # .mha volumes
  
  LabelsTr/   # .mha segmentation masks (same filename stem as image)
  
  mri_statistics.csv   # per-dataset spacing statistics (see below)


Pairing rule

Images and labels are matched by filename stem (e.g., 10000_0001_xxx.mha pairs with 10000_0001_yyy.mha).
Files without a matching pair are skipped (logged as warnings).
Voxel spacing selection (per dataset)
Read from mri_statistics.csv:
median_spacing (z, y, x) as strings like "[1.0 0.8 0.8]".
10th_percentile_spacing (z, y, x).

Target spacing is:
median_spacing unless strong anisotropy is detected, defined as
median_spacing[z] / median_spacing[x] >= 3.
If anisotropic, keep median in-plane spacing and use 10th_percentile_spacing[z] for the through-plane axis:

target_spacing = (median_z_replaced_by_10th, median_y, median_x)


(This mirrors nnU-Net’s handling of anisotropic MRIs.)

Resampling

Images: cubic B-spline interpolation (order=3).
Labels: nearest neighbor (order=0).
Spacing order is handled as (z, y, x).
After resampling, image and label shapes are asserted to match.

Intensity normalisation (MRI)
Compute mean and std only over non-zero voxels of each image.

Z-score normalise: (image - mean) / std.
(If an image has no non-zero voxels, it’s left unchanged as a fallback.)

Label handling
Any negative label values are set to 0 before saving.
Label dtype is kept small (int8 if max label ≤ 127, else int16).

Output format
Preprocessed files are written to:

OUTPUT_ROOT/

  imagesTr/  # {image_stem}.npz
  
  labelsTr/  # {label_stem}.npz


Each .npz contains:

arr_0: the array shaped (1, D, H, W) — a channel-first 3D volume.
spacing: a 3-float numpy array with the target voxel spacing (z, y, x).

What this lets you reproduce

Resampling to a robust, dataset-specific spacing (median; anisotropy-aware on z).
MRI-style z-score normalisation on non-zero voxels.
Safe label resampling.
Deterministic pairing between images/labels and a consistent .npz layout your dataloader can consume.

Notes
The script expects an mri_statistics.csv with at least the columns:
median_spacing
10th_percentile_spacing

If you do not have this file, compute these statistics over the original .mha volumes first.

Attribution

This procedure mirrors the design choices in nnU-Net for MRI preprocessing (spacing selection, interpolation policies, and non-zero intensity normalisation). 
