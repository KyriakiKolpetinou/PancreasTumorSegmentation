# Inference script for SegFormer3D variant (Pancreas Tumor Segmentation)
# Part of PancreasTumorSegmentation (GPLv3).
# Based on SegFormer3D (Perera et al., 2023). 
# Postprocessing inspired by nnU-Net and related challenge heuristics.
# Input:  /input/images/<folder-with-"mri">/*.mha
# Output: /output/images/pancreatic-tumor-segmentation/tumor_seg.mha
# Classes during training: 0=background, 1=tumor, 2=pancreas

from pathlib import Path
import glob, time
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import pandas as pd
from monai.inferers import SlidingWindowInferer
from models.segformer3d_variant import SegFormer3D 
from scipy import ndimage as ndi
import os


def mm_to_vox(r_mm, spacing):
    # r_mm σε mm, spacing=(sx,sy,sz) σε mm → ακτίνα σε voxels ανά άξονα
    sx, sy, sz = spacing
    #  σφαίρα σε ισοτροπικό περίπου grid: μέση τιμή spacing
    s = float((sx + sy + sz) / 3.0 + 1e-6)
    return max(1, int(round(r_mm / s)))

def make_ball(radius_vox):
    r = int(radius_vox)
    if r <= 0: r = 1
    L = 2*r + 1
    zz, yy, xx = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    mask = (xx*xx + yy*yy + zz*zz) <= (r*r)
    return mask.astype(bool)

def postprocess_tumor(
    tumor_prob: np.ndarray,                 # (D,H,W), float32 in [0,1]
    pancreas_prob: np.ndarray | None,       # (D,H,W) or None
    spacing_xyz: tuple[float,float,float],  # (x,y,z) mm
    thr: float = 0.45,
    min_mm3: float = 50.0,
    keep_second_ratio: float = 0.20,
    pancreas_dilate_mm: float = 6.0,
):
    # 1) threshold
    pred = (tumor_prob >= thr).astype(np.uint8)

    if pred.sum() == 0:
        return pred  

    #morphology
    ball1 = make_ball(1)
    pred = ndi.binary_opening(pred, structure=ball1)
    pred = ndi.binary_closing(pred, structure=ball1)
    pred = pred.astype(np.uint8)

    if pred.sum() == 0:
        return pred

    # 3) connected components
    lab, num = ndi.label(pred)
    if num == 0:
        return pred

    
    vox_vol_mm3 = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    sizes_vox = ndi.sum(np.ones_like(lab), labels=lab, index=range(1, num+1))
    sizes_vox = np.asarray(sizes_vox, dtype=np.float64)
    sizes_mm3 = sizes_vox * vox_vol_mm3

    # 3a) exclude small cc
    total_mm3 = float(pred.sum()) * vox_vol_mm3
    dyn_min_mm3 = max(min_mm3, 0.0005 * total_mm3)  # 0.05%
    keep_mask = sizes_mm3 >= dyn_min_mm3

    
    kept_ids = np.where(keep_mask)[0] + 1
    if kept_ids.size == 0:
        # if too strict keep larger c
        kept_ids = [int(np.argmax(sizes_mm3) + 1)]

    # 3b) up to 2 cc
    kept_sizes = sizes_mm3[kept_ids - 1]
    order = np.argsort(-kept_sizes)
    kept_ids = np.asarray(kept_ids)[order]

    final_ids = [int(kept_ids[0])]
    if kept_ids.size >= 2:
        if kept_sizes[order[1]] >= keep_second_ratio * kept_sizes[order[0]]:
            final_ids.append(int(kept_ids[1]))

    pred2 = np.isin(lab, final_ids).astype(np.uint8)

    # 4) close to pancreas
    if pancreas_prob is not None:
        pan = (pancreas_prob >= 0.5).astype(np.uint8)
        if pan.any():
            # dilation ~ pancreas_dilate_mm
            r = mm_to_vox(pancreas_dilate_mm, spacing_xyz)
            pan_dil = ndi.binary_dilation(pan, structure=make_ball(r))

            
            lab2, num2 = ndi.label(pred2)
            keep_ids2 = []
            for cid in range(1, num2+1):
                comp = (lab2 == cid)
                if (comp & pan_dil).any():
                    keep_ids2.append(cid)
            if keep_ids2:
                pred2 = np.isin(lab2, keep_ids2).astype(np.uint8)
           

    return pred2.astype(np.uint8)


# ---------- paths / config ----------
WEIGHTS_PATH =  "./checkpoints/best_model.ckpt"
STATS_CSV    = "./model/mri_statistics.csv"

NUM_CLASSES  = 3
ROI_SIZE     = (32, 160, 208)  
OVERLAP      = 0.5

OUTPUT_DIR   = Path("./output/images/")
OUTPUT_FILE  = OUTPUT_DIR / "tumor_seg.mha"


def build_model() -> torch.nn.Module:
    return SegFormer3D(
        in_channels=1,
        num_classes=NUM_CLASSES,
        embed_dims=[32, 64, 160, 256],
        depths=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        sr_ratios=[4, 2, 1, 1],
        use_hybrid_stem=True,
    )

# ---------- io utils ----------
def find_input_mha() -> Path:
    images_root = Path("/input/images")
    folders = [f for f in images_root.iterdir() if f.is_dir() and "mri" in f.name.lower()]
    if len(folders) == 1:
        fldr = folders[0]
        print("Folder containing eval image:", fldr.name)
    else:
        print("Warning: expected one folder containing 'mri', found", len(folders))
        fldr = images_root / "abdominal-t1-mri"   # baseline fallback
    files = sorted(glob.glob(str(fldr / "*.mha")))
    if not files:
        raise FileNotFoundError(f"No .mha found under {fldr}")
    return Path(files[0])

def read_target_spacing(stats_csv: str) -> tuple[float,float,float]:
    """
    Use training-time rule:
      - start with median spacing (x,y,z)
      - if anisotropy z/x >= 3, replace z with 10th percentile z
    """
    df  = pd.read_csv(stats_csv)
    med = tuple(map(float, df["median_spacing"].values[0].strip("[]").split()))
    p10 = tuple(map(float, df["10th_percentile_spacing"].values[0].strip("[]").split()))
    if med[2] / med[0] >= 3:
        tgt = (med[0], med[1], p10[2])
    else:
        tgt = med
    print("Target spacing (x,y,z):", tgt)
    return tgt

def resample_sitk(img: sitk.Image, new_spacing, is_label: bool) -> sitk.Image:
    old_spacing = np.array(img.GetSpacing(), dtype=float)
    old_size    = np.array(img.GetSize(),    dtype=int)
    new_spacing = np.array(new_spacing,      dtype=float)

    new_size = np.round(old_size * (old_spacing / new_spacing)).astype(int).tolist()

    res = sitk.ResampleImageFilter()
    res.SetOutputSpacing(tuple(new_spacing))
    res.SetSize([int(s) for s in new_size])
    res.SetOutputOrigin(img.GetOrigin())
    res.SetOutputDirection(img.GetDirection())
    res.SetTransform(sitk.Transform())
    res.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return res.Execute(img)

def normalize_nonzero(vol_np: np.ndarray) -> np.ndarray:
    nonzero = vol_np[vol_np > 0]
    if nonzero.size == 0:
        return vol_np.astype(np.float32)
    m, s = float(nonzero.mean()), float(nonzero.std()) + 1e-6
    return ((vol_np - m) / s).astype(np.float32)

# ---------- ckpt loading ----------
def load_state(model: torch.nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    # strip Lightning prefixes if present
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k[6:]] = v
        elif k.startswith("net."):
            new_state[k[4:]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    return model

# ---------- TTA infer ----------
def tta_logits(inferer, model, x):
    flip_sets = [(), (2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]
    acc = None
    with torch.no_grad():
        for flips in flip_sets:
            xin = torch.flip(x, flips) if flips else x
            # NOTE: only return the main logits (index 0)
            out = inferer(xin, lambda t: model(t)[0])
            out = torch.flip(out, flips) if flips else out
            acc = out if acc is None else (acc + out)
    return acc / float(len(flip_sets))


# ---------- main ----------
def main():
    start = time.time()

    # 1) Load input, preserve original geometry
    in_file = find_input_mha()
    ref_img = sitk.ReadImage(str(in_file))
    orig_spacing   = ref_img.GetSpacing()
    orig_origin    = ref_img.GetOrigin()
    orig_direction = ref_img.GetDirection()
    print(f"Original size: {ref_img.GetSize()}, spacing: {orig_spacing}")

    # 2) Resample to training spacing
    tgt_spacing = read_target_spacing(STATS_CSV)  # (x,y,z)
    img_tgt = resample_sitk(ref_img, tgt_spacing, is_label=False)

    # 3) Numpy -> norm
    vol = sitk.GetArrayFromImage(img_tgt).astype(np.float32)  # (D,H,W)
    vol = normalize_nonzero(vol)

    # 4) Build + load + TTA sliding window
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(vol)[None, None].to(device)           # (1,1,D,H,W)

    model = build_model().to(device).eval()
    model = load_state(model, WEIGHTS_PATH, device)

    inferer = SlidingWindowInferer(roi_size=ROI_SIZE, overlap=OVERLAP, sw_batch_size=1, mode="gaussian")

    logits = tta_logits(inferer, model, x) 

    # logits -> probabilities
    probs_t = F.softmax(logits, dim=1)[0]       # torch, shape (C,D,H,W)

# pull to numpy once
    probs = probs_t.detach().cpu().numpy()      # numpy, shape (C,D,H,W)
    tumor_prob    = probs[1]                    # class 1 = tumor
    pancreas_prob = probs[2]                    # class 2 = pancreas

# LIGHT post-processing 
    tumor_bin = postprocess_tumor(
    tumor_prob=tumor_prob,
    pancreas_prob=pancreas_prob,
    spacing_xyz=tgt_spacing,   # (x,y,z) mm from your read_target_spacing
    thr=0.45,
    min_mm3=50.0,
    keep_second_ratio=0.20,
    pancreas_dilate_mm=6.0
).astype(np.uint8)

    # 5) Map back to original spacing/size and save
    lab_tgt = sitk.GetImageFromArray(tumor_bin.astype(np.uint8))
    lab_tgt.SetSpacing(tuple(tgt_spacing))
    lab_tgt.SetOrigin(orig_origin)
    lab_tgt.SetDirection(orig_direction)

    lab_orig = resample_sitk(lab_tgt, orig_spacing, is_label=True)
    lab_orig.SetOrigin(orig_origin)
    lab_orig.SetDirection(orig_direction)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(lab_orig, str(OUTPUT_FILE))

    print(f"Done in {time.time()-start:.1f}s -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
