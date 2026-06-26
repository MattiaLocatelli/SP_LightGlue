import torch
import os
import time
import csv
from pathlib import Path
import torch.nn.functional as F
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    
    fig, axes = plt.subplots(2, 1, figsize=(19, 9), dpi=dpi)
    
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'k'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def resize_frame(image, width=920, height=256):
    if image.dim() == 3:
        image = image.unsqueeze(0)
    elif image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(0)
    
torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

# 1. Config
online_img_pth = "Online_Keyframe/R1257.png"
offline_folder = "Offline_Keyframes_Turn2-3/"
output_dir = "output_matches"
os.makedirs(output_dir, exist_ok=True)

offline_imgs = [f for f in os.listdir(offline_folder) if f.endswith('.png')]

inference_times = []
confidences = []
inliers_number = []
csv_rows = []

# Load online keyframe
image0 = resize_frame(load_image(online_img_pth).to(device))
feats0 = extractor.extract(image0)

# Matching pipeline
for img_name in offline_imgs:
    image1 = resize_frame(load_image(os.path.join(offline_folder, img_name)).to(device))
    feats1 = extractor.extract(image1)
    
    torch.cuda.synchronize() 
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    matches01 = matcher({"image0": feats0, "image1": feats1})
    
    end_event.record()
    
    torch.cuda.synchronize()
    inference_time = start_event.elapsed_time(end_event)
    inference_times.append(inference_time)
    
    # Remove batch dim
    f0, f1, m01 = [rbd(x) for x in [feats0, feats1, matches01]]
    kpts0, kpts1 = f0["keypoints"].cpu().numpy(), f1["keypoints"].cpu().numpy()
    matches = m01["matches"].cpu().numpy()
    
    scores = m01["scores"].cpu().numpy()
    
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    
    img0_np = image0.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img1_np = image1.cpu().squeeze(0).permute(1, 2, 0).numpy()
    
    if img0_np.shape[-1] == 1: img0_np = img0_np.squeeze(-1)
    if img1_np.shape[-1] == 1: img1_np = img1_np.squeeze(-1)

    # out_path = os.path.join(output_dir, f"match_{img_name}")
    
    color = cm.jet(scores)
    
    # filter keypoints
    threshold = 0.7 
    mask = scores > threshold
    mkpts0_filtered = m_kpts0[mask]
    mkpts1_filtered = m_kpts1[mask]
    color_filtered = color[mask]
    
    confidences.append(scores.mean())
    inliers_number.append(len(mkpts0_filtered))
    csv_rows.append({
        "row_type": "match",
        "keyframe": img_name,
        "matches": len(mkpts0_filtered),
        "inference_time_ms": f"{inference_time:.3f}",
        "mean_confidence": f"{scores.mean():.6f}",
        "threshold": threshold,
    })
    
    text = ['LightGlue', 'Matches: {}'.format(len(mkpts0_filtered))]
    fig = make_matching_figure(
        img0_np, img1_np, mkpts0_filtered, mkpts1_filtered, 
        color=color_filtered, 
        text=text
    )
    
    save_path = os.path.join(output_dir, f"match_{img_name}")
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close(fig)
    
    print(f"Keyframe: {img_name} | Matches: {len(mkpts0_filtered)} | Inf Time {inference_time:.3f}ms")
    
    # print(f"Saved: {out_path}")

summary_csv_path = os.path.join(output_dir, "matching_metrics.csv")
if inference_times:
    csv_rows.append({
        "row_type": "summary",
        "keyframe": "",
        "matches": "",
        "inference_time_ms": f"{sum(inference_times[1:])/(len(inference_times)-1):.3f}" if len(inference_times) > 1 else f"{inference_times[0]:.3f}",
        "mean_confidence": f"{np.mean(confidences):.6f}",
        "threshold": threshold,
        "mean_inliers": f"{np.mean(inliers_number):.3f}",
    })

with open(summary_csv_path, "w", newline="", encoding="utf-8") as csv_file:
    fieldnames = ["row_type", "keyframe", "matches", "inference_time_ms", "mean_confidence", "threshold", "mean_inliers"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"Mean Inference Time: {sum(inference_times[1:])/(len(inference_times)-1):.3f}ms")
print(f"Mean Confidence: {np.mean(confidences)}")
print(f"Mean Number Inliers: {np.mean(inliers_number)} with confidence > {threshold}")