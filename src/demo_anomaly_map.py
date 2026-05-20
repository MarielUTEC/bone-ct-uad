import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from inject_synthetic_object import inject_sphere


def normalize_hu(volume, min_hu=-1000, max_hu=3000):
    volume = np.clip(volume, min_hu, max_hu)
    volume = (volume - min_hu) / (max_hu - min_hu)
    return volume.astype(np.float32)


def save_nifti(array, reference_img, path):
    out = nib.Nifti1Image(array.astype(np.float32), reference_img.affine)
    nib.save(out, path)


def main(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img = nib.load(input_path)
    volume = img.get_fdata().astype(np.float32)

    volume_norm = normalize_hu(volume)

    # Inject synthetic object
    corrupted, gt_mask = inject_sphere(
        volume_norm,
        center=None,
        radius=10,
        intensity=1.0
    )

    # Baseline pseudo-healthy reconstruction
    # This is NOT the final model; it is only for the first demo.
    pseudo_healthy = gaussian_filter(corrupted, sigma=2)

    anomaly_map = np.abs(corrupted - pseudo_healthy)

    save_nifti(volume_norm, img, os.path.join(output_dir, "01_original_normalized.nii.gz"))
    save_nifti(corrupted, img, os.path.join(output_dir, "02_synthetic_foreign_object.nii.gz"))
    save_nifti(gt_mask, img, os.path.join(output_dir, "03_ground_truth_mask.nii.gz"))
    save_nifti(pseudo_healthy, img, os.path.join(output_dir, "04_pseudo_healthy_baseline.nii.gz"))
    save_nifti(anomaly_map, img, os.path.join(output_dir, "05_anomaly_map.nii.gz"))

    z = volume.shape[0] // 2

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(volume_norm[z], cmap="gray")
    plt.title("Original CT")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(corrupted[z], cmap="gray")
    plt.title("Synthetic Object")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pseudo_healthy[z], cmap="gray")
    plt.title("Pseudo-healthy")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(anomaly_map[z], cmap="hot")
    plt.title("Anomaly Map")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "demo_result.png"), dpi=200)
    plt.close()

    print("Demo completed.")
    print(f"Outputs saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CT NIfTI file")
    parser.add_argument("--output", default="data/outputs/demo", help="Output directory")
    args = parser.parse_args()

    main(args.input, args.output)