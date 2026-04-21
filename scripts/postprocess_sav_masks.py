import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    return (x * x + y * y) <= radius * radius


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(mask)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    labeled, num = ndi.label(mask)
    if num == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labeled]


def close_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    return ndi.binary_closing(mask, structure=disk(radius))


def majority_temporal_smooth(mask_stack: np.ndarray, window: int) -> np.ndarray:
    """
    mask_stack: (T, H, W) bool
    window must be odd.
    """
    if window <= 1:
        return mask_stack
    assert window % 2 == 1, "window must be odd"
    pad = window // 2
    padded = np.pad(mask_stack.astype(np.uint8), ((pad, pad), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(mask_stack, dtype=bool)
    thresh = window // 2 + 1
    for t in range(mask_stack.shape[0]):
        votes = padded[t:t + window].sum(axis=0)
        out[t] = votes >= thresh
    return out


def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    return arr > 0


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    img.save(path)


def process_object_dir(
    in_obj_dir: Path,
    out_obj_dir: Path,
    min_area: int,
    close_radius: int,
    temporal_window: int,
) -> None:
    frame_paths = sorted([p for p in in_obj_dir.glob("*.png") if "Zone.Identifier" not in p.name])
    if not frame_paths:
        return

    masks = np.stack([load_mask(p) for p in frame_paths], axis=0)

    # Per-frame cleanup
    cleaned = []
    for mask in masks:
        m = fill_holes(mask)
        m = remove_small_components(m, min_area=min_area)
        m = close_mask(m, radius=close_radius)
        cleaned.append(m)
    cleaned = np.stack(cleaned, axis=0)

    # Temporal smoothing
    smoothed = majority_temporal_smooth(cleaned, window=temporal_window)

    for p, mask in zip(frame_paths, smoothed):
        save_mask(out_obj_dir / p.name, mask)


def main():
    parser = argparse.ArgumentParser(description="Post-process SA-V predicted masks.")
    parser.add_argument("--input_root", type=str, required=True, help="Baseline prediction root")
    parser.add_argument("--output_root", type=str, required=True, help="Postprocessed prediction root")
    parser.add_argument("--min_area", type=int, default=64, help="Remove connected components smaller than this")
    parser.add_argument("--close_radius", type=int, default=2, help="Binary closing radius")
    parser.add_argument("--temporal_window", type=int, default=3, help="Odd temporal smoothing window")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])

    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Videos found: {len(video_dirs)}")
    print(f"min_area={args.min_area}, close_radius={args.close_radius}, temporal_window={args.temporal_window}")

    for video_dir in video_dirs:
        obj_dirs = sorted([p for p in video_dir.iterdir() if p.is_dir()])
        for obj_dir in obj_dirs:
            out_obj_dir = output_root / video_dir.name / obj_dir.name
            process_object_dir(
                obj_dir,
                out_obj_dir,
                min_area=args.min_area,
                close_radius=args.close_radius,
                temporal_window=args.temporal_window,
            )

    print("Postprocessing complete.")


if __name__ == "__main__":
    main()