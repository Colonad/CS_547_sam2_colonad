import argparse
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path)) > 0


def save_mask(out_path: Path, mask: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(out_path)


def strict_pngs(directory: Path):
    return tuple(
        sorted(
            filter(
                lambda p: p.is_file()
                and p.suffix.lower() == ".png"
                and "Zone.Identifier" not in p.name,
                directory.iterdir(),
            ),
            key=lambda p: p.name,
        )
    )


def subdirs(directory: Path):
    return tuple(sorted(filter(lambda p: p.is_dir(), directory.iterdir()), key=lambda p: p.name))


def remove_small_components(mask_stack: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask_stack

    def one_mask(mask: np.ndarray) -> np.ndarray:
        labeled, num = ndi.label(mask)
        if num == 0:
            return mask
        sizes = np.bincount(labeled.ravel())
        keep = sizes >= min_area
        keep[0] = False
        return keep[labeled]

    return np.stack(tuple(map(one_mask, mask_stack)), axis=0)


def postprocess_stack(
    mask_stack: np.ndarray,
    min_area: int,
    close_radius: int,
    temporal_window: int,
) -> np.ndarray:
    # 1) fill holes, vectorized across all frames
    masks = ndi.binary_fill_holes(mask_stack)

    # 2) remove small connected components, per-frame
    masks = remove_small_components(masks, min_area=min_area)

    # 3) morphological closing, vectorized across all frames
    if close_radius > 0:
        masks = ndi.binary_closing(
            masks,
            structure=disk(close_radius)[None, :, :],
        )

    # 4) temporal majority smoothing, vectorized along time axis
    if temporal_window > 1:
        assert temporal_window % 2 == 1, "temporal_window must be odd"
        votes = ndi.convolve1d(
            masks.astype(np.uint8),
            weights=np.ones(temporal_window, dtype=np.uint8),
            axis=0,
            mode="nearest",
        )
        masks = votes >= (temporal_window // 2 + 1)

    return masks


def process_object_dir(
    obj_dir: Path,
    output_root: Path,
    min_area: int,
    close_radius: int,
    temporal_window: int,
) -> None:
    frame_paths = strict_pngs(obj_dir)
    if len(frame_paths) == 0:
        return

    masks = np.stack(tuple(map(load_mask, frame_paths)), axis=0)

    processed = postprocess_stack(
        masks,
        min_area=min_area,
        close_radius=close_radius,
        temporal_window=temporal_window,
    )

    out_obj_dir = output_root / obj_dir.parent.name / obj_dir.name

    def save_one(item):
        in_path, mask = item
        save_mask(out_obj_dir / in_path.name, mask)

    tuple(map(save_one, zip(frame_paths, processed)))


def process_video_dir(
    video_dir: Path,
    output_root: Path,
    min_area: int,
    close_radius: int,
    temporal_window: int,
) -> None:
    obj_dirs = subdirs(video_dir)
    worker = partial(
        process_object_dir,
        output_root=output_root,
        min_area=min_area,
        close_radius=close_radius,
        temporal_window=temporal_window,
    )
    tuple(map(worker, obj_dirs))


def main():
    parser = argparse.ArgumentParser(description="Vectorized SA-V mask postprocessing.")
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--min_area", type=int, default=64)
    parser.add_argument("--close_radius", type=int, default=2)
    parser.add_argument("--temporal_window", type=int, default=3)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    video_dirs = subdirs(input_root)

    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Videos found: {len(video_dirs)}")
    print(
        f"min_area={args.min_area}, "
        f"close_radius={args.close_radius}, "
        f"temporal_window={args.temporal_window}"
    )

    worker = partial(
        process_video_dir,
        output_root=output_root,
        min_area=args.min_area,
        close_radius=args.close_radius,
        temporal_window=args.temporal_window,
    )
    tuple(map(worker, video_dirs))

    print("Postprocessing complete.")


if __name__ == "__main__":
    main()