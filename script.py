# extract_geometrical_parameters_xlsx_float32.py

from pathlib import Path
import gc
import numpy as np
import pandas as pd
import trimesh

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ============================================================
# Configuration
# ============================================================

# Use the local copy, not Google Drive, to avoid sync/file-lock issues.
MAIN_FOLDER = Path(r"C:\Terrains")

# Save inside C:\Terrains\Exports.
OUTPUT_XLSX = MAIN_FOLDER / "Exports" / "terrain_geometrical_parameters_no_sinuosity_float32.xlsx"

# Current folder names, as shown in your terminal output.
CATEGORY_FOLDERS = {
    "1. Planos": "plain",
    "2. Irregulares": "irregular",
    "3. Pontiagudos": "sharp",
    "4. Lisos": "smooth",
}

# For your Gaea/glTF terrain files, Y is probably the height axis.
HEIGHT_AXIS = 1

# Main memory-saving option.
# float32 uses half the memory of float64.
FLOAT_DTYPE = np.float32


# ============================================================
# Basic utilities
# ============================================================

def load_mesh(path):
    """
    Loads a .gltf file and joins geometries if needed.
    """
    obj = trimesh.load(path, force="scene")

    if isinstance(obj, trimesh.Scene):
        meshes = [
            geom for geom in obj.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]

        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}")

        mesh = trimesh.util.concatenate(meshes)

    elif isinstance(obj, trimesh.Trimesh):
        mesh = obj

    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")

    mesh.remove_unreferenced_vertices()

    # Keep core mesh arrays lighter where possible.
    # Trimesh may recreate cached arrays internally, so we also cast arrays again
    # in the extraction function before numeric operations.
    mesh.vertices = np.asarray(mesh.vertices, dtype=FLOAT_DTYPE)

    return mesh


def get_axes(vertices, height_axis):
    """
    Separates horizontal coordinates and height values.
    Returns float32 views/copies to reduce memory use.
    """
    axes = [0, 1, 2]
    horizontal_axes = [a for a in axes if a != height_axis]

    vertices = np.asarray(vertices, dtype=FLOAT_DTYPE)

    x = vertices[:, horizontal_axes[0]].astype(FLOAT_DTYPE, copy=False)
    z = vertices[:, horizontal_axes[1]].astype(FLOAT_DTYPE, copy=False)
    h = vertices[:, height_axis].astype(FLOAT_DTYPE, copy=False)

    return x, z, h


def make_height_grid(x, z, h):
    """
    Converts regular terrain vertices into a 2D height grid.
    The grid is stored as float32 to reduce memory use.
    """
    x = np.asarray(x, dtype=FLOAT_DTYPE)
    z = np.asarray(z, dtype=FLOAT_DTYPE)
    h = np.asarray(h, dtype=FLOAT_DTYPE)

    xs = np.unique(x).astype(FLOAT_DTYPE, copy=False)
    zs = np.unique(z).astype(FLOAT_DTYPE, copy=False)

    if len(xs) * len(zs) != len(h):
        return None, None, None

    x_index = {val: i for i, val in enumerate(xs)}
    z_index = {val: i for i, val in enumerate(zs)}

    H = np.full((len(zs), len(xs)), np.nan, dtype=FLOAT_DTYPE)

    for xi, zi, hi in zip(x, z, h):
        H[z_index[zi], x_index[xi]] = hi

    if np.isnan(H).any():
        return None, None, None

    return xs, zs, H


def safe_stats(values, prefix):
    """
    Returns a compact set of useful statistics.
    Input is converted to float32 when possible to reduce temporary memory use.
    Output values are Python floats for clean Excel export.
    """
    values = np.asarray(values, dtype=FLOAT_DTYPE)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p95": np.nan,
        }

    return {
        f"{prefix}_mean": float(np.mean(values, dtype=FLOAT_DTYPE)),
        f"{prefix}_std": float(np.std(values, dtype=FLOAT_DTYPE)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.ptp(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
    }


# ============================================================
# Main feature extraction
# ============================================================

def extract_geometrical_parameters(path, label, category_folder, height_axis=1):
    """
    Extracts the geometrical parameters described in the manuscript.
    Heavy numeric arrays are processed as float32 to reduce memory use.
    """
    path = Path(path)
    mesh = load_mesh(path)

    vertices = np.asarray(mesh.vertices, dtype=FLOAT_DTYPE)
    x, z, h = get_axes(vertices, height_axis)

    features = {
        "file": path.name,
        "terrain_id": path.stem,
        "category_folder": category_folder,
        "label": label,
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
    }

    # --------------------------------------------------------
    # 1. Elevation / relief
    # --------------------------------------------------------
    features.update(safe_stats(h, "height"))

    xs, zs, H = make_height_grid(x, z, h)

    if H is not None:
        features["height_grid_available"] = 1

        dx = float(np.mean(np.diff(xs), dtype=FLOAT_DTYPE)) if len(xs) > 1 else 1.0
        dz = float(np.mean(np.diff(zs), dtype=FLOAT_DTYPE)) if len(zs) > 1 else 1.0

        # ----------------------------------------------------
        # 2. Slope
        # ----------------------------------------------------
        grad_z, grad_x = np.gradient(H, dz, dx)
        grad_z = grad_z.astype(FLOAT_DTYPE, copy=False)
        grad_x = grad_x.astype(FLOAT_DTYPE, copy=False)

        slope = np.sqrt(grad_x ** 2 + grad_z ** 2).astype(FLOAT_DTYPE, copy=False)
        features.update(safe_stats(slope.ravel(), "slope"))

        del grad_z, grad_x, slope
        gc.collect()

        # ----------------------------------------------------
        # 3. Local roughness
        # ----------------------------------------------------
        padded = np.pad(H, 1, mode="edge").astype(FLOAT_DTYPE, copy=False)

        local_mean = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / FLOAT_DTYPE(9.0)
        local_mean = local_mean.astype(FLOAT_DTYPE, copy=False)

        roughness = np.abs(H - local_mean).astype(FLOAT_DTYPE, copy=False)
        features.update(safe_stats(roughness.ravel(), "roughness"))

        del local_mean, roughness
        gc.collect()

        # ----------------------------------------------------
        # 4. Ruggedness
        # ----------------------------------------------------
        ruggedness = np.sqrt(
            (
                (padded[:-2, :-2] - H) ** 2 +
                (padded[:-2, 1:-1] - H) ** 2 +
                (padded[:-2, 2:] - H) ** 2 +
                (padded[1:-1, :-2] - H) ** 2 +
                (padded[1:-1, 2:] - H) ** 2 +
                (padded[2:, :-2] - H) ** 2 +
                (padded[2:, 1:-1] - H) ** 2 +
                (padded[2:, 2:] - H) ** 2
            ) / FLOAT_DTYPE(8.0)
        ).astype(FLOAT_DTYPE, copy=False)

        features.update(safe_stats(ruggedness.ravel(), "ruggedness"))

        del ruggedness, padded
        gc.collect()

        del xs, zs, H
        gc.collect()

    else:
        features["height_grid_available"] = 0

    # --------------------------------------------------------
    # 6. Face adjacency angles
    # --------------------------------------------------------
    adjacency_angles = np.asarray(mesh.face_adjacency_angles, dtype=FLOAT_DTYPE)

    face_adjacency_angle_deg = np.degrees(adjacency_angles).astype(FLOAT_DTYPE, copy=False)
    features.update(safe_stats(face_adjacency_angle_deg, "face_adjacency_angle_deg"))

    del adjacency_angles, face_adjacency_angle_deg
    gc.collect()

    # --------------------------------------------------------
    # 7. Surface-normal tilt
    # --------------------------------------------------------
    vertical = np.zeros(3, dtype=FLOAT_DTYPE)
    vertical[height_axis] = FLOAT_DTYPE(1.0)

    normals = np.asarray(mesh.face_normals, dtype=FLOAT_DTYPE)
    dots = np.clip(np.abs(normals @ vertical), -1.0, 1.0).astype(FLOAT_DTYPE, copy=False)

    normal_tilt_deg = np.degrees(np.arccos(dots)).astype(FLOAT_DTYPE, copy=False)

    features.update(safe_stats(normal_tilt_deg, "normal_tilt_deg"))

    features["normal_vector_variance"] = float(np.var(normals, axis=0, dtype=FLOAT_DTYPE).sum())

    features["prop_faces_tilt_gt_30deg"] = float(np.mean(normal_tilt_deg > 30))
    features["prop_faces_tilt_gt_45deg"] = float(np.mean(normal_tilt_deg > 45))
    features["prop_faces_tilt_gt_60deg"] = float(np.mean(normal_tilt_deg > 60))

    del vertices, x, z, h, normals, dots, normal_tilt_deg, mesh
    gc.collect()

    return features


# ============================================================
# Folder processing
# ============================================================

def extract_all_terrains(main_folder, category_folders, output_xlsx, height_axis=1):
    """
    Extracts geometrical parameters from all .gltf files in all category folders.
    """
    main_folder = Path(main_folder)
    output_xlsx = Path(output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    all_files = []

    for folder_name, label in category_folders.items():
        folder_path = main_folder / "Exports" / folder_name

        if not folder_path.exists():
            print(f"WARNING: folder not found: {folder_path}")
            continue

        gltf_files = sorted(folder_path.glob("*.gltf"))

        print(f"\nCategory: {folder_name}")
        print(f"Found {len(gltf_files)} .gltf files")

        for gltf_file in gltf_files:
            all_files.append((gltf_file, label, folder_name))

    iterator = all_files
    if HAS_TQDM:
        iterator = tqdm(all_files, desc="Extracting terrain parameters", unit="file")

    for i, (gltf_file, label, folder_name) in enumerate(iterator, start=1):
        print(f"[{i}/{len(all_files)}] Processing: {gltf_file.name}")

        try:
            row = extract_geometrical_parameters(
                path=gltf_file,
                label=label,
                category_folder=folder_name,
                height_axis=height_axis,
            )
            rows.append(row)

        except Exception as e:
            print(f"FAILED: {gltf_file.name}")
            print(f"Reason: {e}")

        finally:
            gc.collect()

    df = pd.DataFrame(rows)

    # Save as Excel
    df.to_excel(output_xlsx, index=False)

    print("\nDone.")
    print(f"Total files processed successfully: {len(df)}")
    print(f"Total .gltf files found: {len(all_files)}")
    print(f"Output saved to: {output_xlsx}")

    return df


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    df = extract_all_terrains(
        main_folder=MAIN_FOLDER,
        category_folders=CATEGORY_FOLDERS,
        output_xlsx=OUTPUT_XLSX,
        height_axis=HEIGHT_AXIS,
    )

    print("\nPreview:")
    print(df.head())
