import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib.patheffects as path_effects

from typing import Literal, Optional, Tuple
from pathlib import Path
from src.networks.kohonen_som import SOM
from src.utils.feature_scaling import zscore
from src.utils.config import load_config
from matplotlib.patches import Circle
from collections import defaultdict


def som_size_heuristic(n_samples: int) -> Tuple[int, int]:
    """ Heuristic for (KxK) map size based on N samples."""
    target = int(5 * math.sqrt(n_samples) - 1)
    rows = int(math.sqrt(target))
    cols = int(math.ceil(target / max(1, rows)))
    return max(4, rows), max(4, cols)

def country_abbrev(name: str, k: int = 3) -> str:
    special_cases = {
        "Slovenia": "SVN",
        "Slovakia": "SVK",
        "Switzerland": "CHE",
        "Sweden": "SWE",
        "Austria": "AUT",
    }

    if name in special_cases:
        return special_cases[name]

    t = "".join(c for c in name if c.isalpha())
    return t[:k].upper()

def plot_umatrix_with_hit_circles(ax, umatrix: np.ndarray, hits: np.ndarray, neighbor_type: int, cfg: dict):
    cmap = cfg.get("umatrix_cmap", "plasma")
    hit_color = "white"
    hit_alpha = 0.7
    circle_radius = 0.15
    im = ax.imshow(umatrix, origin="lower", cmap=cmap)
    ax.set_title(f"U-Matrix ({neighbor_type}-neighbors) + hits")
    ax.set_xlabel("j"); ax.set_ylabel("i")

    m, n = hits.shape
    for i in range(m):
        for j in range(n):
            hit_count = hits[i, j]
            if hit_count > 0:
                circle = Circle(
                    (j, i),
                    radius=circle_radius,
                    color=hit_color,
                    alpha=hit_alpha,
                    linewidth=1.0,
                    zorder=10
                )
                ax.add_patch(circle)
                ax.text(
                    j, i, str(int(hit_count)),
                    ha='center', va='center',
                    fontsize=9,
                    fontweight='bold',
                    color='black',
                    zorder=11
                )

    ax.set_aspect("equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(m))

    return im

def plot_hits(ax, hits: np.ndarray, cfg: Optional[dict] = None):
    im = ax.imshow(hits, origin="lower", cmap=cfg["hits_cmap"] if cfg else "Purples")
    ax.set_title("Hits (neuron associations)")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    return im

def plot_labels_on_umatrix(ax, umatrix: np.ndarray, coords: np.ndarray, labels, bmu_dists: np.ndarray, cfg: Optional[dict] = None):
    ax.imshow(umatrix, origin="lower", cmap=cfg.get("umatrix_cmap", "plasma"))
    ax.set_title("U-Matrix with country labels")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    abbrev_to_full = {}
    abbrev_to_dist = {}
    # Group labels by their neuron coordinates
    neuron_labels = defaultdict(list)
    for label, (ri, cj), dist in zip(labels, coords, bmu_dists):
        abbrev = country_abbrev(label)
        neuron_labels[(ri, cj)].append(abbrev)
        abbrev_to_full[abbrev] = label
        abbrev_to_dist[abbrev] = dist

    # Plot stacked labels for each neuron
    for (ri, cj), lab_list in neuron_labels.items():
        if len(lab_list) == 1:
            text = ax.text(cj, ri, lab_list[0], ha="center", va="center",
                   fontsize=10, color=cfg.get("labels_cmap", "deepskyblue"), fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                                   path_effects.Normal()])
        else:
            # multiple labels: stack them vertically
            n_labels = len(lab_list)
            vspace = 0.15
            total_height = (n_labels - 1) * vspace
            start_offset = -total_height / 2
            for idx, label in enumerate(lab_list):
                y_offset = start_offset + idx * vspace
                text = ax.text(cj, ri + y_offset, label, ha="center", va="center",
                       fontsize=10, color=cfg.get("labels_cmap", "deepskyblue"), fontweight='bold')
                text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                                       path_effects.Normal()])

    # legend with abbreviations and full names
    sorted_abbrevs = sorted(abbrev_to_full.items(), key=lambda x: abbrev_to_dist[x[0]], reverse=True)

    # Format with consistent spacing
    max_name_len = max(len(full_name) for _, full_name in sorted_abbrevs)
    legend_lines = []
    for abbrev, full_name in sorted_abbrevs:
        dist_val = abbrev_to_dist[abbrev]
        line = f"{abbrev}:  {full_name:<{max_name_len}}  {dist_val:6.4f}"
        legend_lines.append(line)

    legend_text = "\n".join(legend_lines)
    ax.text(1.25, 0.5, legend_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def plot_component_planes(weights: np.ndarray, feat_names, suptitle: str = "Component planes", save_path: Optional[Path] = None, cfg: Optional[dict] = None):
    m, n, d = weights.shape
    cols = 4
    rows = int(np.ceil(d / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4*cols, 3.4*rows))
    axes = np.atleast_2d(axes)

    for k in range(d):
        r, c = divmod(k, cols)
        plane = weights[:, :, k]
        vmin, vmax = plane.min(), plane.max()
        im = axes[r, c].imshow(plane, origin="lower", cmap=cfg.get("umatrix_cmap", "magma"), vmin=vmin, vmax=vmax)
        axes[r, c].set_title(feat_names[k])
        axes[r, c].set_xticks(range(n)); axes[r, c].set_yticks(range(m))
        cb = plt.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)

    # hide empty axes
    for k in range(d, rows*cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    fig.suptitle(suptitle, y=1.02, fontsize=12)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Component planes plot saved to: {save_path}")

    return fig

if __name__ == "__main__":
    config_path = "configs/europe_som.yaml"
    overrides = []
    if len(sys.argv) > 1:
        if sys.argv[1].startswith("--config="):
            config_path = sys.argv[1].split("=", 1)[1]
            overrides = [arg.split("=", 1)[1] for arg in sys.argv[2:] if arg.startswith("--set=")]
        elif sys.argv[1].startswith("--set="):
            overrides = [arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--set=")]

    cfg = load_config(config_path, overrides)
    dataset_path = cfg["dataset"]["path"]
    country_col = cfg["dataset"]["country_column"]
    features = cfg["dataset"]["features"]
    seed = cfg["experiment"]["seed"]
    output_dir = cfg["experiment"].get("output_dir")
    viz_cfg = cfg["visualization"]

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    standardized_data, mean_vals, std_vals = zscore(df, features)
    map_rows = cfg["som"]["map_rows"]
    map_cols = cfg["som"]["map_cols"]
    if map_rows is None or map_cols is None:
        map_rows, map_cols = som_size_heuristic(len(df))

    print(f"SOM: {map_rows} x {map_cols} neurons")

    som = SOM(
        map_rows, map_cols,
        dim=standardized_data.shape[1],
        lr=cfg["som"]["learning_rate"],
        sigma=cfg["som"]["sigma"],
        seed=seed,
        init_weights_type=cfg["som"]["init_weights_type"],
        bmu_metric=cfg["som"]["bmu_metric"],
        lr_decay_type=cfg["som"]["lr_decay_type"],
        sigma_decay_type=cfg["som"]["sigma_decay_type"],
        neighborhood_type=cfg["som"]["neighborhood_type"]
    )

    som.train(
        standardized_data,
        epochs=cfg["training"]["epochs"],
        shuffle=cfg["training"]["shuffle"]
    )

    # (country -> neuron)
    bmu_coords, bmu_dists = som.bmu_coords_and_dists(standardized_data)
    assignments = pd.DataFrame({
        "Country": df[country_col],
        "BMU_i": bmu_coords[:,0],
        "BMU_j": bmu_coords[:,1],
        "BMU_dist": np.round(bmu_dists, 4)
    }).sort_values("BMU_dist").reset_index(drop=True)
    display_top = cfg["analysis"]["display_top_assignments"]
    print(f"\nCountry top assignments (first {display_top}):")
    print(assignments.head(display_top).to_string(index=False))
    final_qe = som.quantization_error(standardized_data)
    print(f"\n  Quantization Error (QE): {final_qe:.4f}")
    final_te = som.topological_error(standardized_data)
    print(f"  Topological Error (TE):  {final_te:.4f}")
    hit_map = som.hits(standardized_data)
    neighbor_type = cfg["analysis"]["umatrix_neighbor_type"]
    umatrix = som.u_matrix(neighbor_type)

    # Visualization
    viz_cfg = cfg["visualization"]
    fig_width = viz_cfg["figure_size"]["width"]
    fig_height = viz_cfg["figure_size"]["height"]
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))

    im0 = plot_umatrix_with_hit_circles(axes[0], umatrix, hit_map, neighbor_type, viz_cfg)
    plt.colorbar(im0, ax=axes[2], fraction=0.05, pad=0.02)

    im1 = plot_hits(axes[1], hit_map, viz_cfg)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plot_labels_on_umatrix(axes[2], umatrix, bmu_coords, df["Country"], bmu_dists, viz_cfg)
    axes[2].set_xticks(range(map_cols)); axes[2].set_yticks(range(map_rows))

    fig.tight_layout()

    save_path = viz_cfg.get("save_path")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        umatrix_plot = save_path.parent / "kohonen_som_umatrix"
        fig.savefig(umatrix_plot, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {umatrix_plot}")
        component_save_path = save_path.parent / "kohonen_component_planes"

    _ = plot_component_planes(som.weights, features, suptitle="Component planes per variable", save_path=component_save_path, cfg=viz_cfg)
    plt.show()
