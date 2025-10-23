import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from typing import Literal, Optional, Tuple
from pathlib import Path
from src.networks.kohonen_som import SOM
from src.utils.feature_scaling import zscore
from src.utils.config import load_config


def som_size_heuristic(n_samples: int) -> Tuple[int, int]:
    """ Heuristic for (KxK) map size based on N samples."""
    target = int(5 * math.sqrt(n_samples))
    rows = int(math.sqrt(target))
    cols = int(math.ceil(target / max(1, rows)))
    return max(5, rows), max(5, cols)

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
    }).sort_values(["BMU_i","BMU_j","Country"]).reset_index(drop=True)
    display_top = cfg["analysis"]["display_top_assignments"]
    print(f"\nCountry top assignments (first {display_top}):")
    print(assignments.head(display_top).to_string(index=False))
    final_qe = som.quantization_error(standardized_data)
    print(f"\n  Quantization Error (QE): {final_qe:.4f}")
    final_te = som.topological_error(standardized_data)
    print(f"  Topological Error (TE):  {final_te:.4f}")
    hit_map = som.hits(standardized_data)
    umatrix = som.u_matrix(neighbor_type=cfg["analysis"]["umatrix_neighbor_type"])

    # Visualization
    viz_cfg = cfg["visualization"]
    fig_width = viz_cfg["figure_size"]["width"]
    fig_height = viz_cfg["figure_size"]["height"]
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))

    # Hits
    img0 = axes[0].imshow(hit_map, origin="upper", cmap=viz_cfg["hits_cmap"])
    axes[0].set_title("Asociaciones por neurona (hits)")
    axes[0].set_xlabel("j"); axes[0].set_ylabel("i")
    plt.colorbar(img0, ax=axes[0], fraction=0.046, pad=0.04)

    # U-Matrix
    img1 = axes[1].imshow(umatrix, origin="upper", cmap=viz_cfg["umatrix_cmap"])
    neighbor_type = cfg["analysis"]["umatrix_neighbor_type"]
    axes[1].set_title(f"U-Matrix (distancia promedio a {neighbor_type} vecinas)")
    axes[1].set_xlabel("j"); axes[1].set_ylabel("i")
    plt.colorbar(img1, ax=axes[1], fraction=0.046, pad=0.04)

    # Map with country labels
    axes[2].imshow(np.zeros_like(hit_map, dtype=float), origin="upper", cmap=viz_cfg["labels_cmap"], vmin=0, vmax=1)
    axes[2].set_title("Países asignados (BMU)")
    axes[2].set_xlabel("j"); axes[2].set_ylabel("i")
    for country, (row, col) in zip(df[country_col], bmu_coords):
        axes[2].text(col, row, country, ha="center", va="center", fontsize=viz_cfg["label_fontsize"])

    axes[2].set_xticks(range(map_cols)); axes[2].set_yticks(range(map_rows))
    plt.tight_layout()
    save_path = viz_cfg.get("save_path")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    if viz_cfg["show_plot"]:
        plt.show()
