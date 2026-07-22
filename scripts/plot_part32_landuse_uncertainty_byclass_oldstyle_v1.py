#!/usr/bin/env python3
"""Plot class-specific land-use uncertainty with the older map style.

Rows are years and columns are final land-use classes. Each panel only shows
grids assigned to that final GAIA-constrained class. The style follows the
earlier raster uncertainty figure: light-to-dark `magma_r` color ramp, thin
boundary, clean white background, and a compact shared colorbar.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex-cache"))

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "DejaVu Sans"]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
UNCERTAINTY_GPKG = (
    DATA_DIR
    / "mlp_multiyear_5class_joint_tculuWeak_manual2005_gaia_thr10"
    / "uncertainty"
    / "shenzhen_5class_joint_mlp_uncertainty_finalclass.gpkg"
)
UNCERTAINTY_LAYER = "landuse_uncertainty_finalclass"
BOUNDARY_SHP = DATA_DIR / "boundary/sz_street_2_Dissolve_polygon.shp"
OUT_DIR = PROJECT_ROOT / "figures/appendix"

YEARS = [2000, 2005, 2010, 2015, 2020]
CLASSES = ["farmland", "green_space", "living", "production", "water"]
CLASS_LABELS = {
    "farmland": "Farmland",
    "green_space": "Green Space",
    "living": "Living Land",
    "production": "Production Land",
    "water": "Water Bodies",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncertainty-gpkg", type=Path, default=UNCERTAINTY_GPKG)
    parser.add_argument("--uncertainty-layer", default=UNCERTAINTY_LAYER)
    parser.add_argument("--boundary-shp", type=Path, default=BOUNDARY_SHP)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--metric-prefix", default="uncertainty")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=0.8)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[1] Load uncertainty data...")
    gdf = gpd.read_file(args.uncertainty_gpkg, layer=args.uncertainty_layer)
    boundary = gpd.read_file(args.boundary_shp).to_crs(gdf.crs)

    required = []
    for year in YEARS:
        required.extend([f"mlp_pred5_{year}", f"{args.metric_prefix}_{year}"])
    missing = [col for col in required if col not in gdf.columns]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    minx, miny, maxx, maxy = boundary.total_bounds
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.02

    cmap = plt.get_cmap("magma_r")
    norm = mpl.colors.Normalize(vmin=args.vmin, vmax=args.vmax)

    fig, axes = plt.subplots(len(YEARS), len(CLASSES), figsize=(21, 15), constrained_layout=False)

    print("[2] Draw maps...")
    for row, year in enumerate(YEARS):
        class_col = f"mlp_pred5_{year}"
        metric_col = f"{args.metric_prefix}_{year}"
        for col_idx, class_name in enumerate(CLASSES):
            ax = axes[row, col_idx]
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)
            ax.set_aspect("equal")
            ax.axis("off")

            sub = gdf[(gdf[class_col] == class_name) & gdf[metric_col].notna()].copy()
            if not sub.empty:
                sub.plot(
                    ax=ax,
                    column=metric_col,
                    cmap=cmap,
                    norm=norm,
                    linewidth=0,
                    edgecolor="none",
                    zorder=1,
                )
            boundary.plot(ax=ax, facecolor="none", edgecolor="0.35", linewidth=0.45, zorder=2)

            if row == 0:
                ax.set_title(CLASS_LABELS[class_name], fontsize=20, pad=6)
            if col_idx == 0:
                ax.text(
                    -0.05,
                    0.50,
                    str(year),
                    ha="right",
                    va="center",
                    rotation=90,
                    transform=ax.transAxes,
                    fontsize=20,
                )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.30, 0.045, 0.40, 0.018])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("Uncertainty Level", size=17, labelpad=8)
    cbar.ax.xaxis.set_label_position("top")

    plt.subplots_adjust(left=0.045, right=0.99, top=0.955, bottom=0.095, wspace=0.035, hspace=0.08)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "calibrated_oldstyle" if args.metric_prefix == "cal_uncertainty" else "oldstyle"
    pdf_path = args.out_dir / f"FigS_landuse_uncertainty_byclass_{suffix}.pdf"
    png_path = args.out_dir / f"FigS_landuse_uncertainty_byclass_{suffix}.png"
    print(f"[3] Save: {pdf_path}")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", transparent=True, dpi=args.dpi)
    print(f"[3] Save: {png_path}")
    plt.savefig(png_path, format="png", bbox_inches="tight", transparent=False, dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
