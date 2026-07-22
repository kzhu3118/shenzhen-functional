#!/usr/bin/env python3
"""Map MODIS summer LST used in the thermal analysis.

Creates a five-panel map for 2000, 2005, 2010, 2015 and 2020 using a common
color scale. LST values are converted from Kelvin to Celsius.
"""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import frykit.plot as fplt
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


PROJECT = Path(__file__).resolve().parents[1]
BOUNDARY_SHP = PROJECT / "data/boundary/sz_street_2_Dissolve_polygon.shp"
GRID_GPKG = (
    PROJECT
    / "data/mlp_multiyear_5class_joint_tculuWeak_manual2005_gaia_thr10/shenzhen_5class_joint_mlp_multiyear.gpkg"
)
GRID_LAYER = "pred5_joint_mlp"
LST_DIR = PROJECT / "data/GEE/shenzhen_features_2000_2020"
OUT_DIR = PROJECT / "figures/part33_thermal_descriptive"

YEARS = [2000, 2005, 2010, 2015, 2020]


def load_lst_maps() -> gpd.GeoDataFrame:
    grid = gpd.read_file(GRID_GPKG, layer=GRID_LAYER)[["grid_id", "geometry"]].to_crs(epsg=4326).copy()
    for year in YEARS:
        path = LST_DIR / f"shenzhen_LST_MODIS_{year}.csv"
        lst = pd.read_csv(path, usecols=["grid_id", "LST_mean"])
        lst[f"LST_{year}"] = lst["LST_mean"] - 273.15
        grid = grid.merge(lst[["grid_id", f"LST_{year}"]], on="grid_id", how="left")
    return grid


def add_scale_bar(ax) -> None:
    scale_bar = fplt.add_scale_bar(ax, 0.5, 0.1, length=10)
    scale_bar.set_xticks([0, 10])
    scale_bar.set_xticklabels(["0", "10 km"])
    scale_bar.xaxis.get_label().set_fontsize(11)
    scale_bar.tick_params(labelsize=11, length=2)
    for spine in scale_bar.spines.values():
        spine.set_linewidth(0.9)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.titlesize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 0,
        }
    )

    gdf = load_lst_maps()
    boundary_gdf = gpd.read_file(BOUNDARY_SHP).to_crs(epsg=4326)
    boundary_polygon = boundary_gdf.geometry.union_all()
    minx, miny, maxx, maxy = boundary_polygon.bounds

    value_cols = [f"LST_{y}" for y in YEARS]
    vals = gdf[value_cols].to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    vmin, vmax = np.nanpercentile(vals, [2, 98])
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("lst_blue_red", ["#2166ac", "#f7f7f7", "#b2182b"])

    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.TransverseMercator(central_longitude=114.0, central_latitude=22.5)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3)
    axes = [
        fig.add_subplot(gs[0, 0], projection=map_crs),
        fig.add_subplot(gs[0, 1], projection=map_crs),
        fig.add_subplot(gs[0, 2], projection=map_crs),
        fig.add_subplot(gs[1, 0], projection=map_crs),
        fig.add_subplot(gs[1, 1], projection=map_crs),
    ]
    cax = fig.add_subplot(gs[1, 2])

    titles = ["(a) 2000", "(b) 2005", "(c) 2010", "(d) 2015", "(e) 2020"]
    for ax, year, title in zip(axes[:5], YEARS, titles):
        col = f"LST_{year}"
        ax.set_extent([minx - 0.01, maxx + 0.01, miny - 0.01, maxy + 0.01], crs=data_crs)
        ax.axis("off")

        for geom, value in zip(gdf.geometry, gdf[col]):
            facecolor = "#f2f2f2" if not np.isfinite(value) else cmap(norm(value))
            fplt.add_geometries(
                ax,
                [geom],
                crs=data_crs,
                fc=facecolor,
                ec="none",
                lw=0,
            )

        fplt.add_geometries(
            ax,
            boundary_polygon,
            crs=data_crs,
            fc="none",
            ec="black",
            lw=0.5,
        )
        ax.set_title(title, fontsize=16, pad=3, loc="center")
        fplt.add_compass(ax, 0.9, 0.82, size=18)
        add_scale_bar(ax)

    # Draw a vector colorbar to keep the PDF editable in Adobe Illustrator.
    n_steps = 160
    step = (vmax - vmin) / n_steps
    for j in range(n_steps):
        value = vmin + j * step
        cax.add_patch(
            Rectangle(
                (value, 0),
                step,
                1,
                facecolor=cmap(norm(value + step / 2)),
                edgecolor="none",
            )
        )
    cax.set_xlim(vmin, vmax)
    cax.set_ylim(0, 1)
    cax.set_yticks([])
    cax.set_xlabel("MODIS summer LST (°C)", fontsize=13)
    cax.set_xticks(np.linspace(vmin, vmax, 5))
    cax.tick_params(axis="x", labelsize=12, length=3)
    for label in cax.get_xticklabels():
        label.set_fontname("Times New Roman")
    cax.spines["top"].set_visible(False)
    cax.spines["left"].set_visible(False)
    cax.spines["right"].set_visible(False)
    cax.spines["bottom"].set_linewidth(0.8)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.06, wspace=0.02, hspace=0.08)
    fig.savefig(OUT_DIR / "Figure_3_3_MODIS_LST_maps.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_3_3_MODIS_LST_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    stats = []
    for year in YEARS:
        s = gdf[f"LST_{year}"]
        stats.append(
            {
                "year": year,
                "n_valid": int(s.notna().sum()),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
                "p02": float(s.quantile(0.02)),
                "p98": float(s.quantile(0.98)),
            }
        )
    pd.DataFrame(stats).to_csv(OUT_DIR / "Figure_3_3_MODIS_LST_map_stats.csv", index=False, encoding="utf-8-sig")
    print(f"Saved outputs to: {OUT_DIR}")
    print(f"Common color scale: {vmin:.2f} to {vmax:.2f} °C")


if __name__ == "__main__":
    main()
