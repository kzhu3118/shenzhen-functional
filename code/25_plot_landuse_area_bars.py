import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================= Key settings for Illustrator-editable PDF and fonts =================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.compression'] = 0
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']


# ================= 1. Paths and parameters =================
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'data'
landuse_gpkg = (
    data_dir
    / 'mlp_multiyear_5class_joint_tculuWeak_manual2005_gaia_thr10'
    / 'shenzhen_5class_joint_mlp_multiyear.gpkg'
)
landuse_layer = 'pred5_joint_mlp'
output_dir = project_root / 'figures'

years = [2000, 2005, 2010, 2015, 2020]
year_cols = {year: f'mlp_pred5_{year}' for year in years}

type_labels = ['Farmland', 'Green Space', 'Living Land', 'Production Land', 'Water Bodies']
class_map = {
    'farmland': 'Farmland',
    'green_space': 'Green Space',
    'living': 'Living Land',
    'production': 'Production Land',
    'water': 'Water Bodies',
}
type_colors = {
    'Farmland': '#eec747',
    'Green Space': '#8bd486',
    'Living Land': '#e95b5b',
    'Production Land': '#b60add',
    'Water Bodies': '#4aa3df',
}


# ================= 2. Area statistics
stats_list = []
print("Calculating areas...")

try:
    landuse_gdf = gpd.read_file(landuse_gpkg, layer=landuse_layer)
    missing_cols = [col for col in year_cols.values() if col not in landuse_gdf.columns]
    if missing_cols:
        raise ValueError(f"MissingFields: {missing_cols}")
except Exception as e:
    print(f"Failed to read land-use GPKG: {e}")
    exit()

# The GPKG is EPSG:32649, so areas can be calculated directly in square meters.
area_gdf = landuse_gdf if not landuse_gdf.crs.is_geographic else landuse_gdf.to_crs(epsg=32649)
area_km2 = area_gdf.geometry.area / 1e6

for year, col in year_cols.items():
    for class_name, type_name in class_map.items():
        area = area_km2[area_gdf[col] == class_name].sum()
        stats_list.append({'Year': year, 'Type': type_name, 'Area_km2': area})

df = pd.DataFrame(stats_list)
print("Area statistics complete; first five rows:")
print(df.head())


# ================= 3. Plot summary charts
fig = plt.figure(figsize=(12, 5.8), constrained_layout=False)
gs = fig.add_gridspec(1, 5)
ax1 = fig.add_subplot(gs[0, 0:3])
ax2 = fig.add_subplot(gs[0, 3:5])

font_kwargs = {'fontname': 'Times New Roman'}

# --- Figure 1: grouped bar chart
x = np.arange(len(type_labels))
width = 0.15
year_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(years)))

for i, year in enumerate(years):
    year_data = df[df['Year'] == year].set_index('Type').reindex(type_labels)['Area_km2']
    offset = (i - len(years) / 2 + 0.5) * width
    rects = ax1.bar(
        x + offset,
        year_data,
        width,
        label=str(year),
        color=year_colors[i],
        edgecolor='white',
        linewidth=0.5,
    )
    ax1.bar_label(rects, fmt='%.1f', padding=2, fontsize=7, fontname='Times New Roman')

ax1.set_xticks(x)
ax1.set_xticklabels(type_labels, fontsize=11, rotation=18, ha='right', **font_kwargs)
ax1.set_ylabel('Area (km²)', fontsize=12, **font_kwargs)
ax1.set_title('(a)', fontsize=14, pad=15, loc='left', **font_kwargs)
ax1.set_ylim(0, df['Area_km2'].max() * 1.12)
ax1.tick_params(axis='both', labelsize=11)
for label in ax1.get_yticklabels():
    label.set_fontname('Times New Roman')
ax1.legend(
    title='Year',
    fontsize=10,
    title_fontsize=10,
    frameon=False,
    prop={'family': 'Times New Roman', 'size': 10},
)
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Figure 2: percent stacked bar chart
df_pivot = df.pivot(index='Year', columns='Type', values='Area_km2').reindex(columns=type_labels)
df_percent = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100

bottom = np.zeros(len(years))
for type_name in type_labels:
    values = df_percent[type_name].values
    bars = ax2.bar(
        years,
        values,
        bottom=bottom,
        label=type_name,
        color=type_colors[type_name],
        width=3,
        edgecolor='white',
        linewidth=0.5,
    )

    for rect, value in zip(bars, values):
        height = rect.get_height()
        if value > 3:
            ax2.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_y() + height / 2,
                f'{value:.1f}%',
                ha='center',
                va='center',
                color='black',
                fontsize=9,
                fontname='Times New Roman',
            )
    bottom += values

ax2.set_xticks(years)
ax2.set_xticklabels([str(year) for year in years], fontsize=11, **font_kwargs)
ax2.set_xlabel('Year', fontsize=12, **font_kwargs)
ax2.set_ylabel('Percentage (%)', fontsize=12, **font_kwargs)
ax2.set_ylim(0, 100)
ax2.set_title('(b)', fontsize=14, pad=15, loc='left', **font_kwargs)
ax2.tick_params(axis='both', labelsize=11)
for label in ax2.get_yticklabels():
    label.set_fontname('Times New Roman')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Place the land-use legend at the bottom with consistent font styling.
handles, legend_labels = ax2.get_legend_handles_labels()
fig.legend(
    handles[::-1],
    legend_labels[::-1],
    title=' ',
    loc='lower center',
    bbox_to_anchor=(0.5, 0.1),
    ncol=len(type_labels),
    frameon=False,
    prop={'family': 'Times New Roman', 'size': 11},
    title_fontproperties={'family': 'Times New Roman', 'size': 11},
)

fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.27, wspace=0.35)

output_dir.mkdir(parents=True, exist_ok=True)
output_img = output_dir / 'Fig4_Shenzhen_LandUse_Analysis_Charts_Labeled.pdf'
plt.savefig(output_img, format='pdf', bbox_inches='tight')
plt.close(fig)
print(f"Chart saved to: {output_img}")
