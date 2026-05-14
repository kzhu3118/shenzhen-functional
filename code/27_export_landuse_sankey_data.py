import geopandas as gpd
import pandas as pd
from pathlib import Path


# ================= 1. Paths and configuration =================
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'data'
landuse_gpkg = (
    data_dir
    / 'mlp_multiyear_5class_joint_tculuWeak_manual2005_gaia_thr10'
    / 'shenzhen_5class_joint_mlp_multiyear.gpkg'
)
landuse_layer = 'pred5_joint_mlp'
fig_dir = project_root / 'figures'

years = [2000, 2005, 2010, 2015, 2020]
year_cols = {year: f'mlp_pred5_{year}' for year in years}
min_area_km2 = 0.05

class_labels = {
    'farmland': 'Farmland',
    'green_space': 'Green Space',
    'living': 'Living',
    'production': 'Production',
    'water': 'Water',
}
classes = list(class_labels.keys())


# ================= 2. Read data
print("Reading land-use GPKG...")
try:
    gdf = gpd.read_file(landuse_gpkg, layer=landuse_layer)
    missing_cols = [col for col in year_cols.values() if col not in gdf.columns]
    if missing_cols:
        raise ValueError(f"MissingFields: {missing_cols}")
except Exception as e:
    print(f"Failed to read land-use GPKG: {e}")
    exit()

# The GPKG is EPSG:32649, so areas can be calculated directly.
area_gdf = gdf if not gdf.crs.is_geographic else gdf.to_crs(epsg=32649)
area_gdf = area_gdf.copy()
area_gdf['area_km2'] = area_gdf.geometry.area / 1e6


# ================= 3. Calculate Sankey transition results
print("Calculating Sankey transition results...")
rows = []

for start_year, end_year in zip(years[:-1], years[1:]):
    start_col = year_cols[start_year]
    end_col = year_cols[end_year]

    transitions = (
        area_gdf.groupby([start_col, end_col], dropna=False)['area_km2']
        .sum()
        .reset_index()
    )

    for row in transitions.itertuples(index=False):
        source_class = getattr(row, start_col)
        target_class = getattr(row, end_col)
        area = row.area_km2

        # Match the Sankey figure: output only actual transition flows.
        if source_class == target_class:
            continue
        if source_class not in classes or target_class not in classes:
            continue
        if area <= min_area_km2:
            continue

        rows.append({
            'Period': f'{start_year}-{end_year}',
            'Start Year': start_year,
            'End Year': end_year,
            'Source Class': source_class,
            'Target Class': target_class,
            'Source Type': class_labels[source_class],
            'Target Type': class_labels[target_class],
            'Area (km2)': area,
            'Sankey Label': f'{area:.1f}',
        })

df_changes = pd.DataFrame(rows)
df_changes = df_changes.sort_values(
    ['Start Year', 'Source Type', 'Target Type'],
    kind='stable',
).reset_index(drop=True)


# ================= 4. Print and export
print("\n--- Sankey transition data（actual transitions only；area unit：km²） ---")
print(
    df_changes[
        ['Period', 'Source Type', 'Target Type', 'Area (km2)', 'Sankey Label']
    ].to_string(
        index=False,
        formatters={'Area (km2)': '{:.4f}'.format},
    )
)

summary = (
    df_changes.groupby('Period')['Area (km2)']
    .agg(['count', 'sum'])
    .rename(columns={'count': 'Flow Count', 'sum': 'Changed Area (km2)'})
)
print("\n--- Summary by period ---")
print(summary.to_string(formatters={'Changed Area (km2)': '{:.4f}'.format}))

fig_dir.mkdir(parents=True, exist_ok=True)
output_csv = fig_dir / 'Fig5_Shenzhen_LandUse_Sankey_Transition_Data.csv'
df_changes.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"\nSankey transition data saved to: {output_csv}")
