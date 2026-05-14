import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
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

classes = ['farmland', 'green_space', 'living', 'production', 'water']
class_labels = {
    'farmland': 'Farmland',
    'green_space': 'Green Space',
    'living': 'Living',
    'production': 'Production',
    'water': 'Water',
}
class_colors = {
    'farmland': 'rgba(238, 199, 71, 1.0)',
    'green_space': 'rgba(139, 212, 134, 1.0)',
    'living': 'rgba(233, 91, 91, 1.0)',
    'production': 'rgba(182, 10, 221, 1.0)',
    'water': 'rgba(74, 163, 223, 1.0)',
}
link_colors = {
    'farmland': 'rgba(238, 199, 71, 0.35)',
    'green_space': 'rgba(139, 212, 134, 0.35)',
    'living': 'rgba(233, 91, 91, 0.35)',
    'production': 'rgba(182, 10, 221, 0.35)',
    'water': 'rgba(74, 163, 223, 0.35)',
}


# ================= 2. Data processing =================
print("Reading land-use data...")
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


# ================= 3. Build Sankey data
source_indices = []
target_indices = []
values = []
colors = []
customdata = []
label_annotations = []

node_labels = []
node_colors = []
node_x = []
node_y = []

x_positions = np.linspace(0.02, 0.98, len(years))
y_positions = np.linspace(0.03, 0.85, len(classes))
node_lookup = {}

for year_index, year in enumerate(years):
    for class_index, class_name in enumerate(classes):
        node_lookup[(year_index, class_name)] = len(node_labels)
        node_labels.append(class_labels[class_name])
        node_colors.append(class_colors[class_name])
        node_x.append(float(x_positions[year_index]))
        node_y.append(float(y_positions[class_index]))

print("Calculating transitions...")
for i in range(len(years) - 1):
    start_year = years[i]
    end_year = years[i + 1]
    start_col = year_cols[start_year]
    end_col = year_cols[end_year]

    transitions = (
        area_gdf.groupby([start_col, end_col], dropna=False)['area_km2']
        .sum()
        .reset_index()
    )

    for row in transitions.itertuples(index=False):
        start_type = getattr(row, start_col)
        end_type = getattr(row, end_col)
        area = row.area_km2

        # Plot and label only actual transitions.
        if start_type == end_type or start_type not in classes or end_type not in classes:
            continue
        if area <= 0.05:
            continue

        source_indices.append(node_lookup[(i, start_type)])
        target_indices.append(node_lookup[(i + 1, end_type)])
        values.append(float(area))
        colors.append(link_colors[start_type])
        customdata.append(
            f"{class_labels[start_type]} to {class_labels[end_type]}<br>"
            f"{start_year}-{end_year}"
        )

        # Plotly Sankey links cannot always show labels, so annotations show transition areas.
        x_mid = (x_positions[i] + x_positions[i + 1]) / 2
        y_mid = 1 - ((y_positions[classes.index(start_type)] + y_positions[classes.index(end_type)]) / 2)
        y_mid += (classes.index(start_type) - classes.index(end_type)) * 0.006
        label_annotations.append(
            dict(
                x=float(x_mid),
                y=float(y_mid),
                xref='paper',
                yref='paper',
                text=f"{area:.1f}",
                showarrow=False,
                font=dict(size=11, color='black', family='Times New Roman'),
                bgcolor='rgba(255,255,255,0.62)',
                borderpad=1,
            )
        )


# ================= 4. Plotting
fig = go.Figure(data=[go.Sankey(
    arrangement='fixed',
    valueformat='.1f',
    valuesuffix=' km²',
    node=dict(
        pad=20,
        thickness=20,
        line=dict(color='white', width=0.5),
        label=node_labels,
        color=node_colors,
        x=node_x,
        y=node_y,
        hovertemplate='%{label}<br>Total: %{value:.1f} km²<extra></extra>',
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=values,
        color=colors,
        customdata=customdata,
        hovertemplate='%{customdata}<br>Change: %{value:.1f} km²<extra></extra>',
    ),
)])

fig.update_layout(
    title_text='Land Use Transitions (2000-2020)',
    title_font=dict(size=24, family='Times New Roman', color='black'),
    font=dict(size=16, family='Times New Roman', color='black'),
    width=1400,
    height=760,
    margin=dict(l=55, r=55, t=110, b=45),
    plot_bgcolor='white',
    paper_bgcolor='white',
)

# Year labels
for i, year in enumerate(years):
    fig.add_annotation(
        x=float(x_positions[i]),
        y=1.07,
        xref='paper',
        yref='paper',
        text=str(year),
        showarrow=False,
        font=dict(size=20, color='black', family='Times New Roman'),
        xanchor='center',
    )

# Transition areas in km2; only numbers are shown on the figure.
for annotation in label_annotations:
    fig.add_annotation(**annotation)


# ================= 5. Save
fig_dir.mkdir(parents=True, exist_ok=True)
output_html = fig_dir / 'Fig5_Shenzhen_LandUse_Sankey.html'
fig.write_html(output_html)
print(f"HTML saved to: {output_html}")
