import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import geopandas as gpd
from pathlib import Path
import cartopy.crs as ccrs

# Import frykit
import frykit.plot as fplt
import frykit

# ================= Key settings for Illustrator-editable PDF =================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']


# ================= 1. Paths and configuration =================
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'data'
boundary_dir = data_dir / 'boundary'
landuse_gpkg = (
    data_dir
    / 'mlp_multiyear_5class_joint_tculuWeak_manual2005_gaia_thr10'
    / 'shenzhen_5class_joint_mlp_multiyear.gpkg'
)
landuse_layer = 'pred5_joint_mlp'
output_dir = project_root / 'figures'


shp_path = boundary_dir / 'sz_street_2_Dissolve_polygon.shp'
years = [2000, 2005, 2010, 2015, 2020]
titles = ['(a) 2000', '(b) 2005', '(c) 2010', '(d) 2015', '(e) 2020']
year_cols = {year: f'mlp_pred5_{year}' for year in years}

# ================= 2. Color configuration =================
# Five land-use classes: farmland / green_space / living / production / water
classes = ['farmland', 'green_space', 'living', 'production', 'water']
color_list = ["#eec747", "#8bd486", "#e95b5b", "#b60add", "#4aa3df"]
labels = ['Farmland', 'Green Space', 'Living Land', 'Production Land', 'Water Bodies']
class_colors = dict(zip(classes, color_list))
cmap = mcolors.ListedColormap(color_list)
norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

# ================= 3. Read vector boundary =================
print("Reading Shenzhen boundary shapefile...")
try:
    # Keep WGS84 longitude/latitude when reading.
    sz_gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
    sz_polygon = sz_gdf.geometry.union_all()
    # Get original longitude/latitude extent.
    minx, miny, maxx, maxy = sz_polygon.bounds
except Exception as e:
    print(f"Failed to read shapefile: {e}")
    exit()

# CRS of the original data, in longitude/latitude.
data_crs = ccrs.PlateCarree()

# ================= 4. Read land-use data
print("Reading five-class land-use GPKG...")
try:
    landuse_gdf = gpd.read_file(landuse_gpkg, layer=landuse_layer).to_crs(epsg=4326)
    missing_cols = [col for col in year_cols.values() if col not in landuse_gdf.columns]
    if missing_cols:
        raise ValueError(f"MissingFields: {missing_cols}")
except Exception as e:
    print(f"Failed to read land-use GPKG: {e}")
    exit()

# ================= 5. Set map projection
# Use Transverse Mercator centered near Shenzhen longitude.
# Map units are meters, so scale-bar calculations are accurate.
map_crs = ccrs.TransverseMercator(central_longitude=114.0, central_latitude=22.5)

# ================= 6. Main plotting logic
fig, axes = plt.subplots(2, 3, figsize=(18, 9), 
                         subplot_kw={'projection': map_crs})
axes = axes.flatten() 

print("Start plotting...")

for i, year in enumerate(years):
    ax = axes[i]

    # --- A. Set extent and remove frame ---
    # set_extent uses data_crs; cartopy converts to map_crs.
    ax.set_extent([minx-0.01, maxx+0.01, miny-0.01, maxy+0.01], crs=data_crs)
    ax.axis('off')

    # --- B. Plot GPKG polygons ---
    col = year_cols[year]
    for class_name in classes:
        geoms = landuse_gdf.loc[landuse_gdf[col] == class_name, 'geometry']
        if geoms.empty:
            continue
        fplt.add_geometries(
            ax,
            geoms,
            crs=data_crs,
            fc=class_colors[class_name],
            ec='none',
            lw=0,
        )
    
    # Add boundary outline
    fplt.add_geometries(ax, sz_polygon, crs=data_crs, fc='none', ec='black', lw=0.5)

    # --- C. Set title ---
    ax.set_title(titles[i], fontsize=24, pad=3, loc='center')
    
    # --- D. North arrow, shown only on the third panel ---
    if i == 2:
        # Position uses axes coordinates, unaffected by projection.
        fplt.add_compass(ax, 0.9, 0.8, size=25)

    # --- E. Scale bar ---
    # Map units are meters, so length=5000 means 5000 m.
    # Position uses relative axes coordinates.
    scale_bar = fplt.add_scale_bar(ax, 0.5, 0.1, length=10) 
    scale_bar.set_xticks([0, 10]) 
    scale_bar.set_xticklabels(['0', '10 km']) 
    
    # Minor scale-bar style adjustments
    scale_bar.xaxis.get_label().set_fontsize(16)
    scale_bar.tick_params(labelsize=16, length=2) # Tick label size and tick length
    # Slightly thicken the scale-bar line for readability.
    for spine in scale_bar.spines.values():
        spine.set_linewidth(1.0)

# ================= 7. Legend and layout
legend_patches = [mpatches.Patch(color=color_list[i], label=labels[i]) for i in range(len(classes))]
ax_legend = axes[5]
ax_legend.axis('off') 
ax_legend.legend(handles=legend_patches, loc='center', 
                 fontsize=24, title="Legend", title_fontsize=24, frameon=False)

# Compact layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.02, hspace=0.05)

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'Fig3_Shenzhen_LandUse_Frykit_Projected.pdf'
print(f"Saving figure to: {output_path}")
plt.savefig(output_path, format='pdf', bbox_inches='tight', transparent=True, dpi=300)
# plt.close(fig)
