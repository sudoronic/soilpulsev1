import os
import json
import ee
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import transform

# ==========================================================
# 1. CONFIGURATION
# ==========================================================
# Choose AOI_MODE: "file" (.shp / .geojson) OR "coords" (list of [lon, lat] points)
AOI_MODE = "file"

# Option A: File path (.shp, .geojson, .json)
AOI_FILE = r"D:\Data\NASA Space APP\Farm.geojson"

# Option B: Corner coordinates [lon, lat] (used if AOI_MODE = "coords")
AOI_COORDS = [
    [73.65, 30.90],
    [73.72, 30.90],
    [73.72, 30.96],
    [73.65, 30.96],
    [73.65, 30.90],
]

# Output Directory
OUTPUT_DIR = r"D:\Data\NASA Space APP\Output Bare fields"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "fields_report.csv")
OUTPUT_IMG = os.path.join(OUTPUT_DIR, "fields_map.png")

# Analysis Parameters
START_DATE = "2024-01-01"
END_DATE = "2026-11-14"
NDVI_THRESHOLD = 0.20             # Below 0.20 = Bare | 0.20 and above = Cultivated
MIN_FIELD_AREA_HA = 0.20          # Ignore tiny noise slivers (< 0.20 ha)

# SNIC Field Delineation Tuning
SNIC_SIZE = 10                    # Seed spacing in pixels (~100m at 10m resolution)
SNIC_COMPACTNESS = 1              # Shape regularity (1-2 works well for rectangular fields)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ==========================================================
# 2. INITIALIZE GOOGLE EARTH ENGINE
# ==========================================================
def init_ee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


# ==========================================================
# 3. UNIVERSAL AOI LOADER (File, Coordinates, or Web Map GeoJSON)
# ==========================================================
def load_aoi(aoi_source):
    """
    Accepts:
      1. File path string (.shp, .geojson, .json)
      2. List of [lon, lat] coordinate pairs
      3. GeoJSON dict / Feature / FeatureCollection (from interactive web map draw tool)
    """
    if isinstance(aoi_source, str):
        if not os.path.exists(aoi_source):
            raise FileNotFoundError(f"AOI file not found: {os.path.abspath(aoi_source)}")
        gdf = gpd.read_file(aoi_source)

    elif isinstance(aoi_source, list):
        if len(aoi_source) < 3:
            raise ValueError("At least 3 coordinate pairs are required to form a polygon.")
        if aoi_source[0] != aoi_source[-1]:
            aoi_source = aoi_source + [aoi_source[0]]
        gdf = gpd.GeoDataFrame({"geometry": [Polygon(aoi_source)]}, crs="EPSG:4326")

    elif isinstance(aoi_source, dict):
        # Handles GeoJSON drawn directly on a web dashboard (Leaflet / Mapbox / OpenLayers)
        if aoi_source.get("type") == "FeatureCollection":
            gdf = gpd.GeoDataFrame.from_features(aoi_source["features"], crs="EPSG:4326")
        elif aoi_source.get("type") == "Feature":
            gdf = gpd.GeoDataFrame.from_features([aoi_source], crs="EPSG:4326")
        else:
            gdf = gpd.GeoDataFrame({"geometry": [shape(aoi_source)]}, crs="EPSG:4326")
    else:
        raise TypeError("Unsupported AOI input type. Pass a file path, coordinate list, or GeoJSON dict.")

    if gdf.empty:
        raise ValueError("AOI contains no valid features.")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise ValueError("No valid Polygon or MultiPolygon found in AOI.")

    # Repair self-intersections and strip 3D Z coordinates
    gdf["geometry"] = gdf.geometry.buffer(0)
    merged = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union
    merged_2d = transform(lambda x, y, z=None: (x, y), merged)

    if not isinstance(merged_2d, (Polygon, MultiPolygon)):
        raise ValueError(f"Invalid AOI geometry type: {type(merged_2d)}")

    aoi_gdf = gpd.GeoDataFrame({"geometry": [merged_2d]}, crs="EPSG:4326")
    ee_aoi = ee.Geometry(mapping(merged_2d), proj="EPSG:4326", geodesic=False)
    return aoi_gdf, ee_aoi


# ==========================================================
# 4. MAIN PROCESSING PIPELINE
# ==========================================================
def run_pipeline(aoi_source, save_files=True):
    init_ee()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    aoi_gdf, ee_aoi = load_aoi(aoi_source)
    aoi_area_ha = ee_aoi.area(maxError=1).divide(10000).getInfo()
    print(f"[1/5] AOI loaded ({aoi_area_ha:.2f} ha)")

    # A. Isolate Cropland using ESA WorldCereal (removes roads, villages, canals)
    wc = ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
    crop_mask = (
        wc.filter(ee.Filter.eq("product", "temporarycrops"))
        .mosaic()
        .select("classification")
        .eq(100)
        .clip(ee_aoi)
    )

    # B. Build Cloud- & Water-Masked Sentinel-2 NDVI Collection
    def mask_s2(img):
        scl = img.select("SCL")
        clear = (
            scl.neq(3)         # Cloud shadow
            .And(scl.neq(6))   # Water
            .And(scl.neq(8))   # Cloud medium prob
            .And(scl.neq(9))   # Cloud high prob
            .And(scl.neq(10))  # Cirrus
            .And(scl.neq(11))  # Snow/ice
        )
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return ndvi.updateMask(clear).copyProperties(img, ["system:time_start"])

    s2_ndvi = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ee_aoi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(mask_s2)
    )

    # C. Build 12-Month Median NDVI Stack & Delineate Fields via SNIC
    print("[2/5] Delineating individual fields via multi-temporal SNIC...")
    months = ee.List.sequence(1, 12)

    def monthly_median(m):
        m = ee.Number(m)
        col = s2_ndvi.filter(ee.Filter.calendarRange(m, m, "month"))
        return col.median().unmask(0).rename(ee.String("NDVI_").cat(m.format("%d")))

    ndvi_stack = ee.ImageCollection(months.map(monthly_median)).toBands().updateMask(crop_mask)

    seeds = ee.Algorithms.Image.Segmentation.seedGrid(SNIC_SIZE)
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=ndvi_stack,
        size=SNIC_SIZE,
        compactness=SNIC_COMPACTNESS,
        connectivity=8,
        neighborhoodSize=128,
        seeds=seeds,
    )

    clusters = snic.select("clusters").updateMask(crop_mask)

    polygons = clusters.reduceToVectors(
        geometry=ee_aoi,
        scale=10,
        geometryType="polygon",
        eightConnected=False,
        maxPixels=1e13,
        bestEffort=True,
    )

    polygons = polygons.map(
        lambda f: f.set("area_ha", f.geometry().area(maxError=1).divide(10000))
    ).filter(ee.Filter.gte("area_ha", MIN_FIELD_AREA_HA))

    # D. Compute Current NDVI (latest 3 clear images median) + Monthly Bare Frequency
    print("[3/5] Calculating NDVI and bare/cultivated status per field...")
    latest_ndvi = (
        s2_ndvi.sort("system:time_start", False)
        .limit(3)
        .median()
        .rename("mean_ndvi")
    )

    monthly_bands = []
    for m in range(1, 13):
        m_col = s2_ndvi.filter(ee.Filter.calendarRange(m, m, "month"))
        bare_pct = (
            m_col.map(lambda img: img.lt(NDVI_THRESHOLD))
            .mean()
            .multiply(100)
            .rename(MONTH_NAMES[m - 1])
        )
        monthly_bands.append(bare_pct)

    stats_img = latest_ndvi.addBands(ee.Image.cat(monthly_bands))

    fields_fc = stats_img.reduceRegions(
        collection=polygons,
        reducer=ee.Reducer.mean(),
        scale=10,
        tileScale=4,
    ).getInfo()

    # E. Format Data for CSV, Map Image, and Web Dashboard JSON
    records = []
    geo_rows = []

    for idx, feat in enumerate(fields_fc["features"], start=1):
        geom = shape(feat["geometry"])
        props = feat["properties"]

        centroid = geom.centroid
        area_ha = round(props.get("area_ha") or 0.0, 3)
        ndvi_val = props.get("mean_ndvi")

        if ndvi_val is None:
            status = "No Data"
        elif ndvi_val < NDVI_THRESHOLD:
            status = "Bare"
        else:
            status = "Cultivated"

        monthly_dict = {}
        typical_bare = []
        for m in MONTH_NAMES:
            val = props.get(m)
            if val is not None:
                monthly_dict[m] = round(val, 1)
                if val >= 50.0:
                    typical_bare.append(m)
            else:
                monthly_dict[m] = None

        row = {
            "field_id": idx,
            "center_lon": round(centroid.x, 6),
            "center_lat": round(centroid.y, 6),
            "area_ha": area_ha,
            "mean_ndvi": round(ndvi_val, 3) if ndvi_val is not None else None,
            "status": status,
            "typical_bare_months": ", ".join(typical_bare) if typical_bare else "Rarely bare",
            **monthly_dict,
        }
        records.append(row)
        geo_rows.append({"geometry": geom, **row})

    if not records:
        print("No fields detected in this AOI.")
        return None, None

    df = pd.DataFrame(records)
    fields_gdf = gpd.GeoDataFrame(geo_rows, geometry="geometry", crs="EPSG:4326")

    if save_files:
        # ==========================================================
        # OUTPUT 1: CSV REPORT
        # ==========================================================
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"[4/5] Saved CSV report -> {OUTPUT_CSV}")

        # ==========================================================
        # OUTPUT 2: MAP IMAGE (PNG)
        # ==========================================================
        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)

        aoi_gdf.boundary.plot(ax=ax, color="#1e293b", linewidth=1.8, linestyle="--")

        color_map = {
            "Bare": "#d97706",
            "Cultivated": "#16a34a",
            "No Data": "#94a3b8",
        }
        plot_colors = fields_gdf["status"].map(color_map)

        fields_gdf.plot(
            ax=ax,
            color=plot_colors,
            edgecolor="#ffffff",
            linewidth=0.25,
            alpha=0.85,
        )

        # Label ONLY the top 30 largest bare fields so the map stays clean
        bare_labels = fields_gdf[fields_gdf["status"] == "Bare"].nlargest(30, "area_ha")
        for _, r in bare_labels.iterrows():
            ax.annotate(
                text=str(r["field_id"]),
                xy=(r["center_lon"], r["center_lat"]),
                ha="center",
                va="center",
                fontsize=7,
                color="#ffffff",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#9a3412", edgecolor="none", alpha=0.85),
            )

        bare_count = (df["status"] == "Bare").sum()
        cult_count = (df["status"] == "Cultivated").sum()
        bare_ha = df.loc[df["status"] == "Bare", "area_ha"].sum()
        cult_ha = df.loc[df["status"] == "Cultivated", "area_ha"].sum()

        legend_handles = [
            Patch(facecolor="#d97706", edgecolor="white", label=f"Bare (NDVI < {NDVI_THRESHOLD}): {bare_count} fields ({bare_ha:.1f} ha)"),
            Patch(facecolor="#16a34a", edgecolor="white", label=f"Cultivated (NDVI >= {NDVI_THRESHOLD}): {cult_count} fields ({cult_ha:.1f} ha)"),
            Patch(facecolor="none", edgecolor="#1e293b", linestyle="--", label="AOI Boundary"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

        ax.set_title(
            f"Delineated Fields: Bare vs. Cultivated\n"
            f"Total Fields: {len(df)} | Bare: {bare_count} ({bare_ha:.1f} ha) | Cultivated: {cult_count} ({cult_ha:.1f} ha)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", alpha=0.4)

        plt.tight_layout()
        plt.savefig(OUTPUT_IMG, dpi=200)
        plt.close(fig)
        print(f"[5/5] Saved map image -> {OUTPUT_IMG}")

        print("\n--- SUMMARY ---")
        print(f"Total Fields Detected : {len(df)}")
        print(f"Bare Fields           : {bare_count} ({bare_ha:.2f} ha)")
        print(f"Cultivated Fields     : {cult_count} ({cult_ha:.2f} ha)")

    return fields_gdf, df


if __name__ == "__main__":
    if AOI_MODE == "file":
        run_pipeline(AOI_FILE)
    elif AOI_MODE == "coords":
        run_pipeline(AOI_COORDS)
    else:
        raise ValueError("AOI_MODE must be 'file' or 'coords'")