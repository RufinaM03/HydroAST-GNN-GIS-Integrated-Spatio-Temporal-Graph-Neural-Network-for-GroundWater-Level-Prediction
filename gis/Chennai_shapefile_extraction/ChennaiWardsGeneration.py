import geopandas as gpd

# --------------------------------------------
# 1. Load Shapefile + GeoJSON
# --------------------------------------------
chennai = gpd.read_file("Chennai_Shapefile/chennai_shapefile_export.shp")
wards = gpd.read_file("Wards.geojson")

# --------------------------------------------
# 2. Check CRS
# --------------------------------------------
print("Chennai CRS:", chennai.crs)
print("Wards CRS:", wards.crs)

# --------------------------------------------
# 3. Align CRS (Reproject wards to Chennai CRS)
# --------------------------------------------
if wards.crs != chennai.crs:
    wards = wards.to_crs(chennai.crs)
    print("Wards reprojected to match Chennai CRS")

# --------------------------------------------
# 4. Clip wards to Chennai boundary
# --------------------------------------------
clipped_wards = gpd.clip(wards, chennai)

# --------------------------------------------
# 5. Save output GeoJSON
# --------------------------------------------
output_path = "chennai_wards_clipped.geojson"
clipped_wards.to_file(output_path, driver="GeoJSON")

print(f"Clipped wards saved as: {output_path}")
