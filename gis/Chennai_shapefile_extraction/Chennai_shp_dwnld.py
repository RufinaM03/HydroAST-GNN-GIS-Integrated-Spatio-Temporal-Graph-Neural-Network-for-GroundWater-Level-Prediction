import ee
ee.Authenticate()
ee.Initialize()

# ----------------------------
# Load your AOI from GEE assets
# ----------------------------
aoi = ee.FeatureCollection(
    "projects/active-freehold-465811-h1/assets/chennai_shapefile_boundary"
)

# ----------------------------
# Export to Google Drive
# ----------------------------
task = ee.batch.Export.table.toDrive(
    collection=aoi,
    description='chennai_shapefile_export',
    fileFormat='SHP',
    folder='GEE_Exports'   # folder inside Google Drive (must exist)
)

task.start()

print("Export started. Check Google Drive → GEE_Exports folder.")
