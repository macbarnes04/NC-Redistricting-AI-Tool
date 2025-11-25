import geopandas as gpd
import matplotlib.pyplot as plt

# Load GeoJSON file into a GeoDataFrame
gdf = gpd.read_file('"/Users/macbarnes/Downloads/mo_vtds_geo.geojson"')

# Plot the GeoDataFrame
gdf.plot()
plt.title('GeoJSON Data Visualization')
plt.show()