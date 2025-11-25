
from gerrychain import  Graph

import json

import geopandas

shp_file = geopandas.read_file('NC_precs_all_data.shp')
shp_file.to_file('/Users/macbarnes/PycharmProjects/GerryChain/NC_precs_all_data.geojson', driver='GeoJSON')

#
# nc_graph = Graph.from_file(r"/Users/macbarnes/Desktop/Data/NC_precs/NC_precs_all_data.shp")
# total_nodes = len(nc_graph.nodes)
# precinct_identifier = [nc_graph.nodes[y]["loc_prec"] for y in range(total_nodes)]
# json_with_data = {
#     "Precint_ID": precinct_identifier
# }
#
# with open("PrecintNames.json", "w") as data_file:
#     json.dump(json_with_data, data_file, indent=4)



# save as geojson
# nc_precs_dissolve.to_file('C:/Users/amand/Downloads/NC_precs_all_data.geojson', driver='GeoJSON')
# # save as shapefile
# nc_precs_dissolve.to_file('C:/Users/amand/Downloads/NC_precs_all_data.shp')