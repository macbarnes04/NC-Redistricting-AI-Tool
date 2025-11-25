import streamlit as st
import tensorflow as tf
import numpy as np
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import pandas as pd

# ### STREAMLIT SETUP
#
# primaryColor="#FF8B3D"
#
# # SIDEBAR PREFERENCES
# st.sidebar.title("Choose A State:")
#
state_mode = st.sidebar.selectbox(
    'What state would you like to generate a map for?',
    ['North Carolina']
)
#
# st.sidebar.title("Decide Your Metrics:")
#
# st.sidebar.subheader("Cut Edges")
#
# cut_edges_input = st.sidebar.slider(
#     'Select a cut edges score',
#     574.0, 621.0
# )
#
# st.sidebar.subheader("Mean Median")
#
# MM_input = st.sidebar.number_input(
#     "Desired Mean Median score"
# )
#
# st.sidebar.title("Generate Map:")
# generate = st.sidebar.button("Generate Map")
# st.sidebar.header("Download Map")
# shp_dl = st.sidebar.checkbox("Download .SHP")
# geojson_dl = st.sidebar.checkbox("Download .GEOJSON")
# pef_dl = st.sidebar.checkbox("Download Equivalency File")
# download = st.sidebar.button("Download Files")
#
# st.markdown("<h1 style='text-align: center; color: black;"
#             "'>Set Preferences and Click Generate Map to see Districting Plan 🗺️ </h1>",
#             unsafe_allow_html=True)

MM_input = 0.067
cut_edges_input = 500

def prediction(MM_value, CE_value):
    metrics_matrix = [MM_value, CE_value]
    metrics = tf.convert_to_tensor(metrics_matrix)
    metrics = tf.reshape(
        metrics, shape=(1, 2), name=None
    )
    print(metrics)

    latent_tensor = tf.random.normal(shape=(1024,), mean=0, stddev=2, dtype=tf.float32, seed=None, name=None)
    latent_tensor = tf.reshape(
        latent_tensor, shape=(1, 1024), name=None
    )
    print(latent_tensor)

    decoder_inputs = [
        latent_tensor,
        metrics
    ]

    ML_model = tf.keras.models.load_model('mulligans_decoder.h5')
    output = ML_model.predict(decoder_inputs)
    n = np.squeeze(np.argmax(output, axis=-1).tolist())
    # i = range(1, len(n)+1)
    # pd.concat(i, n, headers=[])
    # output = [f.write(f"{i + 1}, {n + 1}" + "\n") for i, n in enumerate(np.argmax(output, axis=-1)[0].tolist())]
    return(n)



### IMPORT

# read in precinct shapefile
nc_precs = gpd.read_file('NC_precs_all_data.shp')
# read in assignment file
test_PEC = pd.read_csv('test_PEC.csv')

### CLEAN

# check for unique ID column to match length of precinct df
nc_precs['loc_prec'].head()
nc_precs['loc_prec'].nunique() #2706=2706
len(nc_precs)

test_PEC['ID'].head()
test_PEC['ID'].nunique() #2706=2706

# subset to only get assignment and ID columns
test_PEC = test_PEC[['assignment', 'ID']]
test_PEC['assignment'].nunique()
test_PEC["assignment"] = prediction(MM_input, cut_edges_input)
# join assignment csv to precinct shapefile
nc_precs_join = nc_precs.merge(test_PEC, left_on='loc_prec', right_on='ID', how='left')

### DISSOLVE

# geographic dissolve of precincts based on assignment column
nc_precs_dissolve = nc_precs_join.dissolve(by='assignment', aggfunc='sum')

nc_precs_dissolve.index.nunique()

nc_precs_dissolve['assign'] = nc_precs_dissolve.index.copy()
null_columns=nc_precs_dissolve.columns[nc_precs_dissolve.isnull().any()]

# change CRS
nc_precs_dissolve = nc_precs_dissolve.to_crs(epsg=4326)

### SAVE

# nc_precs_dissolve.to_file("C:/Users/amand/Documents/PGP/nc_precs_dissolve.geojson", driver='GeoJSON')
# nc_precs_dissolve.to_file("C:/Users/amand/Documents/PGP/NC_precs/nc_precs_dissolve.shp")

### FOLIUM

# create map
m = folium.Map(location=[35.7596, -79.0193], tiles="CartoDB positron", name="Light Map",
               zoom_start=8, attr="My Data Attribution")

# choropleth
folium.Choropleth(
    geo_data=nc_precs_dissolve,
    name='choropleth',
    data=nc_precs_dissolve,
    columns=['assign', 'COUNTY_ID'],
    key_on='feature.properties.assign',
    fill_color='Purples',
    fill_opacity=0.7,
    line_opacity=0.1,
    legend_name=state_mode
).add_to(m)

# folium.features.GeoJson('nc_precs_dissolve.geojson',
#                         name="States", popup=folium.features.GeoJsonPopup(field=["assignment"])).add_to(m)

folium_static(m, width=850, height=500)