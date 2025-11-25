import streamlit as st
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import pandas as pd

primaryColor="#FF8B3D"

# DEFINITIONS
def model_predict():
    user_input = [cut_edges_input, pf_input]
    user_input = tf.convert_to_tensor(user_input)
    user_input = user_input * 0.01
    user_input = tf.reshape(user_input, [1, 2])
    noise = tf.random.uniform(shape=[1, 64])
    prediction = decoder.predict([noise, user_input])
    prediction = np.argmax(prediction, axis=-1)
    prediction = prediction + 1
    prediction = prediction.numpy()
    return prediction

def generate_plan():
    precint_names = pd.read_json("PrecintNames.json")
    pef = [precint_names, model_predict()]
    nc_precs = gpd.read_file('NC_precs_all_data.geojson')
    nc_precs_join = nc_precs.merge(pef, left_on='loc_prec', right_on='Precint_ID', how='left' )
    nc_precs_dissolve = nc_precs_join.dissolve(by='assignment')
    nc_precs_dissolve['assign'] = nc_precs_dissolve.index.copy()
    return nc_precs_dissolve

def create_map():
    m = folium.Map(location=[35.6516, -80.3018], tiles="CartoDB positron", name="Light Map",
                   zoom_start=7, attr="My Data Attribution")
    folium.Choropleth(
        geo_data=nc_precs_dissolve,
        name="choropleth",
        data=nc_precs_dissolve,
        columns=["assign", "ID"],
        key_on='feature.properties.assign',
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.1,
    ).add_to(m)
    return m


# SIDEBAR PREFERENCES

st.sidebar.title("Choose A State:")
state_mode = st.sidebar.selectbox(
    'What state would you like to generate a map for?',
    ['North Carolina']
)
st.sidebar.title("Decide Your Metrics:")
st.sidebar.subheader("Cut Edges")
cut_edges_input = st.sidebar.slider(
    'Select a cut edges score',
    574.0, 621.0
)
st.sidebar.subheader("Mean-Median")
pf_input = st.sidebar.number_input(
    "Desired mean-median score"
)
st.sidebar.title("Generate Map:")
generate = st.sidebar.button("Generate Map")
st.sidebar.header("Download Map")
shape_dl = st.sidebar.checkbox("Download .SHP")
geojson_dl = st.sidebar.checkbox("Download .GEOJSON")
pef_dl = st.sidebar.checkbox("Download Equivalency File")
download = st.sidebar.button("Download Files")
st.markdown("<h1 style='text-align: center; color: black;"
            "'>Set Preferences and Click Generate Map to see Districting Plan 🗺️ </h1>",
            unsafe_allow_html=True)


# read in precinct shapefile
nc_precs = gpd.read_file('NC_precs_all_data.geojson')
# read in assignment file
test_PEC = pd.read_csv('test_PEC.csv')
# check for unique ID column to match length of precinct df
nc_precs['loc_prec'].nunique()
len(nc_precs)


# join assignment csv to precinct shapefile
nc_precs_join = nc_precs.merge(test_PEC, left_on='loc_prec', right_on='ID', how='left')
# geographic dissolve of precincts based on assignment column
nc_precs_dissolve = nc_precs_join.dissolve(by='assignment')
nc_precs_dissolve['assign'] = nc_precs_dissolve.index.copy()
null_columns=nc_precs_dissolve.columns[nc_precs_dissolve.isnull().any()]

#
# nc_precs_dissolve.to_file('r"/Users/macbarnes/Desktop/Data/nc_precs_dissolve.geojson"', driver='GeoJSON')
#
# nc_precs_dissolve.to_file('r"/Users/macbarnes/Desktop/NC_precs/nc_precs_dissolve.shp"')


m = folium.Map(location=[35.6516, -80.3018], tiles="CartoDB positron", name="Light Map",
               zoom_start=7, attr="My Data Attribution")
folium.Choropleth(
    geo_data=nc_precs_dissolve,
    name="choropleth",
    data=nc_precs_dissolve,
    columns=["assign", "ID"],
    # key_on='feature.properties.assign',
    fill_color="BuPu",
    fill_opacity=0.7,
    line_opacity=0.1,
).add_to(m)

# folium.features.GeoJson('nc_precs_dissolve.geojson',
#                         name="States", popup=folium.features.GeoJsonPopup(field=["assignment"])).add_to(m)

folium_static(m, width=850, height=500)