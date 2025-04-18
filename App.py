import streamlit as st
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import requests
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Urban Heat Island Analyzer",
    page_icon="🌡️",
    layout="wide"
)

# Title and description
st.title("Urban Heat Island (UHI) Analyzer")
st.markdown("""
This app analyzes Urban Heat Island effects for any US county using Landsat satellite imagery
and the National Land Cover Database (NLCD). Results show temperature differences between urban
areas and natural/vegetative surfaces over time.
""")

# Initialize Earth Engine
@st.cache_resource
def initialize_ee():
    try:
        # This will use the EARTHENGINE_TOKEN from Streamlit secrets
        geemap.ee_initialize()
        return True
    except Exception as e:
        st.error(f"Error initializing Earth Engine: {str(e)}")
        return False

# Display a loading message while initializing Earth Engine
with st.spinner("Initializing Google Earth Engine..."):
    ee_initialized = initialize_ee()
    if ee_initialized:
        st.success("Earth Engine initialized successfully!")

# Load US states and counties data
@st.cache_data
def load_us_states_counties():
    # Get states from Census Bureau API
    states_url = "https://api.census.gov/data/2019/acs/acs1?get=NAME&for=state:*"
    states_response = requests.get(states_url)
    states_data = states_response.json()
    
    # Convert to a dictionary {state_name: state_id}
    states = {row[0]: row[1] for row in states_data[1:]}
    return states

@st.cache_data
def load_counties(state_id):
    # Get counties for the selected state from Census Bureau API
    counties_url = f"https://api.census.gov/data/2019/acs/acs1?get=NAME&for=county:*&in=state:{state_id}"
    counties_response = requests.get(counties_url)
    counties_data = counties_response.json()
    
    # Extract just the county names (remove ", State" suffix)
    counties = {}
    for row in counties_data[1:]:
        full_name = row[0]
        county_name = full_name.split(",")[0]
        county_id = row[2]
        counties[county_name] = county_id
    
    return counties

# Define left sidebar for input controls
st.sidebar.header("Analysis Parameters")

# Load states
states = load_us_states_counties()
state_names = list(states.keys())
selected_state = st.sidebar.selectbox("Select State", state_names)
state_id = states[selected_state]

# Load counties for selected state
counties = load_counties(state_id)
county_names = list(counties.keys())
selected_county = st.sidebar.selectbox("Select County", county_names)
county_id = counties[selected_county]

# Date range selection (restrict to available data: 2000-2024)
st.sidebar.subheader("Time Period")
start_year = st.sidebar.slider("Start Year", 2000, 2024, 2010)
end_year = st.sidebar.slider("End Year", 2000, 2024, 2023)

# Months selection (default to summer months)
month_options = ['01-Jan', '02-Feb', '03-Mar', '04-Apr', '05-May', '06-Jun', 
                '07-Jul', '08-Aug', '09-Sep', '10-Oct', '11-Nov', '12-Dec']
default_months = ['06-Jun', '07-Jul', '08-Aug']
selected_months = st.sidebar.multiselect("Select Months", 
                                        month_options, 
                                        default=default_months)

# Function to get county boundary from Census API and convert to Earth Engine feature
@st.cache_data
def get_county_boundary(state_id, county_id):
    # This API returns county boundaries as GeoJSON
    tiger_url = f"https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/query?where=STATE={state_id}+AND+COUNTY={county_id}&outFields=*&outSR=4326&f=geojson"
    response = requests.get(tiger_url)
    geojson = response.json()
    
    # Save GeoJSON to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(bytes(str(geojson), 'utf-8'))
    
    # Read with GeoPandas and project to UTM
    gdf = gpd.read_file(tmp_path)
    
    # Determine appropriate UTM zone based on county centroid longitude
    lon = gdf.geometry.centroid.x.mean()
    utm_zone = int(((lon + 180) / 6) % 60) + 1
    
    # Project to the appropriate UTM zone
    epsg_code = 32600 + utm_zone  # North UTM zones
    if gdf.geometry.centroid.y.mean() < 0:
        epsg_code = 32700 + utm_zone  # South UTM zones
    
    gdf = gdf.to_crs(epsg=epsg_code)
    
    # Convert GeoDataFrame to ee.FeatureCollection
    aoi = geemap.geopandas_to_ee(gdf)
    
    # Clean up temporary file
    os.unlink(tmp_path)
    
    return aoi, gdf, epsg_code

# Function to apply scale factors for Landsat 8
def apply_scale_factors_landsat8(image):
    # Scale factors for Landsat 8
    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)

# Function to apply scale factors for Landsat 5
def apply_scale_factors_landsat5(image):
    # For Landsat 5 Collection 2 Level 2 data
    sr_b1 = image.select('SR_B1').multiply(0.0000275).add(-0.2)
    sr_b2 = image.select('SR_B2').multiply(0.0000275).add(-0.2)
    sr_b3 = image.select('SR_B3').multiply(0.0000275).add(-0.2)
    sr_b4 = image.select('SR_B4').multiply(0.0000275).add(-0.2)
    sr_b5 = image.select('SR_B5').multiply(0.0000275).add(-0.2)
    sr_b7 = image.select('SR_B7').multiply(0.0000275).add(-0.2)
    
    # For thermal band
    st_b6 = image.select('ST_B6').multiply(0.00341802).add(149.0)
    
    # Create a new multiband image with the scaled bands
    scaled_bands = ee.Image.cat([sr_b1, sr_b2, sr_b3, sr_b4, sr_b5, sr_b7, st_b6])
    
    return image.addBands(scaled_bands, None, True)

# Cloud masking function for Landsat 8
def cloud_mask_landsat8(image):
    cloud_shadow_bit = 1 << 3
    clouds_bit = 1 << 5
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(cloud_shadow_bit).eq(0).And(qa.bitwiseAnd(clouds_bit).eq(0))
    return image.updateMask(mask)

# Cloud masking function for Landsat 5
def cloud_mask_landsat5(image):
    cloud_shadow_bit = 1 << 3
    clouds_bit = 1 << 5
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(cloud_shadow_bit).eq(0).And(qa.bitwiseAnd(clouds_bit).eq(0))
    return image.updateMask(mask)

# Function to calculate cloud cover percentage within the AOI
def calculate_cloud_cover_within_aoi(image, aoi):
    try:
        qa = image.select('QA_PIXEL')
        
        cloud_shadow_bit = 1 << 3
        clouds_bit = 1 << 5
        
        cloud_mask = qa.bitwiseAnd(clouds_bit).neq(0)
        cloud_shadow_mask = qa.bitwiseAnd(cloud_shadow_bit).neq(0)
        
        combined_cloud_mask = cloud_mask.Or(cloud_shadow_mask)
        
        aoi_mask = ee.Image(1).clip(aoi)
        total_pixels = aoi_mask.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi.geometry(),
            scale=30,
            maxPixels=1e9
        ).get('constant').getInfo()
        
        cloudy_pixels = combined_cloud_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi.geometry(),
            scale=30,
            maxPixels=1e9
        ).get('QA_PIXEL').getInfo()
        
        cloud_cover_percentage = (cloudy_pixels / total_pixels) * 100
        
        return cloud_cover_percentage
    except Exception as e:
        st.warning(f"Could not calculate cloud cover within AOI: {str(e)}")
        return 0

# Function to get the best available NLCD data for any year
def get_best_nlcd_for_year(year):
    available_nlcd_years = [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]
    closest_year = min(available_nlcd_years, key=lambda x: abs(x - year))
    
    # For years up to 2019
    if closest_year <= 2019:
        collection = ee.ImageCollection("USGS/NLCD_RELEASES/2019_REL/NLCD")
        return collection.filter(ee.Filter.eq('system:index', str(closest_year))).first().select('landcover'), closest_year
    # For 2021
    elif closest_year == 2021:
        try:
            collection = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
            nlcd = collection.filter(ee.Filter.eq('system:index', '2021')).first()
            if nlcd is not None:
                return nlcd.select('landcover'), 2021
        except:
            pass
        
        collection = ee.ImageCollection("USGS/NLCD_RELEASES/2019_REL/NLCD")
        return collection.filter(ee.Filter.eq('system:index', '2019')).first().select('landcover'), 2019

# Function to filter LST by 5-95 percentile range
def filter_lst_percentiles(lst_image, aoi):
    try:
        percentiles = lst_image.reduceRegion(
            reducer=ee.Reducer.percentile([5, 95]),
            geometry=aoi.geometry(),
            scale=30,
            maxPixels=1e9
        )
        
        p05 = ee.Number(percentiles.get('LST_p5'))
        p95 = ee.Number(percentiles.get('LST_p95'))
        
        percentile_mask = lst_image.gte(p05).And(lst_image.lte(p95))
        
        filtered_lst = lst_image.updateMask(percentile_mask)
        
        return filtered_lst, p05.getInfo(), p95.getInfo()
    except Exception as e:
        st.warning(f"Could not filter LST by percentiles: {str(e)}")
        return lst_image, None, None

# Main processing function
def process_landsat_data(year, aoi, months):
    results = {}
    
    # Filter months to process
    month_numbers = [m.split('-')[0] for m in months]
    
    # Define date ranges based on selected months
    date_ranges = []
    for month in month_numbers:
        start_date = f'{year}-{month}-01'
        
        # Determine end date based on month
        if month in ['04', '06', '09', '11']:
            end_date = f'{year}-{month}-30'
        elif month == '02':
            # Check for leap year
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                end_date = f'{year}-{month}-29'
            else:
                end_date = f'{year}-{month}-28'
        else:
            end_date = f'{year}-{month}-31'
            
        date_ranges.append((start_date, end_date))
    
    # Determine which satellite to use based on year
    if 2000 <= year <= 2012:
        satellite = 'Landsat 5'
        collection_name = "LANDSAT/LT05/C02/T1_L2"
        scale_factors_func = apply_scale_factors_landsat5
        cloud_mask_func = cloud_mask_landsat5
        thermal_band = 'ST_B6'
        
        # For Landsat 5
        ndvi_bands = ['SR_B4', 'SR_B3']  # NIR, Red
        ndmi_bands = ['SR_B4', 'SR_B5']  # NIR, SWIR
    else:
        satellite = 'Landsat 8'
        collection_name = "LANDSAT/LC08/C02/T1_L2"
        scale_factors_func = apply_scale_factors_landsat8
        cloud_mask_func = cloud_mask_landsat8
        thermal_band = 'ST_B10'
        
        # For Landsat 8
        ndvi_bands = ['SR_B5', 'SR_B4']  # NIR, Red
        ndmi_bands = ['SR_B5', 'SR_B6']  # NIR, SWIR
    
    # Process each date range and find best image
    best_image = None
    best_cloud_cover = 100
    best_date_range = None
    
    for start_date, end_date in date_ranges:
        # Load Landsat image collection
        collection = ee.ImageCollection(collection_name) \
            .filterDate(start_date, end_date) \
            .filterBounds(aoi.geometry()) \
            .sort('CLOUD_COVER')
        
        # Get the image with the least cloud cover
        image = collection.first()
        
        if image is not None:
            try:
                cloud_cover = ee.Number(image.get('CLOUD_COVER')).getInfo()
                if cloud_cover < best_cloud_cover:
                    best_cloud_cover = cloud_cover
                    best_image = image
                    best_date_range = (start_date, end_date)
            except Exception as e:
                continue
    
    # If no images found for any of the date ranges
    if best_image is None:
        return None
    
    # Get image date
    image_date = ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    image_month = image_date.split('-')[1]
    
    # Calculate cloud cover within AOI
    cloud_cover_aoi = calculate_cloud_cover_within_aoi(best_image, aoi)
    
    # Apply scale factors and cloud masking
    image = scale_factors_func(best_image)
    image = cloud_mask_func(image)
    
    # Get best available NLCD for this year
    nlcd, nlcd_year = get_best_nlcd_for_year(year)
    
    # Calculate NDVI
    ndvi = image.normalizedDifference(ndvi_bands).rename('NDVI')
    
    # Calculate NDMI
    ndmi = image.normalizedDifference(ndmi_bands).rename('NDMI')
    
    # LST calculation
    ndvi_min = ndvi.reduceRegion(ee.Reducer.min(), aoi.geometry(), 30).get('NDVI')
    ndvi_max = ndvi.reduceRegion(ee.Reducer.max(), aoi.geometry(), 30).get('NDVI')
    
    # Fractional vegetation
    fv = ndvi.subtract(ee.Number(ndvi_min)).divide(ee.Number(ndvi_max).subtract(ndvi_min)).pow(2)
    
    # Emissivity
    em = fv.multiply(0.004).add(0.986)
    
    # Land surface temperature (celsius)
    lst = image.expression(
        '(T / (1 + (0.00115 * (T / 1.438)) * log(em))) - 273.15',
        {'T': image.select(thermal_band), 'em': em}
    ).rename('LST')
    
    # Clip LST to AOI
    lst_clipped = lst.clip(aoi)
    
    # Filter LST by 5-95 percentile range
    lst_filtered, p05, p95 = filter_lst_percentiles(lst_clipped, aoi)
    
    # Clip NLCD to the AOI
    nlcd_clipped = nlcd.clip(aoi)
    
    # Create masks for different land cover types
    # Urban areas (NLCD classes 22-24, excluding 21 which is urban open space)
    urban_mask = nlcd_clipped.gte(22).And(nlcd_clipped.lte(24))
    
    # Vegetative areas (including urban open space class 21 and water bodies class 11)
    forest_mask = nlcd_clipped.gte(41).And(nlcd_clipped.lte(43))  # Forest (41-43)
    shrub_mask = nlcd_clipped.eq(52)  # Shrub/Scrub (52)
    grass_mask = nlcd_clipped.eq(71)  # Grassland/Herbaceous (71)
    crop_mask = nlcd_clipped.gte(81).And(nlcd_clipped.lte(82))  # Cropland (81-82)
    open_space_mask = nlcd_clipped.eq(21)  # Urban Open Space (21)
    water_mask = nlcd_clipped.eq(11)  # Water (11)
    vegetative_mask = forest_mask.Or(shrub_mask).Or(grass_mask).Or(crop_mask).Or(open_space_mask).Or(water_mask)
    
    # Create a mask for the entire AOI to count total pixels
    aoi_mask = ee.Image(1).clip(aoi)
    
    # Count pixels for each land cover type
    urban_pixel_count = urban_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi.geometry(),
        scale=30,
        maxPixels=1e9
    ).get('landcover').getInfo()
    
    veg_pixel_count = vegetative_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi.geometry(),
        scale=30,
        maxPixels=1e9
    ).get('landcover').getInfo()
    
    total_pixel_count = aoi_mask.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=aoi.geometry(),
        scale=30,
        maxPixels=1e9
    ).get('constant').getInfo()
    
    # Calculate percentages
    urban_percentage = (urban_pixel_count / total_pixel_count) * 100
    veg_percentage = (veg_pixel_count / total_pixel_count) * 100
    other_percentage = 100 - urban_percentage - veg_percentage
    
    # Check if vegetative areas exist
    if veg_pixel_count == 0:
        return None
    
    # Calculate mean LST for urban and vegetative land cover types using the filtered LST
    urban_lst = lst_filtered.updateMask(urban_mask)
    vegetative_lst = lst_filtered.updateMask(vegetative_mask)
    
    urban_lst_mean = urban_lst.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi.geometry(),
        scale=30,
        maxPixels=1e9
    ).get('LST').getInfo()
    
    vegetative_lst_mean = vegetative_lst.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi.geometry(),
        scale=30,
        maxPixels=1e9
    ).get('LST').getInfo()
    
    # Calculate mean LST for the entire AOI using the filtered LST
    mean_lst = lst_filtered.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi.geometry(),
        scale=30,
        maxPixels=1e9
    ).get('LST').getInfo()
    
    # Calculate mean NDVI and NDMI for areas with valid filtered LST
    valid_lst_mask = lst_filtered.mask()
    
    mean_ndvi = ndvi.updateMask(valid_lst_mask).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi.geometry(),
        scale=30,
        maxPixels=1e9
    ).get('NDVI').getInfo()
    
    mean_ndmi = ndmi.updateMask(valid_lst_mask).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi.geometry(),
        scale=30,
        maxPixels=1e9
    ).get('NDMI').getInfo()
    
    # Calculate UHI intensity
    uhi_intensity = urban_lst_mean - vegetative_lst_mean
    
    # Store results
    results = {
        'Year': year,
        'Month': image_month,
        'Urban': urban_lst_mean,
        'Vegetative': vegetative_lst_mean,
        'Mean_LST': mean_lst,
        'Mean_NDVI': mean_ndvi,
        'Mean_NDMI': mean_ndmi,
        'UHI': uhi_intensity,
        'Urban_Percent': urban_percentage,
        'Vegetative_Percent': veg_percentage,
        'Other_Percent': other_percentage,
        'Image_Date': image_date,
        'Satellite': satellite,
        'Cloud_Cover_Scene': best_cloud_cover,
        'Cloud_Cover_AOI': cloud_cover_aoi,
        'LST_p5': p05,
        'LST_p95': p95,
        'NLCD_Year': nlcd_year
    }
    
    return results

# Create a mapping function to fetch county centroid for display
def map_county(aoi_gdf, epsg_code):
    # Create a map centered on the county
    center = aoi_gdf.to_crs(epsg=4326).centroid[0]
    county_map = geemap.Map(center=[center.y, center.x], zoom=9)
    
    # Add the county boundary
    county_map.add_gdf(aoi_gdf.to_crs(epsg=4326), layer_name="County Boundary")
    
    return county_map

# Function to visualize results
def plot_uhi_trend(results_df):
    if results_df.empty:
        return None
    
    # Create UHI trend plot
    fig = px.line(
        results_df, 
        x='Year', 
        y='UHI', 
        title=f'Urban Heat Island Intensity Trend ({start_year}-{end_year})',
        labels={'UHI': 'UHI Intensity (°C)', 'Year': 'Year'},
        markers=True,
        line_shape='linear'
    )
    
    # Add hovering information
    fig.update_traces(
        hovertemplate='<b>Year:</b> %{x}<br><b>UHI:</b> %{y:.2f}°C<br><b>Image Date:</b> %{customdata[0]}<br><b>Satellite:</b> %{customdata[1]}',
        customdata=results_df[['Image_Date', 'Satellite']]
    )
    
    # Add trend line
    fig.add_trace(go.Scatter(
        x=results_df['Year'],
        y=results_df['UHI'].rolling(window=3, min_periods=1).mean(),
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.7)', dash='dash'),
        name='3-Year Moving Average'
    ))
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        yaxis=dict(title='UHI Intensity (°C)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500
    )
    
    return fig

def plot_lst_comparison(results_df):
    if results_df.empty:
        return None
    
    # Create a dual-line plot comparing urban and vegetative temperatures
    fig = go.Figure()
    
    # Add urban temperature line
    fig.add_trace(go.Scatter(
        x=results_df['Year'],
        y=results_df['Urban'],
        mode='lines+markers',
        name='Urban Areas',
        line=dict(color='red', width=2),
        marker=dict(color='red', size=8)
    ))
    
    # Add vegetative temperature line
    fig.add_trace(go.Scatter(
        x=results_df['Year'],
        y=results_df['Vegetative'],
        mode='lines+markers',
        name='Vegetative Areas',
        line=dict(color='green', width=2),
        marker=dict(color='green', size=8)
    ))
    
    # Add hover information
    fig.update_traces(
        hovertemplate='<b>Year:</b> %{x}<br><b>Temperature:</b> %{y:.2f}°C<br><b>Image Date:</b> %{customdata}',
        customdata=results_df['Image_Date']
    )
    
    # Update layout
    fig.update_layout(
        title=f'Urban vs. Vegetative Surface Temperature Comparison ({start_year}-{end_year})',
        xaxis=dict(title='Year', tickmode='linear', dtick=1),
        yaxis=dict(title='Land Surface Temperature (°C)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500
    )
    
    return fig

def plot_ndvi_ndmi_trend(results_df):
    if results_df.empty:
        return None
    
    # Create a dual-axis plot for NDVI and NDMI trends
    fig = go.Figure()
    
    # Add NDVI line
    fig.add_trace(go.Scatter(
        x=results_df['Year'],
        y=results_df['Mean_NDVI'],
        mode='lines+markers',
        name='NDVI',
        line=dict(color='darkgreen', width=2),
        marker=dict(color='darkgreen', size=8)
    ))
    
    # Add NDMI line
    fig.add_trace(go.Scatter(
        x=results_df['Year'],
        y=results_df['Mean_NDMI'],
        mode='lines+markers',
        name='NDMI',
        line=dict(color='blue', width=2),
        marker=dict(color='blue', size=8)
    ))
    
    # Add hover information
    fig.update_traces(
        hovertemplate='<b>Year:</b> %{x}<br><b>Value:</b> %{y:.4f}<br><b>Image Date:</b> %{customdata}',
        customdata=results_df['Image_Date']
    )
    
    # Update layout
    fig.update_layout(
        title=f'Vegetation Indices Trend ({start_year}-{end_year})',
        xaxis=dict(title='Year', tickmode='linear', dtick=1),
        yaxis=dict(title='Index Value'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500
    )
    
    return fig

# Run analysis when user clicks the button
if st.sidebar.button("Run UHI Analysis"):
    if len(selected_months) == 0:
        st.error("Please select at least one month for analysis")
    elif end_year < start_year:
        st.error("End year must be greater than or equal to start year")
    else:
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get county boundary for selected state and county
            status_text.text("Getting county boundary...")
            aoi, aoi_gdf, epsg_code = get_county_boundary(state_id, county_id)
            
            # Process data for each year in range
            all_results = []
            years_to_process = list(range(start_year, end_year + 1))
            total_years = len(years_to_process)
            
            for i, year in enumerate(years_to_process):
                status_text.text(f"Processing {year} data ({i+1}/{total_years})...")
                results = process_landsat_data(year, aoi, selected_months)
                if results is not None:
                    all_results.append(results)
                progress_bar.progress((i + 1) / total_years)
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(all_results)
            
            if results_df.empty:
                st.error("No valid data found for the selected parameters. Try expanding the date range or selecting different months.")
            else:
                # Sort by year
                results_df = results_df.sort_values('Year')
                
                # Success message
                status_text.text("Analysis complete!")
                
                # Create tabs for different outputs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Results Table", "UHI Trend", "Temperature Comparison", "Vegetation Indices", "County Map"])
                
                # Display results table
                with tab1:
                    st.subheader("UHI Analysis Results")
                    display_cols = ['Year', 'Month', 'Image_Date', 'Urban', 'Vegetative', 'UHI', 
                                    'Mean_NDVI', 'Mean_NDMI', 'Urban_Percent', 'Vegetative_Percent',
                                    'Cloud_Cover_AOI', 'Satellite', 'NLCD_Year']
                    
                    # Format the values
                    formatted_df = results_df.copy()
                    for col in ['Urban', 'Vegetative', 'Mean_LST', 'UHI']:
                        formatted_df[col] = formatted_df[col].round(2).astype(str) + ' °C'
                    
                    for col in ['Urban_Percent', 'Vegetative_Percent', 'Other_Percent', 'Cloud_Cover_Scene', 'Cloud_Cover_AOI']:
                        formatted_df[col] = formatted_df[col].round(2).astype(str) + ' %'
                    
                    st.dataframe(formatted_df[display_cols], use_container_width=True)
                    
                    # Download button for CSV
                    csv = results_df.to_csv(index=False)
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{selected_state}_{selected_county}_UHI_{start_year}_{end_year}_{now}.csv"
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=filename,
                        mime="text/csv",
                    )
                
                # Display UHI trend chart
                with tab2:
                    st.subheader("Urban Heat Island Intensity Trend")
                    uhi_fig = plot_uhi_trend(results_df)
                    if uhi_fig:
                        st.plotly_chart(uhi_fig, use_container_width=True)
                        
                        # Add analysis details
                        avg_uhi = results_df['UHI'].mean()
                        max_uhi_row = results_df.loc[results_df['UHI'].idxmax()]
                        min_uhi_row = results_df.loc[results_df['UHI'].idxmin()]
                        
                        st.metric("Average UHI Intensity", f"{avg_uhi:.2f} °C")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Maximum UHI", f"{max_uhi_row['UHI']:.2f} °C", f"Year: {max_uhi_row['Year']}")
                        with col2:
                            st.metric("Minimum UHI", f"{min_uhi_row['UHI']:.2f} °C", f"Year: {min_uhi_row['Year']}")
                            
                        # Calculate trend statistics
                        if len(results_df) > 1:
                            from scipy import stats
                            years = results_df['Year'].values
                            uhi = results_df['UHI'].values
                            slope, intercept, r_value, p_value, std_err = stats.linregress(years, uhi)
                            
                            trend_direction = "increasing" if slope > 0 else "decreasing"
                            trend_significant = "statistically significant" if p_value < 0.05 else "not statistically significant"
                            
                            st.info(f"UHI trend is {trend_direction} at a rate of {slope:.3f} °C/year (p={p_value:.3f}, {trend_significant}).")
                
                # Display temperature comparison chart
                with tab3:
                    st.subheader("Urban vs. Vegetative Surface Temperature")
                    temp_fig = plot_lst_comparison(results_df)
                    if temp_fig:
                        st.plotly_chart(temp_fig, use_container_width=True)
                        
                        # Add temperature statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            avg_urban_temp = results_df['Urban'].mean()
                            avg_veg_temp = results_df['Vegetative'].mean()
                            
                            st.metric("Average Urban Temperature", f"{avg_urban_temp:.2f} °C")
                            st.metric("Average Vegetative Temperature", f"{avg_veg_temp:.2f} °C")
                        
                        with col2:
                            urban_trend = results_df['Urban'].iloc[-1] - results_df['Urban'].iloc[0]
                            veg_trend = results_df['Vegetative'].iloc[-1] - results_df['Vegetative'].iloc[0]
                            
                            st.metric("Urban Temperature Change", f"{urban_trend:.2f} °C", 
                                    f"{results_df['Year'].iloc[0]} to {results_df['Year'].iloc[-1]}")
                            st.metric("Vegetative Temperature Change", f"{veg_trend:.2f} °C", 
                                    f"{results_df['Year'].iloc[0]} to {results_df['Year'].iloc[-1]}")
                
                # Display vegetation indices chart
                with tab4:
                    st.subheader("Vegetation Indices Over Time")
                    veg_fig = plot_ndvi_ndmi_trend(results_df)
                    if veg_fig:
                        st.plotly_chart(veg_fig, use_container_width=True)
                        
                        # Add vegetation indices statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            avg_ndvi = results_df['Mean_NDVI'].mean()
                            ndvi_trend = results_df['Mean_NDVI'].iloc[-1] - results_df['Mean_NDVI'].iloc[0]
                            
                            st.metric("Average NDVI", f"{avg_ndvi:.4f}")
                            st.metric("NDVI Change", f"{ndvi_trend:.4f}", 
                                    f"{results_df['Year'].iloc[0]} to {results_df['Year'].iloc[-1]}")
                        
                        with col2:
                            avg_ndmi = results_df['Mean_NDMI'].mean()
                            ndmi_trend = results_df['Mean_NDMI'].iloc[-1] - results_df['Mean_NDMI'].iloc[0]
                            
                            st.metric("Average NDMI", f"{avg_ndmi:.4f}")
                            st.metric("NDMI Change", f"{ndmi_trend:.4f}", 
                                    f"{results_df['Year'].iloc[0]} to {results_df['Year'].iloc[-1]}")
                            
                        # Explanation
                        with st.expander("What do these indices mean?"):
                            st.markdown("""
                            **NDVI (Normalized Difference Vegetation Index)** measures vegetation health and density. 
                            - Values range from -1 to 1
                            - Higher values indicate healthier, denser vegetation
                            - Values near 0 indicate bare soil or urban areas
                            - Negative values often indicate water
                            
                            **NDMI (Normalized Difference Moisture Index)** measures vegetation water content.
                            - Values range from -1 to 1
                            - Higher values indicate more moisture in vegetation
                            - Lower values indicate drier vegetation or non-vegetated surfaces
                            """)
                
                # Display county map
                with tab5:
                    st.subheader("County Map")
                    county_map = map_county(aoi_gdf, epsg_code)
                    county_map.to_streamlit(height=600)
                    
                    # Display land cover breakdown
                    st.subheader("Land Cover Composition")
                    
                    # Create pie chart of land cover percentages
                    latest_year = results_df['Year'].max()
                    latest_data = results_df[results_df['Year'] == latest_year].iloc[0]
                    
                    labels = ['Urban', 'Vegetative', 'Other']
                    values = [latest_data['Urban_Percent'], 
                            latest_data['Vegetative_Percent'], 
                            latest_data['Other_Percent']]
                    
                    fig = px.pie(
                        values=values,
                        names=labels,
                        title=f"Land Cover Distribution in {selected_county} County ({latest_year}, NLCD {latest_data['NLCD_Year']})",
                        color_discrete_sequence=['grey', 'green', 'lightblue']
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display land cover explanation
                    with st.expander("Land Cover Categories Explanation"):
                        st.markdown("""
                        **Urban** includes:
                        - Developed, Medium Intensity (NLCD class 22)
                        - Developed, High Intensity (NLCD class 23)
                        - Developed, Medium/High Intensity (NLCD class 24)
                        
                        **Vegetative** includes:
                        - Open Water (NLCD class 11)
                        - Developed, Open Space (NLCD class 21)
                        - Forest (NLCD classes 41-43)
                        - Shrub/Scrub (NLCD class 52)
                        - Grassland/Herbaceous (NLCD class 71)
                        - Cropland (NLCD classes 81-82)
                        
                        **Other** includes all remaining NLCD classes.
                        """)
        
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.exception(e)

# Add helper information in the sidebar
with st.sidebar.expander("About the UHI Analysis"):
    st.markdown("""
    **What is Urban Heat Island (UHI)?**
    
    The Urban Heat Island effect refers to the phenomenon where urban areas experience higher temperatures than their surrounding rural areas. This is primarily due to:
    - Dark surfaces like asphalt and buildings that absorb heat
    - Reduced vegetation and tree cover
    - Waste heat from vehicles, air conditioning, and industry
    - Urban geometry that traps heat
    
    **How is UHI calculated in this app?**
    
    UHI intensity is calculated as the temperature difference between urban areas and vegetative/water surfaces:
    
    UHI = Average Urban Temperature - Average Vegetative Temperature
    
    The app uses:
    - Landsat satellite thermal data (30m resolution)
    - NLCD land cover classification to identify urban and vegetative areas
    - 5-95 percentile filtering to remove outlier temperatures
    """)

with st.sidebar.expander("Tips for best results"):
    st.markdown("""
    - **Summer months** (June-August) typically show the strongest UHI effect
    - For areas with seasonal cloud cover, try different months
    - Smaller counties will process faster than larger ones
    - For the most complete time series, include both Landsat 5 years (2000-2012) and Landsat 8 years (2013-2024)
    - Some years may be missing if no suitable imagery (low cloud cover) was available
    """)

# Add credits at the bottom of the sidebar
st.sidebar.markdown("---")
st.sidebar.caption("""
UHI Analysis App | Built with Streamlit, Earth Engine, and geemap
""")
