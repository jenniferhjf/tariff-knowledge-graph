
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Trade Knowledge Graph with Predictions",
    page_icon="🌍",
    layout="wide"
)

# Main title
st.title("🌍 Global Trade Knowledge Graph with 2025 Predictions")
st.markdown("**AI-powered analysis platform with predictive capabilities for trade flows, tariffs, and market trends**")

# Sidebar
st.sidebar.header("🔧 Control Panel")

# Enhanced data generation with predictions
@st.cache_data
def generate_trade_data_with_predictions():
    """Generate mock trade data with 2025 predictions"""
    countries = {
        'China': {'lat': 35.8617, 'lon': 104.1954, 'code': 'CN'},
        'USA': {'lat': 37.0902, 'lon': -95.7129, 'code': 'US'},
        'Germany': {'lat': 51.1657, 'lon': 10.4515, 'code': 'DE'},
        'Japan': {'lat': 36.2048, 'lon': 138.2529, 'code': 'JP'},
        'UK': {'lat': 55.3781, 'lon': -3.4360, 'code': 'GB'},
        'France': {'lat': 46.2276, 'lon': 2.2137, 'code': 'FR'},
        'South Korea': {'lat': 35.9078, 'lon': 127.7669, 'code': 'KR'},
        'Canada': {'lat': 56.1304, 'lon': -106.3468, 'code': 'CA'},
        'Australia': {'lat': -25.2744, 'lon': 133.7751, 'code': 'AU'},
        'Brazil': {'lat': -14.2350, 'lon': -51.9253, 'code': 'BR'}
    }
    
    products = [
        'Electronics', 'Machinery', 'Textiles', 'Chemicals', 
        'Automobiles', 'Steel', 'Agricultural Products', 'Oil & Gas',
        'Pharmaceuticals', 'Furniture'
    ]
    
    # Generate historical data (2021-2024)
    historical_data = []
    for year in [2021, 2022, 2023, 2024]:
        for _ in range(200):
            exporter = np.random.choice(list(countries.keys()))
            importer = np.random.choice(list(countries.keys()))
            while importer == exporter:
                importer = np.random.choice(list(countries.keys()))
            
            # Add trend factors
            year_factor = 1 + (year - 2021) * 0.05  # 5% growth per year
            
            historical_data.append({
                'exporter': exporter,
                'importer': importer,
                'exporter_lat': countries[exporter]['lat'],
                'exporter_lon': countries[exporter]['lon'],
                'importer_lat': countries[importer]['lat'],
                'importer_lon': countries[importer]['lon'],
                'product': np.random.choice(products),
                'trade_value': np.random.uniform(1000000, 1000000000) * year_factor,
                'tariff_rate': np.random.uniform(0, 25),
                'hs_code': f"{np.random.choice([1,2,3,4,5,6,7,8,9])}{np.random.randint(10,99)}{np.random.randint(10,99)}",
                'year': year,
                'month': np.random.randint(1, 13),
                'data_type': 'Historical',
                'confidence': 1.0,
                'prediction_interval_low': None,
                'prediction_interval_high': None
            })
    
    historical_df = pd.DataFrame(historical_data)
    
    # Generate 2025 predictions
    predictions = []
    
    # Simple prediction logic based on historical trends
    for _, country_pair in historical_df[['exporter', 'importer']].drop_duplicates().iterrows():
        exporter = country_pair['exporter']
        importer = country_pair['importer']
        
        # Get historical data for this trade pair
        pair_data = historical_df[
            (historical_df['exporter'] == exporter) & 
            (historical_df['importer'] == importer)
        ]
        
        if len(pair_data) >= 2:  # Need at least 2 data points for prediction
            for product in products:
                product_data = pair_data[pair_data['product'] == product]
                
                if len(product_data) >= 1:
                    # Calculate trend
                    recent_value = product_data['trade_value'].mean()
                    growth_rate = np.random.normal(0.08, 0.15)  # 8% average growth with variation
                    
                    # Predict 2025 value
                    predicted_value = recent_value * (1 + growth_rate)
                    
                    # Add uncertainty
                    confidence = np.random.uniform(0.6, 0.9)
                    uncertainty = 1 - confidence
                    
                    # Prediction intervals
                    interval_width = predicted_value * uncertainty
                    pred_low = max(0, predicted_value - interval_width)
                    pred_high = predicted_value + interval_width
                    
                    # Predict tariff changes
                    avg_tariff = product_data['tariff_rate'].mean()
                    tariff_change = np.random.normal(0, 2)  # Small random changes
                    predicted_tariff = max(0, min(30, avg_tariff + tariff_change))
                    
                    predictions.append({
                        'exporter': exporter,
                        'importer': importer,
                        'exporter_lat': countries[exporter]['lat'],
                        'exporter_lon': countries[exporter]['lon'],
                        'importer_lat': countries[importer]['lat'],
                        'importer_lon': countries[importer]['lon'],
                        'product': product,
                        'trade_value': predicted_value,
                        'tariff_rate': predicted_tariff,
                        'hs_code': product_data['hs_code'].iloc[-1] if len(product_data) > 0 else f"{np.random.choice([1,2,3,4,5,6,7,8,9])}{np.random.randint(10,99)}{np.random.randint(10,99)}",
                        'year': 2025,
                        'month': np.random.randint(1, 13),
                        'data_type': 'Predicted',
                        'confidence': confidence,
                        'prediction_interval_low': pred_low,
                        'prediction_interval_high': pred_high
                    })
    
    # Combine historical and predicted data
    all_data = historical_data + predictions
    
    return pd.DataFrame(all_data), countries, products

# Generate enhanced data
trade_df, countries_info, products_list = generate_trade_data_with_predictions()

# Sidebar controls
view_mode = st.sidebar.selectbox(
    "📊 View Mode",
    ["Geographic Trade Flow", "Knowledge Graph Network", "Comprehensive Dashboard", 
     "Product Analysis", "Tariff Analysis", "2025 Predictions"]
)

# Enhanced year selection including predictions
available_years = sorted(trade_df['year'].unique())
selected_year = st.sidebar.selectbox("📅 Year", available_years)

# Data type filter
data_types = ['All'] + list(trade_df['data_type'].unique())
selected_data_type = st.sidebar.selectbox("📈 Data Type", data_types)

selected_products = st.sidebar.multiselect("📦 Product Categories", products_list, default=products_list[:3])

# Enhanced filtering
filtered_df = trade_df[
    (trade_df['year'] == selected_year) & 
    (trade_df['product'].isin(selected_products))
].copy()

if selected_data_type != 'All':
    filtered_df = filtered_df[filtered_df['data_type'] == selected_data_type]

# Add confidence indicator in sidebar for predictions
if selected_year == 2025:
    if not filtered_df.empty:
        avg_confidence = filtered_df['confidence'].mean()
        st.sidebar.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
        filtered_df = filtered_df[filtered_df['confidence'] >= confidence_threshold]

# Main content area with enhanced views
if view_mode == "Geographic Trade Flow":
    st.header("🌍 Global Trade Flow Map")
    
    # Add prediction indicator
    if selected_year == 2025:
        st.info("🔮 Showing predicted trade flows for 2025 with confidence intervals")
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Enhanced arc layer with prediction styling
            arc_color = [255, 0, 0, 160] if selected_year != 2025 else [0, 0, 255, 120]
            
            arc_layer = pdk.Layer(
                "ArcLayer",
                data=filtered_df,
                get_source_position=["exporter_lon", "exporter_lat"],
                get_target_position=["importer_lon", "importer_lat"],
                get_source_color=arc_color,
                get_target_color=[0, 255, 0, 160],
                get_width="trade_value/100000000",
                width_scale=1,
                width_min_pixels=2,
                pickable=True,
            )
            
            # Enhanced country nodes with prediction indicators
            country_data = []
            for country, info in countries_info.items():
                exports = filtered_df[filtered_df['exporter'] == country]['trade_value'].sum()
                imports = filtered_df[filtered_df['importer'] == country]['trade_value'].sum()
                avg_confidence = filtered_df[
                    (filtered_df['exporter'] == country) | (filtered_df['importer'] == country)
                ]['confidence'].mean() if selected_year == 2025 else 1.0
                
                if exports > 0 or imports > 0:  # Only show countries with trade
                    country_data.append({
                        'country': country,
                        'lat': info['lat'],
                        'lon': info['lon'],
                        'total_trade': exports + imports,
                        'exports': exports,
                        'imports': imports,
                        'confidence': avg_confidence if not pd.isna(avg_confidence) else 1.0
                    })
            
            country_df = pd.DataFrame(country_data)
            
            if not country_df.empty:
                scatter_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=country_df,
                    get_position=["lon", "lat"],
                    get_color=[255, 255, 0, 200],
                    get_radius="total_trade/50000000",
                    radius_scale=1,
                    radius_min_pixels=5,
                    pickable=True,
                )
                
                # Enhanced tooltip for predictions
                tooltip_html = "<b>Trade Flow:</b><br/>From {exporter} to {importer}<br/>Product: {product}<br/>Value: ${trade_value:,.0f}<br/>Tariff Rate: {tariff_rate:.1f}%"
                if selected_year == 2025:
                    tooltip_html += "<br/>Confidence: {confidence:.1%}<br/>Range: ${prediction_interval_low:,.0f} - ${prediction_interval_high:,.0f}"
                
                st.pydeck_chart(
                    pdk.Deck(
                        map_style="mapbox://styles/mapbox/light-v9",
                        initial_view_state=pdk.ViewState(
                            latitude=30,
                            longitude=0,
                            zoom=1.5,
                            pitch=30,
                        ),
                        layers=[arc_layer, scatter_layer],
                        tooltip={
                            "html": tooltip_html,
                            "style": {"backgroundColor": "steelblue", "color": "white"}
                        }
                    )
                )
            else:
                st.warning("No country data to display on map.")
        
        with col2:
            st.subheader("📈 Trade Statistics")
            total_trade = filtered_df['trade_value'].sum()
            st.metric("Total Trade Value", f"${total_trade/1e9:.1f}B")
            
            avg_tariff = filtered_df['tariff_rate'].mean()
            st.metric("Average Tariff Rate", f"{avg_tariff:.1f}%")
            
            if selected_year == 2025 and not filtered_df.empty:
                avg_confidence = filtered_df['confidence'].mean()
                st.metric("Average Confidence", f"{avg_confidence:.1%}")
            
            top_exporters = filtered_df.groupby('exporter')['trade_value'].sum().sort_values(ascending=False).head(5)
            if not top_exporters.empty:
                st.subheader("🏆 Top Exporters")
                for country, value in top_exporters.items():
                    st.write(f"**{country}**: ${value/1e9:.1f}B")

elif view_mode == "2025 Predictions":
    st.header("🔮 2025 Trade Predictions & Analysis")
    
    # Filter for 2025 predictions only
    predictions_df = trade_df[trade_df['year'] == 2025].copy()
    historical_df = trade_df[trade_df['year'] < 2025].copy()
    
    if predictions_df.empty:
        st.warning("No prediction data available.")
    else:
        # Overall prediction metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_predicted_trade = predictions_df['trade_value'].sum()
            st.metric("Predicted 2025 Trade", f"${total_predicted_trade/1e9:.1f}B")
        
        with col2:
            avg_confidence = predictions_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        with col3:
            # Growth rate vs 2024
            trade_2024 = historical_df[historical_df['year'] == 2024]['trade_value'].sum()
            if trade_2024 > 0:
                growth_rate = (total_predicted_trade - trade_2024) / trade_2024 * 100
                st.metric("Predicted Growth", f"{growth_rate:.1f}%")
            else:
                st.metric("Predicted Growth", "N/A")
        
        with col4:
