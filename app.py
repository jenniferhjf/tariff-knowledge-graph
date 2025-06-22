
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
            
            country_data.append({
                'country': country,
                'lat': info['lat'],
                'lon': info['lon'],
                'total_trade': exports + imports,
                'exports': exports,
                'imports': imports,
                'confidence': avg_confidence
            })
        
        country_df = pd.DataFrame(country_data)
        
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
    
    with col2:
        st.subheader("📈 Trade Statistics")
        total_trade = filtered_df['trade_value'].sum()
        st.metric("Total Trade Value", f"${total_trade/1e9:.1f}B")
        
        avg_tariff = filtered_df['tariff_rate'].mean()
        st.metric("Average Tariff Rate", f"{avg_tariff:.1f}%")
        
        if selected_year == 2025:
            avg_confidence = filtered_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        top_exporters = filtered_df.groupby('exporter')['trade_value'].sum().sort_values(ascending=False).head(5)
        st.subheader("🏆 Top Exporters")
        for country, value in top_exporters.items():
            st.write(f"**{country}**: ${value/1e9:.1f}B")

elif view_mode == "2025 Predictions":
    st.header("🔮 2025 Trade Predictions & Analysis")
    
    # Filter for 2025 predictions only
    predictions_df = trade_df[trade_df['year'] == 2025].copy()
    historical_df = trade_df[trade_df['year'] < 2025].copy()
    
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
    
    with col4:
        high_confidence_pct = len(predictions_df[predictions_df['confidence'] > 0.8]) / len(predictions_df) * 100
        st.metric("High Confidence Predictions", f"{high_confidence_pct:.1f}%")
    
    # Prediction visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence distribution - 修复的部分
        fig_conf = px.histogram(predictions_df, x='confidence', nbins=20,
                               title="Prediction Confidence Distribution")
        fig_conf.update_layout(
            xaxis_title="Confidence Level",
            yaxis_title="Number of Predictions"
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        # Growth predictions by product - 修复的部分
        growth_by_product = []
        for product in products_list:
            pred_2025 = predictions_df[predictions_df['product'] == product]['trade_value'].sum()
            hist_2024 = historical_df[(historical_df['year'] == 2024) & 
                                    (historical_df['product'] == product)]['trade_value'].sum()
            if hist_2024 > 0:
                growth = (pred_2025 - hist_2024) / hist_2024 * 100
                growth_by_product.append({'Product': product, 'Growth': growth})
        
        if growth_by_product:
            growth_df = pd.DataFrame(growth_by_product)
            fig_growth = px.bar(growth_df, x='Product', y='Growth',
                               title="Predicted Growth by Product (2024 vs 2025)")
            fig_growth.update_layout(
                xaxis_title="Product Category",
                yaxis_title="Growth Rate (%)"
            )
            st.plotly_chart(fig_growth, use_container_width=True)
    
    # Uncertainty analysis
    st.subheader("📊 Prediction Uncertainty Analysis")
    
    # Create uncertainty bands
    uncertainty_data = []
    for _, row in predictions_df.iterrows():
        uncertainty_data.append({
            'Trade_Route': f"{row['exporter']} → {row['importer']}",
            'Product': row['product'],
            'Predicted_Value': row['trade_value'],
            'Lower_Bound': row['prediction_interval_low'],
            'Upper_Bound': row['prediction_interval_high'],
            'Confidence': row['confidence'],
            'Uncertainty_Range': row['prediction_interval_high'] - row['prediction_interval_low']
        })
    
    uncertainty_df = pd.DataFrame(uncertainty_data).sort_values('Uncertainty_Range', ascending=False).head(20)
    
    # Uncertainty range chart
    fig_uncertainty = go.Figure()
    
    for i, row in uncertainty_df.iterrows():
        fig_uncertainty.add_trace(go.Scatter(
            x=[row['Lower_Bound'], row['Predicted_Value'], row['Upper_Bound']],
            y=[row['Trade_Route']] * 3,
            mode='markers+lines',
            name=row['Product'],
            showlegend=False,
            line=dict(color='rgba(0,100,80,0.5)'),
            marker=dict(size=[6, 10, 6], color=['red', 'blue', 'red'])
        ))
    
    fig_uncertainty.update_layout(
        title="Top 20 Predictions with Highest Uncertainty",
        xaxis_title="Trade Value ($)",
        yaxis_title="Trade Routes",
        showlegend=False
    )
    
    st.plotly_chart(fig_uncertainty, use_container_width=True)
    
    # Detailed predictions table
    st.subheader("🔍 Detailed 2025 Predictions")
    
    display_predictions = predictions_df.copy()
    display_predictions['trade_value'] = display_predictions['trade_value'].apply(lambda x: f"${x:,.0f}")
    display_predictions['confidence'] = display_predictions['confidence'].apply(lambda x: f"{x:.1%}")
    display_predictions['prediction_range'] = display_predictions.apply(
        lambda x: f"${x['prediction_interval_low']:,.0f} - ${x['prediction_interval_high']:,.0f}", axis=1
    )
    
    st.dataframe(
        display_predictions[['exporter', 'importer', 'product', 'trade_value', 'confidence', 'prediction_range']],
        use_container_width=True
    )

elif view_mode == "Knowledge Graph Network":
    st.header("🕸️ Trade Relationship Knowledge Graph")
    
    # Add prediction indicator
    if selected_year == 2025:
        st.info("🔮 Network based on predicted trade relationships")
    
    try:
        # Build network graph
        G = nx.Graph()
        
        # Add country nodes
        for country in countries_info.keys():
            exports = filtered_df[filtered_df['exporter'] == country]['trade_value'].sum()
            imports = filtered_df[filtered_df['importer'] == country]['trade_value'].sum()
            avg_confidence = filtered_df[
                (filtered_df['exporter'] == country) | (filtered_df['importer'] == country)
            ]['confidence'].mean() if selected_year == 2025 else 1.0
            
            G.add_node(country, 
                      type='country',
                      trade_volume=exports + imports,
                      exports=exports,
                      imports=imports,
                      confidence=avg_confidence)
        
        # Add product nodes
        for product in selected_products:
            trade_vol = filtered_df[filtered_df['product'] == product]['trade_value'].sum()
            avg_confidence = filtered_df[filtered_df['product'] == product]['confidence'].mean() if selected_year == 2025 else 1.0
            G.add_node(product, 
                      type='product',
                      trade_volume=trade_vol,
                      confidence=avg_confidence)
        
        # Add trade relationship edges
        for _, row in filtered_df.iterrows():
            # Country-to-country trade relationships
            if G.has_edge(row['exporter'], row['importer']):
                G[row['exporter']][row['importer']]['weight'] += row['trade_value']
            else:
                G.add_edge(row['exporter'], row['importer'], 
                          weight=row['trade_value'],
                          relation='trade',
                          confidence=row['confidence'])
            
            # Country-product relationships
            G.add_edge(row['exporter'], row['product'], 
                      weight=row['trade_value'],
                      relation='export',
                      confidence=row['confidence'])
            G.add_edge(row['importer'], row['product'], 
                      weight=row['trade_value'],
                      relation='import',
                      confidence=row['confidence'])
        
        # Check if graph has nodes
        if G.number_of_nodes() == 0:
            st.warning("No data available for the selected filters")
        else:
            # Use spring layout
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Create Plotly network graph with confidence indicators
            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                   line=dict(width=0.5, color='#888'),
                                   hoverinfo='none',
                                   mode='lines',
                                   showlegend=False)
            
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_sizes = []
            node_info = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                node_data = G.nodes[node]
                
                if node_data['type'] == 'country':
                    # Adjust color intensity based on confidence for predictions
                    if selected_year == 2025:
                        confidence = node_data.get('confidence', 1.0)
                        # Use confidence to adjust color intensity
                        intensity = int(173 + (255-173) * confidence)
                        node_colors.append(f'rgb(173,216,{intensity})')
                    else:
                        node_colors.append('lightblue')
                    node_sizes.append(max(10, node_data['trade_volume']/1e8))
                    
                    # Create hover info
                    exports = node_data.get('exports', 0)
                    imports = node_data.get('imports', 0)
                    conf_text = f"<br>Confidence: {confidence:.1%}" if selected_year == 2025 else ""
                    node_info.append(f"<b>{node}</b><br>Type: Country<br>Exports: ${exports/1e9:.1f}B<br>Imports: ${imports/1e9:.1f}B{conf_text}")
                else:  # product
                    if selected_year == 2025:
                        confidence = node_data.get('confidence', 1.0)
                        intensity = int(240 + (255-240) * confidence)
                        node_colors.append(f'rgb({intensity},128,128)')
                    else:
                        node_colors.append('lightcoral')
                    node_sizes.append(15)
                    
                    trade_vol = node_data.get('trade_volume', 0)
                    conf_text = f"<br>Confidence: {confidence:.1%}" if selected_year == 2025 else ""
                    node_info.append(f"<b>{node}</b><br>Type: Product<br>Trade Volume: ${trade_vol/1e9:.1f}B{conf_text}")
            
            node_trace = go.Scatter(x=node_x, y=node_y,
                                   mode='markers+text',
                                   hoverinfo='text',
                                   hovertext=node_info,
                                   text=node_text,
                                   textposition="middle center",
                                   marker=dict(size=node_sizes,
                                             color=node_colors,
                                             line=dict(width=2, color='white')),
                                   showlegend=False)
            
            # 修复的Layout创建 - 这是关键修复部分
            graph_title = 'Trade Relationship Knowledge Graph'
            if selected_year == 2025:
                graph_title += ' (2025 Predictions)'
            
            # 创建layout参数字典，避免参数冲突
            layout_params = {
                'showlegend': False,
                'hovermode': 'closest',
                'margin': dict(b=20, l=5, r=5, t=40),
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white'
            }
            
            # 安全地添加标题
            layout_params['title'] = {
                'text': graph_title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            }
            
            # 安全地添加坐标轴设置
            layout_params['xaxis'] = {
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False,
                'visible': False
            }
            
            layout_params['yaxis'] = {
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False,
                'visible': False
            }
            
            # 添加注释
            annotation_text = "Node size represents trade volume, Blue=Countries, Red=Products."
            if selected_year == 2025:
                annotation_text += " For predictions: Intensity shows confidence level."
            
            layout_params['annotations'] = [
                {
                    'text': annotation_text,
                    'showarrow': False,
                    'xref': "paper",
                    'yref': "paper",
                    'x': 0.005,
                    'y': -0.002,
                    'xanchor': "left",
                    'yanchor': "bottom",
                    'font': {'color': "black", 'size': 12}
                }
            ]
            
            # 创建图形
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(**layout_params)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Network statistics with prediction metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Nodes", G.number_of_nodes())
            with col2:
                st.metric("Number of Edges", G.number_of_edges())
            with col3:
                if G.number_of_nodes() > 0:
                    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
                    st.metric("Average Degree", f"{avg_degree:.1f}")
    
    except Exception as e:
        st.error(f"Error creating knowledge graph: {str(e)}")
        st.info("Please try adjusting the filters or selecting different parameters.")

elif view_mode == "Comprehensive Dashboard":
    st.header("📊 Comprehensive Trade Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_trade = filtered_df['trade_value'].sum()
        st.metric("Total Trade Volume", f"${total_trade/1e9:.1f}B")
    with col2:
        unique_routes = len(filtered_df.groupby(['exporter', 'importer']))
        st.metric("Active Trade Routes", unique_routes)
    with col3:
        avg_tariff = filtered_df['tariff_rate'].mean()
        st.metric("Average Tariff", f"{avg_tariff:.1f}%")
    with col4:
        if selected_year == 2025:
            avg_conf = filtered_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        else:
            num_products = len(filtered_df['product'].unique())
            st.metric("Product Categories", num_products)
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        # Top trade partners
        trade_partners = filtered_df.groupby(['exporter', 'importer'])['trade_value'].sum().reset_index()
        trade_partners['route'] = trade_partners['exporter'] + ' → ' + trade_partners['importer']
        top_routes = trade_partners.nlargest(10, 'trade_value')
        
        fig_routes = px.bar(top_routes, x='trade_value', y='route', orientation='h',
                           title="Top 10 Trade Routes by Value")
        fig_routes.update_layout(
            xaxis_title="Trade Value ($)",
            yaxis_title="Trade Route"
        )
        st.plotly_chart(fig_routes, use_container_width=True)
    
    with col2:
        # Product distribution
        product_dist = filtered_df.groupby('product')['trade_value'].sum().reset_index()
        fig_products = px.pie(product_dist, values='trade_value', names='product',
                             title="Trade Distribution by Product Category")
        st.plotly_chart(fig_products, use_container_width=True)

elif view_mode == "Product Analysis":
    st.header("📦 Product Category Analysis")
    
    # Product selection
    selected_product = st.selectbox("Select Product for Detailed Analysis", selected_products)
    product_data = filtered_df[filtered_df['product'] == selected_product]
    
    if not product_data.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            product_total = product_data['trade_value'].sum()
            st.metric(f"Total {selected_product} Trade", f"${product_total/1e6:.1f}M")
        with col2:
            avg_tariff = product_data['tariff_rate'].mean()
            st.metric("Average Tariff Rate", f"{avg_tariff:.1f}%")
        with col3:
            if selected_year == 2025:
                avg_conf = product_data['confidence'].mean()
                st.metric("Average Confidence", f"{avg_conf:.1%}")
            else:
                trade_routes = len(product_data)
                st.metric("Number of Trade Routes", trade_routes)
        
        # Product flow visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Top exporters for this product
            top_exporters = product_data.groupby('exporter')['trade_value'].sum().sort_values(ascending=False).head(8)
            fig_exp = px.bar(x=top_exporters.values, y=top_exporters.index, orientation='h',
                            title=f"Top {selected_product} Exporters")
            fig_exp.update_layout(
                xaxis_title="Export Value ($)",
                yaxis_title="Countries"
            )
            st.plotly_chart(fig_exp, use_container_width=True)
        
        with col2:
            # Top importers for this product
            top_importers = product_data.groupby('importer')['trade_value'].sum().sort_values(ascending=False).head(8)
            fig_imp = px.bar(x=top_importers.values, y=top_importers.index, orientation='h',
                            title=f"Top {selected_product} Importers")
            fig_imp.update_layout(
                xaxis_title="Import Value ($)",
                yaxis_title="Countries"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        
        # Tariff analysis for this product
        st.subheader(f"📊 {selected_product} Tariff Analysis")
        tariff_stats = product_data.groupby(['exporter', 'importer']).agg({
            'tariff_rate': 'mean',
            'trade_value': 'sum'
        }).reset_index()
        
        fig_tariff = px.scatter(tariff_stats, x='tariff_rate', y='trade_value',
                               hover_data=['exporter', 'importer'],
                               title=f"{selected_product}: Trade Value vs Tariff Rate")
        fig_tariff.update_layout(
            xaxis_title="Tariff Rate (%)",
            yaxis_title="Trade Value ($)"
        )
        st.plotly_chart(fig_tariff, use_container_width=True)

elif view_mode == "Tariff Analysis":
    st.header("📊 Tariff Analysis Dashboard")
    
    # Tariff statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_tariff = filtered_df['tariff_rate'].mean()
        st.metric("Average Tariff Rate", f"{avg_tariff:.1f}%")
    with col2:
        max_tariff = filtered_df['tariff_rate'].max()
        st.metric("Highest Tariff Rate", f"{max_tariff:.1f}%")
    with col3:
        min_tariff = filtered_df['tariff_rate'].min()
        st.metric("Lowest Tariff Rate", f"{min_tariff:.1f}%")
    with col4:
        tariff_std = filtered_df['tariff_rate'].std()
        st.metric("Tariff Variability", f"{tariff_std:.1f}%")
    
    # Tariff visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Tariff distribution
        fig_hist = px.histogram(filtered_df, x='tariff_rate', nbins=20,
                               title="Distribution of Tariff Rates")
        fig_hist.update_layout(
            xaxis_title="Tariff Rate (%)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Tariff vs Trade Value correlation
        fig_scatter = px.scatter(filtered_df, x='tariff_rate', y='trade_value',
                                color='product', title="Tariff Rate vs Trade Value")
        fig_scatter.update_layout(
            xaxis_title="Tariff Rate (%)",
            yaxis_title="Trade Value ($)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Country-wise tariff analysis
    st.subheader("🌍 Country-wise Tariff Analysis")
    
    # Average tariffs by country (as importer)
    country_tariffs = filtered_df.groupby('importer').agg({
        'tariff_rate': ['mean', 'std', 'count'],
        'trade_value': 'sum'
    }).round(2)
    
    country_tariffs.columns = ['Avg_Tariff', 'Tariff_Std', 'Trade_Routes', 'Total_Imports']
    country_tariffs = country_tariffs.sort_values('Avg_Tariff', ascending=False).head(10)
    
    fig_country_tariff = px.bar(country_tariffs.reset_index(), 
                               x='importer', y='Avg_Tariff',
                               title="Average Tariff Rates by Importing Country (Top 10)")
    fig_country_tariff.update_layout(
        xaxis_title="Importing Country",
        yaxis_title="Average Tariff Rate (%)"
    )
    st.plotly_chart(fig_country_tariff, use_container_width=True)

# Enhanced data download with prediction metadata
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Data Export")

if st.sidebar.button("Download Trade Data"):
    csv = filtered_df.to_csv(index=False)
    filename_suffix = f"_predictions_{selected_year}" if selected_year == 2025 else f"_{selected_year}"
    st.sidebar.download_button(
        label="Download CSV File",
        data=csv,
        file_name=f'trade_data{filename_suffix}.csv',
        mime='text/csv'
    )

# Enhanced usage instructions
with st.expander("📖 User Guide"):
    st.markdown("""
    ### 🌟 Enhanced Platform Features:
    
    #### 🔮 **NEW: 2025 Predictions**
    - 📈 AI-powered trade flow predictions
    - 📊 Confidence intervals and uncertainty analysis
    - 🎯 Growth rate forecasts by product and country
    - 📉 Risk assessment for trade relationships
    
    #### 📊 **Prediction Methodology**
    - **Historical Trend Analysis**: Based on 2021-2024 data patterns
    - **Confidence Scoring**: 60-90% confidence levels
    - **Uncertainty Intervals**: Upper and lower prediction bounds
    - **Growth Modeling**: Incorporates economic factors and volatility
    
    #### 🎮 **Enhanced Controls**
    - **Prediction Filters**: Filter by confidence threshold
    - **Data Type Selection**: Historical vs Predicted data
    - **Uncertainty Visualization**: Color-coded confidence levels
    - **Interactive Tooltips**: Detailed prediction metadata
    
    ### 💡 **Use Cases for Predictions:**
    - 🎯 **Strategic Planning**: Plan for 2025 trade opportunities
    - 📊 **Risk Management**: Identify high-uncertainty trade routes
    - 💼 **Investment Decisions**: Focus on high-confidence predictions
    - 📈 **Policy Analysis**: Understand predicted trade pattern changes
    
    ### ⚠️ **Prediction Disclaimers:**
    - Based on simulated data for demonstration purposes
    - Real-world predictions require comprehensive economic models
    - Confidence levels indicate model uncertainty, not market guarantees
    """)
