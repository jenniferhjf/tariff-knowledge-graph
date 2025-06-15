
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Trade Knowledge Graph Visualization",
    page_icon="🌍",
    layout="wide"
)

# Main title
st.title("🌍 Global Trade Knowledge Graph Visualization Platform")
st.markdown("**Comprehensive analysis platform integrating geographical data, trade flows, commodity classification and tariff information**")

# Sidebar
st.sidebar.header("🔧 Control Panel")

# Generate mock trade data
@st.cache_data
def generate_trade_data():
    """Generate mock trade data"""
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
    
    trade_data = []
    for _ in range(200):
        exporter = np.random.choice(list(countries.keys()))
        importer = np.random.choice(list(countries.keys()))
        while importer == exporter:
            importer = np.random.choice(list(countries.keys()))
        
        trade_data.append({
            'exporter': exporter,
            'importer': importer,
            'exporter_lat': countries[exporter]['lat'],
            'exporter_lon': countries[exporter]['lon'],
            'importer_lat': countries[importer]['lat'],
            'importer_lon': countries[importer]['lon'],
            'product': np.random.choice(products),
            'trade_value': np.random.uniform(1000000, 1000000000),
            'tariff_rate': np.random.uniform(0, 25),
            'hs_code': f"{np.random.choice([1,2,3,4,5,6,7,8,9])}{np.random.randint(10,99)}{np.random.randint(10,99)}",
            'year': np.random.choice([2021, 2022, 2023, 2024]),
            'month': np.random.randint(1, 13)
        })
    
    return pd.DataFrame(trade_data), countries, products

# Generate data
trade_df, countries_info, products_list = generate_trade_data()

# Sidebar controls
view_mode = st.sidebar.selectbox(
    "📊 View Mode",
    ["Geographic Trade Flow", "Knowledge Graph Network", "Comprehensive Dashboard", "Product Analysis", "Tariff Analysis"]
)

selected_year = st.sidebar.selectbox("📅 Year", sorted(trade_df['year'].unique()))
selected_products = st.sidebar.multiselect("📦 Product Categories", products_list, default=products_list[:3])

# Filter data
filtered_df = trade_df[
    (trade_df['year'] == selected_year) & 
    (trade_df['product'].isin(selected_products))
].copy()

# Main content area
if view_mode == "Geographic Trade Flow":
    st.header("🌍 Global Trade Flow Map")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create trade flow map
        arc_layer = pdk.Layer(
            "ArcLayer",
            data=filtered_df,
            get_source_position=["exporter_lon", "exporter_lat"],
            get_target_position=["importer_lon", "importer_lat"],
            get_source_color=[255, 0, 0, 160],
            get_target_color=[0, 255, 0, 160],
            get_width="trade_value/100000000",
            width_scale=1,
            width_min_pixels=2,
            pickable=True,
        )
        
        # Country nodes layer
        country_data = []
        for country, info in countries_info.items():
            exports = filtered_df[filtered_df['exporter'] == country]['trade_value'].sum()
            imports = filtered_df[filtered_df['importer'] == country]['trade_value'].sum()
            country_data.append({
                'country': country,
                'lat': info['lat'],
                'lon': info['lon'],
                'total_trade': exports + imports,
                'exports': exports,
                'imports': imports
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
                    "html": "<b>Trade Flow:</b><br/>From {exporter} to {importer}<br/>Product: {product}<br/>Value: ${trade_value:,.0f}<br/>Tariff Rate: {tariff_rate:.1f}%",
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
        
        top_exporters = filtered_df.groupby('exporter')['trade_value'].sum().sort_values(ascending=False).head(5)
        st.subheader("🏆 Top Exporters")
        for country, value in top_exporters.items():
            st.write(f"**{country}**: ${value/1e9:.1f}B")

elif view_mode == "Knowledge Graph Network":
    st.header("🕸️ Trade Relationship Knowledge Graph")
    
    # Build network graph
    G = nx.Graph()
    
    # Add country nodes
    for country in countries_info.keys():
        exports = filtered_df[filtered_df['exporter'] == country]['trade_value'].sum()
        imports = filtered_df[filtered_df['importer'] == country]['trade_value'].sum()
        G.add_node(country, 
                  type='country',
                  trade_volume=exports + imports,
                  exports=exports,
                  imports=imports)
    
    # Add product nodes
    for product in selected_products:
        trade_vol = filtered_df[filtered_df['product'] == product]['trade_value'].sum()
        G.add_node(product, 
                  type='product',
                  trade_volume=trade_vol)
    
    # Add trade relationship edges
    for _, row in filtered_df.iterrows():
        # Country-to-country trade relationships
        if G.has_edge(row['exporter'], row['importer']):
            G[row['exporter']][row['importer']]['weight'] += row['trade_value']
        else:
            G.add_edge(row['exporter'], row['importer'], 
                      weight=row['trade_value'],
                      relation='trade')
        
        # Country-product relationships
        G.add_edge(row['exporter'], row['product'], 
                  weight=row['trade_value'],
                  relation='export')
        G.add_edge(row['importer'], row['product'], 
                  weight=row['trade_value'],
                  relation='import')
    
    # Use spring layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Create Plotly network graph
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_info.append(f"{edge[0]} - {edge[1]}")
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=0.5, color='#888'),
                           hoverinfo='none',
                           mode='lines')
    
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        if G.nodes[node]['type'] == 'country':
            node_colors.append('lightblue')
            node_sizes.append(max(10, G.nodes[node]['trade_volume']/1e8))
        else:  # product
            node_colors.append('lightcoral')
            node_sizes.append(15)
    
    node_trace = go.Scatter(x=node_x, y=node_y,
                           mode='markers+text',
                           hoverinfo='text',
                           text=node_text,
                           textposition="middle center",
                           marker=dict(size=node_sizes,
                                     color=node_colors,
                                     line=dict(width=2)))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Trade Relationship Knowledge Graph',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Node size represents trade volume, Blue=Countries, Red=Products",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor="left", yanchor="bottom",
                           font=dict(color="black", size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Network statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Nodes", G.number_of_nodes())
    with col2:
        st.metric("Number of Edges", G.number_of_edges())
    with col3:
        if G.number_of_nodes() > 0:
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            st.metric("Average Degree", f"{avg_degree:.1f}")

elif view_mode == "Comprehensive Dashboard":
    st.header("📊 Comprehensive Trade Data Dashboard")
    
    # First row: Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_value = filtered_df['trade_value'].sum()
        st.metric("Total Trade Value", f"${total_value/1e9:.1f}B")
    
    with col2:
        unique_routes = len(filtered_df[['exporter', 'importer']].drop_duplicates())
        st.metric("Trade Routes", f"{unique_routes}")
    
    with col3:
        avg_tariff = filtered_df['tariff_rate'].mean()
        st.metric("Average Tariff", f"{avg_tariff:.1f}%")
    
    with col4:
        unique_products = len(filtered_df['product'].unique())
        st.metric("Product Categories", f"{unique_products}")
    
    # Second row: Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Trade value distribution by product
        product_trade = filtered_df.groupby('product')['trade_value'].sum().sort_values(ascending=False)
        fig_pie = px.pie(values=product_trade.values, names=product_trade.index, 
                        title="Trade Value Distribution by Product")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Monthly trade trends
        monthly_trade = filtered_df.groupby('month')['trade_value'].sum()
        fig_line = px.line(x=monthly_trade.index, y=monthly_trade.values, 
                          title="Monthly Trade Trends")
        fig_line.update_xaxis(title="Month")
        fig_line.update_yaxis(title="Trade Value ($)")
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Third row: Detailed table
    st.subheader("🔍 Detailed Trade Records")
    display_df = filtered_df.copy()
    display_df['trade_value'] = display_df['trade_value'].apply(lambda x: f"${x:,.0f}")
    display_df['tariff_rate'] = display_df['tariff_rate'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        display_df[['exporter', 'importer', 'product', 'trade_value', 'tariff_rate', 'hs_code']],
        use_container_width=True
    )

elif view_mode == "Product Analysis":
    st.header("📦 Product Trade Analysis")
    
    # Product selection
    selected_product = st.selectbox("Select Product for Analysis", products_list)
    product_data = filtered_df[filtered_df['product'] == selected_product]
    
    if not product_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top exporters
            exporters = product_data.groupby('exporter')['trade_value'].sum().sort_values(ascending=False)
            fig_bar = px.bar(x=exporters.index, y=exporters.values, 
                            title=f"{selected_product} - Top Exporters")
            fig_bar.update_xaxis(title="Country")
            fig_bar.update_yaxis(title="Export Value ($)")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Top importers
            importers = product_data.groupby('importer')['trade_value'].sum().sort_values(ascending=False)
            fig_bar2 = px.bar(x=importers.index, y=importers.values, 
                             title=f"{selected_product} - Top Importers")
            fig_bar2.update_xaxis(title="Country")
            fig_bar2.update_yaxis(title="Import Value ($)")
            st.plotly_chart(fig_bar2, use_container_width=True)
        
        # Tariff analysis
        st.subheader("📊 Tariff Analysis")
        tariff_stats = product_data.groupby(['exporter', 'importer']).agg({
            'tariff_rate': 'mean',
            'trade_value': 'sum'
        }).reset_index()
        
        fig_scatter = px.scatter(tariff_stats, x='tariff_rate', y='trade_value',
                               hover_data=['exporter', 'importer'],
                               title=f"{selected_product} - Tariff Rate vs Trade Value")
        fig_scatter.update_xaxis(title="Tariff Rate (%)")
        fig_scatter.update_yaxis(title="Trade Value ($)")
        st.plotly_chart(fig_scatter, use_container_width=True)

elif view_mode == "Tariff Analysis":
    st.header("💰 Tariff Impact Analysis")
    
    # Tariff impact on trade
    tariff_impact = filtered_df.copy()
    tariff_impact['tariff_category'] = pd.cut(tariff_impact['tariff_rate'], 
                                            bins=[0, 5, 10, 15, 25], 
                                            labels=['Low Tariff (0-5%)', 'Medium-Low (5-10%)', 
                                                   'Medium-High (10-15%)', 'High Tariff (15-25%)'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tariff distribution
        tariff_dist = tariff_impact['tariff_category'].value_counts()
        fig_pie = px.pie(values=tariff_dist.values, names=tariff_dist.index,
                        title="Tariff Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Tariff vs trade volume relationship
        tariff_trade = tariff_impact.groupby('tariff_category')['trade_value'].sum()
        fig_bar = px.bar(x=tariff_trade.index, y=tariff_trade.values,
                        title="Trade Value by Tariff Level")
        fig_bar.update_xaxis(title="Tariff Level")
        fig_bar.update_yaxis(title="Trade Value ($)")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Bilateral tariff heatmap
    st.subheader("🌡️ Bilateral Tariff Heatmap")
    tariff_matrix = filtered_df.pivot_table(values='tariff_rate', 
                                           index='exporter', 
                                           columns='importer', 
                                           aggfunc='mean')
    
    fig_heatmap = px.imshow(tariff_matrix, 
                           title="Average Bilateral Tariff Rates",
                           labels=dict(x="Importer", y="Exporter", color="Tariff Rate (%)"))
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Data download
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Data Export")

if st.sidebar.button("Download Trade Data"):
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV File",
        data=csv,
        file_name=f'trade_data_{selected_year}.csv',
        mime='text/csv'
    )

# Usage instructions
with st.expander("📖 User Guide"):
    st.markdown("""
    ### 🌟 Platform Features:
    
    #### 1. **Geographic Trade Flow**
    - 🗺️ Global trade flow visualization
    - 📊 Country trade volume display
    - 🔍 Interactive trade information viewer
    
    #### 2. **Knowledge Graph Network**
    - 🕸️ Country and product relationship network
    - 📈 Trade relationship strength visualization
    - 🎯 Network structure analysis
    
    #### 3. **Comprehensive Dashboard**
    - 📊 Multi-dimensional data statistics
    - 📈 Trend analysis charts
    - 📋 Detailed data tables
    
    #### 4. **Product Analysis**
    - 📦 In-depth single product analysis
    - 🏆 Major trading partner identification
    - 💰 Tariff impact assessment
    
    #### 5. **Tariff Analysis**
    - 📊 Tariff distribution statistics
    - 🌡️ Bilateral tariff heatmap
    - 📈 Tariff impact on trade
    
    ### 🔧 Data Description:
    - **Data Source**: Simulated global trade data
    - **Time Range**: 2021-2024
    - **Components**: Trade flows, commodity classification, tariff rates, HS codes
    
    ### 💡 Use Cases:
    - 🌍 International trade analysis
    - 📊 Supply chain optimization
    - 💼 Investment decision support
    - 📈 Policy impact assessment
    
    ### 🎮 Controls:
    - **Year Filter**: Select specific year for analysis
    - **Product Filter**: Choose product categories to analyze
    - **Interactive Maps**: Click and hover for detailed information
    - **Data Export**: Download filtered data as CSV
    """)
