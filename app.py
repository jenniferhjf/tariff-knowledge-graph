
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime
import json

# 页面配置
st.set_page_config(
    page_title="关税知识图谱",
    page_icon="🌐",
    layout="wide"
)

# 标题
st.title("🌐 关税知识图谱可视化")

# 侧边栏控制
st.sidebar.header("图谱控制")

# 示例数据生成函数
@st.cache_data
def generate_sample_data():
    """生成示例关税数据"""
    countries = ['中国', '美国', '欧盟', '日本', '韩国', '东盟', '印度', '巴西']
    products = ['电子产品', '汽车', '纺织品', '化工产品', '农产品', '钢铁', '机械设备']
    
    nodes = []
    edges = []
    
    # 添加国家节点
    for i, country in enumerate(countries):
        nodes.append({
            'id': country,
            'label': country,
            'type': 'country',
            'size': 20,
            'color': 'lightblue'
        })
    
    # 添加产品节点
    for i, product in enumerate(products):
        nodes.append({
            'id': product,
            'label': product,
            'type': 'product',
            'size': 15,
            'color': 'lightgreen'
        })
    
    # 添加关税关系边
    np.random.seed(42)
    for country in countries:
        for product in products:
            if np.random.random() > 0.6:  # 随机生成关系
                tariff_rate = np.random.uniform(0, 25)
                edges.append({
                    'source': country,
                    'target': product,
                    'weight': tariff_rate,
                    'label': f'{tariff_rate:.1f}%'
                })
    
    # 添加国家间贸易关系
    for i, country1 in enumerate(countries):
        for j, country2 in enumerate(countries[i+1:], i+1):
            if np.random.random() > 0.7:
                trade_volume = np.random.uniform(100, 1000)
                edges.append({
                    'source': country1,
                    'target': country2,
                    'weight': trade_volume,
                    'label': f'${trade_volume:.0f}B'
                })
    
    return nodes, edges

# 创建网络图函数
def create_network_graph(nodes, edges, layout_type='spring'):
    """创建网络图"""
    try:
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加节点
        for node in nodes:
            G.add_node(node['id'], **node)
        
        # 添加边
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # 选择布局算法
        if layout_type == 'spring':
            pos = nx.spring_layout(G, k=3, iterations=50)
        elif layout_type == 'circular':
            pos = nx.circular_layout(G)
        elif layout_type == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.random_layout(G)
        
        # 准备边的数据
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # 获取边的信息
            edge_data = G.edges[edge]
            weight = edge_data.get('weight', 0)
            label = edge_data.get('label', '')
            edge_info.append(f"{edge[0]} - {edge[1]}: {label}")
        
        # 创建边的轨迹
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 准备节点数据
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # 获取节点信息
            node_data = G.nodes[node]
            node_type = node_data.get('type', 'unknown')
            node_colors.append(node_data.get('color', 'lightblue'))
            node_sizes.append(node_data.get('size', 10))
            
            # 计算连接数
            adjacencies = list(G.neighbors(node))
            node_info.append(f'{node}<br>类型: {node_type}<br>连接数: {len(adjacencies)}')
            node_text.append(node)
        
        # 创建节点轨迹
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.05,
                    title="节点类型"
                ),
                line=dict(width=2)
            )
        )
        
        # 创建图形
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title={
                    'text': '关税知识图谱',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24}
                },
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="交互式网络图 - 悬停查看详细信息",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color='rgb(150,150,150)', size=10)
                    )
                ],
                xaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    visible=False
                ),
                yaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    visible=False
                ),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
        )
        
        return fig, G
        
    except Exception as e:
        st.error(f"创建网络图时出错: {str(e)}")
        return None, None

# 主应用逻辑
def main():
    # 侧边栏选项
    layout_options = {
        'spring': 'Spring布局',
        'circular': '圆形布局', 
        'kamada_kawai': 'Kamada-Kawai布局'
    }
    
    selected_layout = st.sidebar.selectbox(
        "选择布局算法",
        options=list(layout_options.keys()),
        format_func=lambda x: layout_options[x]
    )
    
    # 生成数据
    with st.spinner('正在生成数据...'):
        nodes, edges = generate_sample_data()
    
    # 显示统计信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("节点数量", len(nodes))
    with col2:
        st.metric("边数量", len(edges))
    with col3:
        st.metric("连通性", f"{(len(edges)/len(nodes)):.1f}")
    
    # 创建和显示网络图
    with st.spinner('正在创建网络图...'):
        fig, graph = create_network_graph(nodes, edges, selected_layout)
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, height=600)
        else:
            st.error("无法创建网络图")
    
    # 数据表格展示
    st.subheader("📊 数据详情")
    
    tab1, tab2 = st.tabs(["节点数据", "边数据"])
    
    with tab1:
        nodes_df = pd.DataFrame(nodes)
        st.dataframe(nodes_df, use_container_width=True)
    
    with tab2:
        edges_df = pd.DataFrame(edges)
        st.dataframe(edges_df, use_container_width=True)
    
    # 图谱分析
    if graph is not None:
        st.subheader("🔍 图谱分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 度分布
            degrees = dict(graph.degree())
            degree_df = pd.DataFrame(
                list(degrees.items()), 
                columns=['节点', '度数']
            ).sort_values('度数', ascending=False)
            
            fig_degree = px.bar(
                degree_df.head(10), 
                x='节点', y='度数',
                title='节点度数排名 (Top 10)'
            )
            st.plotly_chart(fig_degree, use_container_width=True)
        
        with col2:
            # 中心性分析
            try:
                centrality = nx.betweenness_centrality(graph)
                centrality_df = pd.DataFrame(
                    list(centrality.items()),
                    columns=['节点', '中介中心性']
                ).sort_values('中介中心性', ascending=False)
                
                fig_centrality = px.bar(
                    centrality_df.head(10),
                    x='节点', y='中介中心性',
                    title='中介中心性排名 (Top 10)'
                )
                st.plotly_chart(fig_centrality, use_container_width=True)
            except:
                st.info("图谱不连通，无法计算中介中心性")

# 运行应用
if __name__ == "__main__":
    main()
