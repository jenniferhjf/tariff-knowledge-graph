import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
import requests
import json
import time
from typing import Dict, List, Any
import os

# 页面配置
st.set_page_config(
    page_title="关税知识图谱系统",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitTariffSystem:
    """Streamlit关税知识图谱系统"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.countries = ["China", "USA", "Germany", "Japan", "Canada", "UK", "France", "Italy"]
        self.products = ["Steel", "Electronics", "Textiles", "Agriculture", "Automobiles", "Chemicals"]
        
        # 初始化会话状态
        if 'graph_built' not in st.session_state:
            st.session_state.graph_built = False
        if 'graph_data' not in st.session_state:
            st.session_state.graph_data = None
    
    def get_sample_tariff_data(self) -> Dict[str, Any]:
        """获取示例关税数据"""
        return {
            "entities": [
                {"id": "china", "name": "中国", "type": "country", "region": "Asia"},
                {"id": "usa", "name": "美国", "type": "country", "region": "North America"},
                {"id": "germany", "name": "德国", "type": "country", "region": "Europe"},
                {"id": "japan", "name": "日本", "type": "country", "region": "Asia"},
                {"id": "steel", "name": "钢铁", "type": "product", "hs_code": "72"},
                {"id": "electronics", "name": "电子产品", "type": "product", "hs_code": "85"},
                {"id": "textiles", "name": "纺织品", "type": "product", "hs_code": "61"},
                {"id": "tariff_25", "name": "25%关税", "type": "tariff", "rate": 25},
                {"id": "tariff_15", "name": "15%关税", "type": "tariff", "rate": 15},
                {"id": "tariff_10", "name": "10%关税", "type": "tariff", "rate": 10},
                {"id": "usmca", "name": "美墨加协定", "type": "agreement", "year": 2020},
                {"id": "rcep", "name": "RCEP", "type": "agreement", "year": 2022}
            ],
            "relationships": [
                {"source": "china", "target": "steel", "relation": "exports", "volume": 100000000},
                {"source": "china", "target": "electronics", "relation": "exports", "volume": 500000000},
                {"source": "usa", "target": "tariff_25", "relation": "imposes", "year": 2018},
                {"source": "tariff_25", "target": "steel", "relation": "applies_to", "scope": "imports"},
                {"source": "usa", "target": "tariff_15", "relation": "imposes", "year": 2019},  
                {"source": "tariff_15", "target": "electronics", "relation": "applies_to", "scope": "imports"},
                {"source": "china", "target": "rcep", "relation": "member_of", "status": "active"},
                {"source": "japan", "target": "rcep", "relation": "member_of", "status": "active"},
                {"source": "usa", "target": "usmca", "relation": "member_of", "status": "active"}
            ]
        }
    
    def build_knowledge_graph(self, data: Dict[str, Any]):
        """构建知识图谱"""
        self.graph.clear()
        
        # 添加节点
        for entity in data["entities"]:
            self.graph.add_node(
                entity["id"],
                name=entity["name"],
                type=entity["type"],
                **{k: v for k, v in entity.items() if k not in ["id", "name", "type"]}
            )
        
        # 添加边
        for rel in data["relationships"]:
            self.graph.add_edge(
                rel["source"],
                rel["target"],
                relation=rel["relation"],
                **{k: v for k, v in rel.items() if k not in ["source", "target", "relation"]}
            )
        
        st.session_state.graph_built = True
        st.session_state.graph_data = data
    
    def create_plotly_graph(self):
        """创建Plotly交互式图谱"""
        if self.graph.number_of_nodes() == 0:
            return None
        
        # 计算布局
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # 准备边数据
        edge_x, edge_y = [], []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # 创建边轨迹
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 准备节点数据
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        
        color_map = {
            'country': '#3B82F6',
            'product': '#10B981', 
            'tariff': '#EF4444',
            'agreement': '#F59E0B'
        }
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_info = self.graph.nodes[node]
            node_text.append(node_info['name'])
            node_color.append(color_map.get(node_info.get('type'), '#999999'))
            node_size.append(30 if node_info.get('type') == 'country' else 20)
        
        # 创建节点轨迹
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
        
        # 创建图形
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='关税知识图谱',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig
    
    def create_pyvis_graph(self):
        """创建Pyvis交互式网络图"""
        if self.graph.number_of_nodes() == 0:
            return None
        
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        
        color_map = {
            'country': '#3B82F6',
            'product': '#10B981', 
            'tariff': '#EF4444',
            'agreement': '#F59E0B'
        }
        
        # 添加节点
        for node_id, data in self.graph.nodes(data=True):
            net.add_node(
                node_id,
                label=data['name'],
                color=color_map.get(data.get('type'), '#999999'),
                title=f"类型: {data.get('type')}\n名称: {data['name']}",
                size=25 if data.get('type') == 'country' else 15
            )
        
        # 添加边
        for source, target, data in self.graph.edges(data=True):
            net.add_edge(
                source, target,
                title=data.get('relation', ''),
                color='#999999'
            )
        
        # 设置物理效果
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
          }
        }
        """)
        
        # 保存为HTML
        net.save_graph("temp_graph.html")
        
        # 读取HTML内容
        with open("temp_graph.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        
        return html_content
    
    def query_tariff_rate(self, export_country: str, import_country: str, product: str) -> Dict[str, Any]:
        """查询关税税率"""
        # 模拟数据库查询
        tariff_db = {
            ("China", "USA", "Steel"): {"rate": "25%", "effective_date": "2018-07-06", "status": "Active"},
            ("China", "USA", "Electronics"): {"rate": "15%", "effective_date": "2019-05-10", "status": "Active"},
            ("Germany", "USA", "Steel"): {"rate": "10%", "effective_date": "2018-06-01", "status": "Active"},
            ("Japan", "USA", "Electronics"): {"rate": "5%", "effective_date": "2020-01-01", "status": "Active"}
        }
        
        key = (export_country, import_country, product)
        return tariff_db.get(key, {"message": "未找到相关关税信息"})
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # 节点类型统计
        type_counts = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        # 关系类型统计
        relation_counts = {}
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get('relation', 'unknown')
            relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "node_types": type_counts,
            "relation_types": relation_counts
        }

def main():
    """主函数"""
    
    # 标题
    st.markdown('<h1 class="main-header">🌐 关税知识图谱系统</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 初始化系统
    tariff_system = StreamlitTariffSystem()
    
    # 侧边栏
    with st.sidebar:
        st.header("🔧 系统控制")
        
        # 数据加载
        if st.button("📊 加载示例数据", type="primary"):
            with st.spinner("正在加载数据..."):
                data = tariff_system.get_sample_tariff_data()
                tariff_system.build_knowledge_graph(data)
                time.sleep(1)  # 模拟加载时间
            st.success("数据加载完成！")
        
        st.markdown("---")
        
        # 关税查询
        st.header("🔍 关税查询")
        export_country = st.selectbox("出口国", tariff_system.countries)
        import_country = st.selectbox("进口国", tariff_system.countries) 
        product = st.selectbox("产品类别", tariff_system.products)
        
        if st.button("查询关税"):
            result = tariff_system.query_tariff_rate(export_country, import_country, product)
            
            if "message" in result:
                st.warning(result["message"])
            else:
                st.success("查询成功！")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("关税税率", result["rate"])
                with col2:
                    st.metric("状态", result["status"])
                st.info(f"生效日期: {result['effective_date']}")
    
    # 主要内容区域
    if not st.session_state.graph_built:
        # 欢迎页面
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👈 请点击左侧 '加载示例数据' 开始使用系统")
            
            # 功能介绍
            st.markdown("""
            ### 🚀 系统功能
            - **知识图谱可视化**: 交互式图谱展示关税关系
            - **实时关税查询**: 快速查询双边关税信息  
            - **数据分析报告**: 自动生成统计分析
            - **多种可视化方式**: Plotly + Pyvis双重展示
            """)
    else:
        # 图谱展示区域
        tab1, tab2, tab3, tab4 = st.tabs(["📊 统计概览", "🌐 知识图谱", "📈 数据分析", "📋 详细数据"])
        
        with tab1:
            st.header("📊 图谱统计概览")
            
            stats = tariff_system.get_graph_statistics()
            if stats:
                # 关键指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("节点总数", stats["total_nodes"])
                with col2:
                    st.metric("边总数", stats["total_edges"])  
                with col3:
                    st.metric("图密度", f"{stats['density']:.3f}")
                with col4:
                    st.metric("连通性", "Strong" if stats["density"] > 0.3 else "Moderate")
                
                # 节点类型分布
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("节点类型分布")
                    node_df = pd.DataFrame(list(stats["node_types"].items()), 
                                         columns=["类型", "数量"])
                    fig_pie = px.pie(node_df, values="数量", names="类型", 
                                   title="节点类型分布")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.subheader("关系类型分布")
                    rel_df = pd.DataFrame(list(stats["relation_types"].items()), 
                                        columns=["关系", "数量"])
                    fig_bar = px.bar(rel_df, x="关系", y="数量", 
                                   title="关系类型分布")
                    st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            st.header("🌐 交互式知识图谱")
            
            # 可视化方式选择
            viz_type = st.radio("选择可视化方式", ["Plotly图谱", "Pyvis网络图"], horizontal=True)
            
            if viz_type == "Plotly图谱":
                fig = tariff_system.create_plotly_graph()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 图例
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown("🔵 **国家**")
                    with col2:
                        st.markdown("🟢 **产品**") 
                    with col3:
                        st.markdown("🔴 **关税**")
                    with col4:
                        st.markdown("🟡 **协定**")
                        
            else:  # Pyvis网络图
                html_content = tariff_system.create_pyvis_graph()
                if html_content:
                    components.html(html_content, height=600)
                    st.info("💡 提示: 可以拖拽节点，滚轮缩放，点击查看详情")
        
        with tab3:
            st.header("📈 数据分析报告")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # 生成分析报告
                stats = tariff_system.get_graph_statistics()
                if stats:
                    st.markdown(f"""
                    ### 分析摘要
                    
                    **图谱规模**
                    - 节点数量: {stats['total_nodes']}个
                    - 边数量: {stats['total_edges']}条
                    - 图密度: {stats['density']:.4f}
                    
                    **结构特征**
                    - 国家节点: {stats['node_types'].get('country', 0)}个
                    - 产品节点: {stats['node_types'].get('product', 0)}个  
                    - 关税节点: {stats['node_types'].get('tariff', 0)}个
                    - 协定节点: {stats['node_types'].get('agreement', 0)}个
                    
                    **关系分布**
                    """)
                    
                    for rel_type, count in stats['relation_types'].items():
                        st.markdown(f"- {rel_type}: {count}条")
            
            with col2:
                st.info("""
                📊 **分析建议**
                
                1. 图谱密度适中，结构清晰
                2. 国家间关税关系复杂
                3. 贸易协定影响显著
                4. 建议进一步细化产品分类
                """)
        
        with tab4:
            st.header("📋 详细数据表")
            
            if st.session_state.graph_data:
                # 实体数据表
                st.subheader("实体数据")
                entities_df = pd.DataFrame(st.session_state.graph_data["entities"])
                st.dataframe(entities_df, use_container_width=True)
                
                # 关系数据表  
                st.subheader("关系数据")
                relationships_df = pd.DataFrame(st.session_state.graph_data["relationships"])
                st.dataframe(relationships_df, use_container_width=True)
                
                # 数据导出
                col1, col2 = st.columns(2)
                with col1:
                    entities_csv = entities_df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载实体数据",
                        data=entities_csv,
                        file_name="tariff_entities.csv",
                        mime="text/csv"
                    )
                with col2:
                    relationships_csv = relationships_df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载关系数据", 
                        data=relationships_csv,
                        file_name="tariff_relationships.csv",
                        mime="text/csv"
                    )
    
    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            💡 关税知识图谱系统 | 基于Streamlit构建 | 
            <a href='https://github.com/your-username/tariff-knowledge-graph' target='_blank'>GitHub</a>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
