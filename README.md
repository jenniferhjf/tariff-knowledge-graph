
# 🌐 关税知识图谱系统

一个基于Streamlit的交互式关税知识图谱可视化系统，支持实时关税查询、知识图谱构建和数据分析。

## ✨ 功能特点

- 🌐 **交互式知识图谱**: 支持Plotly和Pyvis双重可视化
- 🔍 **实时关税查询**: 快速查询双边关税信息
- 📊 **数据分析报告**: 自动生成统计分析和图谱指标
- 📱 **响应式设计**: 适配不同屏幕尺寸
- 🚀 **易于部署**: 支持Streamlit Cloud一键部署

## 🖥️ 在线演示

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 🛠️ 本地安装

### 前置要求

- Python 3.8+
- pip

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/your-username/tariff-knowledge-graph.git
cd tariff-knowledge-graph
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **运行应用**
```bash
streamlit run app.py
```

5. **访问应用**
打开浏览器访问 `http://localhost:8501`

## 🚀 部署到Streamlit Cloud

### 方法1: GitHub连接部署

1. **Fork此仓库**到您的GitHub账户

2. **登录Streamlit Cloud**
   - 访问 [share.streamlit.io](https://share.streamlit.io)
   - 使用GitHub账户登录

3. **创建新应用**
   - 点击 "New app"
   - 选择您fork的仓库
   - 主文件路径: `app.py`
   - 点击 "Deploy!"

4. **等待部署完成**
   - 通常需要2-5分钟
   - 部署成功后会生成公开访问链接

### 方法2: 直接上传部署

1. **打包项目文件**
```bash
zip -r tariff-kg.zip app.py requirements.txt README.md
```

2. **在Streamlit Cloud上传**
   - 选择 "Upload files"
   - 上传zip文件
   - 设置主文件为 `app.py`

## 📁 项目结构

```
tariff-knowledge-graph/
├── app.py              # 主应用文件
├── requirements.txt    # Python依赖包
├── README.md          # 项目说明文档
├── .gitignore         # Git忽略文件
└── assets/            # 静态资源（可选）
    └── screenshots/   # 应用截图
```

## 🎯 使用指南

### 基本操作

1. **加载数据**: 点击左侧边栏的"加载示例数据"按钮
2. **查看图谱**: 在"知识图谱"标签页中选择可视化方式
3. **查询关税**: 在左侧边栏选择国家和产品进行查询
4. **分析报告**: 查看"数据分析"标签页的统计信息

### 高级功能

- **节点交互**: 在Pyvis图谱中可以拖拽节点、缩放视图
- **数据导出**: 在"详细数据"标签页下载CSV格式数据
- **多维分析**: 查看不同维度的统计图表

## 🔧 配置选项

### 自定义数据源

修改 `app.py` 中的数据获取函数：

```python
def get_sample_tariff_data(self) -> Dict[str, Any]:
    # 在这里添加您的数据源
    return your_data
```

### API集成

添加真实API调用：

```python
def fetch_real_tariff_data(self, country1, country2):
    # 集成WTO API或其他数据源
    response = requests.get(f"your-api-endpoint")
    return response.json()
```

## 📊 数据格式

### 实体数据格式
```json
{
  "id": "unique_identifier",
  "name": "显示名称", 
  "type": "实体类型",
  "properties": {}
}
```

### 关系数据格式
```json
{
  "source": "源实体ID",
  "target": "目标实体ID",
  "relation": "关系类型",
  "properties": {}
}
```

## 🤝 贡献指南

1. Fork此仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📝 更新日志

### v1.0.0 (2024-12-xx)
- ✨ 初始版本发布
- 🌐 基础知识图谱可视化
- 🔍 关税查询功能
- 📊 数据统计分析

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🆘 支持与反馈

- 📧 邮箱: your-email@example.com
- 🐛 问题反馈: [GitHub Issues](https://github.com/your-username/tariff-knowledge-graph/issues)
- 💬 讨论: [GitHub Discussions](https://github.com/your-username/tariff-knowledge-graph/discussions)

## 🙏 致谢

- [Streamlit](https://streamlit.io/) - 出色的Web应用框架
- [NetworkX](https://networkx.org/) - 强大的图分析库
- [Plotly](https://plotly.com/) - 交互式图表库
- [Pyvis](https://pyvis.readthedocs.io/) - 网络可视化库

---

⭐ 如果这个项目对您有帮助，请给它一个Star！
