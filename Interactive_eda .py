"""
TDM 互動式資料探索分析 (Interactive EDA) - 完整修正版
包含所有優化:動畫曲線、3D圖例縮寫
使用 Plotly 製作互動式圖表
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import sys
import io

# 設定 UTF-8 輸出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("TDM 互動式 EDA 生成程式 - 完整版")
print("=" * 60)
print()

# ============================================
# 模擬 TDM 資料
# ============================================

np.random.seed(42)
n_total = 1745

data = {
    'Patient_ID': range(1, n_total + 1),
    'Drug': np.random.choice(['Vancomycin', 'Digoxin', 'Phenytoin', 'Theophylline', 
                              'Gentamicin', 'Lithium', 'Tacrolimus', 'Cyclosporine',
                              'Carbamazepine', 'Valproic Acid'], n_total),
    'Age': np.random.normal(60, 15, n_total).clip(18, 95),
    'Gender': np.random.choice(['M', 'F'], n_total, p=[0.55, 0.45]),
    'Dose': np.random.uniform(100, 1000, n_total),
    'Level': np.random.uniform(5, 50, n_total),
    'Time': np.random.choice(['Peak', 'Trough'], n_total, p=[0.3, 0.7]),
    'Department': np.random.choice(['ICU', 'Internal Medicine', 'Surgery', 'Pediatrics', 
                                    'Emergency', 'Nephrology'], n_total, p=[0.3, 0.25, 0.15, 0.1, 0.1, 0.1])
}

df = pd.DataFrame(data)

# 加入 Accept 欄位
df['Accept'] = None
accept_indices = np.random.choice(n_total, size=int(n_total * 0.812), replace=False)
accept_values = np.random.choice(['Yes', 'No'], size=len(accept_indices), p=[0.933, 0.067])
for idx, val in zip(accept_indices, accept_values):
    df.loc[idx, 'Accept'] = val

# 加入 Medicine 欄位
df['Medicine'] = None
medicine_indices = np.random.choice(n_total, size=int(n_total * 0.387), replace=False)
medicine_values = np.random.choice(['Adjusted', 'Maintained', 'Changed'], size=len(medicine_indices))
for idx, val in zip(medicine_indices, medicine_values):
    df.loc[idx, 'Medicine'] = val

print(f"資料集大小: {len(df)} 筆")
print()

# ============================================
# 互動式圖表 1: 缺失值分析
# ============================================

def create_interactive_missing_analysis():
    """創建互動式缺失值分析"""
    
    missing_data = df.isnull().sum() / len(df) * 100
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    colors_list = ['#e74c3c' if x > 50 else '#f39c12' if x > 15 else '#3498db' 
                   for x in missing_data.values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=missing_data.index,
        x=missing_data.values,
        orientation='h',
        marker=dict(
            color=colors_list,
            line=dict(color='rgba(0,0,0,0.5)', width=2)
        ),
        text=[f'{val:.1f}%' for val in missing_data.values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>缺失率: %{x:.1f}%<extra></extra>'
    ))
    
    fig.add_vline(x=20, line_dash="dash", line_color="red", 
                  annotation_text="Critical (>20%)", annotation_position="top")
    fig.add_vline(x=10, line_dash="dash", line_color="orange",
                  annotation_text="Warning (>10%)", annotation_position="top")
    
    fig.update_layout(
        title=dict(
            text='<b>Missing Data Analysis</b><br>TDM Dataset (N=1,745)',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis_title='Missing Rate (%)',
        yaxis_title='Field',
        height=500,
        template='plotly_white',
        hovermode='closest',
        showlegend=False
    )
    
    return fig

# ============================================
# 互動式圖表 2: 3D 散點圖 (使用縮寫)
# ============================================

def create_3d_scatter():
    """創建 3D 互動散點圖 - 使用藥名縮寫"""
    
    df_complete = df.dropna(subset=['Accept']).copy()
    
    # 藥物名稱縮寫對照
    drug_short = {
        'Vancomycin': 'Vanc',
        'Digoxin': 'Dig',
        'Phenytoin': 'Phen',
        'Theophylline': 'Theo',
        'Gentamicin': 'Gent',
        'Lithium': 'Li',
        'Tacrolimus': 'Tacro',
        'Cyclosporine': 'Cyclo',
        'Carbamazepine': 'Carba',
        'Valproic Acid': 'VPA'
    }
    
    df_complete.loc[:, 'Drug_Short'] = df_complete['Drug'].map(drug_short)
    df_complete.loc[:, 'Drug_Full'] = df_complete['Drug']
    
    fig = px.scatter_3d(
        df_complete,
        x='Age',
        y='Dose',
        z='Level',
        color='Drug_Short',
        symbol='Accept',
        hover_data=['Drug_Full', 'Department', 'Time'],
        title='<b>3D Interactive Analysis</b><br>Age x Dose x Drug Level',
        labels={
            'Age': 'Age (years)',
            'Dose': 'Dose (mg)',
            'Level': 'Drug Level (ug/mL)',
            'Drug_Short': 'Drug',
            'Drug_Full': 'Drug (Full Name)'
        }
    )
    
    fig.update_traces(marker=dict(size=4, line=dict(width=0.3, color='white')))
    
    fig.update_layout(
        height=850,
        width=1400,
        margin=dict(r=180, l=20, t=80, b=20),
        
        # 調整 3D 場景的顯示區域
        scene=dict(
            domain=dict(
                x=[0, 0.85],  # 3D 圖只佔 85% 寬度
                y=[0, 1]
            ),
            xaxis_title='Age (years)',
            yaxis_title='Dose (mg)',
            zaxis_title='Drug Level (ug/mL)',
            camera=dict(
                eye=dict(x=1.3, y=1.3, z=1.1)
            )
        ),
        
        template='plotly_white',
        
        # 圖例配置
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.87,  # 從 85% 的位置開始
            bgcolor="rgba(255, 255, 255, 0.98)",
            bordercolor="#2c3e50",
            borderwidth=2,
            font=dict(size=12, family="Arial"),
            title=dict(
                text='<b>Legend</b>',
                font=dict(size=14, family='Arial Black', color='#2c3e50')
            ),
            itemsizing='constant',
            tracegroupgap=8
        )
    )
    
    return fig

# ============================================
# 互動式圖表 3: 數據收集動畫 (修正版)
# ============================================

def create_animated_data_collection():
    """創建數據收集過程動畫 - 顯示完整率上升曲線"""
    
    # 創建時間序列資料
    dates = pd.date_range(start='2024-01-01', periods=len(df), freq='2h')
    
    # 模擬資料收集過程 - 完整率逐漸上升
    completeness_rates = []
    for i in range(1, len(df) + 1):
        base_rate = 60 + (81.2 - 60) * (i / len(df))
        noise = np.random.normal(0, 1)
        rate = np.clip(base_rate + noise, 0, 100)
        completeness_rates.append(rate)
    
    df_anim = pd.DataFrame({
        'Collection_Time': dates,
        'Cumulative_Count': range(1, len(df) + 1),
        'Completeness_Rate': completeness_rates
    })
    
    df_plot = df_anim.iloc[::10].copy()
    
    fig = px.line(
        df_plot,
        x='Collection_Time',
        y='Completeness_Rate',
        title='<b>Data Collection Animation</b><br>Completeness Rate Over Time',
        labels={
            'Collection_Time': 'Collection Time',
            'Completeness_Rate': 'Completeness Rate (%)'
        },
        range_y=[0, 100]
    )
    
    fig.update_traces(
        line=dict(color='#3498db', width=3),
        mode='lines+markers',
        marker=dict(size=8)
    )
    
    fig.add_hline(
        y=80, 
        line_dash="dash", 
        line_color="green",
        annotation_text="Target: 80%", 
        annotation_position="right"
    )
    
    fig.add_hline(
        y=81.2, 
        line_dash="dot", 
        line_color="blue",
        annotation_text="Final: 81.2%", 
        annotation_position="left"
    )
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

# ============================================
# 互動式圖表 4: Sunburst 圖
# ============================================

def create_sunburst():
    """創建階層式 Sunburst 圖"""
    
    df_complete = df.dropna(subset=['Accept'])
    
    fig = px.sunburst(
        df_complete,
        path=['Department', 'Drug', 'Accept'],
        title='<b>Hierarchical Distribution</b><br>Department -> Drug -> Acceptance',
        color='Accept',
        color_discrete_map={'Yes': '#27ae60', 'No': '#e74c3c'}
    )
    
    fig.update_traces(
        textinfo='label+percent parent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent}<extra></extra>'
    )
    
    fig.update_layout(
        height=700,
        template='plotly_white'
    )
    
    return fig

# ============================================
# 互動式圖表 5: 平行座標圖
# ============================================

def create_parallel_coordinates():
    """創建平行座標圖"""
    
    df_complete = df.dropna(subset=['Accept']).copy()
    
    df_complete.loc[:, 'Drug_Code'] = pd.Categorical(df_complete['Drug']).codes
    df_complete.loc[:, 'Dept_Code'] = pd.Categorical(df_complete['Department']).codes
    df_complete.loc[:, 'Accept_Code'] = (df_complete['Accept'] == 'Yes').astype(int)
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df_complete['Accept_Code'],
                colorscale=[[0, '#e74c3c'], [1, '#27ae60']],
                showscale=True,
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title="Accept",
                    tickvals=[0, 1],
                    ticktext=['No', 'Yes']
                )
            ),
            dimensions=[
                dict(range=[df_complete['Age'].min(), df_complete['Age'].max()],
                     label='Age', values=df_complete['Age']),
                dict(range=[df_complete['Dose'].min(), df_complete['Dose'].max()],
                     label='Dose', values=df_complete['Dose']),
                dict(range=[df_complete['Level'].min(), df_complete['Level'].max()],
                     label='Level', values=df_complete['Level']),
                dict(range=[0, df_complete['Drug_Code'].max()],
                     label='Drug', values=df_complete['Drug_Code'],
                     tickvals=list(range(df_complete['Drug_Code'].max() + 1)),
                     ticktext=df_complete.groupby('Drug_Code')['Drug'].first().tolist()),
                dict(range=[0, df_complete['Dept_Code'].max()],
                     label='Department', values=df_complete['Dept_Code'],
                     tickvals=list(range(df_complete['Dept_Code'].max() + 1)),
                     ticktext=df_complete.groupby('Dept_Code')['Department'].first().tolist()),
            ]
        )
    )
    
    fig.update_layout(
        title='<b>Parallel Coordinates Plot</b><br>Multi-dimensional Data Analysis',
        height=600,
        template='plotly_white'
    )
    
    return fig

# ============================================
# 互動式圖表 6: 統計檢定力
# ============================================

def create_interactive_power_analysis():
    """創建互動式統計檢定力分析"""
    
    sample_sizes = np.linspace(1000, 1745, 50)
    power = np.sqrt(sample_sizes / 1745)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sample_sizes,
        y=power,
        mode='lines',
        name='Statistical Power',
        line=dict(color='#3498db', width=4),
        fill='tonexty',
        fillcolor='rgba(52, 152, 219, 0.2)',
        hovertemplate='Sample Size: %{x:.0f}<br>Power: %{y:.2%}<extra></extra>'
    ))
    
    fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                  annotation_text="Target: 100%", annotation_position="right")
    
    fig.add_trace(go.Scatter(
        x=[1417],
        y=[np.sqrt(1417/1745)],
        mode='markers',
        name='Current State',
        marker=dict(size=15, color='#e74c3c', symbol='star'),
        hovertemplate='Current: 1,417 samples<br>Power: %{y:.2%}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[1745],
        y=[1.0],
        mode='markers',
        name='Full Dataset',
        marker=dict(size=15, color='#27ae60', symbol='star'),
        hovertemplate='Full: 1,745 samples<br>Power: 100%<extra></extra>'
    ))
    
    fig.update_layout(
        title='<b>Interactive Statistical Power Analysis</b><br>Sample Size vs Statistical Power',
        xaxis_title='Sample Size (n)',
        yaxis_title='Statistical Power',
        height=600,
        template='plotly_white',
        hovermode='x unified',
        yaxis=dict(tickformat='.0%')
    )
    
    return fig

# ============================================
# 整合儀表板
# ============================================

def create_dashboard():
    """創建完整的互動式儀表板"""
    
    print("生成互動式圖表...")
    print()
    
    try:
        # 生成所有圖表
        print("[1/6] 生成缺失值分析...")
        fig1 = create_interactive_missing_analysis()
        fig1.write_html('interactive_missing_analysis.html')
        print("     [OK] interactive_missing_analysis.html")
        
        print("[2/6] 生成 3D 散點圖...")
        fig2 = create_3d_scatter()
        fig2.write_html('interactive_3d_scatter.html')
        print("     [OK] interactive_3d_scatter.html")
        
        print("[3/6] 生成數據收集動畫...")
        fig3 = create_animated_data_collection()
        fig3.write_html('interactive_animation.html')
        print("     [OK] interactive_animation.html")
        
        print("[4/6] 生成 Sunburst 圖...")
        fig4 = create_sunburst()
        fig4.write_html('interactive_sunburst.html')
        print("     [OK] interactive_sunburst.html")
        
        print("[5/6] 生成平行座標圖...")
        fig5 = create_parallel_coordinates()
        fig5.write_html('interactive_parallel.html')
        print("     [OK] interactive_parallel.html")
        
        print("[6/6] 生成統計檢定力圖...")
        fig6 = create_interactive_power_analysis()
        fig6.write_html('interactive_power_analysis.html')
        print("     [OK] interactive_power_analysis.html")
        
        # 創建整合儀表板
        print()
        print("創建整合儀表板...")
        
        dashboard_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>TDM Interactive EDA Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            font-size: 1.2em;
            margin-bottom: 30px;
        }
        .info-box {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .info-box h2 {
            margin-top: 0;
            color: #2c3e50;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .feature {
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            text-align: center;
        }
        .feature h3 {
            margin-top: 0;
        }
        .nav {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        .nav-btn {
            padding: 12px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .nav-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .nav-btn.active {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .chart-container {
            display: none;
            margin-top: 20px;
        }
        .chart-container.active {
            display: block;
        }
        .drug-legend {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 0.9em;
        }
        .drug-legend h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .drug-table {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .drug-item {
            padding: 5px;
        }
        .drug-item strong {
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 TDM Interactive EDA Dashboard</h1>
        <div class="subtitle">互動式資料探索分析儀表板 - 完整版</div>
        
        <div class="info-box">
            <h2>💡 使用說明</h2>
            <p>點擊下方按鈕切換不同的互動式圖表。所有圖表都支援:</p>
            <ul>
                <li><strong>懸停顯示詳細資訊</strong> - 滑鼠移到圖表上查看數據</li>
                <li><strong>縮放與平移</strong> - 可以放大、縮小、拖曳圖表</li>
                <li><strong>選擇與篩選</strong> - 點擊圖例可以隱藏/顯示資料</li>
                <li><strong>匯出圖片</strong> - 點擊相機圖示可以下載圖表</li>
            </ul>
        </div>
        
        <div class="features">
            <div class="feature">
                <h3>📊 6 個互動圖表</h3>
                <p>涵蓋所有重要分析面向</p>
            </div>
            <div class="feature">
                <h3>🎨 3D 視覺化</h3>
                <p>可旋轉的三維散點圖</p>
            </div>
            <div class="feature">
                <h3>🎬 動態動畫</h3>
                <p>數據收集過程模擬</p>
            </div>
            <div class="feature">
                <h3>🔍 深度互動</h3>
                <p>完全可探索的數據</p>
            </div>
        </div>
        
        <div class="nav">
            <button class="nav-btn active" onclick="showChart(0)">1️⃣ 缺失值分析</button>
            <button class="nav-btn" onclick="showChart(1)">2️⃣ 3D 散點圖</button>
            <button class="nav-btn" onclick="showChart(2)">3️⃣ 數據收集動畫</button>
            <button class="nav-btn" onclick="showChart(3)">4️⃣ 階層分布圖</button>
            <button class="nav-btn" onclick="showChart(4)">5️⃣ 平行座標圖</button>
            <button class="nav-btn" onclick="showChart(5)">6️⃣ 統計檢定力</button>
        </div>
        
        <div id="chart0" class="chart-container active">
            <iframe src="interactive_missing_analysis.html" width="100%" height="600" frameborder="0"></iframe>
        </div>
        
        <div id="chart1" class="chart-container">
            <iframe src="interactive_3d_scatter.html" width="100%" height="900" frameborder="0"></iframe>
            <div class="drug-legend">
                <h3>📋 藥物代碼對照表</h3>
                <p>3D 圖表使用縮寫以保持視覺清晰,完整名稱請參考下表或將滑鼠移到資料點上查看:</p>
                <div class="drug-table">
                    <div class="drug-item"><strong>Vanc</strong> = Vancomycin</div>
                    <div class="drug-item"><strong>Dig</strong> = Digoxin</div>
                    <div class="drug-item"><strong>Phen</strong> = Phenytoin</div>
                    <div class="drug-item"><strong>Theo</strong> = Theophylline</div>
                    <div class="drug-item"><strong>Gent</strong> = Gentamicin</div>
                    <div class="drug-item"><strong>Li</strong> = Lithium</div>
                    <div class="drug-item"><strong>Tacro</strong> = Tacrolimus</div>
                    <div class="drug-item"><strong>Cyclo</strong> = Cyclosporine</div>
                    <div class="drug-item"><strong>Carba</strong> = Carbamazepine</div>
                    <div class="drug-item"><strong>VPA</strong> = Valproic Acid</div>
                </div>
            </div>
        </div>
        
        <div id="chart2" class="chart-container">
            <iframe src="interactive_animation.html" width="100%" height="700" frameborder="0"></iframe>
        </div>
        
        <div id="chart3" class="chart-container">
            <iframe src="interactive_sunburst.html" width="100%" height="800" frameborder="0"></iframe>
        </div>
        
        <div id="chart4" class="chart-container">
            <iframe src="interactive_parallel.html" width="100%" height="700" frameborder="0"></iframe>
        </div>
        
        <div id="chart5" class="chart-container">
            <iframe src="interactive_power_analysis.html" width="100%" height="700" frameborder="0"></iframe>
        </div>
    </div>
    
    <script>
        function showChart(index) {
            const charts = document.querySelectorAll('.chart-container');
            charts.forEach(chart => chart.classList.remove('active'));
            
            const btns = document.querySelectorAll('.nav-btn');
            btns.forEach(btn => btn.classList.remove('active'));
            
            document.getElementById('chart' + index).classList.add('active');
            btns[index].classList.add('active');
        }
    </script>
</body>
</html>"""
        
        with open('interactive_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        print("     [OK] interactive_dashboard.html")
        print()
        print("=" * 60)
        print("[SUCCESS] 所有互動式圖表生成完成!")
        print("=" * 60)
        print()
        print("生成的檔案:")
        print("  1. interactive_missing_analysis.html - 缺失值分析")
        print("  2. interactive_3d_scatter.html - 3D 散點圖 (藥名縮寫)")
        print("  3. interactive_animation.html - 數據收集動畫 (有曲線)")
        print("  4. interactive_sunburst.html - 階層分布圖")
        print("  5. interactive_parallel.html - 平行座標圖")
        print("  6. interactive_power_analysis.html - 統計檢定力")
        print("  7. interactive_dashboard.html - 整合儀表板 [STAR]")
        print()
        print("✨ 特別優化:")
        print("  - 3D 圖表使用藥物縮寫,圖例清晰不會被切")
        print("  - 動畫顯示完整率上升曲線 (60% -> 81.2%)")
        print("  - 儀表板內含藥物代碼對照表")
        print()
        print("請用瀏覽器開啟 interactive_dashboard.html 查看!")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("[ERROR] 生成過程發生錯誤:")
        print(str(e))
        import traceback
        traceback.print_exc()

# ============================================
# 主程式
# ============================================

if __name__ == "__main__":
    create_dashboard()