# 🎯 TDM Interactive EDA Dashboard
## Therapeutic Drug Monitoring - 互動式探索性資料分析儀表板

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Visualization-green.svg)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-yellow.svg)](https://pandas.pydata.org/)
[![HTML5](https://img.shields.io/badge/HTML5-Dashboard-orange.svg)](https://developer.mozilla.org/en-US/docs/Web/HTML)

### 📋 專案概述

本專案開發了一個全功能的治療藥物監測（TDM）互動式資料分析儀表板，整合了6個不同類型的視覺化圖表，提供醫療專業人員和研究人員深入探索TDM資料的工具。透過先進的互動式視覺化技術，使用者可以全方位分析藥物濃度、患者特徵、科別分布等關鍵指標。

### 🏥 TDM系統背景

**Therapeutic Drug Monitoring (TDM)** 是現代個人化醫療的核心技術：
- **目的**：確保藥物濃度維持在治療窗口內
- **重要性**：避免毒性反應，確保治療效果
- **監測藥物**：包含Vancomycin、Digoxin、Phenytoin等10種重要藥物
- **臨床價值**：支援精準醫療決策制定

### 📊 資料集特徵

#### **模擬資料規格**
- **總樣本數**：1,745筆TDM記錄
- **監測藥物**：10種常用TDM藥物
- **患者特徵**：年齡、性別、科別分布
- **監測參數**：藥物劑量、血中濃度、採樣時間
- **治療結果**：接受率81.2%，用藥調整38.7%

#### **包含藥物類型**
```
抗生素類：Vancomycin, Gentamicin
心血管藥物：Digoxin
抗癲癇藥物：Phenytoin, Carbamazepine, Valproic Acid
呼吸道藥物：Theophylline
精神科藥物：Lithium
免疫抑制劑：Tacrolimus, Cyclosporine
```

### 🎨 六大互動式視覺化模組

#### **1️⃣ 缺失值分析** (`interactive_missing_analysis.html`)
- **功能**：視覺化資料完整性
- **特色**：顏色分級警示系統（紅色>20%，橙色>10%）
- **應用**：資料品質評估，識別需要改進的欄位

#### **2️⃣ 3D互動散點圖** (`interactive_3d_scatter.html`)
- **功能**：三維度藥物-患者-濃度關係探索
- **特色**：可旋轉、縮放的3D視覺化，藥物代碼對照表
- **應用**：多維度模式識別，異常值檢測

#### **3️⃣ 資料收集動畫** (`interactive_animation.html`)
- **功能**：模擬資料收集過程的時間序列動畫
- **特色**：完整率上升曲線，目標達成視覺化
- **應用**：資料收集進度監控，品質改進追蹤

#### **4️⃣ 階層分布圖** (`interactive_sunburst.html`)
- **功能**：科別→藥物→接受率的階層式分析
- **特色**：Sunburst多層次圓形圖，直覺的階層導航
- **應用**：科別差異分析，決策模式探索

#### **5️⃣ 平行座標圖** (`interactive_parallel.html`)
- **功能**：多維度資料的平行軸視覺化
- **特色**：多變數同時比較，模式識別
- **應用**：複雜關係分析，患者分群研究

#### **6️⃣ 統計檢定力分析** (`interactive_power_analysis.html`)
- **功能**：樣本大小與統計檢定力關係分析
- **特色**：動態檢定力曲線，樣本數優化建議
- **應用**：研究設計優化，樣本數計算

### 📁 專案檔案結構

```
TDM_Interactive_EDA/
│
├── README.md                          # 專案說明文件
├── interactive_eda_gemini.py          # 主要程式檔案
├── interactive_dashboard.html         # 整合式儀表板
│
├── Individual_Charts/                 # 個別圖表檔案
│   ├── interactive_missing_analysis.html
│   ├── interactive_3d_scatter.html
│   ├── interactive_animation.html
│   ├── interactive_sunburst.html
│   ├── interactive_parallel.html
│   └── interactive_power_analysis.html
│
└── Documentation/                     # 說明文件
    └── 142216015_劉玳如.pptx         # 專案簡報
```

### 🛠️ 技術架構

#### **後端技術**
```python
# 資料處理核心
pandas >= 1.3.0          # 資料框架處理
numpy >= 1.21.0          # 數值計算

# 視覺化引擎
plotly >= 5.0.0          # 互動式圖表
plotly.graph_objects     # 進階圖表物件
plotly.express           # 快速圖表生成
```

#### **前端技術**
```html
<!-- 視覺化核心 -->
Plotly.js                <!-- 客戶端互動引擎 -->
HTML5                    <!-- 結構化標記 -->
CSS3                     <!-- 響應式樣式設計 -->
JavaScript               <!-- 互動邏輯控制 -->
```

#### **儀表板特色**
- **響應式設計**：自適應不同螢幕尺寸
- **分頁式導航**：快速切換不同分析視圖
- **統一樣式**：一致的視覺設計語言
- **效能優化**：iframe分離載入，避免記憶體衝突

### 🚀 快速開始

#### **環境需求**
```bash
Python 3.8+
現代瀏覽器 (Chrome/Firefox/Safari/Edge)
```

#### **1. 安裝依賴套件**
```bash
pip install plotly pandas numpy
```

#### **2. 執行主程式**
```bash
python interactive_eda_gemini.py
```

#### **3. 開啟儀表板**
```bash
# 程式執行完成後，開啟生成的檔案
open interactive_dashboard.html  # macOS
start interactive_dashboard.html # Windows
xdg-open interactive_dashboard.html # Linux
```

### 🎯 核心功能特色

#### **✨ 進階互動性**
- **智慧懸停提示**：詳細的數據標籤和統計資訊
- **動態篩選**：點擊圖例隱藏/顯示特定資料
- **縮放與平移**：無限制的視角調整
- **高解析度匯出**：PNG/SVG格式下載

#### **📱 使用者體驗**
- **一鍵切換**：分頁式圖表導航
- **視覺回饋**：按鈕狀態與載入指示
- **資訊豐富**：完整的圖表說明與使用指南
- **錯誤處理**：優雅的錯誤提示機制

#### **🔬 專業分析能力**
- **多維度探索**：同時分析多個變數關係
- **統計深度**：包含檢定力分析等進階統計
- **醫學導向**：針對TDM場景優化的分析邏輯
- **決策支援**：提供可行動的洞察建議

### 📈 實際應用場景

#### **🏥 臨床應用**
```
藥師工作站：
├── 劑量調整決策支援
├── 濃度異常警示系統  
├── 科別間差異比較
└── 患者風險分層管理

醫師臨床決策：
├── 個人化用藥方案
├── 療效監控追蹤
├── 毒性風險評估
└── 多重用藥管理
```

#### **📊 研究應用**
```
藥物動力學研究：
├── 人群PK參數分析
├── 協變量效應探索
├── 模型驗證視覺化
└── 劑量優化研究

臨床試驗分析：
├── 樣本數計算
├── 中期效果評估
├── 安全性信號偵測
└── 族群差異分析
```

#### **🎓 教育訓練**
```
醫學教育：
├── TDM概念教學
├── 互動式案例學習
├── 視覺化數據解讀
└── 決策思維培養

專業培訓：
├── 藥師繼續教育
├── 臨床藥學培訓
├── 資料分析技能
└── 數位工具應用
```

### 🔍 技術實現細節

#### **資料模擬邏輯**
```python
# 真實TDM場景模擬
n_total = 1745  # 符合大型醫院年度TDM量

# 藥物選擇基於臨床使用頻率
drugs = ['Vancomycin', 'Digoxin', 'Phenytoin', ...]

# 患者特徵符合真實分布
age_distribution = normal(60, 15)  # 住院患者年齡分布
gender_ratio = [0.55, 0.45]       # 男女比例

# 科別權重反映TDM需求
department_weights = {
    'ICU': 0.3,              # 重症最高
    'Internal_Medicine': 0.25, # 內科次之
    'Surgery': 0.15,         # 外科中等
    ...
}
```

#### **視覺化優化技術**
```python
# 3D圖表效能優化
scene=dict(
    camera=dict(eye=dict(x=1.3, y=1.3, z=1.1)),  # 最佳視角
    bgcolor='rgba(0,0,0,0)',                      # 透明背景
    xaxis=dict(backgroundcolor='rgb(245,245,245)') # 網格優化
)

# 顏色映射策略
color_schemes = {
    'acceptance': {'Yes': '#27ae60', 'No': '#e74c3c'},
    'departments': 'Set3',  # 科別區分
    'drugs': 'tab10'        # 藥物分類
}
```

### ⚡ 效能與優化

#### **📊 載入效能**
- **分離式架構**：iframe隔離避免圖表衝突
- **延遲載入**：按需載入圖表內容
- **記憶體管理**：自動釋放非活躍圖表資源
- **快取策略**：瀏覽器快取靜態資源

#### **🔧 技術優化**
- **編碼處理**：UTF-8全程支援，避免亂碼
- **錯誤恢復**：異常處理機制，確保穩定性
- **相容性**：跨瀏覽器測試，確保一致體驗
- **響應式**：多裝置適配，行動裝置友善

### 📚 使用指南

#### **基本操作**
```
1. 開啟儀表板後，預設顯示缺失值分析
2. 點擊頂部按鈕切換不同圖表
3. 滑鼠懸停查看詳細數據
4. 點擊圖例篩選特定資料
5. 使用縮放工具深入分析
6. 點擊相機圖示匯出圖表
```

#### **進階功能**
```
3D散點圖：
├── 滑鼠拖曳旋轉視角
├── 滾輪縮放調整距離
├── 點擊重置到預設視角
└── 參考藥物代碼對照表

平行座標圖：
├── 拖曳軸線調整範圍
├── 點擊軸標籤重新排序
├── 刷選操作篩選資料
└── 色彩編碼識別模式
```

### 🎯 關鍵洞察與發現

#### **📊 資料品質洞察**
- **完整率達成**：整體資料完整率81.2%
- **關鍵缺失**：Accept欄位缺失率61.3%，Medicine欄位缺失率18.8%
- **品質建議**：需要加強資料收集流程標準化

#### **🏥 臨床使用模式**
- **科別分布**：ICU使用率最高（30%），符合重症監護需求
- **藥物偏好**：Vancomycin和Digoxin為最常監測藥物
- **接受率分析**：整體接受率93.3%，顯示良好的臨床依從性

#### **⚕️ 醫療決策支援**
- **劑量調整**：38.7%案例需要用藥調整
- **監測時機**：Trough濃度監測佔70%，符合臨床指引
- **風險管理**：建立基於濃度的預警機制

