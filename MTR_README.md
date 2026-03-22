# 🎯 TDM Interactive EDA Dashboard
### Therapeutic Drug Monitoring — 互動式探索性資料分析儀表板

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Visualization-green.svg)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-yellow.svg)](https://pandas.pydata.org/)
[![HTML5](https://img.shields.io/badge/HTML5-Dashboard-orange.svg)](https://developer.mozilla.org/en-US/docs/Web/HTML)

## 🌐 互動式儀表板

🎯 **Live Demo**：[tai-ju.github.io/Introduction-to-Information-Science_MTR](https://tai-ju.github.io/Introduction-to-Information-Science_MTR/)

> 包含 6 個互動式 Plotly 圖表：缺失值分析、3D 散點圖、數據收集動畫、階層分布圖、平行座標圖、統計檢定力分析。

---

## 📋 專案概述

本專案開發了一個全功能的治療藥物監測（TDM）互動式資料分析儀表板，整合 6 個不同類型的視覺化圖表，提供醫療專業人員和研究人員深入探索 TDM 資料的工具。

**Therapeutic Drug Monitoring (TDM)** 是現代個人化醫療的核心技術：
- **目的**：確保藥物濃度維持在治療窗口內
- **重要性**：避免毒性反應，確保治療效果
- **監測藥物**：Vancomycin、Digoxin、Phenytoin 等 10 種重要藥物

---

## 📊 六大互動式視覺化模組

| # | 圖表類型 | 功能說明 |
|---|---------|---------|
| 1️⃣ | 缺失值分析 | 視覺化資料完整性，顏色分級警示系統 |
| 2️⃣ | 3D 互動散點圖 | 三維藥物-患者-濃度關係，可旋轉縮放 |
| 3️⃣ | 資料收集動畫 | 模擬資料收集過程的時間序列動畫 |
| 4️⃣ | 階層分布圖 | 科別→藥物→接受率的 Sunburst 分析 |
| 5️⃣ | 平行座標圖 | 多維度資料的平行軸視覺化 |
| 6️⃣ | 統計檢定力分析 | 樣本大小與統計檢定力關係分析 |

---

## 📁 專案結構

```
Introduction-to-Information-Science_MTR/
├── index.html                          # 整合式儀表板（GitHub Pages 首頁）
├── interactive_dashboard.html          # 儀表板主檔
├── interactive_missing_analysis.html   # 缺失值分析圖
├── interactive_3d_scatter.html         # 3D 散點圖
├── interactive_animation.html          # 資料收集動畫
├── interactive_sunburst.html           # 階層分布圖
├── interactive_parallel.html           # 平行座標圖
├── interactive_power_analysis.html     # 統計檢定力分析
├── interactive_eda_gemini.py           # 主要程式檔案
├── 142216015 劉玳如.pptx              # 專案簡報
└── README.md
```

---

## 🛠️ 技術棧

- **資料處理**：Python · pandas · numpy
- **視覺化**：Plotly · Plotly Express · Plotly Graph Objects
- **前端**：HTML5 · CSS3 · JavaScript
- **部署**：GitHub Pages

---

## 🚀 本地運行

```bash
# 安裝依賴
pip install plotly pandas numpy

# 執行主程式生成圖表
python interactive_eda_gemini.py

# 開啟儀表板
start index.html   # Windows
```

---

## 📈 資料集特徵

- **總樣本數**：1,745 筆 TDM 記錄
- **監測藥物**：10 種常用 TDM 藥物
- **治療接受率**：93.3%
- **用藥調整率**：38.7%

---

*142216015 劉玳如 · 資訊科學概論 期中作業 · 國立臺北護理健康大學*
