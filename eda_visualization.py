"""
TDM 資料探索性分析 (EDA) - 視覺化程式碼 (Windows 完全相容版)
使用 Matplotlib 和 Seaborn 製作三個重要圖表
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import os
import sys

# 設定 UTF-8 輸出 (避免編碼錯誤)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================================
# 設定
# ============================================

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 11

# 設定 Seaborn 樣式
sns.set_style("whitegrid")
sns.set_palette("husl")

# 定義配色
colors = {
    'primary': '#3498db',
    'danger': '#e74c3c',
    'success': '#27ae60',
    'warning': '#f39c12',
    'purple': '#9b59b6',
    'gray': '#95a5a6'
}

# 創建輸出資料夾
OUTPUT_DIR = '.'  # 儲存在當前目錄
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("=" * 60)
print("TDM EDA 圖表生成程式")
print("=" * 60)
print(f"圖表將儲存在: {os.path.abspath(OUTPUT_DIR)}")
print()

# ============================================
# 模擬 TDM 資料
# ============================================

# 設定隨機種子以便重現
np.random.seed(42)

# 總樣本數
n_total = 1745

# 創建模擬資料
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

# 加入 Accept 欄位 (模擬缺失 18.8%)
df['Accept'] = None
accept_indices = np.random.choice(n_total, size=int(n_total * 0.812), replace=False)
accept_values = np.random.choice(['Yes', 'No'], size=len(accept_indices), p=[0.933, 0.067])
for idx, val in zip(accept_indices, accept_values):
    df.loc[idx, 'Accept'] = val

# 加入 Medicine 欄位 (模擬缺失 61.3%)
df['Medicine'] = None
medicine_indices = np.random.choice(n_total, size=int(n_total * 0.387), replace=False)
medicine_values = np.random.choice(['Adjusted', 'Maintained', 'Changed'], size=len(medicine_indices))
for idx, val in zip(medicine_indices, medicine_values):
    df.loc[idx, 'Medicine'] = val

print(f"資料集大小: {len(df)} 筆")
print()
print("各欄位缺失率:")
missing_info = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print(missing_info[missing_info > 0])
print()

# ============================================
# 圖表 1: 缺失值分析
# ============================================

def plot_missing_analysis():
    """繪製缺失值分析圖"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 計算缺失率
    missing_data = df.isnull().sum() / len(df) * 100
    missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
    
    # 繪製水平條形圖
    bars = ax.barh(missing_data.index, missing_data.values, 
                   color=[colors['danger'] if x > 50 else colors['warning'] if x > 15 else colors['primary'] 
                          for x in missing_data.values])
    
    # 添加數值標籤
    for i, (idx, val) in enumerate(missing_data.items()):
        ax.text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold')
    
    # 添加警戒線
    ax.axvline(x=20, color=colors['danger'], linestyle='--', linewidth=2, alpha=0.5, label='Critical (>20%)')
    ax.axvline(x=10, color=colors['warning'], linestyle='--', linewidth=2, alpha=0.5, label='Warning (>10%)')
    
    ax.set_xlabel('Missing Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Missing Data Analysis\nTDM Dataset (N=1,745)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'eda_missing_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 圖表1已儲存: {output_path}")
    return fig

# ============================================
# 圖表 2: 特徵分布
# ============================================

def plot_feature_distribution():
    """繪製特徵分布圖"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Distribution Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. 藥物分布
    drug_counts = df['Drug'].value_counts().head(8)
    axes[0, 0].bar(range(len(drug_counts)), drug_counts.values, color=colors['primary'])
    axes[0, 0].set_xticks(range(len(drug_counts)))
    axes[0, 0].set_xticklabels(drug_counts.index, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Drug Distribution (Top 8)', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. 年齡分布
    axes[0, 1].hist(df['Age'], bins=30, color=colors['success'], edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df['Age'].mean(), color=colors['danger'], linestyle='--', linewidth=2, label=f'Mean: {df["Age"].mean():.1f}')
    axes[0, 1].set_xlabel('Age (years)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Age Distribution', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. 性別分布
    gender_counts = df['Gender'].value_counts()
    explode = (0.05, 0)
    axes[0, 2].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                   colors=[colors['primary'], colors['warning']], explode=explode, startangle=90)
    axes[0, 2].set_title('Gender Distribution', fontweight='bold')
    
    # 4. 科別分布
    dept_counts = df['Department'].value_counts()
    axes[1, 0].barh(dept_counts.index, dept_counts.values, color=colors['purple'])
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_title('Department Distribution', fontweight='bold')
    for i, v in enumerate(dept_counts.values):
        axes[1, 0].text(v + 10, i, str(v), va='center')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 5. 採樣時間分布
    time_counts = df['Time'].value_counts()
    axes[1, 1].bar(time_counts.index, time_counts.values, color=[colors['warning'], colors['success']])
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Sampling Time Distribution', fontweight='bold')
    for i, (idx, val) in enumerate(time_counts.items()):
        axes[1, 1].text(i, val + 20, str(val), ha='center', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # 6. Accept 建議接受率
    accept_counts = df['Accept'].value_counts()
    colors_accept = [colors['success'], colors['danger']]
    bars = axes[1, 2].bar(accept_counts.index, accept_counts.values, color=colors_accept)
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title(f'Recommendation Acceptance\n(Available: {accept_counts.sum()}/{len(df)})', fontweight='bold')
    
    # 計算接受率
    if 'Yes' in accept_counts.index:
        acceptance_rate = accept_counts['Yes'] / accept_counts.sum() * 100
        axes[1, 2].text(0.5, 0.95, f'Acceptance Rate: {acceptance_rate:.1f}%', 
                       transform=axes[1, 2].transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=11, fontweight='bold')
    
    for i, (idx, val) in enumerate(accept_counts.items()):
        axes[1, 2].text(i, val + 20, str(val), ha='center', fontweight='bold')
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'eda_feature_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 圖表2已儲存: {output_path}")
    return fig

# ============================================
# 圖表 3: 相關性分析
# ============================================

def plot_correlation_analysis():
    """繪製相關性矩陣熱圖"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
    
    # 選擇數值型欄位
    numeric_cols = ['Age', 'Dose', 'Level']
    corr_data = df[numeric_cols].corr()
    
    # 左圖: 相關係數熱圖
    sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=2, cbar_kws={"shrink": 0.8},
                ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('Correlation Matrix\n(Numeric Features)', fontweight='bold', pad=15)
    
    # 右圖: 藥物濃度 vs 劑量
    top_drugs = df['Drug'].value_counts().head(5).index
    df_top = df[df['Drug'].isin(top_drugs)]
    
    for drug in top_drugs:
        drug_data = df_top[df_top['Drug'] == drug]
        axes[1].scatter(drug_data['Dose'], drug_data['Level'], 
                       label=drug, alpha=0.6, s=50)
    
    axes[1].set_xlabel('Dose (mg)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Drug Level (ug/mL)', fontsize=11, fontweight='bold')
    axes[1].set_title('Dose vs Drug Level\n(Top 5 Drugs)', fontweight='bold', pad=15)
    axes[1].legend(loc='upper left', framealpha=0.9)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'eda_correlation_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 圖表3已儲存: {output_path}")
    return fig

# ============================================
# 圖表 4: 資料品質總覽
# ============================================

def plot_data_quality_overview():
    """繪製資料品質總覽圖"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Data Quality Overview - TDM Dataset', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. 完整度堆疊條形圖
    fields = ['Accept', 'Medicine', 'Dose', 'Level', 'Time', 'Department', 'Drug', 'Age']
    complete_pct = [(~df[col].isnull()).sum() / len(df) * 100 for col in fields]
    missing_pct = [100 - pct for pct in complete_pct]
    
    x = np.arange(len(fields))
    width = 0.6
    
    p1 = axes[0, 0].bar(x, complete_pct, width, label='Complete', color=colors['success'])
    p2 = axes[0, 0].bar(x, missing_pct, width, bottom=complete_pct, label='Missing', color=colors['danger'])
    
    axes[0, 0].set_ylabel('Percentage (%)', fontweight='bold')
    axes[0, 0].set_title('Data Completeness by Field', fontweight='bold', pad=15)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(fields, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.5)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 添加百分比標籤
    for i, (c, m) in enumerate(zip(complete_pct, missing_pct)):
        if c > 5:
            axes[0, 0].text(i, c/2, f'{c:.1f}%', ha='center', va='center', fontweight='bold', color='white')
        if m > 5:
            axes[0, 0].text(i, c + m/2, f'{m:.1f}%', ha='center', va='center', fontweight='bold', color='white')
    
    # 2. 樣本可用性分析
    complete_samples = df.dropna(subset=['Accept']).shape[0]
    incomplete_samples = len(df) - complete_samples
    
    labels = ['Complete\nRecords', 'Incomplete\nRecords']
    sizes = [complete_samples, incomplete_samples]
    explode = (0.1, 0)
    
    axes[0, 1].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                   colors=[colors['success'], colors['danger']], startangle=90,
                   textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[0, 1].set_title(f'Sample Usability\n(Total: {len(df)} samples)', fontweight='bold', pad=15)
    
    # 添加數量標籤
    axes[0, 1].text(0, -1.5, f'Complete: {complete_samples}\nIncomplete: {incomplete_samples}',
                   ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. 藥物分布與缺失率關係
    drug_stats = df.groupby('Drug').agg({
        'Accept': lambda x: x.isnull().sum() / len(x) * 100
    }).sort_values('Accept', ascending=False).head(8)
    
    axes[1, 0].barh(drug_stats.index, drug_stats['Accept'], color=colors['warning'])
    axes[1, 0].set_xlabel('Missing Rate (%)', fontweight='bold')
    axes[1, 0].set_title('Missing Rate by Drug Type\n(Top 8)', fontweight='bold', pad=15)
    axes[1, 0].axvline(x=18.8, color='red', linestyle='--', linewidth=2, label='Overall Missing Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 添加百分比標籤
    for i, (idx, val) in enumerate(drug_stats['Accept'].items()):
        axes[1, 0].text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold')
    
    # 4. 統計檢定力影響視覺化
    sample_sizes = np.array([1417, 1500, 1600, 1700, 1745])
    power = np.sqrt(sample_sizes / 1745)
    
    axes[1, 1].plot(sample_sizes, power, marker='o', linewidth=3, markersize=10, color=colors['primary'])
    axes[1, 1].axhline(y=1.0, color=colors['success'], linestyle='--', linewidth=2, label='Full Power (100%)')
    axes[1, 1].axhline(y=power[0], color=colors['danger'], linestyle='--', linewidth=2, label=f'Current ({power[0]:.1%})')
    axes[1, 1].axvline(x=1417, color=colors['danger'], linestyle=':', linewidth=2, alpha=0.5)
    axes[1, 1].axvline(x=1745, color=colors['success'], linestyle=':', linewidth=2, alpha=0.5)
    
    axes[1, 1].fill_between(sample_sizes, power, 1.0, alpha=0.3, color=colors['warning'])
    axes[1, 1].set_xlabel('Sample Size (n)', fontweight='bold')
    axes[1, 1].set_ylabel('Statistical Power', fontweight='bold')
    axes[1, 1].set_title('Statistical Power vs Sample Size\n(Power ~ sqrt(n))', fontweight='bold', pad=15)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # 標註損失
    axes[1, 1].annotate(f'Power Loss:\n{(1-power[0])*100:.1f}%',
                       xy=(1417, power[0]), xytext=(1500, 0.92),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'eda_data_quality_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 進階圖表已儲存: {output_path}")
    return fig

# ============================================
# 主程式
# ============================================

def generate_all_charts():
    """生成所有 EDA 圖表"""
    print("=" * 60)
    print("開始生成 TDM EDA 圖表...")
    print("=" * 60)
    print()
    
    try:
        # 生成圖表
        fig1 = plot_missing_analysis()
        plt.close(fig1)
        
        fig2 = plot_feature_distribution()
        plt.close(fig2)
        
        fig3 = plot_correlation_analysis()
        plt.close(fig3)
        
        fig4 = plot_data_quality_overview()
        plt.close(fig4)
        
        print()
        print("=" * 60)
        print("[SUCCESS] 所有圖表生成完成!")
        print("=" * 60)
        print()
        print("生成的檔案:")
        print(f"  1. eda_missing_analysis.png")
        print(f"  2. eda_feature_distribution.png")
        print(f"  3. eda_correlation_analysis.png")
        print(f"  4. eda_data_quality_overview.png")
        print()
        print(f"檔案位置: {os.path.abspath(OUTPUT_DIR)}")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("[ERROR] 生成過程發生錯誤:")
        print(str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_all_charts()