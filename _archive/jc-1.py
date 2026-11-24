"""
GENERALIZED JC-1 MITOCHONDRIAL MEMBRANE POTENTIAL ANALYSIS
=========================================================

Publication-quality analysis for JC-1 fluorescence data across multiple conditions.
Creates 4 distinct publication-ready plots:

1. Pure scatter plot (individual cells + means)
2. Pure density/contour plot (population distributions)  
3. Side-by-side density comparison (condition-specific panels)
4. Joint plot with marginal distributions

JC-1 Biological Context:
- Red fluorescence (aggregates): Healthy, hyperpolarized mitochondria
- Green fluorescence (monomers): Depolarized/damaged mitochondria  
- Red/Green ratio: Primary metric for mitochondrial membrane potential
- Higher ratio = healthier mitochondria

Designed for flexible input with automatic column detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# =================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# =================================================================

# File path to your CSV data
DATA_FILE = '/home/adb/Documents/Projects/image analysis/Analysis Images - SA/Hacat SA JC1 VC OM(2)/Output/ctcf_analysis_cellular_Data_20250909_041227.csv'  # Update this path


# Column name mappings - the script will auto-detect CTCF columns only
# Supports common naming patterns: mRFP_CTCF, GFP_CTCF, Red_CTCF, Green_CTCF, etc.
COLUMN_KEYWORDS = {
    'condition': ['condition', 'treatment', 'group', 'sample'],
    'jc1_red': ['mrfp_ctcf', 'rfp_ctcf', 'red_ctcf', 'jc1_red_ctcf', 'jc-1_red_ctcf', 'aggregate_ctcf'],
    'jc1_green': ['gfp_ctcf', 'green_ctcf', 'jc1_green_ctcf', 'jc-1_green_ctcf', 'monomer_ctcf'],
    'area': ['area', 'cell_area', 'size']
}

# Analysis parameters
OUTLIER_FACTOR = 1.5  # IQR multiplier for outlier removal (1.5 = standard, 2.0 = conservative)
MIN_AREA_THRESHOLD = 50  # Minimum cell area to include (adjust based on your imaging)
# Note: Zero CTCF removal is mandatory for JC-1 analysis (breaks ratio calculations)

# Plot styling
PLOT_DPI = 300
FIGURE_FORMAT = ['png', 'pdf']  # Output formats
POINT_SIZE = 45
TRANSPARENCY = 0.6

def detect_columns(df, keywords_dict):
    """
    Automatically detect relevant CTCF columns based on keyword matching.
    Prioritizes CTCF measurements for accurate fluorescence quantification.
    Returns dictionary with detected column names.
    """
    detected = {}
    
    print("Available columns:", df.columns.tolist())
    print("\nSearching for CTCF columns...")
    
    for key, keywords in keywords_dict.items():
        detected[key] = None
        
        # For fluorescence columns, prioritize CTCF measurements
        if key in ['jc1_red', 'jc1_green']:
            # First, look specifically for CTCF columns
            ctcf_matches = [col for col in df.columns if 'ctcf' in col.lower()]
            print(f"Found CTCF columns: {ctcf_matches}")
            
            for keyword in keywords:
                matches = [col for col in ctcf_matches if keyword.lower() in col.lower()]
                if matches:
                    detected[key] = matches[0]  # Take first match
                    print(f"✓ Detected {key}: {matches[0]} (CTCF measurement)")
                    break
        else:
            # For non-fluorescence columns, use standard matching
            for keyword in keywords:
                matches = [col for col in df.columns if keyword.lower() in col.lower()]
                if matches:
                    detected[key] = matches[0]  # Take first match
                    print(f"✓ Detected {key}: {matches[0]}")
                    break
    
    # Validation for CTCF columns
    for key in ['jc1_red', 'jc1_green']:
        if detected[key] and 'ctcf' not in detected[key].lower():
            print(f"⚠️  Warning: {detected[key]} may not be a CTCF measurement!")
            print("   CTCF (Corrected Total Cell Fluorescence) is recommended for quantitative analysis.")
    
    return detected

def load_and_prepare_data(file_path):
    """
    Load data and perform initial preparation with automatic column detection.
    """
    # Load data
    df = pd.read_csv(file_path)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Detect columns
    columns = detect_columns(df, COLUMN_KEYWORDS)
    
    # Verify essential columns were found
    if not columns['condition']:
        print("Error: Could not detect condition column. Available columns:")
        print(df.columns.tolist())
        return None, None
    
    if not columns['jc1_red'] or not columns['jc1_green']:
        print("Error: Could not detect JC-1 CTCF fluorescence columns.")
        print("Looking for columns containing:")
        print("  Red channel: mRFP_CTCF, RFP_CTCF, Red_CTCF, JC1_Red_CTCF")
        print("  Green channel: GFP_CTCF, Green_CTCF, JC1_Green_CTCF")
        print("\nAvailable columns:")
        print(df.columns.tolist())
        print("\n⚠️  Important: Only CTCF (Corrected Total Cell Fluorescence) columns are supported")
        print("   for accurate quantitative analysis. Avoid MeanIntensity or IntegratedDensity.")
        return None, None
    
    # Create standardized column names
    df_clean = df.copy()
    df_clean = df_clean.rename(columns={
        columns['condition']: 'Condition',
        columns['jc1_red']: 'JC1_Red_CTCF',
        columns['jc1_green']: 'JC1_Green_CTCF'
    })
    
    # Add area column if detected
    if columns['area']:
        df_clean = df_clean.rename(columns={columns['area']: 'Cell_Area'})
    
    print(f"Conditions found: {df_clean['Condition'].value_counts().to_dict()}")
    
    return df_clean, columns

def calculate_jc1_ratio(df):
    """
    Calculate JC-1 red/green ratio - the primary metric for mitochondrial membrane potential.
    """
    # Avoid division by zero
    df['JC1_Ratio'] = np.where(df['JC1_Green_CTCF'] > 0, 
                               df['JC1_Red_CTCF'] / df['JC1_Green_CTCF'], 
                               np.nan)
    return df

def clean_data(df, min_area=MIN_AREA_THRESHOLD):
    """
    Clean data by removing zeros, outliers, and small cells.
    Zero removal is mandatory for JC-1 analysis (breaks ratio calculations).
    """
    print(f"\n=== DATA CLEANING ===")
    print(f"Starting with: {len(df)} cells")
    
    # Remove cells below area threshold if area column exists
    if 'Cell_Area' in df.columns:
        df_clean = df[df['Cell_Area'] >= min_area].copy()
        print(f"After area filter (>={min_area}): {len(df_clean)} cells")
    else:
        df_clean = df.copy()
        print("No area column detected - skipping area filter")
    
    # MANDATORY: Remove zeros and negatives (critical for JC-1 ratio analysis)
    before_count = len(df_clean)
    df_clean = df_clean[(df_clean['JC1_Red_CTCF'] > 0) & 
                       (df_clean['JC1_Green_CTCF'] > 0)].copy()
    zeros_removed = before_count - len(df_clean)
    print(f"Removed {zeros_removed} cells with zero/negative CTCF values (mandatory for ratio calculation)")
    print(f"Remaining cells: {len(df_clean)}")
    
    if len(df_clean) == 0:
        print("ERROR: No cells remaining after removing zeros! Check your CTCF data.")
        return None
    
    # Calculate ratio after cleaning
    df_clean = calculate_jc1_ratio(df_clean)
    
    # Remove infinite ratios (shouldn't happen after zero removal, but safety check)
    before_count = len(df_clean)
    df_clean = df_clean[np.isfinite(df_clean['JC1_Ratio'])].copy()
    inf_removed = before_count - len(df_clean)
    if inf_removed > 0:
        print(f"Removed {inf_removed} cells with infinite ratios")
    
    return df_clean

def remove_outliers_iqr(df, columns, factor=OUTLIER_FACTOR):
    """
    Remove outliers using IQR method - standard approach for fluorescence imaging.
    """
    df_filtered = df.copy()
    
    for col in columns:
        Q1 = df_filtered[col].quantile(0.25)
        Q3 = df_filtered[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers_before = len(df_filtered)
        df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & 
                                 (df_filtered[col] <= upper_bound)]
        outliers_removed = outliers_before - len(df_filtered)
        print(f"  {col}: Removed {outliers_removed} outliers (bounds: {lower_bound:.1f} - {upper_bound:.1f})")
    
    return df_filtered

def setup_plotting_style():
    """
    Configure publication-quality plotting style.
    """
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'DejaVu Sans',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'text.color': '#333333'
    })

def get_condition_colors(conditions):
    """
    Assign publication-quality colors to conditions.
    """
    prism_colors = [
        '#FF6B9D', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
    ]
    return {cond: prism_colors[i % len(prism_colors)] for i, cond in enumerate(conditions)}

def create_scatter_plot(df, condition_colors, mean_data):
    """
    Create Plot 1: Pure scatter plot with individual cells and means.
    Uses log scale for better visualization of CTCF data.
    """
    print("Creating Plot 1: JC-1 Scatter Plot (Log Scale)...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    conditions = df['Condition'].unique()
    
    # Plot individual points
    for condition in conditions:
        condition_data = df[df['Condition'] == condition]
        ax.scatter(condition_data['JC1_Red_CTCF'], 
                  condition_data['JC1_Green_CTCF'],
                  c=condition_colors[condition],
                  alpha=TRANSPARENCY,
                  s=POINT_SIZE,
                  edgecolors='white',
                  linewidths=0.5,
                  label=f'{condition} (n={len(condition_data)})',
                  zorder=2)
    
    # Plot means
    for data in mean_data:
        ax.scatter(data['Mean_Red'], data['Mean_Green'],
                  c=condition_colors[data['Condition']],
                  s=120, alpha=1.0, edgecolors='white',
                  linewidths=2, marker='D', zorder=3)
    
    # Apply log scale to both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Styling
    ax.set_xlabel('JC-1 Red CTCF (Aggregates) [log scale]', fontsize=13, fontweight='bold')
    ax.set_ylabel('JC-1 Green CTCF (Monomers) [log scale]', fontsize=13, fontweight='bold')
    ax.set_title('JC-1 Mitochondrial Membrane Potential: Individual Cell Analysis', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Legend and grid
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                      framealpha=0.9, edgecolor='#CCCCCC')
    legend.get_frame().set_facecolor('white')
    ax.grid(True, alpha=0.3, linewidth=0.5, color='#CCCCCC')
    ax.set_axisbelow(True)
    
    # Add ratio information
    correlation, p_value = stats.pearsonr(np.log10(df['JC1_Red_CTCF']), np.log10(df['JC1_Green_CTCF']))
    ax.text(0.02, 0.98, f'Log correlation: r = {correlation:.3f}, p = {p_value:.2e}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    for fmt in FIGURE_FORMAT:
        plt.savefig(f'jc1_plot1_scatter.{fmt}', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.show()

def create_density_plot(df, condition_colors):
    """
    Create Plot 2: Pure density/contour plot showing population distributions.
    """
    print("Creating Plot 2: JC-1 Density Plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    conditions = df['Condition'].unique()
    
    # Calculate global limits
    x_min, x_max = df['JC1_Red_CTCF'].min(), df['JC1_Red_CTCF'].max()
    y_min, y_max = df['JC1_Green_CTCF'].min(), df['JC1_Green_CTCF'].max()
    
    x_range, y_range = x_max - x_min, y_max - y_min
    x_min -= 0.1 * x_range; x_max += 0.1 * x_range
    y_min -= 0.1 * y_range; y_max += 0.1 * y_range
    
    xx, yy = np.mgrid[x_min:x_max:150j, y_min:y_max:150j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    legend_elements = []
    
    for condition in conditions:
        condition_data = df[df['Condition'] == condition]
        
        if len(condition_data) > 10:
            try:
                x_data = condition_data['JC1_Red_CTCF'].values
                y_data = condition_data['JC1_Green_CTCF'].values
                
                values = np.vstack([x_data, y_data])
                kernel = gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)
                
                contour_levels = np.percentile(f[f > 0], [10, 25, 40, 55, 70, 85, 95])
                
                ax.contourf(xx, yy, f, levels=contour_levels,
                           colors=[condition_colors[condition]], 
                           alpha=0.4, zorder=1)
                
                ax.contour(xx, yy, f, levels=contour_levels[2::2],
                          colors=[condition_colors[condition]], 
                          alpha=0.8, linewidths=1.5, zorder=2)
                
                legend_elements.append(plt.Line2D([0], [0], color=condition_colors[condition], 
                                                linewidth=3, label=f'{condition} (n={len(condition_data)})'))
                
            except Exception as e:
                print(f"  Could not generate contours for {condition}: {e}")
    
    ax.set_xlabel('JC-1 Red CTCF (Aggregates)', fontsize=13, fontweight='bold')
    ax.set_ylabel('JC-1 Green CTCF (Monomers)', fontsize=13, fontweight='bold')
    ax.set_title('JC-1 Population Density Distribution by Condition', 
                fontsize=16, fontweight='bold', pad=20)
    
    legend = ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
                      fancybox=True, shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    
    ax.grid(True, alpha=0.3, linewidth=0.5, color='#CCCCCC')
    ax.set_axisbelow(True)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.tight_layout()
    for fmt in FIGURE_FORMAT:
        plt.savefig(f'jc1_plot2_density.{fmt}', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.show()

def create_ratio_analysis_plot(df, condition_colors):
    """
    Create Plot 3: JC-1 Ratio analysis - the key metric for mitochondrial health.
    """
    print("Creating Plot 3: JC-1 Ratio Analysis...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')
    
    conditions = df['Condition'].unique()
    
    # Left panel: Ratio vs Red (shows efficiency of membrane potential)
    for condition in conditions:
        condition_data = df[df['Condition'] == condition]
        ax1.scatter(condition_data['JC1_Red_CTCF'], 
                   condition_data['JC1_Ratio'],
                   c=condition_colors[condition],
                   alpha=TRANSPARENCY, s=POINT_SIZE,
                   edgecolors='white', linewidths=0.5,
                   label=f'{condition} (n={len(condition_data)})')
    
    ax1.set_xlabel('JC-1 Red CTCF (Aggregates)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('JC-1 Ratio (Red/Green)', fontsize=13, fontweight='bold')
    ax1.set_title('Mitochondrial Efficiency vs Total Signal', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, framealpha=0.9)
    
    # Right panel: Ratio distributions by condition
    ratio_data = [df[df['Condition'] == cond]['JC1_Ratio'].values for cond in conditions]
    colors_list = [condition_colors[cond] for cond in conditions]
    
    box_plot = ax2.boxplot(ratio_data, labels=conditions, patch_artist=True, 
                          notch=True, showmeans=True)
    
    for patch, color in zip(box_plot['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('JC-1 Ratio (Red/Green)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Condition', fontsize=13, fontweight='bold')
    ax2.set_title('Mitochondrial Membrane Potential by Condition', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistical annotation
    if len(conditions) == 2:
        ratio1 = df[df['Condition'] == conditions[0]]['JC1_Ratio']
        ratio2 = df[df['Condition'] == conditions[1]]['JC1_Ratio']
        stat, p_val = stats.mannwhitneyu(ratio1, ratio2, alternative='two-sided')
        
        if p_val < 0.001:
            sig_text = "***"
        elif p_val < 0.01:
            sig_text = "**"
        elif p_val < 0.05:
            sig_text = "*"
        else:
            sig_text = "ns"
        
        y_max = max([max(data) for data in ratio_data])
        ax2.text(0.5, 0.95, f'p = {p_val:.2e} ({sig_text})', 
                transform=ax2.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    for fmt in FIGURE_FORMAT:
        plt.savefig(f'jc1_plot3_ratio_analysis.{fmt}', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.show()

def create_joint_plot(df, condition_colors, mean_data):
    """
    Create Plot 4: Joint plot with marginal distributions.
    Uses log scale for main scatter plot.
    """
    print("Creating Plot 4: Joint Plot with Marginal Distributions (Log Scale)...")
    
    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor('white')
    
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 3, 0.4], height_ratios=[1, 3, 0.2], 
                         hspace=0.08, wspace=0.08)
    
    ax_main = fig.add_subplot(gs[1, 1])
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    
    conditions = df['Condition'].unique()
    
    # Main scatter plot with log scale
    for condition in conditions:
        condition_data = df[df['Condition'] == condition]
        ax_main.scatter(condition_data['JC1_Red_CTCF'], 
                       condition_data['JC1_Green_CTCF'],
                       c=condition_colors[condition], alpha=TRANSPARENCY,
                       s=40, edgecolors='white', linewidths=0.5,
                       label=f'{condition} (n={len(condition_data)})')
    
    # Apply log scale to main plot
    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    
    # Plot means
    for data in mean_data:
        ax_main.scatter(data['Mean_Red'], data['Mean_Green'],
                       c=condition_colors[data['Condition']],
                       s=100, alpha=1.0, edgecolors='white',
                       linewidths=2, marker='D', zorder=3)
    
    # Marginal distributions (on linear scale for better visualization)
    for condition in conditions:
        condition_data = df[df['Condition'] == condition]
        
        # Top: Red distribution (linear scale for marginal)
        ax_top.hist(condition_data['JC1_Red_CTCF'], bins=50, alpha=0.6, 
                   color=condition_colors[condition], density=True,
                   edgecolor='white', linewidth=0.5)
        
        if len(condition_data) > 10:
            try:
                kde = gaussian_kde(condition_data['JC1_Red_CTCF'])
                x_range = np.linspace(condition_data['JC1_Red_CTCF'].min(), 
                                     condition_data['JC1_Red_CTCF'].max(), 200)
                ax_top.plot(x_range, kde(x_range), 
                           color=condition_colors[condition], linewidth=2.5, alpha=0.8)
            except:
                pass
        
        # Right: Green distribution (linear scale for marginal)
        ax_right.hist(condition_data['JC1_Green_CTCF'], bins=50, alpha=0.6, 
                     color=condition_colors[condition], density=True,
                     orientation='horizontal', edgecolor='white', linewidth=0.5)
        
        if len(condition_data) > 10:
            try:
                kde = gaussian_kde(condition_data['JC1_Green_CTCF'])
                y_range = np.linspace(condition_data['JC1_Green_CTCF'].min(), 
                                     condition_data['JC1_Green_CTCF'].max(), 200)
                ax_right.plot(kde(y_range), y_range, 
                             color=condition_colors[condition], linewidth=2.5, alpha=0.8)
            except:
                pass
    
    # Styling
    ax_main.set_xlabel('JC-1 Red CTCF (Aggregates) [log scale]', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('JC-1 Green CTCF (Monomers) [log scale]', fontsize=13, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    
    ax_top.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax_top.set_title('Red Fluorescence Distribution', fontsize=12, fontweight='bold')
    ax_top.set_xscale('log')  # Match main plot scale
    
    ax_right.set_xlabel('Density', fontsize=10, fontweight='bold')
    ax_right.text(0.5, 1.02, 'Green Fluorescence\nDistribution', 
                 transform=ax_right.transAxes, fontsize=11, fontweight='bold',
                 rotation=270, ha='center', va='bottom')
    ax_right.set_yscale('log')  # Match main plot scale
    
    # Remove spines and add legends
    for ax in [ax_main, ax_top, ax_right]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax != ax_main:
            ax.grid(True, alpha=0.2)
    
    ax_top.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
    
    plt.tight_layout()
    for fmt in FIGURE_FORMAT:
        plt.savefig(f'jc1_plot4_joint.{fmt}', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.show()

def perform_statistical_analysis(df):
    """
    Comprehensive statistical analysis of JC-1 data.
    """
    print(f"\n=== JC-1 STATISTICAL ANALYSIS ===")
    
    conditions = df['Condition'].unique()
    
    # Summary statistics
    print(f"Total cells analyzed: {len(df)}")
    print(f"Conditions: {len(conditions)}")
    
    summary_data = []
    for condition in conditions:
        condition_data = df[df['Condition'] == condition]
        
        summary = {
            'Condition': condition,
            'N_cells': len(condition_data),
            'Red_mean': condition_data['JC1_Red_CTCF'].mean(),
            'Red_median': condition_data['JC1_Red_CTCF'].median(),
            'Green_mean': condition_data['JC1_Green_CTCF'].mean(),
            'Green_median': condition_data['JC1_Green_CTCF'].median(),
            'Ratio_mean': condition_data['JC1_Ratio'].mean(),
            'Ratio_median': condition_data['JC1_Ratio'].median(),
            'Ratio_std': condition_data['JC1_Ratio'].std()
        }
        summary_data.append(summary)
        
        print(f"\n{condition} (n={summary['N_cells']}):")
        print(f"  Red CTCF: {summary['Red_mean']:.0f} ± {condition_data['JC1_Red_CTCF'].std():.0f}")
        print(f"  Green CTCF: {summary['Green_mean']:.0f} ± {condition_data['JC1_Green_CTCF'].std():.0f}")
        print(f"  Ratio: {summary['Ratio_mean']:.2f} ± {summary['Ratio_std']:.2f}")
    
    # Statistical tests
    if len(conditions) == 2:
        cond1_data = df[df['Condition'] == conditions[0]]
        cond2_data = df[df['Condition'] == conditions[1]]
        
        # Test all three metrics
        metrics = ['JC1_Red_CTCF', 'JC1_Green_CTCF', 'JC1_Ratio']
        print(f"\n=== STATISTICAL COMPARISON ({conditions[0]} vs {conditions[1]}) ===")
        
        for metric in metrics:
            stat, p_val = stats.mannwhitneyu(cond1_data[metric], cond2_data[metric], 
                                           alternative='two-sided')
            print(f"{metric}: Mann-Whitney U = {stat:.1f}, p = {p_val:.2e}")
            
            if p_val < 0.05:
                direction = "higher" if cond1_data[metric].median() > cond2_data[metric].median() else "lower"
                print(f"  → {conditions[0]} has significantly {direction} {metric}")
    
    elif len(conditions) > 2:
        metrics = ['JC1_Red_CTCF', 'JC1_Green_CTCF', 'JC1_Ratio']
        print(f"\n=== KRUSKAL-WALLIS TEST (all {len(conditions)} conditions) ===")
        
        for metric in metrics:
            groups = [df[df['Condition'] == cond][metric] for cond in conditions]
            stat, p_val = stats.kruskal(*groups)
            print(f"{metric}: H = {stat:.3f}, p = {p_val:.2e}")
    
    # Correlation analysis
    correlation_red_green, p_val_corr = stats.pearsonr(df['JC1_Red_CTCF'], df['JC1_Green_CTCF'])
    print(f"\n=== CORRELATION ANALYSIS ===")
    print(f"Red vs Green correlation: r = {correlation_red_green:.3f}, p = {p_val_corr:.2e}")
    
    if correlation_red_green > 0.7:
        print("  → Strong positive correlation suggests coordinated mitochondrial response")
    elif correlation_red_green < -0.3:
        print("  → Negative correlation suggests opposing red/green dynamics")
    else:
        print("  → Weak correlation suggests independent red/green regulation")
    
    return summary_data

# =================================================================
# MAIN ANALYSIS PIPELINE
# =================================================================

def main():
    """
    Main analysis pipeline for JC-1 data.
    """
    print("=== JC-1 MITOCHONDRIAL MEMBRANE POTENTIAL ANALYSIS ===")
    print("Publication-quality analysis pipeline\n")
    
    # Load and prepare data
    df_raw, detected_columns = load_and_prepare_data(DATA_FILE)
    if df_raw is None:
        return
    
    # Clean data (zero removal is mandatory for JC-1)
    df_clean = clean_data(df_raw)
    
    # Remove outliers
    print(f"\nRemoving outliers using IQR method ({OUTLIER_FACTOR}x IQR):")
    df_final = remove_outliers_iqr(df_clean, ['JC1_Red_CTCF', 'JC1_Green_CTCF', 'JC1_Ratio'], 
                                   factor=OUTLIER_FACTOR)
    print(f"Final dataset: {len(df_final)} cells")
    
    # Setup plotting
    setup_plotting_style()
    conditions = df_final['Condition'].unique()
    condition_colors = get_condition_colors(conditions)
    
    # Calculate means for plotting
    mean_data = []
    for condition in conditions:
        condition_data = df_final[df_final['Condition'] == condition]
        mean_data.append({
            'Condition': condition,
            'Mean_Red': condition_data['JC1_Red_CTCF'].mean(),
            'Mean_Green': condition_data['JC1_Green_CTCF'].mean(),
            'Mean_Ratio': condition_data['JC1_Ratio'].mean(),
            'N_cells': len(condition_data)
        })
    
    # Create all plots
    print(f"\n=== CREATING PUBLICATION PLOTS ===")
    create_scatter_plot(df_final, condition_colors, mean_data)
    create_density_plot(df_final, condition_colors)
    create_ratio_analysis_plot(df_final, condition_colors)
    create_joint_plot(df_final, condition_colors, mean_data)
    
    # Statistical analysis
    summary_data = perform_statistical_analysis(df_final)
    
    # Summary
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Generated 4 publication-ready plots:")
    print(f"1. jc1_plot1_scatter.png/pdf - Individual cell scatter plot (LOG SCALE)")
    print(f"2. jc1_plot2_density.png/pdf - Population density distributions")
    print(f"3. jc1_plot3_ratio_analysis.png/pdf - JC-1 ratio analysis")
    print(f"4. jc1_plot4_joint.png/pdf - Joint plot with marginals (LOG SCALE)")
    print(f"\n📊 Key Insights:")
    print(f"• JC-1 ratio (red/green) is the primary metric for mitochondrial membrane potential")
    print(f"• Higher ratios indicate healthier, more polarized mitochondria")
    print(f"• Log scale plots reveal population structure across orders of magnitude")
    print(f"• Zero CTCF values automatically removed (mandatory for ratio calculations)")
    print(f"• Use Plot 3 (ratio analysis) for the clearest biological interpretation")
    print(f"• Compare condition means and distributions for treatment effects")

if __name__ == "__main__":
    main()