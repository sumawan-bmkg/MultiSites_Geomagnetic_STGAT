"""
Figure 1: System Architecture - Multimodal Tensor Flow
=======================================================
Publication-quality architecture diagram for Elsevier Q1 journal (ESWA)

This script generates a visual representation of the complete system architecture
including CNN backbone, SupCon detection head, and Physics Sidecar with gated fusion.

Author: Earthquake Prediction Research Team
Date: May 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL CONFIGURATION FOR Q1 PUBLICATION
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 320,
    'savefig.dpi': 320,
    'savefig.bbox': 'tight',
})

# Professional color palette (pastel, colorblind-safe)
COLORS = {
    'input': '#E8F4F8',      # Light Blue
    'backbone': '#FFE5CC',   # Light Orange
    'supcon': '#D4EDDA',     # Light Green
    'physics': '#FFF3CD',    # Light Yellow
    'fusion': '#E2D4F0',     # Light Purple
    'output': '#F8D7DA',     # Light Red
    'arrow': '#555555',      # Dark Gray
}


def create_box(ax, x, y, width, height, text, color, fontsize=9, bold=False):
    """
    Create a rounded box with text.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
    x, y : float
        Bottom-left corner coordinates
    width, height : float
        Box dimensions
    text : str
        Label text
    color : str
        Fill color
    fontsize : int
        Font size for text
    bold : bool
        Whether to use bold font
    """
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor=color,
                          linewidth=1.2, zorder=2)
    ax.add_patch(box)
    
    # Add text in center of box
    fontweight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, zorder=3)


def create_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    """
    Create an arrow between two points.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
    x1, y1 : float
        Start coordinates
    x2, y2 : float
        End coordinates
    label : str
        Optional label for arrow
    style : str
        Arrow style
    """
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=COLORS['arrow'],
                            linewidth=1.5, zorder=1,
                            mutation_scale=20)
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.15, label,
                ha='center', va='bottom', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='none', alpha=0.8), zorder=3)


def plot_architecture_diagram(output_dir='eswa'):
    """
    Generate Figure 1: Complete system architecture flowchart (Horizontal Layout).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create figure - Wider and shorter for horizontal flow
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # ========================================================================
    # X-COORDINATES FOR STAGES
    # ========================================================================
    x_input = 0.5
    x_backbone = 4.0
    x_heads = 7.5
    x_fusion = 12.0
    x_output = 16.0
    
    # Y-COORDINATES FOR ROWS
    y_main = 1.0
    y_side = 4.0
    
    # ========================================================================
    # STAGE 1: INPUT DATA
    # ========================================================================
    
    # HDF5 Scalogram Input
    create_box(ax, x_input, y_main, 2.5, 1.5,
               'HDF5 Scalogram\nInput\n[B, 3, 128, 1440]',
               COLORS['input'], fontsize=9, bold=True)
    
    # Kp-Index Input
    create_box(ax, x_input, y_side, 2.5, 1.5,
               'Kp-Index\nCosmic Gating\n[B, 1]',
               COLORS['input'], fontsize=9, bold=True)
    
    # ========================================================================
    # STAGE 2: BACKBONE
    # ========================================================================
    
    create_box(ax, x_backbone, y_main, 2.5, 1.5,
               'CNN Backbone\n(ResNet-34)\n\nLatent: [B, 128]',
               COLORS['backbone'], fontsize=9, bold=True)
    
    # Arrow from Scalogram to Backbone
    create_arrow(ax, x_input + 2.5, y_main + 0.75, x_backbone, y_main + 0.75, 'Conv Layers')
    
    # ========================================================================
    # STAGE 3: DETECTION HEAD & PHYSICS SIDECAR
    # ========================================================================
    
    # MAIN BRANCH: SupCon Detection Head
    create_box(ax, x_heads, y_main, 3.2, 1.8,
               'SupCon Detection Head\n\n' +
               'Supervised Contrastive Loss\n' +
               'Binary Classification\n' +
               'Output: [B, 2]',
               COLORS['supcon'], fontsize=8.5, bold=True)
    
    # Arrow from Backbone to SupCon Head
    create_arrow(ax, x_backbone + 2.5, y_main + 0.75, x_heads, y_main + 0.75, 'Latent [128]')
    
    # SIDE BRANCH: Physics Sidecar
    create_box(ax, x_heads, y_side, 2.8, 1.8,
               'Physics Sidecar\n\n' +
               'Dobrovolsky Strain\n' +
               'Polarization Tensor\n' +
               'Output: [B, 16]',
               COLORS['physics'], fontsize=8.5, bold=True)
    
    # Arrow from Kp-Index to Physics Sidecar
    create_arrow(ax, x_input + 2.5, y_side + 0.75, x_heads, y_side + 0.75, 'Gating Signal')
    
    # ========================================================================
    # STAGE 4: DYNAMIC PRIOR & GATED FUSION
    # ========================================================================
    
    # Dynamic Prior Module
    create_box(ax, x_fusion, y_side, 2.8, 1.8,
               'Dynamic Prior\n\n' +
               'Bayesian Update\n' +
               'Von Mises Prior\n' +
               'μ, κ',
               COLORS['physics'], fontsize=8.5, bold=True)
    
    # Arrow from Physics Sidecar to Dynamic Prior
    create_arrow(ax, x_heads + 2.8, y_side + 0.9, x_fusion, y_side + 0.9, 'Physics [16]')
    
    # Gated Fusion Module
    create_box(ax, x_fusion, y_main, 2.8, 1.5,
               'Gated Fusion\n\n' +
               'Attention Gate\n' +
               'α ∈ [0, 1]',
               COLORS['fusion'], fontsize=8.5, bold=True)
    
    # Arrow from SupCon Head to Gated Fusion
    create_arrow(ax, x_heads + 3.2, y_main + 0.9, x_fusion, y_main + 0.9, 'Detection')
    
    # Arrow from Dynamic Prior to Gated Fusion
    create_arrow(ax, x_fusion + 1.4, y_side, x_fusion + 1.4, y_main + 1.5, 'Prior Update', style='-|>')
    
    # ========================================================================
    # STAGE 5: AZIMUTH HEAD (Output)
    # ========================================================================
    
    create_box(ax, x_output, y_main, 2.5, 1.5,
               'Von Mises\nAzimuth Head\n\nθ ∈ [0°, 360°)',
               COLORS['output'], fontsize=9, bold=True)
    
    # Arrow from Gated Fusion to Azimuth Head
    create_arrow(ax, x_fusion + 2.8, y_main + 0.75, x_output, y_main + 0.75, 'Fused Flow')
    
    # ========================================================================
    # ANNOTATIONS & LEGEND
    # ========================================================================
    
    # Title
    ax.text(9.5, 6.5, 'Figure 1: Multimodal Unified Tensor Architecture (UMTA) Workflow',
            ha='center', va='top', fontsize=14, fontweight='bold')
    
    ax.text(9.5, 6.1, 'Horizontal Parallel Processing Pipeline for physics-guided earthquake prediction',
            ha='center', va='top', fontsize=11, style='italic')
    
    # Add tensor dimension annotations
    ax.text(0.5, 0.5, 'Tensor Info: [Batch, Channels, Frequencies, TimeSteps]',
            ha='left', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='gray', linewidth=0.5, alpha=0.9))
    
    # Add legend for color coding
    legend_x = 16.0
    legend_y = 6.2
    legend_elements = [
        ('Input Data', COLORS['input']),
        ('CNN Backbone', COLORS['backbone']),
        ('Detection Head', COLORS['supcon']),
        ('Physics Module', COLORS['physics']),
        ('Fusion Layer', COLORS['fusion']),
        ('Output Head', COLORS['output']),
    ]
    
    ax.text(legend_x, legend_y, 'Module Key:',
            ha='left', va='top', fontsize=8, fontweight='bold')
    
    for i, (label, color) in enumerate(legend_elements):
        y_pos = legend_y - 0.4 - i * 0.35
        rect = mpatches.Rectangle((legend_x, y_pos - 0.15), 0.4, 0.25,
                                   facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(legend_x + 0.5, y_pos, label, ha='left', va='center', fontsize=7.5)
    
    # ========================================================================
    # Save figure
    # ========================================================================
    plt.tight_layout()
    
    png_path = output_dir / 'figure1_architecture_diagram.png'
    pdf_path = output_dir / 'figure1_architecture_diagram.pdf'
    
    fig.savefig(png_path, dpi=320, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    
    print(f"\n[DONE] Figure 1 saved (Horizontal Layout):")
    print(f"  - PNG: {png_path}")
    print(f"  - PDF: {pdf_path}")
    
    plt.close(fig)
    
    # Save architecture description to text file
    description = """
ARCHITECTURE DESCRIPTION
========================

1. INPUT LAYER
   - HDF5 Scalogram: [Batch, 3 Channels, 128 Frequencies, 1440 Time Steps]
   - Kp-Index: [Batch, 1] - Geomagnetic activity indicator

2. CNN BACKBONE (ResNet-34)
   - Extracts latent representations from scalogram
   - Output: [Batch, 128] latent vector

3. DUAL-HEAD ARCHITECTURE

   A. DETECTION HEAD (SupCon)
      - Supervised Contrastive Learning
      - Binary classification (Noise vs Precursor)
      - Output: [Batch, 2] class probabilities
   
   B. PHYSICS SIDECAR
      - Dobrovolsky Strain Model
      - Polarization Tensor Analysis
      - Output: [Batch, 16] physics features

4. DYNAMIC PRIOR MODULE
   - Bayesian update mechanism
   - Von Mises distribution parameters (μ, κ)
   - Incorporates physics constraints

5. GATED FUSION
   - Attention-based gating mechanism
   - Adaptive weight α ∈ [0, 1]
   - Balances data-driven and physics-guided predictions

6. AZIMUTH HEAD
   - Von Mises distribution output
   - Circular regression for azimuth angle
   - Output: θ ∈ [0°, 360°)

KEY INNOVATIONS
===============
- Multimodal fusion of seismic and geomagnetic data
- Physics-guided learning via Dobrovolsky strain model
- Supervised contrastive learning for robust detection
- Gated fusion for adaptive physics integration
- Circular statistics for azimuth prediction
"""
    
    desc_path = output_dir / 'architecture_description.txt'
    with open(desc_path, 'w', encoding='utf-8') as f:
        f.write(description)
    
    print(f"  - Description: {desc_path}")


if __name__ == '__main__':
    print("=" * 70)
    print("FIGURE 1: SYSTEM ARCHITECTURE DIAGRAM")
    print("=" * 70)
    print("Generating architecture flowchart (320 DPI)...\n")
    
    plot_architecture_diagram(output_dir='eswa')
    
    print("\n" + "=" * 70)
    print(" [DONE] TASK COMPLETED")
    print("=" * 70)
    print("\nUsage for manuscript:")
    print("  LaTeX: \\includegraphics[width=\\textwidth]{figure1_architecture_diagram.pdf}")
    print("  Word:  Insert figure1_architecture_diagram.png at 100% scale")
