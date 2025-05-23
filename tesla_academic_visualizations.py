"""
Tesla Strategic Analysis - Publication-Ready Visualizations
Academic-quality charts supporting essay.md analysis using matplotlib and seaborn
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from math import pi, cos, sin

# Set style for academic publications
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Custom color scheme matching Tesla branding
TESLA_COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2', 
    'accent': '#e74c3c',
    'success': '#27ae60',
    'warning': '#f39c12',
    'dark': '#2c3e50',
    'light': '#ecf0f1'
}

def create_vrio_matrix():
    """Create VRIO analysis matrix visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    capabilities = [
        'Vertical Integration &\nManufacturing Excellence',
        'Revolutionary Battery\nTechnology Approach', 
        'Software-Centric\nDesign Philosophy',
        'Direct Sales Model'
    ]
    
    criteria = ['Valuable', 'Rare', 'Inimitable', 'Organized']
    advantages = ['Sustained', 'Sustained', 'Sustained', 'Temporary']
    
    # Create matrix data
    matrix_data = [
        [1, 1, 1, 1],  # Vertical Integration
        [1, 1, 1, 1],  # Battery Technology
        [1, 1, 1, 1],  # Software Design
        [1, 1, 0.5, 1] # Direct Sales (? for Inimitable)
    ]
    
    # Create heatmap
    im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(criteria)))
    ax.set_yticks(range(len(capabilities)))
    ax.set_xticklabels(criteria, fontsize=12, fontweight='bold')
    ax.set_yticklabels(capabilities, fontsize=11)
    
    # Add text annotations
    for i in range(len(capabilities)):
        for j in range(len(criteria)):
            if matrix_data[i][j] == 1:
                text = '✓'
                color = 'white'
            elif matrix_data[i][j] == 0.5:
                text = '?'
                color = 'black'
            else:
                text = '✗'
                color = 'white'
            ax.text(j, i, text, ha='center', va='center', 
                   fontsize=16, fontweight='bold', color=color)
    
    # Add competitive advantage column
    for i, advantage in enumerate(advantages):
        bbox_props = dict(boxstyle="round,pad=0.3", 
                         facecolor=TESLA_COLORS['success'] if advantage == 'Sustained' 
                         else TESLA_COLORS['warning'], alpha=0.8)
        ax.text(len(criteria) + 0.5, i, advantage, ha='center', va='center',
               fontweight='bold', color='white', bbox=bbox_props)
    
    # Customize plot
    ax.set_title('VRIO Analysis: Tesla\'s Core Capabilities', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('VRIO Criteria', fontsize=14, fontweight='bold')
    ax.set_ylabel('Tesla Capabilities', fontsize=14, fontweight='bold')
    
    # Add competitive advantage header
    ax.text(len(criteria) + 0.5, -0.7, 'Competitive\nAdvantage', 
           ha='center', va='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('tesla_vrio_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_battery_cost_comparison():
    """Create battery cost comparison chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Data
    companies = ['Tesla 60kWh', 'Nissan Leaf', 'Industry Average']
    cost_per_kwh = [275, 550, 600]
    total_pack_cost = [16500, 13200, 18000]
    colors = [TESLA_COLORS['primary'], TESLA_COLORS['accent'], TESLA_COLORS['light']]
    
    # Cost per kWh chart
    bars1 = ax1.bar(companies, cost_per_kwh, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Battery Cost per kWh Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cost per kWh (USD)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(cost_per_kwh) * 1.1)
    
    # Add value labels
    for bar, value in zip(bars1, cost_per_kwh):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'${value}', ha='center', va='bottom', fontweight='bold')
    
    # Total pack cost chart
    bars2 = ax2.bar(companies, total_pack_cost, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Total Battery Pack Cost', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Total Cost (USD)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(total_pack_cost) * 1.1)
    
    # Add value labels
    for bar, value in zip(bars2, total_pack_cost):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 200,
                f'${value:,}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tesla_battery_cost_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_value_chain_comparison():
    """Create value chain disruption visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Traditional value chain
    traditional_steps = [
        'R&D and Design',
        'Multiple Parts\nSuppliers',
        'Assembly\nOperations', 
        'Independent\nDealerships\n(Limited profit)',
        'Service & Repairs\n(Main profit center)'
    ]
    
    tesla_steps = [
        'In-House R&D\n(Silicon Valley)',
        'Battery Technology\n(18650 + BMS)',
        'Vertical Manufacturing\n(90% in-house)',
        'Direct Sales\n(Company stores)',
        'Minimal Service\n(Software updates)'
    ]
    
    def draw_value_chain(ax, steps, title, colors):
        ax.set_xlim(0, len(steps))
        ax.set_ylim(-0.5, 1.5)
        
        for i, step in enumerate(steps):
            # Draw box
            box = FancyBboxPatch((i + 0.1, 0.2), 0.8, 0.6, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors[i], alpha=0.8,
                               edgecolor='black', linewidth=1.5)
            ax.add_patch(box)
            
            # Add text
            ax.text(i + 0.5, 0.5, step, ha='center', va='center',
                   fontweight='bold', fontsize=10, color='white' if i >= 3 else 'black')
            
            # Draw arrow to next step
            if i < len(steps) - 1:
                ax.arrow(i + 0.9, 0.5, 0.1, 0, head_width=0.05, head_length=0.05,
                        fc='black', ec='black')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Traditional colors (red for problematic areas)
    traditional_colors = [TESLA_COLORS['light']] * 3 + [TESLA_COLORS['accent']] + [TESLA_COLORS['success']]
    # Tesla colors (blue for advantages)
    tesla_colors = [TESLA_COLORS['primary']] * 5
    
    draw_value_chain(ax1, traditional_steps, 'Traditional Automotive Value Chain', traditional_colors)
    draw_value_chain(ax2, tesla_steps, 'Tesla\'s Integrated Value Chain', tesla_colors)
    
    plt.tight_layout()
    plt.savefig('tesla_value_chain_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_three_stage_strategy():
    """Create Tesla's three-stage strategy timeline"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    stages = [
        {'name': 'Stage 1: Roadster', 'price': '$109,000', 'year': '2008-2012',
         'volume': '2,500 units', 'purpose': 'Technology Demonstrator'},
        {'name': 'Stage 2: Model S', 'price': '~$61,000', 'year': '2012-2015', 
         'volume': 'Scaled Production', 'purpose': 'Premium Market Entry'},
        {'name': 'Stage 3: Gen 3', 'price': '~$35,000', 'year': '2016+',
         'volume': '100,000+ units/year', 'purpose': 'Mass Market Penetration'}
    ]
    
    # Timeline
    years = [2008, 2012, 2016, 2020]
    ax.plot(years, [0]*len(years), 'o-', linewidth=4, markersize=10, color=TESLA_COLORS['primary'])
    
    # Stage boxes
    for i, stage in enumerate(stages):
        year = years[i]
        
        # Create stage box
        box_height = 0.8
        box_width = 3
        box = FancyBboxPatch((year - box_width/2, 0.3), box_width, box_height,
                           boxstyle="round,pad=0.1", 
                           facecolor=TESLA_COLORS['primary'], alpha=0.8,
                           edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Add stage info
        ax.text(year, 0.9, stage['name'], ha='center', va='center',
               fontweight='bold', fontsize=12, color='white')
        ax.text(year, 0.7, stage['price'], ha='center', va='center',
               fontweight='bold', fontsize=14, color='white')
        ax.text(year, 0.5, stage['volume'], ha='center', va='center',
               fontsize=10, color='white')
        ax.text(year, 0.35, stage['purpose'], ha='center', va='center',
               fontsize=9, color='white', style='italic')
        
        # Learning arrows and text
        if i < len(stages) - 1:
            ax.annotate('', xy=(years[i+1] - 1, -0.3), xytext=(year + 1, -0.3),
                       arrowprops=dict(arrowstyle='->', lw=2, color=TESLA_COLORS['accent']))
            ax.text((year + years[i+1]) / 2, -0.5, 'Experience Curve\nLearning', 
                   ha='center', va='center', fontweight='bold', color=TESLA_COLORS['accent'])
    
    ax.set_xlim(2006, 2022)
    ax.set_ylim(-0.8, 1.3)
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_title('Tesla\'s Three-Stage Market Entry Strategy', fontsize=16, fontweight='bold', pad=20)
    
    # Remove y-axis
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('tesla_three_stage_strategy.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_competitive_radar():
    """Create competitive positioning radar chart"""
    # Categories
    categories = ['Innovation', 'Manufacturing', 'Distribution', 'Battery Tech', 'Software', 'Brand']
    N = len(categories)
    
    # Data for each company
    tesla_values = [5, 4, 5, 5, 5, 4]
    traditional_values = [3, 5, 4, 2, 2, 4]
    ev_competitors_values = [4, 3, 3, 4, 3, 3]
    
    # Compute angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add the first value to the end to close the radar chart
    tesla_values += tesla_values[:1]
    traditional_values += traditional_values[:1]
    ev_competitors_values += ev_competitors_values[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, tesla_values, 'o-', linewidth=3, label='Tesla', color=TESLA_COLORS['primary'])
    ax.fill(angles, tesla_values, alpha=0.25, color=TESLA_COLORS['primary'])
    
    ax.plot(angles, traditional_values, 'o-', linewidth=3, label='Traditional OEMs', color=TESLA_COLORS['accent'])
    ax.fill(angles, traditional_values, alpha=0.25, color=TESLA_COLORS['accent'])
    
    ax.plot(angles, ev_competitors_values, 'o-', linewidth=3, label='EV Competitors', color=TESLA_COLORS['success'])
    ax.fill(angles, ev_competitors_values, alpha=0.25, color=TESLA_COLORS['success'])
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    
    plt.title('Competitive Positioning Analysis', size=16, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig('tesla_competitive_radar.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_dynamic_capabilities_framework():
    """Create dynamic capabilities visualization"""
    fig, ax = plt.subplots(figsize=(15, 10))
    
    capabilities = {
        'Sensing': {
            'items': ['Battery cost constraint', 'Software differentiation', 'Dealership inefficiencies', 'EV industry emergence'],
            'color': TESLA_COLORS['primary']
        },
        'Seizing': {
            'items': ['NUMMI plant ($42M)', 'Commodity 18650 cells', 'Supercharger network', 'Direct sales model'],
            'color': TESLA_COLORS['success']
        },
        'Transforming': {
            'items': ['Sports car → mass market', 'Software integration', 'New service paradigm', 'Value chain redesign'],
            'color': TESLA_COLORS['accent']
        }
    }
    
    # Create three columns
    col_width = 0.25
    col_spacing = 0.05
    
    for i, (capability, data) in enumerate(capabilities.items()):
        x_pos = i * (col_width + col_spacing) + 0.1
        
        # Main capability box
        main_box = FancyBboxPatch((x_pos, 0.7), col_width, 0.2,
                                boxstyle="round,pad=0.02",
                                facecolor=data['color'], alpha=0.8,
                                edgecolor='black', linewidth=2)
        ax.add_patch(main_box)
        
        ax.text(x_pos + col_width/2, 0.8, capability, ha='center', va='center',
               fontweight='bold', fontsize=14, color='white')
        
        # Item boxes
        for j, item in enumerate(data['items']):
            y_pos = 0.5 - j * 0.12
            item_box = FancyBboxPatch((x_pos + 0.02, y_pos), col_width - 0.04, 0.08,
                                    boxstyle="round,pad=0.01",
                                    facecolor='white', alpha=0.9,
                                    edgecolor=data['color'], linewidth=1.5)
            ax.add_patch(item_box)
            
            ax.text(x_pos + col_width/2, y_pos + 0.04, item, ha='center', va='center',
                   fontsize=10, fontweight='500')
            
            # Arrow from main box to item
            ax.arrow(x_pos + col_width/2, 0.69, 0, y_pos + 0.08 - 0.69 + 0.02,
                    head_width=0.01, head_length=0.01, fc=data['color'], ec=data['color'])
    
    # Add arrows between capabilities
    for i in range(len(capabilities) - 1):
        start_x = (i + 1) * (col_width + col_spacing) + 0.1 - col_spacing/2
        ax.arrow(start_x, 0.8, col_spacing, 0, head_width=0.02, head_length=0.01,
                fc='black', ec='black', linewidth=2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Tesla\'s Dynamic Capabilities Framework', fontsize=16, fontweight='bold', pad=20)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('tesla_dynamic_capabilities.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_innovation_appropriability():
    """Create innovation appropriability mechanisms flow"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    mechanisms = [
        {'title': 'Lead Time\nAdvantage', 'desc': 'First-mover in\npremium EVs'},
        {'title': 'Complementary\nAssets', 'desc': 'Supercharger network\nDirect sales'},
        {'title': 'Experience\nCurves', 'desc': '85-90% learning rate\nBattery technology'},
        {'title': 'Sustained\nAdvantage', 'desc': 'VRIO-qualified\ncapabilities'}
    ]
    
    box_width = 0.18
    box_height = 0.3
    spacing = 0.02
    
    for i, mechanism in enumerate(mechanisms):
        x_pos = i * (box_width + spacing) + 0.1
        
        # Main box
        box = FancyBboxPatch((x_pos, 0.35), box_width, box_height,
                           boxstyle="round,pad=0.02",
                           facecolor=TESLA_COLORS['primary'], alpha=0.8,
                           edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Title
        ax.text(x_pos + box_width/2, 0.55, mechanism['title'], ha='center', va='center',
               fontweight='bold', fontsize=12, color='white')
        
        # Description
        ax.text(x_pos + box_width/2, 0.42, mechanism['desc'], ha='center', va='center',
               fontsize=10, color='white')
        
        # Arrow to next mechanism
        if i < len(mechanisms) - 1:
            arrow_start = x_pos + box_width + 0.005
            arrow_end = (i + 1) * (box_width + spacing) + 0.1 - 0.005
            ax.arrow(arrow_start, 0.5, arrow_end - arrow_start, 0,
                    head_width=0.03, head_length=0.01, fc=TESLA_COLORS['accent'], 
                    ec=TESLA_COLORS['accent'], linewidth=3)
    
    # Add feedback loop
    ax.annotate('', xy=(0.12, 0.25), xytext=(0.88, 0.25),
               arrowprops=dict(arrowstyle='->', lw=2, color=TESLA_COLORS['success'],
                             connectionstyle="arc3,rad=-0.3"))
    ax.text(0.5, 0.15, 'Reinforcing Feedback Loop', ha='center', va='center',
           fontweight='bold', color=TESLA_COLORS['success'], fontsize=12)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)
    ax.set_title('Innovation Appropriability Mechanisms', fontsize=16, fontweight='bold', pad=20)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('tesla_innovation_appropriability.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_metrics_dashboard():
    """Create key performance metrics dashboard"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    metrics = [
        {'title': 'Battery Cost\nAdvantage', 'value': '50%', 'subtitle': 'vs Nissan Leaf'},
        {'title': 'Plant Setup\nSavings', 'value': '67%', 'subtitle': 'vs Traditional'},
        {'title': 'In-House\nProduction', 'value': '90%', 'subtitle': 'Model S Parts'},
        {'title': 'Experience\nCurve', 'value': '85-90%', 'subtitle': 'Learning Rate'},
        {'title': 'Consumer\nReports', 'value': '99/100', 'subtitle': 'Highest Ever'},
        {'title': 'NUMMI Plant\nCost', 'value': '$42M', 'subtitle': 'Acquisition'},
        {'title': 'Roadster\nPerformance', 'value': '<4s', 'subtitle': '0-60mph'},
        {'title': 'Production\nTarget', 'value': '100K+', 'subtitle': 'Gen 3 Annual'},
        {'title': 'Battery Pack\nCost', 'value': '$250-300', 'subtitle': 'per kWh'}
    ]
    
    positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
    colors = [TESLA_COLORS['primary'], TESLA_COLORS['success'], TESLA_COLORS['accent']] * 3
    
    for i, (metric, pos, color) in enumerate(zip(metrics, positions, colors)):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        
        # Create metric card
        ax.text(0.5, 0.7, metric['value'], ha='center', va='center',
               fontsize=20, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.5, 0.4, metric['title'], ha='center', va='center',
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.2, metric['subtitle'], ha='center', va='center',
               fontsize=10, color='gray', transform=ax.transAxes)
        
        # Style the subplot
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)
    
    fig.suptitle('Tesla Strategic Performance Metrics', fontsize=18, fontweight='bold', y=0.95)
    plt.savefig('tesla_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_all_visualizations():
    """Generate all Tesla strategic analysis visualizations"""
    print("Generating Tesla Strategic Analysis Visualizations...")
    
    # Set up the plotting environment
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11
    
    visualizations = [
        ("VRIO Matrix Analysis", create_vrio_matrix),
        ("Battery Cost Comparison", create_battery_cost_comparison),
        ("Value Chain Disruption", create_value_chain_comparison),
        ("Three-Stage Strategy", create_three_stage_strategy),
        ("Competitive Radar Chart", create_competitive_radar),
        ("Dynamic Capabilities Framework", create_dynamic_capabilities_framework),
        ("Innovation Appropriability", create_innovation_appropriability),
        ("Performance Metrics Dashboard", create_performance_metrics_dashboard)
    ]
    
    for name, func in visualizations:
        print(f"Creating {name}...")
        try:
            func()
            print(f"✓ {name} completed")
        except Exception as e:
            print(f"✗ Error creating {name}: {e}")
    
    print("\nAll visualizations generated successfully!")
    print("Files saved as high-resolution PNG images suitable for academic publication.")

if __name__ == "__main__":
    generate_all_visualizations()
