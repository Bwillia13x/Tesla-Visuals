"""
Tesla charts with minimalist, elegant design inspired by Johnny Ive.
- Minimalist color palette and whitespace
- Subtle, thin lines and soft accent colors
- Modern sans-serif fonts, increased padding
- No chart borders or heavy gridlines
- Direct, clean value labels and annotations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Minimalist Apple-inspired style
plt.style.use('default')
sns.set_palette(["#222222", "#A3C1DA", "#E74C3C", "#B2BABB", "#F7CAC9", "#92A8D1"])
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'axes.edgecolor': '#DDDDDD',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'axes.grid': True,
    'grid.color': '#EEEEEE',
    'grid.linestyle': '-',
    'grid.linewidth': 0.7,
    'xtick.color': '#888888',
    'ytick.color': '#888888',
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.titleweight': 'medium',
    'axes.labelweight': 'regular',
    'legend.fontsize': 12,
    'legend.loc': 'best',
    'lines.linewidth': 2.2,
    'lines.markersize': 8
})

# Create figure with subplots
fig = plt.figure(figsize=(20, 24))

# 1. Tesla Financial Performance Evolution
ax1 = plt.subplot(3, 3, 1)
years = [2010, 2011, 2012, 2013]
revenue = [117, 204, 413, 967]
gross_profit = [31, 62, 30, 197]
gross_margin = [26, 30, 7, 20]

ax1_twin = ax1.twinx()
bars1 = ax1.bar([y-0.2 for y in years], revenue, width=0.4, label='Revenue', alpha=0.8, color='#2E86AB')
bars2 = ax1.bar([y+0.2 for y in years], gross_profit, width=0.4, label='Gross Profit', alpha=0.8, color='#A23B72')
line = ax1_twin.plot(years, gross_margin, 'ro-', linewidth=3, markersize=8, label='Gross Margin %', color='#F18F01')

ax1.set_xlabel('Year')
ax1.set_ylabel('Revenue & Gross Profit ($M)')
ax1_twin.set_ylabel('Gross Margin (%)')
ax1.set_title('Tesla Financial Performance Evolution\n2010-2013', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# Remove chart spines for minimalism
for spine in ax1.spines.values():
    spine.set_visible(False)
for spine in ax1_twin.spines.values():
    spine.set_visible(False)
ax1.grid(True, axis='y', color='#EEEEEE', linewidth=0.7)
ax1_twin.grid(False)

# Add value labels to bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 10, f'{int(height)}', ha='center', va='bottom', fontsize=11, color='#888888')
# Add value labels to gross margin line
for x, y in zip(years, gross_margin):
    ax1_twin.text(x, y + 1, f'{y}%', color='#E74C3C', fontsize=11, ha='center')

# 2. Business Model Comparison - Cost Structure
ax2 = plt.subplot(3, 3, 2)
categories = ['Materials\n& Parts', 'Mfg &\nAssembly', 'R&D', 'SG&A', 'Marketing\n& Sales', 'Dealer\nMargin']
traditional_costs = [50, 10, 7, 2, 5, 5]  # Traditional auto manufacturer
tesla_costs = [60, 12, 15, 8, 3, 0]  # Tesla (no dealer margin, higher R&D)

x = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x - width/2, traditional_costs, width, label='Traditional OEM', alpha=0.8, color='#95A5A6')
bars2 = ax2.bar(x + width/2, tesla_costs, width, label='Tesla Model', alpha=0.8, color='#E74C3C')

ax2.set_xlabel('Cost Categories')
ax2.set_ylabel('% of Vehicle Price')
ax2.set_title('Cost Structure Comparison:\nTraditional vs Tesla Model', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, rotation=45, ha='right')
ax2.legend()

# Remove chart spines for minimalism
for spine in ax2.spines.values():
    spine.set_visible(False)
ax2.grid(True, axis='y', color='#EEEEEE', linewidth=0.7)

# Add value labels to bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height}%', ha='center', va='bottom', fontsize=10)

# 3. Vehicle Comparison Matrix
ax3 = plt.subplot(3, 3, 3)
vehicles = ['Tesla Model S', 'BMW 528i', 'Nissan Leaf']
price = [61070, 48725, 19650]
performance = [5.6, 6.1, 10.3]  # 0-60 mph time (lower is better)
range_miles = [208, 400, 75]  # EPA range

# Create scatter plot with bubble size representing range
scatter = ax3.scatter(price, performance, s=[r*2 for r in range_miles], 
                     alpha=0.6, c=['#E74C3C', '#3498DB', '#2ECC71'])

ax3.set_xlabel('MSRP ($)')
ax3.set_ylabel('0-60 mph Time (seconds)')
ax3.set_title('Vehicle Positioning Matrix\n(Bubble size = Range)', fontsize=14, fontweight='bold')
ax3.invert_yaxis()  # Lower acceleration time is better

# Remove chart spines for minimalism
for spine in ax3.spines.values():
    spine.set_visible(False)
ax3.grid(True, axis='y', color='#EEEEEE', linewidth=0.7)

# Add labels
for i, vehicle in enumerate(vehicles):
    ax3.annotate(vehicle, (price[i], performance[i]), 
                xytext=(10, 10), textcoords='offset points', fontsize=10)

# 4. Tesla's Sequential Strategy Timeline
ax4 = plt.subplot(3, 3, 4)
phases = ['Roadster\n2008-2012', 'Model S\n2012-2016', 'Gen 3\n2016+']
volumes = [2.5, 40, 500]  # in thousands
prices = [109, 61, 35]  # in thousands

# Create timeline chart
x_pos = [1, 2, 3]
bars = ax4.bar(x_pos, volumes, alpha=0.7, color=['#F39C12', '#E74C3C', '#27AE60'])

# Remove chart spines for minimalism
for spine in ax4.spines.values():
    spine.set_visible(False)
ax4.grid(True, axis='y', color='#EEEEEE', linewidth=0.7)

# Add price labels on bars
for i, (volume, price) in enumerate(zip(volumes, prices)):
    ax4.text(x_pos[i], volume + 20, f'${price}k', ha='center', fontweight='bold', fontsize=12)
    ax4.annotate(f'{volume}k', (x_pos[i], volume), textcoords="offset points", xytext=(0, -20), ha='center', fontsize=10, color='#34495E')

ax4.set_xlabel('Strategic Phase')
ax4.set_ylabel('Target Annual Volume (thousands)')
ax4.set_title('Tesla Sequential Strategy:\nVolume & Price Evolution', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(phases)

# 5. Battery Cost Comparison
ax5 = plt.subplot(3, 3, 5)
companies = ['Tesla\nModel S', 'Nissan\nLeaf', 'Industry\nAverage']
cost_per_kwh = [275, 625, 800]  # Estimated costs per kWh
colors = ['#E74C3C', '#2ECC71', '#95A5A6']

bars = ax5.bar(companies, cost_per_kwh, color=colors, alpha=0.8)
ax5.set_ylabel('Cost per kWh ($)')
ax5.set_title('Battery Cost Comparison\n(2013 Estimates)', fontsize=14, fontweight='bold')

# Remove chart spines for minimalism
for spine in ax5.spines.values():
    spine.set_visible(False)
ax5.grid(True, axis='y', color='#EEEEEE', linewidth=0.7)

# Add value labels on bars
for bar, cost in zip(bars, cost_per_kwh):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 10, f'${cost}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 6. Traditional Dealer Revenue Structure
ax6 = plt.subplot(3, 3, 6)
dealer_revenue = ['New Vehicle\nSales (56%)', 'Used Vehicle\nSales (32%)', 'Parts &\nService (12%)']
dealer_margins = [30, 26, 44]
revenue_share = [56, 32, 12]

# Create pie chart
colors = ['#3498DB', '#E74C3C', '#2ECC71']
# Fix pie unpacking for compatibility
pie_result = ax6.pie(revenue_share, labels=dealer_revenue, autopct='%1.0f%%', colors=colors, startangle=90)
ax6.set_title('Traditional Dealer\nRevenue Structure', fontsize=14, fontweight='bold')

# 7. EV vs ICE Maintenance Requirements
ax7 = plt.subplot(3, 3, 7)
maintenance_items = ['Oil Changes', 'Transmission\nService', 'Engine\nRepairs', 'Brake\nMaintenance', 'Tire\nRotation']
ice_frequency = [4, 2, 3, 2, 2]  # Annual frequency
ev_frequency = [0, 0, 0, 1, 2]   # Annual frequency for EVs

x = np.arange(len(maintenance_items))
width = 0.35

bars1 = ax7.bar(x - width/2, ice_frequency, width, label='ICE Vehicles', alpha=0.8, color='#95A5A6')
bars2 = ax7.bar(x + width/2, ev_frequency, width, label='Electric Vehicles', alpha=0.8, color='#E74C3C')

ax7.set_xlabel('Maintenance Type')
ax7.set_ylabel('Annual Frequency')
ax7.set_title('Maintenance Requirements:\nICE vs Electric Vehicles', fontsize=14, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(maintenance_items, rotation=45, ha='right')
ax7.legend()

# Remove chart spines for minimalism
for spine in ax7.spines.values():
    spine.set_visible(False)
ax7.grid(True, axis='y', color='#EEEEEE', linewidth=0.7)

# Add value labels to bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(height)}', ha='center', va='bottom', fontsize=10)

# 8. Market Share Evolution
ax8 = plt.subplot(3, 3, 8)
quarters = ['Q1 2012', 'Q2 2012', 'Q3 2012', 'Q4 2012', 'Q1 2013', 'Q2 2013']
tesla_sales = [0, 1000, 2500, 4750, 4900, 5500]
nissan_leaf = [1000, 2000, 2800, 1800, 2600, 5100]
chevy_volt = [1500, 1500, 2500, 2400, 1600, 1800]

ax8.plot(quarters, tesla_sales, 'o-', linewidth=3, markersize=8, label='Tesla Model S', color='#E74C3C')
ax8.plot(quarters, nissan_leaf, 's-', linewidth=3, markersize=8, label='Nissan Leaf', color='#2ECC71')
ax8.plot(quarters, chevy_volt, '^-', linewidth=3, markersize=8, label='Chevy Volt', color='#3498DB')

ax8.set_xlabel('Quarter')
ax8.set_ylabel('Quarterly Sales (Units)')
ax8.set_title('EV Market Share Evolution\n2012-2013', fontsize=14, fontweight='bold')
ax8.legend()
plt.xticks(rotation=45)

# Remove chart spines for minimalism
for spine in ax8.spines.values():
    spine.set_visible(False)
ax8.grid(True, axis='y', color='#EEEEEE', linewidth=0.7)

# Add value labels to lines
for i, (t, n, c) in enumerate(zip(tesla_sales, nissan_leaf, chevy_volt)):
    ax8.annotate(str(t), (i, t), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#E74C3C')
    ax8.annotate(str(n), (i, n), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#2ECC71')
    ax8.annotate(str(c), (i, c), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#3498DB')
ax8.set_xticks(range(len(quarters)))
ax8.set_xticklabels(quarters, rotation=45)

# 9. Tesla Value Chain Analysis
ax9 = plt.subplot(3, 3, 9)

# Create value chain diagram
value_chain_data = {
    'R&D': 15,
    'Design': 8,
    'Manufacturing': 35,
    'Marketing': 5,
    'Sales': 8,
    'Service': 12,
    'Software': 17
}

# Create horizontal bar chart
y_pos = np.arange(len(value_chain_data))
values = list(value_chain_data.values())
labels = list(value_chain_data.keys())

bars = ax9.barh(y_pos, values, alpha=0.8)

# Color bars based on Tesla's strategic focus
colors = ['#E74C3C', '#F39C12', '#3498DB', '#9B59B6', '#E74C3C', '#2ECC71', '#E74C3C']
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax9.set_yticks(y_pos)
ax9.set_yticklabels(labels)
ax9.set_xlabel('Strategic Importance Score')
ax9.set_title('Tesla Value Chain Analysis\n(Strategic Focus Areas)', fontsize=14, fontweight='bold')

# Remove chart spines for minimalism
for spine in ax9.spines.values():
    spine.set_visible(False)
ax9.grid(True, axis='y', color='#EEEEEE', linewidth=0.7)

# Add value labels to bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax9.text(width + 1, bar.get_y() + bar.get_height()/2., f'{width}', va='center', fontsize=10)

plt.tight_layout(pad=2.0)
plt.show()

# Create additional detailed charts
fig2, ((ax10, ax11), (ax12, ax13)) = plt.subplots(2, 2, figsize=(16, 12))

# 10. Learning Curve Analysis
ax10.set_title('Manufacturing Learning Curve\nDefects & Assembly Time Reduction', fontsize=14, fontweight='bold')
months = np.arange(1, 13)
defect_reduction = 100 * (0.9 ** months)  # 90% learning curve
assembly_time = 100 * (0.92 ** months)   # 92% learning curve for assembly time

ax10.plot(months, defect_reduction, 'r-', linewidth=3, label='Defect Rate', marker='o')
ax10.plot(months, assembly_time, 'b-', linewidth=3, label='Assembly Time', marker='s')
ax10.set_xlabel('Months of Production')
ax10.set_ylabel('Relative Performance (Month 1 = 100%)')
ax10.legend()
ax10.grid(True, alpha=0.3)

# Add value labels to lines
for x, y in zip(months, defect_reduction):
    ax10.text(x, y - 3, f'{y:.1f}%', color='r', fontsize=9, ha='center')
for x, y in zip(months, assembly_time):
    ax10.text(x, y + 2, f'{y:.1f}%', color='b', fontsize=9, ha='center')

# 11. Investment Comparison
ax11.set_title('Capital Investment Comparison\nTesla vs Traditional Approach', fontsize=14, fontweight='bold')
categories = ['Plant\nAcquisition', 'Equipment', 'R&D', 'Working\nCapital', 'Total']
traditional = [1500, 800, 500, 300, 3100]  # Millions
tesla_actual = [42, 200, 500, 150, 892]     # Millions

x = np.arange(len(categories))
width = 0.35

bars1 = ax11.bar(x - width/2, traditional, width, label='Traditional Approach', alpha=0.8, color='#95A5A6')
bars2 = ax11.bar(x + width/2, tesla_actual, width, label='Tesla Approach', alpha=0.8, color='#E74C3C')

ax11.set_ylabel('Investment ($M)')
ax11.set_xticks(x)
ax11.set_xticklabels(categories)
ax11.legend()

# Remove chart spines for minimalism
for spine in ax11.spines.values():
    spine.set_visible(False)
ax11.grid(True, axis='y', color='#EEEEEE', linewidth=0.7)

# Add value labels to bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax11.text(bar.get_x() + bar.get_width()/2., height + 30, f'${int(height)}M', ha='center', va='bottom', fontsize=10)

# 12. Customer Experience Journey
ax12.set_title('Customer Experience: Traditional vs Tesla Model', fontsize=14, fontweight='bold')
touchpoints = ['Research', 'Test Drive', 'Purchase', 'Delivery', 'Service', 'Support']
traditional_satisfaction = [7, 6, 5, 6, 5, 6]  # Satisfaction scores out of 10
tesla_satisfaction = [9, 9, 8, 9, 8, 9]

x = np.arange(len(touchpoints))
ax12.plot(x, traditional_satisfaction, 'o-', linewidth=3, markersize=8, 
         label='Traditional Dealer', color='#95A5A6')
ax12.plot(x, tesla_satisfaction, 'o-', linewidth=3, markersize=8, 
         label='Tesla Direct', color='#E74C3C')

ax12.set_xlabel('Customer Journey Stage')
ax12.set_ylabel('Satisfaction Score (1-10)')
ax12.set_xticks(x)
ax12.set_xticklabels(touchpoints, rotation=45)
ax12.legend()
ax12.grid(True, alpha=0.3)

# Add value labels to lines
x_ticks = list(range(len(touchpoints)))
for i, y in enumerate(traditional_satisfaction):
    ax12.text(i, y - 0.5, f'{y}', color='#95A5A6', fontsize=10, ha='center')
for i, y in enumerate(tesla_satisfaction):
    ax12.text(i, y + 0.3, f'{y}', color='#E74C3C', fontsize=10, ha='center')

# 13. Competitive Positioning Matrix
ax13.set_title('Strategic Positioning Matrix\nElectric Vehicle Market', fontsize=14, fontweight='bold')

# Position vehicles on innovation vs market position
innovation_score = [9, 7, 6, 4, 8]  # Tesla, Nissan, BMW, GM, Fisker
market_position = [8, 9, 7, 8, 3]   # Market strength
companies = ['Tesla', 'Nissan', 'BMW i-series', 'GM Volt', 'Fisker']
colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#9B59B6']

for i, company in enumerate(companies):
    ax13.scatter(innovation_score[i], market_position[i], s=300, alpha=0.7, c=colors[i], label=company, edgecolor='black', linewidth=1.5)
    ax13.annotate(company, (innovation_score[i], market_position[i]), xytext=(10, 10), textcoords='offset points', fontsize=12, fontweight='bold')

ax13.set_xlabel('Innovation Leadership (1-10)')
ax13.set_ylabel('Market Position (1-10)')
ax13.grid(True, alpha=0.3)

# Add quadrant labels
ax13.text(2, 9, 'Established\nPlayers', fontsize=12, ha='center', alpha=0.7)
ax13.text(9, 9, 'Innovation\nLeaders', fontsize=12, ha='center', alpha=0.7)
ax13.text(2, 2, 'Weak\nPosition', fontsize=12, ha='center', alpha=0.7)
ax13.text(9, 2, 'Niche\nInnovators', fontsize=12, ha='center', alpha=0.7)

plt.tight_layout(pad=2.0)
plt.show()

# Create final summary dashboard
fig3, ax14 = plt.subplots(1, 1, figsize=(14, 10))

# Tesla Business Model Canvas Visualization
ax14.set_xlim(0, 10)
ax14.set_ylim(0, 8)
ax14.axis('off')

# Define rectangles for business model canvas
rectangles = [
    {'xy': (0, 6), 'width': 2, 'height': 2, 'label': 'Key Partners\n• Panasonic\n• Toyota\n• Daimler', 'color': '#3498DB'},
    {'xy': (2, 6), 'width': 2, 'height': 2, 'label': 'Key Activities\n• R&D\n• Manufacturing\n• Software Dev', 'color': '#E74C3C'},
    {'xy': (4, 6), 'width': 2, 'height': 2, 'label': 'Value Propositions\n• Performance\n• Technology\n• Sustainability', 'color': '#F39C12'},
    {'xy': (6, 6), 'width': 2, 'height': 2, 'label': 'Customer Relations\n• Direct Sales\n• Community\n• Service', 'color': '#2ECC71'},
    {'xy': (8, 6), 'width': 2, 'height': 2, 'label': 'Customer Segments\n• Luxury Buyers\n• Tech Enthusiasts\n• Early Adopters', 'color': '#9B59B6'},
    {'xy': (2, 4), 'width': 2, 'height': 2, 'label': 'Key Resources\n• Engineering\n• Brand\n• IP\n• Facilities', 'color': '#E67E22'},
    {'xy': (6, 4), 'width': 2, 'height': 2, 'label': 'Channels\n• Company Stores\n• Online\n• Service Centers', 'color': '#1ABC9C'},
    {'xy': (0, 0), 'width': 5, 'height': 2, 'label': 'Cost Structure\n• R&D (High) • Manufacturing • Sales & Marketing', 'color': '#95A5A6'},
    {'xy': (5, 0), 'width': 5, 'height': 2, 'label': 'Revenue Streams\n• Vehicle Sales • ZEV Credits • Powertrains • Service', 'color': '#34495E'}
]

# For business model canvas, use soft rectangles and more whitespace
for rect in rectangles:
    rectangle = Rectangle(rect['xy'], rect['width'], rect['height'], facecolor=rect['color'], alpha=0.13, edgecolor='#BBBBBB', linewidth=1.2)
    ax14.add_patch(rectangle)
    ax14.text(rect['xy'][0] + rect['width']/2, rect['xy'][1] + rect['height']/2, rect['label'], ha='center', va='center', fontsize=13, fontweight='regular', wrap=True, color='#222222')

ax14.set_title('Tesla Business Model Canvas\nIntegrated Strategic Framework', fontsize=19, fontweight='medium', pad=28, color='#222222')
plt.tight_layout(pad=3.0)
plt.show()

print("All charts have been generated successfully!")
print("\nChart Summary:")
print("1. Tesla Financial Performance Evolution (2010-2013)")
print("2. Cost Structure Comparison (Traditional vs Tesla)")
print("3. Vehicle Positioning Matrix")
print("4. Tesla's Sequential Strategy Timeline")
print("5. Battery Cost Comparison")
print("6. Traditional Dealer Revenue Structure")
print("7. Maintenance Requirements (ICE vs EV)")
print("8. EV Market Share Evolution")
print("9. Tesla Value Chain Analysis")
print("10. Manufacturing Learning Curve")
print("11. Capital Investment Comparison")
print("12. Customer Experience Journey")
print("13. Strategic Positioning Matrix")
print("14. Tesla Business Model Canvas")