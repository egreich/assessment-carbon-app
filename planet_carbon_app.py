### streamlit app for planet
# To run locally in terminal: streamlit run planet_carbon_app.py
# To view online, visit https://assessment-carbon-app.fly.dev/

# 1. Import Libraries
import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import glob
import os
import re # for extract_year
import time # for automatic raster animation

# 2. Load and Process Raster Data
@st.cache_data

def extract_year(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'_([0-9]{4})\.tiff$', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def load_carbon_data(data_folder):
    files = sorted(glob.glob(os.path.join(data_folder, "Brazil-pv-forests-diligence-aboveground-carbon-density-v1.2.0_*.tiff")))
    print(f"Found raster files: {files}") 
    years = []
    mean_carbon = []
    raster_layers = {}

    for file in files:
        year = extract_year(file)
        if year is None:
            print(f"Warning: Could not extract year from file {file}")
            continue
        with rasterio.open(file) as src:
            data = src.read(1)
            data = np.where(data == src.nodata, np.nan, data)  # Mask no-data
            mean_val = np.nanmean(data)
            mean_carbon.append(mean_val)
            raster_layers[year] = data
            years.append(year)

    return years, mean_carbon, raster_layers

# 3. Main App
st.title("Forest Carbon Stock in Brazil (2013-2023)")
st.markdown("---")

# Path to your raster data folder
data_folder = "data" 

years, mean_carbon, raster_layers = load_carbon_data(data_folder) # main data we're working with

# Adjsutable "carbon price" slider. Can be modified during demo for most accurate estimates.
carbon_price = st.slider(
    label="Carbon Price ($ per Mg CO₂e) - adjusts estimated carbon price for all plots and tables",
    min_value=5,
    max_value=100,
    value=15,  # default is $15
    step=1,
    help="Adjust to reflect current or projected market value per metric ton CO₂ equivalent."
)

# Explain assumptions for the carbon price slider
with st.expander("💡 What does this price mean?", expanded=False):
    st.markdown(f"""
    The carbon price reflects the market value of avoiding or removing one metric ton of CO₂ emissions.
    We're currently assuming **${carbon_price} per Mg CO₂e**, which is a typical value for voluntary carbon markets.
    You can adjust the slider above to explore how different carbon prices affect the estimated value of your forest's carbon stocks.
    """)

# 4. Show Animation
st.subheader("Carbon Value Density Decrease from 2013-2023")

# User option: change plot view + animate or manual
# four potential map options based on change in value, value density, change in carbon density, or raw carbon density
view_option = st.radio("View Mode:", ["Carbon Gain/Loss (Mg C/ha)", "Carbon Value Density ($/ha)", "Carbon Value Gain/Loss ($/ha)", "Carbon Density (Mg C/ha)",])
animate = st.checkbox("Play animation over time")

if animate:
    # Create a placeholder for the animation
    slider_placeholder = st.empty()    # For dynamic year slider
    #progress_bar = st.progress(0)  # If we want to create a more traditional progress bar
    placeholder = st.empty()
    caption1_placeholder = st.empty() # for gain/loss caption

    years_sorted = sorted(raster_layers.keys())
    total_frames = len(years_sorted)

    baseline_data = raster_layers[2013] # for carbon accumulation or decline

    # Auto-play animation
    for i, year in enumerate(years_sorted):

        carbon_data = raster_layers[year]

        # Standardized color scale
        vmin = -10  # You can tweak this if needed based on data
        vmax = 10
        
        fig, ax = plt.subplots(figsize=(10, 8))

        if view_option == "Carbon Gain/Loss (Mg C/ha)":
            if year == min(years_sorted): # If we are on year 1 (2013), then there is no change
                # 2013: Show all-zero (neutral) map
                zero_change = np.zeros_like(carbon_data)
                cax = ax.imshow(zero_change, cmap='BrBG', vmin=vmin, vmax=vmax)
                ax.set_title(f"Carbon Change from Baseline (2013)", fontsize=16)
                ax.axis('off')  # Hide axis ticks and labels
                cb = plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
                cb.set_label('Change in Carbon Density (Mg C/ha)')
            else:
                # Later years: Change relative to 2013
                diff = carbon_data - baseline_data
                cax = ax.imshow(diff, cmap='BrBG', vmin=vmin, vmax=vmax)
                ax.set_title(f"Carbon Change from 2013 Baseline ({year})", fontsize=16)
                ax.axis('off')  # Hide axis ticks and labels
                cb = plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
                cb.set_label('Change in Carbon Density (Mg C/ha)')
        if view_option == "Carbon Density (Mg C/ha)":
            # Plot normal carbon density
            cax = ax.imshow(carbon_data, cmap='viridis', vmin=np.nanmin(carbon_data), vmax=np.nanmax(carbon_data))
            ax.set_title(f"Aboveground Carbon Density ({year})", fontsize=16)
            ax.axis('off')  # Hide axis ticks and labels
            cb = plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
            cb.set_label('Carbon Density (Mg C/ha)')
        if view_option == "Carbon Value Density ($/ha)":
            # Convert carbon rasters to dollar value
            carbon_value_raster = carbon_data * 3.67 * carbon_price  # still in $/ha
            # Plot normal carbon density
            cax = ax.imshow(carbon_value_raster, cmap='cividis', vmin=np.nanmin(carbon_value_raster), vmax=np.nanmax(carbon_value_raster))
            ax.set_title(f"Carbon Value ({year})", fontsize=16)
            ax.axis('off')  # Hide axis ticks and labels
            cb = plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
            cb.set_label('Carbon Value Density ($/ha)')
        if view_option == "Carbon Value Gain/Loss ($/ha)":
            # Convert carbon rasters to change in dollar value
            carbon_value_raster = carbon_data * 3.67 * carbon_price  # still in $/ha
            baseline_value = baseline_data * 3.67 * carbon_price
            if year == min(years_sorted): # If we are on year 1 (2013), then there is no change
                # 2013: Show all-zero (neutral) map 
                zero_change = np.zeros_like(carbon_value_raster)
                cax = ax.imshow(zero_change, cmap='PuOr', vmin=-np.nanmax(carbon_value_raster), vmax=np.nanmax(carbon_value_raster))
                ax.set_title(f"Carbon Value ($) Change from Baseline (2013)", fontsize=16)
                ax.axis('off')  # Hide axis ticks and labels
                cb = plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
                cb.set_label('Change in Carbon Value ($/ha)')
            else:
                # Later years: Change relative to 2013
                diff = carbon_value_raster - baseline_value
                cax = ax.imshow(diff, cmap='PuOr', vmin=-np.nanmax(carbon_value_raster), vmax=np.nanmax(carbon_value_raster))
                ax.set_title(f"Carbon Value ($) Change from Baseline (2023)", fontsize=16)
                ax.axis('off')  # Hide axis ticks and labels
                cb = plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
                cb.set_label('Change in Carbon Value ($/ha)')

        # Dynamically update the year slider
        slider_placeholder.slider(
            "Current Year", 
            min_value=min(years_sorted), 
            max_value=max(years_sorted), 
            value=year, 
            disabled=True  # Disable manual dragging during animation
        )
        
        # Update the placeholder with the new figure
        placeholder.pyplot(fig)
        
        if view_option == "Carbon Gain/Loss (Mg C/ha)":
            # Label carbon gain or loss
            caption1_placeholder.markdown(
            "<div style='text-align:center; font-size:14px;'>Brown = carbon loss &nbsp;&nbsp;|&nbsp;&nbsp; Green = carbon gain<br>Compared to 2013 baseline</div>",
            unsafe_allow_html=True
            )
        if view_option == "Carbon Value Gain/Loss ($/ha)":
            # Label carbon value gain or loss
            caption1_placeholder.markdown(
            "<div style='text-align:center; font-size:14px;'>Orange = carbon value loss &nbsp;&nbsp;|&nbsp;&nbsp; Purple = carbon value gain<br>Compared to 2013 baseline</div>",
            unsafe_allow_html=True
            )

        if view_option == "Carbon Density (Mg C/ha)":
            caption1_placeholder.empty()  # hide caption in non-diff mode
        if view_option == "Carbon Value Density ($/ha)":
            caption1_placeholder.empty()  # hide caption in non-diff mode


        #progress_bar.progress((i + 1) / total_frames)  # Update progress bar
        time.sleep(.5)  # Pause .5 second between frames

        previous_data = carbon_data.copy()  # Save for next diff

else: # Else if map is static w/ year selection
    selected_year = st.slider("Select a year:", min_value=min(years), max_value=max(years), value=max(years))
    carbon_data = raster_layers[selected_year]
    baseline_data = raster_layers[2013] # for carbon accumulation or decline

    # Standardized color scale
    vmin = -10  # You can tweak this if needed based on data
    vmax = 10

    fig, ax = plt.subplots(figsize=(10, 8))

    if view_option == "Carbon Gain/Loss (Mg C/ha)":
        if selected_year == min(years): # If we are on year 1 (2013), then there is no change
            # 2013: Show all-zero (neutral) map
            zero_change = np.zeros_like(carbon_data)
            cax = ax.imshow(zero_change, cmap='BrBG', vmin=vmin, vmax=vmax)
            ax.set_title(f"Carbon Change from Baseline (2013)", fontsize=16)
            ax.axis('off')  # Hide axis ticks and labels
            cb = plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
            cb.set_label('Change in Carbon Density (Mg C/ha)')
        else:
            # Later years: Change relative to 2013
            diff = carbon_data - baseline_data
            cax = ax.imshow(diff, cmap='BrBG', vmin=vmin, vmax=vmax)
            ax.set_title(f"Carbon Change from 2013 Baseline ({selected_year})", fontsize=16)
            ax.axis('off')  # Hide axis ticks and labels
            cb = plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
            cb.set_label('Change in Carbon Density (Mg C/ha)')
    if view_option == "Carbon Density (Mg C/ha)":
        # Plot normal carbon density
        cax = ax.imshow(carbon_data, cmap='viridis', vmin=np.nanmin(carbon_data), vmax=np.nanmax(carbon_data))
        ax.set_title(f"Aboveground Carbon Density ({selected_year})", fontsize=16)
        ax.axis('off')  # Hide axis ticks and labels
        cb = plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
        cb.set_label('Carbon Density (Mg C/ha)')
    if view_option == "Carbon Value Density ($/ha)":
        # Convert carbon raster (e.g., from 2023) to dollar value
        # raster_2023: 2D array in Mg C/ha
        carbon_value_raster = carbon_data * 3.67 * carbon_price  # still in $/ha
        # Plot normal carbon density
        cax = ax.imshow(carbon_value_raster, cmap='cividis', vmin=np.nanmin(carbon_value_raster), vmax=np.nanmax(carbon_value_raster))
        ax.set_title(f"Carbon Value ({selected_year})", fontsize=16)
        ax.axis('off')  # Hide axis ticks and labels
        cb = plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
        cb.set_label('Carbon Value ($/ha)')
    if view_option == "Carbon Value Gain/Loss ($/ha)":
        # Convert carbon rasters to change in dollar value
        carbon_value_raster = carbon_data * 3.67 * carbon_price  # still in $/ha
        baseline_value = baseline_data * 3.67 * carbon_price
        if selected_year == min(years): # If we are on year 1 (2013), then there is no change
            # 2013: Show all-zero (neutral) map
            zero_change = np.zeros_like(carbon_value_raster)
            cax = ax.imshow(zero_change, cmap='PuOr', vmin=-np.nanmax(carbon_value_raster), vmax=np.nanmax(carbon_value_raster))
            ax.set_title(f"Carbon Value ($) Change from Baseline (2013)", fontsize=16)
            ax.axis('off')  # Hide axis ticks and labels
            cb = plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
            cb.set_label('Change in Carbon Value ($/ha)')
        else:
            # Later years: Change relative to 2013
            diff = carbon_value_raster - baseline_value
            cax = ax.imshow(diff, cmap='PuOr', vmin=-np.nanmax(carbon_value_raster), vmax=np.nanmax(carbon_value_raster))
            ax.set_title(f"Carbon Value ($) Change from Baseline ({selected_year})", fontsize=16)
            ax.axis('off')  # Hide axis ticks and labels
            cb = plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
            cb.set_label('Change in Carbon Value ($/ha)')


    st.pyplot(fig)

    if view_option == "Carbon Gain/Loss (Mg C/ha)":
        # Label carbon gain or loss
        st.markdown("<div style='text-align:center; font-size:14px;'>Brown = carbon loss &nbsp;&nbsp;|&nbsp;&nbsp; Green = carbon gain<br>Compared to 2013 baseline</div>", unsafe_allow_html=True)
    if view_option == "Carbon Value Gain/Loss ($/ha)":
        # Label carbon gain or loss
        st.markdown("<div style='text-align:center; font-size:14px;'>Orange = carbon value loss &nbsp;&nbsp;|&nbsp;&nbsp; Purple = carbon value gain<br>Compared to 2013 baseline</div>", unsafe_allow_html=True)



# 5. Calculate net carbon change
first_year = min(raster_layers.keys())
last_year = max(raster_layers.keys())

first_carbon = raster_layers[first_year]
last_carbon = raster_layers[last_year]

# Mask no-data values properly
valid_mask = (~np.isnan(first_carbon)) & (~np.isnan(last_carbon))

# Calculate change
net_change = last_carbon[valid_mask] - first_carbon[valid_mask]
total_change = np.nanmean(net_change)
annual_change = total_change / (last_year - first_year)

# 6. Show Time Series + Projection
st.subheader("Carbon Stock Over Time and Future Projection")

# Prepare data
df = pd.DataFrame({'Year': years, 'MeanCarbon': mean_carbon})

# Simple Linear Regression for projection
X = np.array(years).reshape(-1, 1)
y = np.array(mean_carbon)
model = LinearRegression()
model.fit(X, y)

# Predict future
future_years = np.arange(max(years)+1, max(years)+11)
predicted_carbon = model.predict(future_years.reshape(-1, 1))

# Combine data
future_df = pd.DataFrame({'Year': future_years, 'MeanCarbon': predicted_carbon})
full_df = pd.concat([df, future_df])

# Compute $ value of carbon loss from baseline year
baseline_carbon = full_df['MeanCarbon'].iloc[0]
carbon_loss = full_df['MeanCarbon'] - baseline_carbon
dollar_loss = carbon_loss * 3.67 * carbon_price  # $15/ton CO₂e
full_df['DollarValue'] = dollar_loss

# Plotly Dual-Axis Interactive Plot
fig2 = go.Figure()

# Line plot: Carbon stock
fig2.add_trace(go.Scatter(
    x=full_df['Year'], y=full_df['MeanCarbon'],
    mode='lines+markers',
    name='Carbon Density (Mg C/ha)',
    line=dict(color='green'),
    yaxis='y1'
))

# Bar plot: $ value
fig2.add_trace(go.Bar(
    x=full_df['Year'], y=full_df['DollarValue'],
    name='Carbon Value Density ($/ha)',
    marker_color='rgba(0, 123, 255, 0.5)',
    yaxis='y2'
))

# Add vertical line to mark projection start
fig2.add_vline(x=max(years), line_dash='dash', line_color='gray')

# Layout with dual y-axes
fig2.update_layout(
    title='Observed and Projected Carbon Stock with Estimated Dollar Value',
    xaxis=dict(title='Year'),
    yaxis=dict(
        title=dict(
            text='Carbon Density (Mg C/ha)',
            font=dict(color='green')
        ),
        tickfont=dict(color='green'),
    ),
    yaxis2=dict(
        title=dict(
            text='Carbon Value ($/ha)',
            font=dict(color='rgba(0, 123, 255, 1)')
        ),
        tickfont=dict(color='rgba(0, 123, 255, 1)'),
        overlaying='y',
        side='right'
    ),
    legend=dict(
    x=0,
    y=-0.3,
    xanchor='left',
    yanchor='top',
    orientation='h'
    ),
    bargap=0.2,
)

st.plotly_chart(fig2, use_container_width=True)

# 7. Tabular Summary

# Calculate Dollar Equivalent
# 1 Mg C = 3.67 Mg Co2 emission
# Carbon offset price is ~ $15 per ton Co2 emission, but slider can also adjust this
avg_carbon = np.nanmean(mean_carbon)
mean_carbon_dollar = avg_carbon * 3.67 * carbon_price
predicted_carbon_dollar = predicted_carbon[-1] * 3.67 * carbon_price
annual_change_dollar = annual_change * 3.67 * carbon_price
total_change_dollar = total_change * 3.67 * carbon_price

st.subheader("Summary Statistics")
summary_data = {
    "Statistic": [
                  "Average Carbon Density (2013-2023)", 
                  "Projected Carbon Density (2033)", 
                  "Average Annual Change", 
                  "Total Net Change (2013-2023)"],
    "Carbon Density": [
        f"{avg_carbon:.2f} Mg C/ha",
        f"{predicted_carbon[-1]:.2f} Mg C/ha",
        f"{annual_change:.2f} Mg C/ha/year",
        f"{total_change:.2f} Mg C/ha/year",
    ],
    "Value Equivalent": [
        f" ${mean_carbon_dollar:.0f} /ha",
        f" ${predicted_carbon_dollar:.0f}  /ha",
        f" ${annual_change_dollar:.0f}  /ha/year",
        f" ${total_change_dollar:.0f}  /ha/year",
    ]
}
summary_df = pd.DataFrame(summary_data)

# Highlight the "Value Equivalent" column
def highlight_column(s):
    return ['background-color: #e8f5e9' for _ in s]  # very light green

styled_df = summary_df.style.apply(highlight_column, subset=['Value Equivalent'])


st.dataframe(styled_df, use_container_width=True)

# 8. Narrative
st.markdown("---")
st.markdown("""
#### Why Planet's Data Matters
Planet's Forest Carbon Diligence product provides reliable, high-resolution data critical for identifying and monitoring carbon stock changes over time. With consistent coverage and clear trends in forest carbon accumulation/loss, developers can confidently select, validate, and manage forest carbon projects in Brazil.
""")
