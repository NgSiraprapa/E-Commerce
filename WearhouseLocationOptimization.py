import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
file_path = "E-Commerce_Analytics_Dataset_Term_Project.csv"
df = pd.read_csv(file_path)

# Convert 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Manually map customer locations to latitude & longitude
location_coords = {
    "Bangkok": {"lat": 13.7563, "lon": 100.5018},
    "Chiang Mai": {"lat": 18.7883, "lon": 98.9853},
    "Pattaya": {"lat": 12.9236, "lon": 100.8825},
    "Phuket": {"lat": 7.8804, "lon": 98.3923},
    "Khon Kaen": {"lat": 16.4419, "lon": 102.8359},
    "Hat Yai": {"lat": 7.0084, "lon": 100.4746}
}

# Add latitude and longitude columns
df["Latitude"] = df["Customer Location"].map(lambda x: location_coords.get(x, {}).get("lat"))
df["Longitude"] = df["Customer Location"].map(lambda x: location_coords.get(x, {}).get("lon"))

# Streamlit app layout
st.set_page_config(page_title="Warehouse Location Optimization", layout="wide")

# Title
st.title("📦 Warehouse Location Optimization Dashboard")
st.markdown("### Analyzing logistics and customer behavior for better warehouse placement.")

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
date_range = st.sidebar.date_input("Select Date Range", [df["Order Date"].min(), df["Order Date"].max()])
location_filter = st.sidebar.multiselect("Filter by Customer Location", df["Customer Location"].unique(), default=df["Customer Location"].unique())

# Apply filters
filtered_df = df[(df["Order Date"] >= pd.to_datetime(date_range[0])) & (df["Order Date"] <= pd.to_datetime(date_range[1]))]
filtered_df = filtered_df[filtered_df["Customer Location"].isin(location_filter)]

# Key Metrics
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Orders", len(filtered_df))
col2.metric("Total Revenue (USD)", f"${filtered_df['Purchase Amount (USD)'].sum():,.2f}")
col3.metric("Average Shipping Time (Days)", f"{filtered_df['Shipping Time (Days)'].mean():.1f}")

# Orders by Warehouse
st.subheader("🏭 Orders Distribution by Warehouse")
warehouse_orders = filtered_df["Warehouse Location"].value_counts().reset_index()
warehouse_orders.columns = ["Warehouse", "Orders"]
fig_warehouse = px.bar(warehouse_orders, x="Warehouse", y="Orders", color="Warehouse", title="Orders Processed by Each Warehouse")
st.plotly_chart(fig_warehouse, use_container_width=True)

# Improved Customer Locations Map
st.subheader("🌍 Customer Location Distribution")

# Filter data for visualization
df_filtered = filtered_df.dropna(subset=["Latitude", "Longitude"])  

# Aggregate order counts per location
location_summary = df_filtered.groupby(["Customer Location", "Latitude", "Longitude"])["Order ID"].count().reset_index()
location_summary.columns = ["Customer Location", "Latitude", "Longitude", "Total Orders"]

# Create a scatter plot with a better map style
fig_map = px.scatter_mapbox(
    location_summary, 
    lat="Latitude", 
    lon="Longitude", 
    size="Total Orders",
    color="Total Orders",  # Color based on demand
    hover_name="Customer Location",
    hover_data={"Latitude": False, "Longitude": False, "Total Orders": True},
    title="Customer Demand by Location",
    color_continuous_scale="thermal",  # Vibrant color scale
    mapbox_style="open-street-map",  # More detailed background
    zoom=4  # Adjust zoom for better visibility
)

# Improve font size and layout
fig_map.update_layout(
    font=dict(size=14),  # Increase font size
    title_font=dict(size=18, family="Arial Bold"),  # Bigger and bold title
    margin=dict(l=0, r=0, t=50, b=10),  # Adjust margins
)

st.plotly_chart(fig_map, use_container_width=True)

# Shipping Time Analysis
st.subheader("⏳ Shipping Time Analysis")
fig_shipping = px.histogram(filtered_df, x="Shipping Time (Days)", nbins=10, title="Shipping Time Distribution", color_discrete_sequence=["#FF5733"])
st.plotly_chart(fig_shipping, use_container_width=True)

# Predictive Insights: Recommended Warehouse Locations
st.subheader("🔮 Recommended Warehouse Expansion")
warehouse_suggestions = filtered_df.groupby("Customer Location")["Purchase Amount (USD)"].sum().reset_index()
warehouse_suggestions = warehouse_suggestions.sort_values(by="Purchase Amount (USD)", ascending=False)

st.write("Based on demand, consider expanding warehouses in these locations:")
st.dataframe(warehouse_suggestions.head(5))

# Conclusion
st.markdown("### 📌 Insights & Recommendations")
st.markdown("""
- **High-demand areas** should be prioritized for new warehouse locations.
- **Long shipping times** indicate potential inefficiencies in current logistics.
- **Warehouse performance** should be continuously monitored to reduce delays and costs.
""")