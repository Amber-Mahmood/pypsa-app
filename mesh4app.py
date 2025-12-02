import streamlit as st
import pandas as pd
import numpy as np
import pypsa
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
import datetime

# NEW: for interactive map bus placement
import folium
from streamlit_folium import st_folium

# --- NEW ADDITION START: Timezone List ---

# A comprehensive list of common timezones for selection
COMMON_TIMEZONES = [
    "UTC",
    "America/New_York",
    "America/Los_Angeles",
    "Europe/London",
    "Europe/Berlin",
    "Asia/Shanghai",
    "Asia/Kolkata",
    "Australia/Sydney",
    "Africa/Lagos",
    "Africa/Johannesburg",
    "Asia/Dubai",
    "Asia/Islamabad",
    "America/Jamaica",
    "Asia/Karachi",
    "Africa/Nairobi",  # IANA name for Kenya
    "Europe/Rome",
]

# --- NEW ADDITION END ---

# --- GLOBAL ASSUMPTIONS & CONSTANTS ---
DISCOUNT_RATE = 0.05  # 5%
LIFE_PV = 25  # Years
LIFE_BATTERY = 10  # Years
LIFE_BIOGAS = 20  # Years
GRID_EMISSION_FACTOR = 561  # gCO₂e/kWh
BIOGAS_EMISSION_FACTOR = 10  # gCO₂e/kWh (Small factor for non-ideal operation)

# --- NEW BIOGAS CONVERSION FACTORS ---
BIOGAS_FEEDSTOCK_ENERGY_CONTENT = 2.0  # MWh_thermal / tonne of feedstock (Example value: Biomass)
BIOGAS_GENERATOR_EFFICIENCY = 0.35  # Electrical efficiency of the generator
BIOGAS_ELECTRIC_ENERGY_PER_TONNE = BIOGAS_FEEDSTOCK_ENERGY_CONTENT * BIOGAS_GENERATOR_EFFICIENCY
# --- END NEW BIOGAS CONSTANTS ---

# --- FIXED ECONOMIC CONSTANTS (Simplified for User) ---
# These are fixed internally and not exposed to the user in the sidebar
SOLAR_FIXED_OPEX_RATE = 0.005  # 0.5% of CAPEX/year
BIOGAS_FIXED_OPEX_RATE = 0.02  # 2.0% of CAPEX/year
BATTERY_FIXED_OPEX_RATE = 0.01  # 1.0% of CAPEX/year
BATTERY_POWER_CAPEX_PER_MW = 200000  # $/MW

# --- PAGE CONFIG ---
st.set_page_config(page_title="Global Rural Microgrid", layout="wide")

st.title("🌍 Global Rural Mesh Grid Simulator (Capacity Expansion)")
st.markdown(
    "Optimization Model (PyPSA) determines **optimal capacity, sustainable and cost efficient system** "
    "based on inputs parameters for a user-defined network."
)

# --- HELPER FUNCTIONS ---


def calculate_crf(r, n):
    """Calculates the Capital Recovery Factor (CRF)."""
    if r == 0:
        return 1 / n
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def haversine(coord1, coord2):
    """Calculates the distance between two latitude/longitude points in km."""
    R = 6371.0
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2 +
         cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def load_synthetic_shape(index):
    """
    Generates the 8760-hour synthetic Load profile shape (unit-less).
    The resulting series has a mean value of approximately 1.0.
    """
    hours = np.arange(24)
    # Simple double-peak load curve (morning and evening)
    base_profile = (0.4 + 0.6 * np.sin(2 * np.pi * (hours - 7) / 24) +
                    1.0 * np.sin(2 * np.pi * (hours - 17) / 24) ** 2)
    base_profile = np.maximum(base_profile, 0.2)
    base_profile = base_profile / base_profile.max()
    daily_curve = np.tile(base_profile, 365)

    # Apply a slight weekday/weekend factor
    weekday_factor = np.where(index.weekday < 5, 0.9, 1.1)
    rng = np.random.default_rng(42)
    load_pu = daily_curve * weekday_factor * rng.normal(1.0, 0.08, size=8760)
    load_pu = np.maximum(load_pu, 0.01)  # Ensure positive load

    # Normalize the entire 8760 series so its mean power is 1.0 MW (used as a shape)
    load_shape = pd.Series(load_pu / load_pu.mean(), index=index)
    return load_shape


def load_synthetic_profile(index):
    """Generates the 8760-hour synthetic Load profile scaled to 4.9 GWh annually (MW)."""
    annual_energy_MWh = 4900
    average_power_MW = annual_energy_MWh / 8760
    load_shape = load_synthetic_shape(index)
    load_MW = load_shape * average_power_MW
    return load_MW


def generate_synthetic_solar(index):
    """Simple synthetic solar curve if TMY file fails or is missing."""
    hours = index.hour
    irrad = np.maximum(0, np.cos((hours - 12) / 24 * np.pi * 3))
    rng = np.random.default_rng(42)
    noise = rng.normal(1, 0.1, len(index))

    p_max_pu = irrad * noise
    return np.clip(p_max_pu, 0, 1)


@st.cache_data
def read_topology_data(uploaded_topology_file):
    """Reads the bus coordinates and line definitions from the uploaded CSV."""
    if uploaded_topology_file is None:
        # Return empty but valid structures if no file is uploaded
        return {}, []

    try:
        df_topo = pd.read_csv(uploaded_topology_file)
        # Ensure minimum required columns are present
        required_cols = ['Bus', 'Latitude', 'Longitude']
        if not all(col in df_topo.columns for col in required_cols):
            st.error(f"Topology file must contain columns: {', '.join(required_cols)}.")
            return {}, []

        bus_data = {}
        for _, row in df_topo.iterrows():
            bus_name = str(row['Bus'])
            bus_data[bus_name] = (row['Latitude'], row['Longitude'])

        # Extract lines: assumes columns starting with 'Line_' contain connections (Bus0-Bus1)
        line_data = []
        for col in df_topo.columns:
            if col.startswith('Line_'):
                # We are using a simple sequential connection fallback for now
                pass

        st.sidebar.success(f"Loaded **{len(bus_data)}** bus locations from topology file.")

        # Fallback: create simple sequential lines based on the order of buses
        buses_list = list(bus_data.keys())
        edges_fallback = []
        for i in range(len(buses_list)):
            u = buses_list[i]
            v = buses_list[(i + 1) % len(buses_list)]  # Connect back to form a loop
            if u != v:
                edges_fallback.append((u, v))

        if not edges_fallback and len(buses_list) > 0:
            if len(buses_list) == 2:
                edges_fallback = [(buses_list[0], buses_list[1])]
            elif len(buses_list) > 0:
                st.sidebar.warning("No explicit line definitions found; using simple sequential/loop connections.")

        return bus_data, edges_fallback

    except Exception as e:
        st.error(f"Error reading topology file: {e}")
        return {}, []


def interactive_topology_builder(center_lat, center_lon, map_zoom):
    """
    Interactive way for the user to define buses by clicking on a Folium map.

    Returns:
        bus_coords: dict {bus_name: (lat, lon)}
        edges: list of (bus_i, bus_j) tuples (simple loop order)
    """
    st.markdown(
        "**Interactive topology mode:** click on the map to add buses (Bus1, Bus2, ...). "
        "You can rename or delete buses in the table below."
    )

    # Persist bus list + last click in session_state
    if "manual_buses" not in st.session_state:
        st.session_state.manual_buses = []
    if "last_click" not in st.session_state:
        st.session_state.last_click = None

    # 1) Base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=map_zoom)

    # 2) Existing markers
    for bus in st.session_state.manual_buses:
        folium.Marker(
            location=[bus["lat"], bus["lon"]],
            tooltip=bus["name"],
        ).add_to(m)

    # 3) Render Folium map and capture interactions
    map_state = st_folium(
        m,
        width=700,
        height=450,
        key="topology_picker",
        returned_objects=["last_clicked", "zoom", "center"],
    )

    # 4) On new click, append a bus
    if map_state and map_state.get("last_clicked"):
        click = map_state["last_clicked"]
        click_tuple = (click["lat"], click["lng"])

        if st.session_state.last_click != click_tuple:
            st.session_state.last_click = click_tuple
            new_id = len(st.session_state.manual_buses) + 1
            st.session_state.manual_buses.append(
                {
                    "name": f"Bus{new_id}",
                    "lat": click["lat"],
                    "lon": click["lng"],
                }
            )

    # 5) Editable table
    if st.session_state.manual_buses:
        df_buses = pd.DataFrame(st.session_state.manual_buses)
        edited_df = st.data_editor(
            df_buses,
            num_rows="dynamic",
            use_container_width=True,
            key="manual_buses_editor",
        )
        st.session_state.manual_buses = edited_df.to_dict(orient="records")
    else:
        st.info("Click on the map to add the first bus.")

    # 6) Convert to bus_coords + edges
    bus_coords = {
        row["name"]: (row["lat"], row["lon"])
        for row in st.session_state.manual_buses
        if "name" in row and pd.notna(row["name"])
    }

    buses_list = list(bus_coords.keys())
    edges = []
    if len(buses_list) >= 2:
        for i in range(len(buses_list)):
            u = buses_list[i]
            v = buses_list[(i + 1) % len(buses_list)]
            if u != v:
                edges.append((u, v))

    return bus_coords, edges


@st.cache_data
def generate_profiles(uploaded_tmy_file, uploaded_hourly_load_file, uploaded_monthly_load_file,
                      uploaded_biogas_feedstock_file, user_timezone):
    """
    Generates the 8760-hour Load, PV, and Biogas profiles.
    The biogas profile is the hourly *maximum available power* based on monthly feedstock.
    """
    # PyPSA standard index: 8760 hours for a non-leap-year simulation
    index = pd.date_range("2020-01-01", periods=8760, freq="h")

    # Get the hourly shape profile (used for monthly disaggregation and scaling)
    load_shape_normalized = load_synthetic_shape(index)
    load_MW = None

    # 1. Load Profile (Priority: Hourly > Monthly > Synthetic)
    if uploaded_hourly_load_file is not None:
        try:
            df_load = pd.read_csv(uploaded_hourly_load_file)
            power_col = next(
                (col for col in df_load.columns
                 if any(keyword in col.lower() for keyword in ['load', 'power', 'mw'])),
                None,
            )
            if power_col is None and len(df_load.columns) >= 2:
                power_col = df_load.columns[1]

            if power_col and len(df_load) >= 8760:
                load_MW = pd.Series(df_load[power_col].values[:8760], index=index)
                st.sidebar.success(f"Loaded custom HOURLY load file using column '{power_col}'.")
            else:
                st.sidebar.warning(
                    f"Uploaded hourly load file is invalid or too short ({len(df_load)} data points). "
                    f"Checking for monthly data."
                )

        except Exception as e:
            st.sidebar.warning(
                f"Error processing custom hourly load file: {e}. Checking for monthly data."
            )

    # PRIORITY 2: Custom Monthly CSV (12 points)
    if load_MW is None and uploaded_monthly_load_file is not None:
        try:
            df_monthly = pd.read_csv(uploaded_monthly_load_file)
            energy_col = next(
                (col for col in df_monthly.columns
                 if any(keyword in col.lower() for keyword in ['energy', 'mwh', 'consumption'])),
                None,
            )

            if energy_col and len(df_monthly) >= 12:
                monthly_energy_MWh = pd.Series(df_monthly[energy_col].values[:12])
                monthly_energy_MWh.index = range(1, 13)
                monthly_shape_sum = load_shape_normalized.groupby(load_shape_normalized.index.month).sum()
                hourly_scale_factor = pd.Series(index=index, dtype=float)

                for month in range(1, 13):
                    scaling_ratio = (monthly_energy_MWh[month] / monthly_shape_sum[month])
                    hourly_scale_factor.loc[load_shape_normalized.index.month == month] = scaling_ratio

                load_MW = load_shape_normalized * hourly_scale_factor
                st.sidebar.success(
                    f"Loaded custom MONTHLY load file ({monthly_energy_MWh.sum():,.0f} MWh) "
                    f"and disaggregated to 8760 hours."
                )
            else:
                st.sidebar.warning(
                    "Could not identify energy column or data is less than 12 months in Monthly CSV. "
                    "Using synthetic load."
                )

        except Exception as e:
            st.sidebar.warning(f"Error processing custom monthly load file: {e}. Using synthetic load.")

    # PRIORITY 3: Synthetic Fallback
    if load_MW is None:
        load_MW = load_synthetic_profile(index)
        st.sidebar.info("Using default **synthetic load profile** (4.9 GWh annual).")

    # 2. Solar Profile (Generation)
    if uploaded_tmy_file is not None:
        try:
            df = pd.read_csv(uploaded_tmy_file)
            df['datetime_utc'] = pd.to_datetime(df['time(UTC)'], format='%Y%m%d:%H%M', errors='coerce')
            df = df.dropna(subset=['datetime_utc'])

            # Localize the UTC time and convert to the user's local timezone
            df['datetime_local'] = (
                df['datetime_utc'].dt.tz_localize('UTC').dt.tz_convert(user_timezone)
            )

            df = df.set_index(df['datetime_local'])

            # Remove timezone information and set year to 2020 for PyPSA's index alignment
            df.index = df.index.tz_localize(None).map(lambda t: t.replace(year=2020))

            # Handle DST Duplicates
            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='first')]
                st.sidebar.info("Removed duplicate index entries caused by DST transition.")

            ghi = df['G(h)'].clip(lower=0)

            # Reindex to the standard 8760-hour PyPSA index
            ghi_8760 = ghi.reindex(index, fill_value=0)

            p_max_pu = (ghi_8760 / ghi_8760.max()).values
            st.sidebar.success(
                f"Loaded **TMY file** and reconciled {len(ghi)} data points to 8760 hours "
                f"using timezone: **{user_timezone}**."
            )

        except Exception as e:
            st.sidebar.warning(f"Error processing TMY file: {e}. Using synthetic solar data.")
            p_max_pu = generate_synthetic_solar(index)
    else:
        p_max_pu = generate_synthetic_solar(index)
        st.sidebar.info("Using default **synthetic solar profile**.")

    # 3. Biogas Profile (Max available fuel)
    # Default to a high capacity (10 MW) with p_max_pu = 1.0 (unconstrained fuel)
    biogas_max_MW_available = pd.Series(10.0, index=index)
    st.sidebar.info("Using default **maximum biogas capacity (10 MW)** for 8760 hours.")

    if uploaded_biogas_feedstock_file is not None:
        try:
            df_biogas = pd.read_csv(uploaded_biogas_feedstock_file)
            feedstock_col = next(
                (col for col in df_biogas.columns
                 if any(keyword in col.lower() for keyword in ['feedstock', 'tonne', 'tonnes'])),
                None,
            )

            if feedstock_col and len(df_biogas) >= 12:
                # 1. Monthly Tonnes
                monthly_feedstock_tonnes = pd.Series(df_biogas[feedstock_col].values[:12])
                monthly_feedstock_tonnes.index = range(1, 13)

                # 2. Convert to Monthly Energy Limit (MWh)
                monthly_energy_limit_MWh = monthly_feedstock_tonnes * BIOGAS_ELECTRIC_ENERGY_PER_TONNE

                # 3. Disaggregate to Hourly Power Limit (MW)
                monthly_shape_sum = load_shape_normalized.groupby(load_shape_normalized.index.month).sum()
                hourly_scale_factor = pd.Series(index=index, dtype=float)

                for month in range(1, 13):
                    # Scaling ratio: (Monthly Biogas Energy / Monthly Load Shape Sum)
                    scaling_ratio = (monthly_energy_limit_MWh[month] / monthly_shape_sum[month])
                    hourly_scale_factor.loc[load_shape_normalized.index.month == month] = scaling_ratio

                # The result is the max biogas power available (MW) based on fuel
                biogas_max_MW_available = load_shape_normalized * hourly_scale_factor

                st.sidebar.success(
                    f"Loaded custom **MONTHLY biogas feedstock** ({monthly_feedstock_tonnes.sum():,.0f} tonnes) "
                    f"and converted to hourly MW profile."
                )
            else:
                st.sidebar.warning(
                    "Could not identify feedstock column or data is less than 12 months in Biogas CSV."
                )

        except Exception as e:
            st.sidebar.warning(f"Error processing biogas feedstock file: {e}. Using default max capacity.")

    return load_MW, pd.Series(p_max_pu, index=index), biogas_max_MW_available


@st.cache_resource
def build_and_solve_network(
    load_MW, p_max_pu, biogas_max_MW_available, bus_coords, edges,
    pv_max_cap, battery_max_power, battery_hours, fixed_biogas_cap,
    solar_capex_per_mw,
    biogas_capex_per_mw,
    biogas_fuel_cost,
    battery_capex_energy,
    grid_import_price,
):
    """
    Builds and solves the PyPSA network (Capacity Expansion OPF) using dynamic inputs.
    Biogas p_nom is now extendable, constrained by fixed_biogas_cap (p_nom_max) and the fuel profile (p_max_pu).
    """
    # PULL FIXED CONSTANTS (Hidden from User)
    solar_opex_rate = SOLAR_FIXED_OPEX_RATE
    biogas_opex_rate = BIOGAS_FIXED_OPEX_RATE
    battery_capex_power = BATTERY_POWER_CAPEX_PER_MW
    battery_opex_rate = BATTERY_FIXED_OPEX_RATE

    if not bus_coords:
        raise ValueError("Network topology (bus coordinates) is missing.")

    n = pypsa.Network()
    n.set_snapshots(load_MW.index)
    n.add("Carrier", "AC")

    # 1. Calculate Annualized Capital Cost for PyPSA
    crf_pv = calculate_crf(DISCOUNT_RATE, LIFE_PV)
    crf_biogas = calculate_crf(DISCOUNT_RATE, LIFE_BIOGAS)
    crf_battery = calculate_crf(DISCOUNT_RATE, LIFE_BATTERY)

    pv_capital_cost = (solar_capex_per_mw * crf_pv + solar_capex_per_mw * solar_opex_rate)
    battery_p_capital_cost = (battery_capex_power * crf_battery + battery_capex_power * battery_opex_rate)
    battery_e_capital_cost = (battery_capex_energy * crf_battery + battery_capex_energy * battery_opex_rate)
    biogas_capital_cost = (biogas_capex_per_mw * crf_biogas + biogas_capex_per_mw * biogas_opex_rate)

    # 2. Add Buses, Lines, and Loads
    for i, (bus_name, coords) in enumerate(bus_coords.items()):
        n.add("Bus", bus_name, x=coords[1], y=coords[0], v_nom=13.8, carrier="AC")

    # Add lines based on dynamic edges
    for u, v in edges:
        if u in n.buses.index and v in n.buses.index:
            coord_u = bus_coords[u]
            coord_v = bus_coords[v]
            dist = haversine(coord_u, coord_v)
            n.add(
                "Line", f"line_{u}_{v}", bus0=u, bus1=v,
                length=dist, r=0.4, x=0.35, s_nom=10, carrier="AC",
            )
        else:
            st.warning(f"Line definition skipped: Bus {u} or {v} not found.")

    total_load_t = load_MW.values
    all_buses = n.buses.index.tolist()
    num_buses = len(all_buses)

    if num_buses == 0:
        raise ValueError("Network has no buses defined. Cannot distribute load.")

    load_per_bus = total_load_t / num_buses

    # Distribute load uniformly across all buses
    for b in all_buses:
        n.add("Load", f"load_{b}", bus=b, p_set=load_per_bus)

    # 3. Add Generators & Storage (Capacity Expansion Logic)

    # 3.1. Solar PV (Optimized Size)
    pv_bus = all_buses[1] if num_buses > 1 else all_buses[0]
    n.add(
        "Generator", "pv", bus=pv_bus,
        p_nom_extendable=True,
        p_nom_max=pv_max_cap,
        capital_cost=pv_capital_cost,
        p_max_pu=p_max_pu,
        marginal_cost=0.0,
        carrier="AC",
    )

    # 3.2. Battery Storage (Optimized Size with fixed duration)
    battery_bus = all_buses[2] if num_buses > 2 else (all_buses[1] if num_buses > 1 else all_buses[0])
    n.add(
        "StorageUnit", "battery", bus=battery_bus,
        p_nom_extendable=True,
        p_nom_max=battery_max_power,  # MW
        capital_cost=battery_p_capital_cost,  # cost per MW (annualised)
        e_capital_cost=battery_e_capital_cost,  # cost per MWh (annualised)
        max_hours=battery_hours,  # fixes energy = p_nom * max_hours
        efficiency_store=0.95, efficiency_dispatch=0.95,
        marginal_cost=0.0,
        cyclic_state_of_charge=True,
        carrier="AC",
    )

    # 3.3. Biogas Generator (Optimized Size constrained by Fuel)
    biogas_bus = all_buses[3] if num_buses > 3 else (all_buses[0] if num_buses > 0 else "bus0_fallback")
    if biogas_bus in n.buses.index:
        # p_max_pu based on fuel availability vs physical capacity
        p_max_pu_biogas = biogas_max_MW_available.values / fixed_biogas_cap
        p_max_pu_biogas = np.clip(p_max_pu_biogas, 0, 1.0)

        n.add(
            "Generator", "biogas", bus=biogas_bus,
            p_nom_extendable=True,
            p_nom_max=fixed_biogas_cap,  # Acts as the physical size limit
            capital_cost=biogas_capital_cost,
            marginal_cost=biogas_fuel_cost * 1000,  # $/kWh -> $/MWh
            efficiency=BIOGAS_GENERATOR_EFFICIENCY,
            p_max_pu=p_max_pu_biogas,
            carrier="AC",
        )

    # 3.4. Grid Connection (SLACK BUS) - Located at the first bus
    slack_bus = all_buses[0]
    n.add(
        "Generator", "grid", bus=slack_bus,
        control="Slack",
        p_nom=1000,
        marginal_cost=grid_import_price * 1000,  # $/kWh -> $/MWh
        carrier="AC",
    )

    # 4. SOLVE Capacity Expansion OPF
    status = n.optimize(solver_name="highs")

    # Update biogas capacity extraction
    opt_biogas_cap = n.generators.at["biogas", "p_nom_opt"] if "biogas" in n.generators.index else 0.0

    return n, status, opt_biogas_cap


# --- SIDEBAR: STEP 1 - Data Upload & Location (Priority 1) ---
st.sidebar.header("Step 1: Data Upload & Location 🌎")

st.sidebar.subheader("Network Topology")

# NEW: topology input mode
topology_input_mode = st.sidebar.radio(
    "Topology Input Mode",
    ("Upload CSV", "Pick buses on map"),
    help="Either upload a topology CSV or pick bus locations interactively on a map.",
)

uploaded_topology_file = None
bus_coords = {}
edges = []

if topology_input_mode == "Upload CSV":
    uploaded_topology_file = st.sidebar.file_uploader(
        "1. Upload Network Topology CSV",
        type=["csv"],
        key="topology_uploader",
        help="Must contain columns: Bus (name), Latitude, Longitude. "
             "Lines will be sequentially/loop connected as a fallback.",
    )

    # Load from CSV
    bus_coords, edges = read_topology_data(uploaded_topology_file)
else:
    st.sidebar.info("Use the map in the main area to place buses in interactive mode.")

st.sidebar.subheader("Location Settings")
# Timezone Selection
user_timezone = st.sidebar.selectbox(
    "IANA Timezone",
    options=COMMON_TIMEZONES,
    index=COMMON_TIMEZONES.index("UTC"),  # Set 'UTC' as the default selection
    help="Crucial for aligning solar/load profiles to real time.",
)

st.sidebar.subheader("Load Profile")
uploaded_hourly_load_file = st.sidebar.file_uploader(
    "2. Upload HOURLY Load Profile CSV (MW, 8760 points)",
    type=["csv"], key="hourly_load_uploader",
    help="Highest priority. If uploaded, monthly/synthetic are ignored."
)
uploaded_monthly_load_file = st.sidebar.file_uploader(
    "3. Upload MONTHLY Load Profile CSV (MWh, 12 points)",
    type=["csv"], key="monthly_load_uploader",
    help="Second priority. If uploaded, disaggregated using synthetic shape."
)
st.sidebar.markdown("""
    **Monthly Data Format:** Must be a CSV with **12 rows** (Jan-Dec) and an 'Energy' column in MWh.
""")

st.sidebar.subheader("Generation Profile")
uploaded_tmy_file = st.sidebar.file_uploader(
    "4. Upload TMY CSV (Solar Irradiance, 8760 points)",
    type=["csv"], key="tmy_uploader", help="If none, synthetic solar is used."
)
# --- BIOGAS UPLOADER ---
uploaded_biogas_feedstock_file = st.sidebar.file_uploader(
    "5. Upload MONTHLY Biogas Feedstock CSV (tonnes/month)",
    type=["csv"], key="biogas_feedstock_uploader",
    help="If uploaded, the biogas generator dispatch is constrained by fuel availability."
)
# --- END NEW BIOGAS UPLOADER ---

# Call the profile generation function
load_MW, solar_pu, biogas_max_MW_available = generate_profiles(
    uploaded_tmy_file, uploaded_hourly_load_file, uploaded_monthly_load_file,
    uploaded_biogas_feedstock_file, user_timezone
)

# --- SIDEBAR: STEP 2 - Input Parameters (Priority 2) ---
st.sidebar.header("Step 2: Economic Inputs 💰 & Limits")

st.sidebar.subheader("Cost Inputs (CAPEX & OPEX)")

# --- USER-FACING ECONOMIC INPUTS ---
# 1. Solar CAPEX
solar_capex_per_mw = st.sidebar.number_input(
    "**1. PV CAPEX ($/MW)**",
    500000, 2000000, 1000000,
    help="The initial investment cost to build 1 MW of Solar PV capacity."
)

# 2. Biogas CAPEX
biogas_capex_per_mw = st.sidebar.number_input(
    "**2. Biogas CAPEX ($/MW)**",
    1000000, 5000000, 3000000,
    help="The initial investment cost to build 1 MW of Biogas generator capacity."
)

# 3. Battery Energy CAPEX
battery_capex_energy = st.sidebar.number_input(
    "**3. Battery Energy CAPEX ($/MWh)**",
    100000, 500000, 300000,
    help="The initial investment cost for 1 MWh of battery storage capacity (the 'tank' size)."
)

# 4. Biogas Fuel Cost (Marginal)
biogas_fuel_cost = st.sidebar.number_input(
    "**4. Biogas OPEX ($/kWh)**",
    0.01, 0.5, 0.08, format="%.3f",
    help="The cost of fuel (biogas) consumed per kWh of electricity generated."
)

# 5. Grid Import Price (Marginal)
grid_import_price = st.sidebar.number_input(
    "**5. Grid Import Price ($/kWh)**",
    0.01, 1.0, 0.50, format="%.2f",
    help="The cost to import 1 kWh of electricity from the external grid (marginal cost)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Capacity Limits 📐")
st.sidebar.markdown("_The solver will choose the optimal capacity up to this limit._")

pv_max_cap = st.sidebar.number_input("PV Max. Capacity Limit (MW)", 0.0, 10.0, 5.0)
battery_max_power = st.sidebar.number_input("Battery Max. Power Limit (MW)", 0.0, 10.0, 3.0)
battery_hours = st.sidebar.slider("Battery Duration (Hours)", 1, 10, 4)
# This input is now p_nom_max (Max Physical Size) for the biogas generator
fixed_biogas_cap = st.sidebar.number_input("Biogas Max. Physical Capacity (MW)", 0.0, 5.0, 0.248)

# --- SIDEBAR: STEP 3 - Map Center (Global Input) ---
st.sidebar.header("Step 3: Map View Center 🗺️")
# Default center is based on the first bus coordinate, or a generic global point if no bus is uploaded yet.
if bus_coords and list(bus_coords.keys()):
    default_lat = bus_coords[list(bus_coords.keys())[0]][0]
    default_lon = bus_coords[list(bus_coords.keys())[0]][1]
else:
    default_lat = 20.0
    default_lon = 0.0

center_lat = st.sidebar.number_input("Center Latitude", -90.0, 90.0, default_lat, format="%.2f")
center_lon = st.sidebar.number_input("Center Longitude", -180.0, 180.0, default_lon, format="%.2f")
map_zoom = st.sidebar.slider("Map Zoom Level", 1, 15, 11)

# --- MAIN AREA: GRID TOPOLOGY (Visible as soon as data is there) ---
st.markdown("---")
st.header("📍 Grid Topology (Input Topology)")

if topology_input_mode == "Upload CSV":
    if bus_coords:
        topo_df = pd.DataFrame(
            {
                "Bus": list(bus_coords.keys()),
                "lat": [coords[0] for coords in bus_coords.values()],
                "lon": [coords[1] for coords in bus_coords.values()],
            }
        )

        fig_topology = px.scatter_mapbox(
            topo_df,
            lat="lat",
            lon="lon",
            hover_name="Bus",
            height=450,
            zoom=map_zoom,
            center={"lat": center_lat, "lon": center_lon},
        )
        fig_topology.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )

        # Draw lines using edges
        for u, v in edges:
            if u in bus_coords and v in bus_coords:
                lat0, lon0 = bus_coords[u]
                lat1, lon1 = bus_coords[v]
                fig_topology.add_trace(
                    go.Scattermapbox(
                        mode="lines",
                        lat=[lat0, lat1],
                        lon=[lon0, lon1],
                        line=dict(width=2, color="blue"),
                        showlegend=False,
                    )
                )

        st.plotly_chart(fig_topology, use_container_width=True)
    else:
        st.info("Upload a Network Topology CSV in the sidebar to see the grid here.")
else:
    # Interactive mode: user clicks on map to create buses
    bus_coords, edges = interactive_topology_builder(center_lat, center_lon, map_zoom)
    if not bus_coords:
        st.warning("Add at least one bus by clicking on the map before running the optimization.")

# --- MAIN EXECUTION: STEP 4 - Run Optimization (Priority 3) ---
st.header("Step 4: Run Optimization & View Results")

# 1. Initialize Session State
if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False
if 'network_results' not in st.session_state:
    st.session_state.network_results = None
if 'opt_biogas_cap' not in st.session_state:
    st.session_state.opt_biogas_cap = 0.0
if 'solver_status' not in st.session_state:
    st.session_state.solver_status = None

run_button = st.button("Run Capacity Optimization (PyPSA-Highs)")

if run_button:
    if not bus_coords:
        st.error(
            "Cannot run simulation: define a network topology "
            "(CSV upload or interactive map) in Step 1."
        )
        st.session_state.simulation_done = False
    else:
        with st.spinner("Optimizing Capacity Expansion Model..."):
            try:
                # UPDATED CALL: passing biogas_max_MW_available, receiving opt_biogas_cap
                network, status, opt_biogas_cap = build_and_solve_network(
                    load_MW, solar_pu, biogas_max_MW_available, bus_coords, edges,
                    pv_max_cap, battery_max_power, battery_hours, fixed_biogas_cap,
                    solar_capex_per_mw,
                    biogas_capex_per_mw,
                    biogas_fuel_cost,
                    battery_capex_energy,
                    grid_import_price,
                )
                st.session_state.network_results = network
                st.session_state.simulation_done = True
                st.session_state.solver_status = status
                st.session_state.opt_biogas_cap = opt_biogas_cap  # Store optimized biogas capacity
            except Exception as e:
                st.error(
                    "Optimization failed! Ensure PyPSA and the 'highs' solver are correctly installed "
                    f"and topology is valid. Error: {e}"
                )
                st.session_state.simulation_done = False

# 2. Display Results
if st.session_state.simulation_done:
    network = st.session_state.network_results
    status = st.session_state.solver_status
    opt_biogas_cap = st.session_state.opt_biogas_cap
    st.success(f"Optimization Completed! Solver Status: **{status}**")

    # Helper to safely extract optimized capacity
    def safe_extract_capacity(df, component, cap_type):
        opt_col = f"{cap_type}_opt"
        if opt_col in df.columns:
            return df[opt_col].loc[component]
        elif cap_type in df.columns:
            return df[cap_type].loc[component]
        return 0.0

    # --- EXTRACT OPTIMIZED CAPACITIES ---
    opt_pv_cap = safe_extract_capacity(network.generators, "pv", "p_nom")
    opt_battery_power = safe_extract_capacity(network.storage_units, "battery", "p_nom")
    battery_hours_used = network.storage_units.at["battery", "max_hours"]
    opt_battery_energy = opt_battery_power * battery_hours_used

    # --- CALCULATIONS (LCOE, Costs, Energy) ---
    gen = network.generators_t.p
    total_load_annual_MWh = network.loads_t.p.sum().sum()
    solar_gen = gen["pv"].sum()
    grid_import = gen["grid"].sum()
    biogas_gen = gen["biogas"].sum() if "biogas" in gen.columns else 0

    # CRF for each component (recalculated/copied for local scope)
    crf_pv = calculate_crf(DISCOUNT_RATE, LIFE_PV)
    crf_biogas = calculate_crf(DISCOUNT_RATE, LIFE_BIOGAS)
    crf_battery = calculate_crf(DISCOUNT_RATE, LIFE_BATTERY)

    # Annualized CAPEX
    annual_capex_pv = opt_pv_cap * solar_capex_per_mw * crf_pv
    annual_capex_biogas = opt_biogas_cap * biogas_capex_per_mw * crf_biogas
    annual_capex_battery_p = opt_battery_power * BATTERY_POWER_CAPEX_PER_MW * crf_battery
    annual_capex_battery_e = opt_battery_energy * battery_capex_energy * crf_battery
    annual_capex = annual_capex_pv + annual_capex_biogas + annual_capex_battery_p + annual_capex_battery_e

    # Fixed OPEX
    fixed_opex_pv = opt_pv_cap * solar_capex_per_mw * SOLAR_FIXED_OPEX_RATE
    fixed_opex_biogas = opt_biogas_cap * biogas_capex_per_mw * BIOGAS_FIXED_OPEX_RATE
    fixed_opex_battery = (
        (opt_battery_power * BATTERY_POWER_CAPEX_PER_MW + opt_battery_energy * battery_capex_energy)
        * BATTERY_FIXED_OPEX_RATE
    )
    total_fixed_opex = fixed_opex_pv + fixed_opex_biogas + fixed_opex_battery

    # Variable OPEX
    variable_opex_biogas = biogas_gen * biogas_fuel_cost * 1000
    variable_opex_grid = grid_import * grid_import_price * 1000

    total_annual_cost = annual_capex + total_fixed_opex + variable_opex_biogas + variable_opex_grid
    lcoe = total_annual_cost / (total_load_annual_MWh * 1000)

    total_system_capex = (
        opt_pv_cap * solar_capex_per_mw
        + opt_biogas_cap * biogas_capex_per_mw
        + opt_battery_power * BATTERY_POWER_CAPEX_PER_MW
        + opt_battery_energy * battery_capex_energy
    )

    st.markdown("---")

    # ⚙️ Optimized System Configuration
    colA, colB, colC, colD = st.columns(4)
    colA.metric("**PV Capacity (MW)**", f"{opt_pv_cap:.3f}")
    colB.metric("**Biogas Capacity (MW)**", f"{opt_biogas_cap:.3f}")
    colC.metric("**Battery Power (MW)**", f"{opt_battery_power:.3f}")
    colD.metric("**Battery Energy (MWh)**", f"{opt_battery_energy:.3f}")

    st.markdown("---")

    # 📊 Key Performance Indicators (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("**LCOE (USD/kWh)**", f"${lcoe:.4f}")
    col2.metric(
        "**Renewable Fraction**",
        f"{(solar_gen + biogas_gen) / total_load_annual_MWh:.1%}",
    )
    col3.metric("**Total Annual Cost**", f"${total_annual_cost:,.0f} USD")
    col4.metric("**Total System CAPEX**", f"${total_system_capex:,.0f} USD")

    st.subheader("Annual Energy Supply Breakdown (MWh)")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Load Demand", f"{total_load_annual_MWh:,.0f}")
    col6.metric("Solar Generation", f"{solar_gen:,.0f}")
    col7.metric("Biogas Generation", f"{biogas_gen:,.0f}")
    col8.metric("Grid Purchase", f"{grid_import:,.0f}")
    
    # ---------- DOWNLOAD RESULTS SECTION ----------
    st.markdown("---")
    st.subheader("⬇️ Download Results")

    # Summary table
    summary_df = pd.DataFrame({
    "pv_capacity_MW": [opt_pv_cap],
    "biogas_capacity_MW": [opt_biogas_cap],
    "battery_power_MW": [opt_battery_power],
    "battery_energy_MWh": [opt_battery_energy],
    "lcoe_USD_per_kWh": [lcoe],
    "total_annual_cost_USD": [total_annual_cost],
    "total_system_capex_USD": [total_system_capex],
    "annual_load_MWh": [total_load_annual_MWh],
    "annual_solar_gen_MWh": [solar_gen],
    "annual_biogas_gen_MWh": [biogas_gen],
    "annual_grid_import_MWh": [grid_import],
    })

    st.download_button(
    label="Download summary (CSV)",
    data=summary_df.to_csv(index=False).encode("utf-8"),
    file_name="microgrid_summary.csv",
    mime="text/csv",
    )

    # Full-year dispatch timeseries
    full_dispatch_df = pd.DataFrame(
    {
        "timestamp": network.snapshots,
        "Solar_MW": gen["pv"].values,
        "Biogas_MW": (
            gen["biogas"].values
            if "biogas" in gen.columns
            else np.zeros_like(gen["pv"].values)
        ),
        "Grid_Import_MW": gen["grid"].values,
        "Battery_Dispatch_MW": (
            network.storage_units_t.p_dispatch["battery"].values
            - network.storage_units_t.p_store["battery"].values
        ),
        "Load_MW": network.loads_t.p.sum(axis=1).values,
    }
    )

    st.download_button(
    label="Download full-year dispatch (CSV, 8760 rows)",
    data=full_dispatch_df.to_csv(index=False).encode("utf-8"),
    file_name="microgrid_dispatch_8760.csv",
    mime="text/csv",
    )
# ---------- END DOWNLOAD SECTION ----------


    st.markdown("---")
    # ⚡ System Dispatch Analysis
    st.header("⚡ System Dispatch Analysis")

    df_plot = pd.DataFrame({
        "Solar": gen["pv"],
        "Biogas": gen["biogas"] if "biogas" in gen.columns else 0,
        "Grid Import": gen["grid"],
        "Battery Dispatch": (
            network.storage_units_t.p_dispatch["battery"]
            - network.storage_units_t.p_store["battery"]
        ),
        "Load": network.loads_t.p.sum(axis=1),
    })

    # Define all available days and months
    all_months = {i: datetime.date(2020, i, 1).strftime('%B') for i in range(1, 13)}
    all_days = list(range(1, 32))

    st.write("Select a date range for Dispatch and GHG charts (Year is fixed to 2020):")

    # INPUTS (Using selectbox for Month and Day)
    col_start_m, col_start_d, col_end_m, col_end_d = st.columns(4)

    with col_start_m:
        start_month_name = st.selectbox("Start Month", options=list(all_months.values()), index=0)
        start_month_num = list(all_months.keys())[list(all_months.values()).index(start_month_name)]

    with col_start_d:
        start_day = st.selectbox("Start Day", options=all_days, index=0)

    with col_end_m:
        end_month_name = st.selectbox("End Month", options=list(all_months.values()), index=0)
        end_month_num = list(all_months.keys())[list(all_months.values()).index(end_month_name)]

    with col_end_d:
        end_day = st.selectbox("End Day", options=all_days, index=6)  # Default 7th day

    # INTERNAL DATE CONSTRUCTION
    valid_date_range = False
    try:
        start_d = datetime.date(2020, start_month_num, start_day)
        end_d = datetime.date(2020, end_month_num, end_day)

        if start_d > end_d:
            st.warning("Start date cannot be after the end date. Please adjust your selection.")
            valid_date_range = False
        else:
            valid_date_range = True

    except ValueError as e:
        st.error(f"Invalid date selection: {e}. Please correct the month and day combination.")
        valid_date_range = False

    # PLOTTING LOGIC
    if valid_date_range:
        start_ts = pd.Timestamp(start_d, tz=user_timezone).tz_convert(None)
        end_ts = pd.Timestamp(end_d, tz=user_timezone).tz_convert(None) + pd.Timedelta(hours=23, minutes=59)

        # Dispatch plot
        filtered_df = df_plot.loc[start_ts:end_ts]
        fig = px.line(
            filtered_df,
            title=f"Dispatch Profile: {start_d.strftime('%b %d')} to {end_d.strftime('%b %d')} ({user_timezone})"
        )
        fig.update_layout(xaxis_title="Time", yaxis_title="Power (MW)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # 🌍 GHG Emissions Analysis
        st.header("🌍 GHG Emissions Analysis")

        # Calculate hourly net emissions (tCO₂e per hour)
        ghg_grid = gen["grid"] * GRID_EMISSION_FACTOR * 1000 / 1e6
        ghg_biogas = (
            (gen["biogas"] if "biogas" in gen.columns else 0)
            * BIOGAS_EMISSION_FACTOR * 1000 / 1e6
        )

        ghg_hourly_t = ghg_grid + ghg_biogas
        total_annual_ghg_t = ghg_hourly_t.sum()
        st.metric("**Total Annual GHG Emissions (tCO₂e)**", f"{total_annual_ghg_t:,.0f}")

        # Plot filtered GHG
        filtered_ghg_ts = ghg_hourly_t.loc[start_ts:end_ts]
        ghg_df = filtered_ghg_ts.reset_index()
        ghg_df.columns = ["timestamp", "tCO2e_per_hour"]

        fig_ghg = px.bar(
            ghg_df, x="timestamp", y="tCO2e_per_hour",
            title=(
                "Hourly GHG Emissions from Grid Import: "
                f"{start_d.strftime('%b %d')} to {end_d.strftime('%b %d')}"
            ),
            color_discrete_sequence=['#4CAF50'],
        )
        fig_ghg.update_layout(xaxis_title="Time (hourly)", yaxis_title="tCO₂e per hour")
        st.plotly_chart(fig_ghg, use_container_width=True)

        st.markdown("---")

        # 📍 Grid Topology (Optimized network)
        st.header("📍 Grid Topology (Optimized Network)")
        st.markdown("Visual representation of the optimized user-defined mesh grid.")

        bus_df = network.buses.copy()
        fig_map = px.scatter_mapbox(
            bus_df, lat="y", lon="x", hover_name=bus_df.index,
            zoom=map_zoom, height=500,
            center={"lat": center_lat, "lon": center_lon}
        )
        fig_map.update_layout(mapbox_style="open-street-map")

        # Draw lines on the map
        for line_name, line in network.lines.iterrows():
            b0 = network.buses.loc[line.bus0]
            b1 = network.buses.loc[line.bus1]
            fig_map.add_trace(
                go.Scattermapbox(
                    mode="lines",
                    lon=[b0.x, b1.x],
                    lat=[b0.y, b1.y],
                    line=dict(width=2, color='blue'),
                    name=line_name,
                    showlegend=False,
                )
            )
        st.plotly_chart(fig_map, use_container_width=True)

else:
    st.info(
        "👈 Please complete **Step 1** (Data Upload & Location), **Step 2** (Input Parameters), "
        "and **Step 3** (Map View) in the sidebar, then click the 'Run Capacity Optimization' button above to "
        "find the optimal system size and view the results."
    )
