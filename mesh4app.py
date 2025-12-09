import streamlit as st
import pandas as pd
import numpy as np
import pypsa
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
import datetime

def get_topology_csv_template():
    """Template for the case-study CSV."""
    template_df = pd.DataFrame(
        {
            "Bus": ["bus1", "bus2", "bus3"],
            "Latitude": [17.96812, 17.95116, 17.96488],
            "Longitude": [-77.30337, -77.29616, -77.27498],
            "NumHouseholds": [80, 50, 140],
            "Load_kWh_month": [30717, 21472, 59181],
            "Feedstock_tonne_month": [70, 53, 53],
        }
    )
    return template_df


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

st.title("🌍 Global Rural Mesh Grid Simulator")
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
def build_meshed_edges(bus_coords, k_nearest_extra=1, max_line_km=None):
    """
    Build a more realistic *meshed* set of edges:

    1. First build a Minimum Spanning Tree (MST) so all buses are connected
       with the shortest possible total line length.
    2. Then add 'k_nearest_extra' shortest additional connections
       per bus to create loops (mesh).

    Args:
        bus_coords: dict {bus_name: (lat, lon)}
        k_nearest_extra: how many extra neighbours (beyond the MST) each bus
                         should try to connect to (default 1).
        max_line_km: if given, ignore candidate lines longer than this distance.

    Returns:
        edges: list of (bus_i, bus_j) tuples.
    """
    buses = list(bus_coords.keys())
    n = len(buses)
    if n < 2:
        return []

    # --- pairwise distance matrix (km) ---
    coords = [bus_coords[b] for b in buses]
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(coords[i], coords[j])
            dist[i, j] = dist[j, i] = d

    # --- 1) Minimum Spanning Tree via a simple Prim's algorithm ---
    in_tree = [False] * n
    in_tree[0] = True
    edges = []

    while sum(in_tree) < n:
        best_i = best_j = None
        best_d = float("inf")
        for i in range(n):
            if not in_tree[i]:
                continue
            for j in range(n):
                if in_tree[j]:
                    continue
                d = dist[i, j]
                if d == 0:
                    continue
                if max_line_km is not None and d > max_line_km:
                    continue
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j

        if best_i is None:  # no more valid links
            break

        u = buses[best_i]
        v = buses[best_j]
        edges.append((u, v))
        in_tree[best_j] = True

    # --- 2) Add extra nearest-neighbour links to create loops ---
    def has_edge(a, b):
        return (a, b) in edges or (b, a) in edges

    for i in range(n):
        # distances from bus i to all others
        order = np.argsort(dist[i, :])
        added = 0
        for j in order:
            if i == j:
                continue
            d = dist[i, j]
            if d == 0:
                continue
            if max_line_km is not None and d > max_line_km:
                continue

            u, v = buses[i], buses[j]
            if not has_edge(u, v):
                edges.append((u, v))
                added += 1
            if added >= k_nearest_extra:
                break

    return edges


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
    """
    Reads a case-study CSV with columns:
    Bus, Latitude, Longitude, NumHouseholds, Load_kWh_month, Feedstock_tonne_month

    Returns:
        bus_coords: dict {bus_name: (lat, lon)}
        edges: list of (bus_i, bus_j)
        bus_meta_df: DataFrame indexed by Bus with NumHouseholds, Load_kWh_month, Feedstock_tonne_month
    """
    if uploaded_topology_file is None:
        return {}, [], None

    try:
        df = pd.read_csv(uploaded_topology_file)

        # Tolerant renaming for some common variants
        rename_map = {
            "BUS#": "Bus",
            "bus#": "Bus",
            "Latitude ": "Latitude",
            "Longitude ": "Longitude",
            "Load_kWh_month ": "Load_kWh_month",
            "Feedstock_tonne_month ": "Feedstock_tonne_month",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        required = ["Bus", "Latitude", "Longitude"]
        if not all(c in df.columns for c in required):
            st.error("Topology file must contain columns: Bus, Latitude, Longitude.")
            return {}, [], None

        # Coordinates
        bus_coords = {}
        for _, row in df.iterrows():
            bus_name = str(row["Bus"])
            bus_coords[bus_name] = (float(row["Latitude"]), float(row["Longitude"]))

        # Metadata (households, load, feedstock)
        meta_cols = [
            c
            for c in ["NumHouseholds", "Load_kWh_month", "Feedstock_tonne_month"]
            if c in df.columns
        ]
        if meta_cols:
            bus_meta_df = df.set_index(df["Bus"])[meta_cols].copy()
        else:
            bus_meta_df = None

        st.sidebar.success(f"Loaded **{len(bus_coords)}** buses from case-study CSV.")

        # Build a more realistic *meshed* network based on geographic proximity
        edges = build_meshed_edges(
            bus_coords,
            k_nearest_extra=1,    # try 1 or 2 for more loops
            max_line_km=40.0      # cap line length; adjust to your scale
        )

        return bus_coords, edges, bus_meta_df


    except Exception as e:
        st.error(f"Error reading topology/case-study file: {e}")
        return {}, [], None


def interactive_topology_builder(center_lat, center_lon, map_zoom):
    """
    Interactive way for the user to define buses by clicking on a Folium map.

    Returns:
        bus_coords: dict {bus_name: (lat, lon)}
        edges: list of (bus_i, bus_j) tuples (simple loop order)
    """
    st.markdown(
        "**Interactive topology mode:** click on the map to add buses (Bus1, Bus2, ...). "
        "You can rename or delete buses, and edit households/load/feedstock in the table below."
    )

    # One single source of truth for buses in interactive mode
    if "bus_table_df" not in st.session_state:
        st.session_state.bus_table_df = pd.DataFrame(
            columns=[
                "name",
                "lat",
                "lon",
                "NumHouseholds",
                "Load_kWh_month",
                "Feedstock_tonne_month",
            ]
        )
    if "last_click" not in st.session_state:
        st.session_state.last_click = None

    df = st.session_state.bus_table_df.copy()

    # 1) Base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=map_zoom)

    # 2) Existing markers
    for _, row in df.iterrows():
        if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
            folium.Marker(
                location=[row["lat"], row["lon"]],
                tooltip=str(row.get("name", "")),
            ).add_to(m)

    # 3) Render Folium map and capture interactions
    map_state = st_folium(
        m,
        width=700,
        height=450,
        key="topology_picker",
        returned_objects=["last_clicked", "zoom", "center"],
    )

    # 4) On new click, append a bus with default meta values (before editor)
    if map_state and map_state.get("last_clicked"):
        click = map_state["last_clicked"]
        click_tuple = (click["lat"], click["lng"])

        if st.session_state.last_click != click_tuple:
            st.session_state.last_click = click_tuple
            new_id = len(df) + 1
            new_row = {
                "name": f"Bus{new_id}",
                "lat": click["lat"],
                "lon": click["lng"],
                "NumHouseholds": 0,
                "Load_kWh_month": 0.0,
                "Feedstock_tonne_month": 0.0,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # 5) Ensure all required columns exist & in right order
    for col in ["name", "lat", "lon", "NumHouseholds", "Load_kWh_month", "Feedstock_tonne_month"]:
        if col not in df.columns:
            if col == "name":
                df[col] = ""
            elif col in ["lat", "lon"]:
                df[col] = np.nan
            else:
                df[col] = 0.0

    df = df[
        ["name", "lat", "lon", "NumHouseholds", "Load_kWh_month", "Feedstock_tonne_month"]
    ]

    # 6) Editable table
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="manual_buses_editor",
    )
    st.caption(
        "Edit bus name, coordinates, number of households, monthly load (kWh/month), "
        "and feedstock (tonne/month) directly in the table."
    )

    # Save back the edited table as the new truth
    st.session_state.bus_table_df = edited_df

        # 7) Convert to bus_coords + edges
    bus_coords = {}
    for _, row in edited_df.iterrows():
        if (
            pd.notna(row.get("name"))
            and pd.notna(row.get("lat"))
            and pd.notna(row.get("lon"))
        ):
            bus_coords[str(row["name"])] = (float(row["lat"]), float(row["lon"]))

    # build meshed edges *after* collecting all buses
    if len(bus_coords) >= 2:
        edges = build_meshed_edges(
            bus_coords,
            k_nearest_extra=1,
            max_line_km=40.0,
        )
    else:
        edges = []

    return bus_coords, edges



@st.cache_data
def generate_profiles(uploaded_tmy_file, user_timezone, bus_meta_df=None):
    """
    Generates:
      - load_MW: total system load (8760h)
      - p_max_pu: solar per-unit availability
      - biogas_max_MW_available: hourly max biogas power (fuel-limited)
      - bus_load_profiles: dict {bus_name: Series(MW, 8760h)} if bus_meta_df is given
    """
    index = pd.date_range("2020-01-01", periods=8760, freq="h")

    # Base shape (unitless, mean ≈ 1)
    load_shape_normalized = load_synthetic_shape(index)

    bus_load_profiles = {}
    load_MW = None

    # --- Per-bus load from case-study CSV ---
    if bus_meta_df is not None and "Load_kWh_month" in bus_meta_df.columns:
        shape_sum = load_shape_normalized.sum()  # ≈8760
        for bus_name, row in bus_meta_df.iterrows():
            monthly_kwh = row.get("Load_kWh_month", 0.0)
            if pd.isna(monthly_kwh) or monthly_kwh <= 0:
                continue

            # Assume Load_kWh_month is typical month; convert to annual MWh
            annual_MWh = 12.0 * float(monthly_kwh) / 1000.0  # kWh/month → MWh/year
            scale = annual_MWh / shape_sum  # MW scaling factor

            bus_load_profiles[str(bus_name)] = load_shape_normalized * scale

        if bus_load_profiles:
            # System load is the sum of all bus loads
            load_MW = sum(bus_load_profiles.values())
            st.sidebar.success("Used Load_kWh_month from case-study CSV for per-bus load.")
        else:
            st.sidebar.warning("Case-study CSV had no valid Load_kWh_month; using synthetic load.")

    # --- Fallback to synthetic total load if no bus-level data ---
    if load_MW is None:
        load_MW = load_synthetic_profile(index)
        bus_load_profiles = {}  # triggers equal split later
        st.sidebar.info("Using default **synthetic load profile** (4.9 GWh annual).")

    # --- Solar Profile (TMY or synthetic) ---
    if uploaded_tmy_file is not None:
        try:
            df = pd.read_csv(uploaded_tmy_file)
            df["datetime_utc"] = pd.to_datetime(
                df["time(UTC)"], format="%Y%m%d:%H%M", errors="coerce"
            )
            df = df.dropna(subset=["datetime_utc"])

            # Localize UTC and convert to user timezone
            df["datetime_local"] = (
                df["datetime_utc"].dt.tz_localize("UTC").dt.tz_convert(user_timezone)
            )

            df = df.set_index(df["datetime_local"])

            # Remove timezone info and set year to 2020 for alignment
            df.index = df.index.tz_localize(None).map(lambda t: t.replace(year=2020))

            # Handle DST duplicates
            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep="first")]
                st.sidebar.info("Removed duplicate index entries caused by DST transition.")

            ghi = df["G(h)"].clip(lower=0)
            ghi_8760 = ghi.reindex(index, fill_value=0)

            if ghi_8760.max() > 0:
                p_max_pu = (ghi_8760 / ghi_8760.max()).values
            else:
                p_max_pu = generate_synthetic_solar(index)
                st.sidebar.warning("TMY file had zero GHI; using synthetic solar.")

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

    # --- Biogas max profile from Feedstock_tonne_month in case-study CSV ---
    biogas_max_MW_available = pd.Series(10.0, index=index)  # default 10 MW flat

    if bus_meta_df is not None and "Feedstock_tonne_month" in bus_meta_df.columns:
        total_feedstock_month = bus_meta_df["Feedstock_tonne_month"].fillna(0.0).sum()
        if total_feedstock_month > 0:
            annual_tonnes = 12.0 * total_feedstock_month
            annual_energy_MWh = annual_tonnes * BIOGAS_ELECTRIC_ENERGY_PER_TONNE

            shape_sum = load_shape_normalized.sum()
            scale_bio = annual_energy_MWh / shape_sum
            biogas_max_MW_available = load_shape_normalized * scale_bio

            st.sidebar.success(
                f"Using Feedstock_tonne_month from CSV "
                f"(total {total_feedstock_month:.0f} t/month → "
                f"{annual_energy_MWh:,.0f} MWh/year biogas potential)."
            )
        else:
            st.sidebar.info("Feedstock_tonne_month is zero; using default max biogas capacity (10 MW).")
    else:
        st.sidebar.info("Using default **maximum biogas capacity (10 MW)** for 8760 hours.")

    return load_MW, pd.Series(p_max_pu, index=index), biogas_max_MW_available, bus_load_profiles


@st.cache_resource
def build_and_solve_network(
    load_MW,
    p_max_pu,
    biogas_max_MW_available,
    bus_coords,
    edges,
    bus_load_profiles,
    pv_max_cap,
    battery_max_power,
    battery_hours,
    fixed_biogas_cap,
    solar_capex_per_mw,
    biogas_capex_per_mw,
    biogas_fuel_cost,
    battery_capex_energy,
    grid_import_price,
):
    """
    Builds and solves the PyPSA network (Capacity Expansion OPF) using dynamic inputs.
    Biogas p_nom is extendable, constrained by fixed_biogas_cap (p_nom_max) and the fuel profile (p_max_pu).
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

    pv_capital_cost = solar_capex_per_mw * (crf_pv + solar_opex_rate)
    battery_p_capital_cost = battery_capex_power * (crf_battery + battery_opex_rate)
    battery_e_capital_cost = battery_capex_energy * (crf_battery + battery_opex_rate)
    biogas_capital_cost = biogas_capex_per_mw * (crf_biogas + biogas_opex_rate)

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
                "Line",
                f"line_{u}_{v}",
                bus0=u,
                bus1=v,
                length=dist,
                r=0.4,
                x=0.35,
                s_nom=10,
                carrier="AC",
            )
        else:
            st.warning(f"Line definition skipped: Bus {u} or {v} not found.")

    all_buses = n.buses.index.tolist()
    num_buses = len(all_buses)

    if num_buses == 0:
        raise ValueError("Network has no buses defined. Cannot distribute load.")

    # If we have per-bus profiles from the case-study CSV, use them
    if bus_load_profiles:
        for b in all_buses:
            if b in bus_load_profiles:
                p_set = bus_load_profiles[b]
            else:
                p_set = pd.Series(0.0, index=load_MW.index)
            n.add("Load", f"load_{b}", bus=b, p_set=p_set)
    else:
        # Fallback: equally distribute total load across buses
        total_load_t = load_MW.values
        load_per_bus = total_load_t / num_buses
        for b in all_buses:
            n.add("Load", f"load_{b}", bus=b, p_set=load_per_bus)

    # 3. Add Generators & Storage (Capacity Expansion Logic)

    # 3.1. Solar PV (Optimized Size)
    pv_bus = all_buses[1] if num_buses > 1 else all_buses[0]
    n.add(
        "Generator",
        "pv",
        bus=pv_bus,
        p_nom_extendable=True,
        p_nom_max=pv_max_cap,
        capital_cost=pv_capital_cost,
        p_max_pu=p_max_pu,
        marginal_cost=0.0,
        carrier="AC",
    )

    # 3.2. Battery Storage (Optimized Size with fixed duration)
    battery_bus = (
        all_buses[2] if num_buses > 2 else (all_buses[1] if num_buses > 1 else all_buses[0])
    )
    n.add(
        "StorageUnit",
        "battery",
        bus=battery_bus,
        p_nom_extendable=True,
        p_nom_max=battery_max_power,  # MW
        capital_cost=battery_p_capital_cost,  # cost per MW (annualised)
        e_capital_cost=battery_e_capital_cost,  # cost per MWh (annualised)
        max_hours=battery_hours,  # fixes energy = p_nom * max_hours
        efficiency_store=0.95,
        efficiency_dispatch=0.95,
        marginal_cost=0.0,
        cyclic_state_of_charge=True,
        carrier="AC",
    )

    # 3.3. Biogas Generator (Optimized Size constrained by Fuel)
    biogas_bus = all_buses[3] if num_buses > 3 else all_buses[0]
    if biogas_bus in n.buses.index and fixed_biogas_cap > 0:
        p_max_pu_biogas = biogas_max_MW_available.values / fixed_biogas_cap
        p_max_pu_biogas = np.clip(p_max_pu_biogas, 0, 1.0)

        n.add(
            "Generator",
            "biogas",
            bus=biogas_bus,
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
        "Generator",
        "grid",
        bus=slack_bus,
        control="Slack",
        p_nom=1000,
        marginal_cost=grid_import_price * 1000,  # $/kWh -> $/MWh
        carrier="AC",
    )

    # 4. SOLVE Capacity Expansion OPF
    status = n.optimize(solver_name="highs")

    # Update biogas capacity extraction
    if "biogas" in n.generators.index:
        opt_biogas_cap = n.generators.at["biogas", "p_nom_opt"]
    else:
        opt_biogas_cap = 0.0

    return n, status, opt_biogas_cap


# --- SIDEBAR: STEP 1 - Data Upload & Location (Priority 1) ---
st.sidebar.header("Step 1: Case-Study Data & Location 🌎")

st.sidebar.subheader("Network & Demand Topology")

topology_input_mode = st.sidebar.radio(
    "Topology Input Mode",
    ("Upload CSV (recommended)", "Pick buses on map"),
    help="Either upload a case-study CSV or pick bus locations interactively.",
)

uploaded_topology_file = None
bus_coords = {}
edges = []
bus_meta_df = None

if topology_input_mode == "Upload CSV (recommended)":
    uploaded_topology_file = st.sidebar.file_uploader(
        "1. Upload Case-Study CSV",
        type=["csv"],
        key="topology_uploader",
        help="Columns: Bus, Latitude, Longitude, NumHouseholds, Load_kWh_month, Feedstock_tonne_month",
    )

    bus_coords, edges, bus_meta_df = read_topology_data(uploaded_topology_file)
    # --- Show & download CSV template ---
    with st.sidebar.expander("📄 View CSV template"):
        template_df = get_topology_csv_template()
        st.dataframe(template_df, use_container_width=True)
        st.caption(
            "Required columns:\n"
            "Bus, Latitude, Longitude, NumHouseholds, Load_kWh_month, Feedstock_tonne_month"
        )

        template_csv = template_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV template",
            data=template_csv,
            file_name="case_study_template.csv",
            mime="text/csv",
        )
else:
    st.sidebar.info("Use the map in the main area to place buses in interactive mode.")
    # bus_coords & edges will be set later in the main area
    bus_meta_df = None

st.sidebar.subheader("Location Settings")
# Timezone Selection
user_timezone = st.sidebar.selectbox(
    "IANA Timezone",
    options=COMMON_TIMEZONES,
    index=COMMON_TIMEZONES.index("UTC"),  # Set 'UTC' as the default selection
    help="Crucial for aligning solar/load profiles to real time.",
)

st.sidebar.subheader("Generation Profile")
uploaded_tmy_file = st.sidebar.file_uploader(
    "2. Upload TMY CSV (Solar Irradiance, 8760 points)",
    type=["csv"],
    key="tmy_uploader",
    help="If none, synthetic solar is used.",
)

# Generate profiles (may use bus_meta_df if CSV uploaded)
load_MW, solar_pu, biogas_max_MW_available, bus_load_profiles = generate_profiles(
    uploaded_tmy_file, user_timezone, bus_meta_df
)

# --- SIDEBAR: STEP 2 - Input Parameters (Priority 2) ---
st.sidebar.header("Step 2: Economic Inputs 💰 & Limits")

st.sidebar.subheader("Cost Inputs (CAPEX & OPEX)")

# --- USER-FACING ECONOMIC INPUTS ---
# 1. Solar CAPEX
solar_capex_per_mw = st.sidebar.number_input(
    "**1. PV CAPEX ($/MW)**",
    500000,
    2000000,
    1000000,
    help="The initial investment cost to build 1 MW of Solar PV capacity.",
)

# 2. Biogas CAPEX
biogas_capex_per_mw = st.sidebar.number_input(
    "**2. Biogas CAPEX ($/MW)**",
    1000000,
    5000000,
    3000000,
    help="The initial investment cost to build 1 MW of Biogas generator capacity.",
)

# 3. Battery Energy CAPEX
battery_capex_energy = st.sidebar.number_input(
    "**3. Battery Energy CAPEX ($/MWh)**",
    100000,
    500000,
    300000,
    help="The initial investment cost for 1 MWh of battery storage capacity (the 'tank' size).",
)

# 4. Biogas Fuel Cost (Marginal)
biogas_fuel_cost = st.sidebar.number_input(
    "**4. Biogas OPEX ($/kWh)**",
    0.01,
    0.5,
    0.08,
    format="%.3f",
    help="The cost of fuel (biogas) consumed per kWh of electricity generated.",
)

# 5. Grid Import Price (Marginal)
grid_import_price = st.sidebar.number_input(
    "**5. Grid Import Price ($/kWh)**",
    0.01,
    1.0,
    0.50,
    format="%.2f",
    help="The cost to import 1 kWh of electricity from the external grid (marginal cost).",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Emission Factor 🌍")

grid_emission_factor = st.sidebar.number_input(
    "Grid Emission Factor (gCO₂e/kWh)",
    min_value=0.0,
    max_value=1500.0,
    value=float(GRID_EMISSION_FACTOR),  # uses your constant 561 as default
    step=10.0,
    help="Operational emission factor for imported grid electricity."
)

st.sidebar.subheader("Capacity Limits 📐")
st.sidebar.markdown("_The solver will choose the optimal capacity up to this limit._")

pv_max_cap = st.sidebar.number_input("PV Max. Capacity Limit (MW)", 0.0, 10.0, 5.0)
battery_max_power = st.sidebar.number_input("Battery Max. Power Limit (MW)", 0.0, 10.0, 3.0)
battery_hours = st.sidebar.slider("Battery Duration (Hours)", 1, 10, 4)
fixed_biogas_cap = st.sidebar.number_input(
    "Biogas Max. Physical Capacity (MW)", 0.0, 20.0, 5.0
)

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

if topology_input_mode == "Upload CSV (recommended)":
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

        if bus_meta_df is not None:
            st.subheader("Bus-Level Metadata (from CSV)")
            st.dataframe(bus_meta_df)
    else:
        st.info("Upload a Case-Study CSV in the sidebar to see the grid here.")
else:
    # Interactive mode: user clicks on map to create buses
    bus_coords, edges = interactive_topology_builder(center_lat, center_lon, map_zoom)

    # If we have manually defined buses, build a bus_meta_df from the table
    if (
        bus_coords
        and "bus_table_df" in st.session_state
        and not st.session_state.bus_table_df.empty
    ):
        mb = st.session_state.bus_table_df.copy()

        # Make sure the metadata columns exist
        for col in ["NumHouseholds", "Load_kWh_month", "Feedstock_tonne_month"]:
            if col not in mb.columns:
                mb[col] = 0.0

        # Index by bus name (same as used in the network)
        bus_meta_df = mb.set_index(mb["name"])[
            ["NumHouseholds", "Load_kWh_month", "Feedstock_tonne_month"]
        ]

        # 🔁 Re-generate profiles using this manual metadata
        load_MW, solar_pu, biogas_max_MW_available, bus_load_profiles = generate_profiles(
            uploaded_tmy_file, user_timezone, bus_meta_df
        )
    else:
        bus_meta_df = None
        if not bus_coords:
            st.warning("Add at least three buses by clicking on the map before running the optimization.")


# --- MAIN EXECUTION: STEP 4 - Run Optimization (Priority 3) ---
st.header("Step 4: Run Optimization & View Results")

# 1. Initialize Session State
if "simulation_done" not in st.session_state:
    st.session_state.simulation_done = False
if "network_results" not in st.session_state:
    st.session_state.network_results = None
if "opt_biogas_cap" not in st.session_state:
    st.session_state.opt_biogas_cap = 0.0
if "solver_status" not in st.session_state:
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
                network, status, opt_biogas_cap = build_and_solve_network(
                    load_MW,
                    solar_pu,
                    biogas_max_MW_available,
                    bus_coords,
                    edges,
                    bus_load_profiles,
                    pv_max_cap,
                    battery_max_power,
                    battery_hours,
                    fixed_biogas_cap,
                    solar_capex_per_mw,
                    biogas_capex_per_mw,
                    biogas_fuel_cost,
                    battery_capex_energy,
                    grid_import_price,
                )
                st.session_state.network_results = network
                st.session_state.simulation_done = True
                st.session_state.solver_status = status
                st.session_state.opt_biogas_cap = opt_biogas_cap
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
    biogas_gen = gen["biogas"].sum() if "biogas" in gen.columns else 0.0

    # NEW: total electricity generation (MWh) and grid capacity (MW)
    total_generation_MWh = solar_gen + biogas_gen + grid_import
    grid_peak_import_MW = gen["grid"].max()  # effective used grid "capacity"

    # CRF for each component (recalculated/copied for local scope)
    crf_pv = calculate_crf(DISCOUNT_RATE, LIFE_PV)
    crf_biogas = calculate_crf(DISCOUNT_RATE, LIFE_BIOGAS)
    crf_battery = calculate_crf(DISCOUNT_RATE, LIFE_BATTERY)

    # Annualized CAPEX
    annual_capex_pv = opt_pv_cap * solar_capex_per_mw * crf_pv
    annual_capex_biogas = opt_biogas_cap * biogas_capex_per_mw * crf_biogas
    annual_capex_battery_p = opt_battery_power * BATTERY_POWER_CAPEX_PER_MW * crf_battery
    annual_capex_battery_e = opt_battery_energy * battery_capex_energy * crf_battery
    annual_capex = (
        annual_capex_pv
        + annual_capex_biogas
        + annual_capex_battery_p
        + annual_capex_battery_e
    )

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

    if total_load_annual_MWh > 0:
        lcoe = total_annual_cost / (total_load_annual_MWh * 1000)
    else:
        lcoe = 0.0

    # NEW: Average generation cost (USD/kWh) based on total generation
    if total_generation_MWh > 0:
        gen_cost_USD_per_kWh = total_annual_cost / (total_generation_MWh * 1000.0)
    else:
        gen_cost_USD_per_kWh = 0.0

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

    # NEW: grid capacity and total generation
    colA2, colB2, colC2, colD2 = st.columns(4)
    colA2.metric("**Grid Capacity (MW)**", f"{grid_peak_import_MW:.3f}")
    colB2.metric("**Total Generation (MWh)**", f"{total_generation_MWh:,.0f}")
    colC2.write("")
    colD2.write("")

    st.markdown("---")

    # 📊 Key Performance Indicators (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("**LCOE (USD/kWh)**", f"${lcoe:.4f}")
    col2.metric("**Avg Generation Cost (USD/kWh)**", f"${gen_cost_USD_per_kWh:.4f}")
    col3.metric("**Total Annual Cost**", f"${total_annual_cost:,.0f} USD")
    col4.metric("**Total System CAPEX**", f"${total_system_capex:,.0f} USD")

    # Optional extra KPIs in a second row
    col5, col6 = st.columns(2)
    if total_load_annual_MWh > 0:
        renew_frac = (solar_gen + biogas_gen) / total_load_annual_MWh
    else:
        renew_frac = 0.0
    col5.metric("**Renewable Fraction**", f"{renew_frac:.1%}")

    st.subheader("Annual Energy Supply Breakdown (MWh/yr)")
    col7, col8, col9, col10 = st.columns(4)
    col7.metric("Load Demand", f"{total_load_annual_MWh:,.0f}")
    col8.metric("Solar Generation", f"{solar_gen:,.0f}")
    col9.metric("Biogas Generation", f"{biogas_gen:,.0f}")
    col10.metric("Grid Purchase", f"{grid_import:,.0f}")

    # ---------- DOWNLOAD RESULTS SECTION ----------
    st.markdown("---")
    st.subheader("⬇️ Download Results")

    # Summary table
    summary_df = pd.DataFrame(
        {
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
        }
    )

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
                gen["biogas"].values if "biogas" in gen.columns else np.zeros_like(gen["pv"].values)
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

    df_plot = pd.DataFrame(
        {
            "Solar": gen["pv"],
            "Biogas": gen["biogas"] if "biogas" in gen.columns else 0,
            "Grid Import": gen["grid"],
            "Battery Dispatch": (
                network.storage_units_t.p_dispatch["battery"]
                - network.storage_units_t.p_store["battery"]
            ),
            "Load": network.loads_t.p.sum(axis=1),
        }
    )

    # Define all available days and months
    all_months = {i: datetime.date(2020, i, 1).strftime("%B") for i in range(1, 13)}
    all_days = list(range(1, 32))

    st.write("Select a date range for Dispatch and GHG chart")

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
        end_ts = pd.Timestamp(end_d, tz=user_timezone).tz_convert(None) + pd.Timedelta(
            hours=23, minutes=59
        )

        # Dispatch plot
        filtered_df = df_plot.loc[start_ts:end_ts]
        fig = px.line(
            filtered_df,
            title=f"Dispatch Profile: {start_d.strftime('%b %d')} to {end_d.strftime('%b %d')} ({user_timezone})",
        )
        fig.update_layout(xaxis_title="Time", yaxis_title="Power (MW)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

                # 🌍 GHG Emissions Analysis (Grid Only)
        st.header("🌍 GHG Emissions Analysis (Grid Only)")

        # Use the user-selected grid emission factor from the sidebar (gCO2e/kWh)
        ef_grid = grid_emission_factor

        # Hourly grid import power (MW)
        grid_power_MW = gen["grid"]

        # Hourly GHG emissions from grid only (tCO2e per hour)
        # gen["grid"] is MW = MWh/h → multiply by 1000 to get kWh,
        # then by ef_grid (gCO2e/kWh), then convert g → t (1e6 g per tonne)
        ghg_grid_hourly_t = grid_power_MW * ef_grid * 1000.0 / 1e6

        # Annual GHG emissions from grid (project case)
        total_annual_ghg_grid_t = ghg_grid_hourly_t.sum()

        # Baseline GHG emissions: all load supplied by the grid
        baseline_ghg_t = total_load_annual_MWh * ef_grid * 1000.0 / 1e6

        # GHG savings compared to baseline
        ghg_savings_t = baseline_ghg_t - total_annual_ghg_grid_t

        # KPIs
        col_ghg1, col_ghg2 = st.columns(2)
        col_ghg1.metric(
            "**Total Annual GHG Emissions (tCO₂e/yr)**",
            f"{total_annual_ghg_grid_t:,.0f}",
        )
        col_ghg2.metric(
            "**GHG Savings (tCO₂e/yr)**",
            f"{ghg_savings_t:,.0f}",
        )

        # Plot hourly grid GHG emissions for the selected date range
        filtered_ghg_grid_ts = ghg_grid_hourly_t.loc[start_ts:end_ts]
        ghg_df = filtered_ghg_grid_ts.reset_index()
        ghg_df.columns = ["timestamp", "tCO2e_per_hour"]

        fig_ghg = px.bar(
            ghg_df,
            x="timestamp",
            y="tCO2e_per_hour",
            title=(
                "Hourly GHG Emissions from Grid: "
                f"{start_d.strftime('%b %d')} to {end_d.strftime('%b %d')}"
            ),
        )
        fig_ghg.update_layout(
            xaxis_title="Time (hourly)",
            yaxis_title="tCO₂e per hour",
        )
        st.plotly_chart(fig_ghg, use_container_width=True)

        st.markdown("---")

        # 📍 Grid Topology (Optimized network)
        st.header("📍 Grid Topology (Optimized Network)")
        st.markdown("Visual representation of the optimized user-defined mesh grid.")

        bus_df = network.buses.copy()
        fig_map = px.scatter_mapbox(
            bus_df,
            lat="y",
            lon="x",
            hover_name=bus_df.index,
            zoom=map_zoom,
            height=500,
            center={"lat": center_lat, "lon": center_lon},
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
                    line=dict(width=2, color="blue"),
                    name=line_name,
                    showlegend=False,
                )
            )
        st.plotly_chart(fig_map, use_container_width=True)

else:
    st.info(
        "👈 Please complete **Step 1** (Case-Study data & Location), **Step 2** (Input Parameters), "
        "and **Step 3** (Map View) in the sidebar, then click the 'Run Capacity Optimization' button above to "
        "find the optimal system size and view the results."
    )
