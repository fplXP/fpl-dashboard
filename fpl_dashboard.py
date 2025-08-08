import streamlit as st
import pandas as pd
import numpy as np
from typing import List

st.set_page_config(page_title="FPL Prediction Dashboard", layout="wide")

st.title("⚽ FPL Prediction Dashboard — Multi‑Week Expected Points (Poisson goals/assists, 2025 rules)")
st.write(
    "Now using **Poisson mode** for scoring contributions: you enter expected counts μ for goals and assists per GW. "
    "Removed Own Goals and Pen Miss (negligible impact). Still supports action thresholds: DEF ≥10 (CBI+T) = +2; MID/FWD ≥12 (CBI+T+Recoveries) = +2."
)

# ---------------- Constants ----------------
WEEKS = 5
POSITION_OPTIONS = ["FWD", "MID", "DEF", "GK"]
GOAL_POINTS = {"FWD": 4, "MID": 5, "DEF": 6, "GK": 6}
CS_POINTS = {"FWD": 0, "MID": 1, "DEF": 4, "GK": 4}
ASSIST_POINTS = 3
APPEAR_90_POINTS = 2
YELLOW_POINTS = -1
DEF_ACTION_THRESHOLD_POINTS = 2     # DEF reaches ≥10 (CBI + Tackles)
ATT_ACTION_THRESHOLD_POINTS = 2     # MID/FWD reaches ≥12 (CBI + Tackles + Recoveries)

BASE_COLS = ["Player", "Team", "Position", "Price"]
# Events per GW (Poisson inputs for goals/assists)
EVENTS = [
    ("Goals μ", "Expected goals (xG) for the match — use 0.00–1.50+"),
    ("Assists μ", "Expected assists (xA) for the match — use 0.00–1.00+"),
    ("CleanSheet%", "Clean sheet chance"),
    ("Play90%", "Chance of playing 90 minutes"),
    ("YellowCard%", "Yellow card chance"),
    ("CBI+T ≥10 % (DEF)", "Defenders only"),
    ("CBI+T+Rec ≥12 % (MID/FWD)", "Midfielders/Forwards only"),
]

# Build required columns for weeks 1..WEEKS
REQUIRED_COLS: List[str] = BASE_COLS.copy()
for w in range(1, WEEKS + 1):
    for e, _ in EVENTS:
        REQUIRED_COLS.append(f"{e}_GW{w}")

# ---------------- Session State Init ----------------

def _default_row(player: str, team: str, pos: str, price: float) -> dict:
    row = {"Player": player, "Team": team, "Position": pos, "Price": price}
    for w in range(1, WEEKS + 1):
        row.update({
            f"Goals μ_GW{w}": 0.0, f"Assists μ_GW{w}": 0.0,
            f"CleanSheet%_GW{w}": 0, f"Play90%_GW{w}": 0, f"YellowCard%_GW{w}": 0,
            f"CBI+T ≥10 % (DEF)_GW{w}": 0, f"CBI+T+Rec ≥12 % (MID/FWD)_GW{w}": 0,
        })
    return row

if "players" not in st.session_state:
    st.session_state.players = pd.DataFrame([
        _default_row("Saka", "Arsenal", "MID", 9.0),
        _default_row("Watkins", "Aston Villa", "FWD", 9.0),
        _default_row("Saliba", "Arsenal", "DEF", 6.0),
    ])[REQUIRED_COLS]

# Ensure required columns exist (and drop legacy columns if any)
for col in REQUIRED_COLS:
    if col not in st.session_state.players.columns:
        st.session_state.players[col] = 0
legacy_cols = [c for c in st.session_state.players.columns if c.endswith("OwnGoal%") or c.endswith("PenMiss%")]
if legacy_cols:
    st.session_state.players = st.session_state.players.drop(columns=legacy_cols, errors="ignore")

# Reorder
st.session_state.players = st.session_state.players[REQUIRED_COLS]

# ---------------- Sidebar: Data Management ----------------
st.sidebar.header("Data management")

# CSV Template & Downloads
if st.sidebar.button("Create CSV template"):
    template_df = pd.DataFrame(columns=REQUIRED_COLS)
    st.sidebar.download_button(
        label="Download template.csv",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="fpl_template_multiweek_poisson.csv",
        mime="text/csv",
    )

uploaded = st.sidebar.file_uploader("Upload players CSV", type=["csv"], help="Must include required columns. Price in £m (e.g., 6.5)")
replace_mode = st.sidebar.radio("When uploading:", ["Replace all rows", "Append rows", "Update by Player name"], index=0)

if uploaded is not None:
    try:
        up = pd.read_csv(uploaded)
        missing = [c for c in REQUIRED_COLS if c not in up.columns]
        if missing:
            st.sidebar.error(f"CSV missing required columns: {missing}")
        else:
            up = up[REQUIRED_COLS]
            if replace_mode == "Replace all rows":
                st.session_state.players = up.copy()
                st.sidebar.success("Replaced all rows with your CSV.")
            elif replace_mode == "Append rows":
                st.session_state.players = pd.concat([st.session_state.players, up], ignore_index=True)
                st.sidebar.success("Appended uploaded rows.")
            else:  # Update by Player name
                base = st.session_state.players.set_index("Player")
                inc = up.set_index("Player")
                base.update(inc)
                st.session_state.players = base.reset_index()
                st.sidebar.success("Updated rows where Player matched.")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")

# Official FPL import (players + prices)
st.sidebar.subheader("Official FPL import")
if st.sidebar.button("Import from official FPL (players + prices)"):
    import requests
    try:
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        elements = pd.DataFrame(data.get("elements", []))
        teams = pd.DataFrame(data.get("teams", []))

        # Maps
        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        team_map = teams.set_index("id")["name"].to_dict()

        df = pd.DataFrame({
            "Player": elements["web_name"],
            "Team": elements["team"].map(team_map),
            "Position": elements["element_type"].map(pos_map),
            "Price": elements["now_cost"].astype(float) / 10.0,
        })
        # Fill week columns with zeros
        for w in range(1, WEEKS + 1):
            for e, _ in EVENTS:
                df[f"{e}_GW{w}"] = 0
        df = df[REQUIRED_COLS]

        if replace_mode == "Replace all rows":
            st.session_state.players = df.copy()
        elif replace_mode == "Append rows":
            st.session_state.players = pd.concat([st.session_state.players, df], ignore_index=True)
        else:  # Update by Player name
            base = st.session_state.players.set_index("Player")
            inc = df.set_index("Player")
            base.update(inc)
            st.session_state.players = base.reset_index()

        st.sidebar.success("Imported players and prices from FPL.")
    except Exception as e:
        st.sidebar.error(f"FPL import failed: {e}")

# Add single player via form
st.sidebar.subheader("Add a player")
with st.sidebar.form("add_player_form", clear_on_submit=True):
    p_name = st.text_input("Name", "")
    p_team = st.text_input("Team", "")
    p_pos = st.selectbox("Position", POSITION_OPTIONS, index=1)
    p_price = st.number_input("Price (£m)", 3.5, 15.0, 6.0, 0.5)
    submitted = st.form_submit_button("Add Player")

if submitted and p_name.strip():
    new_row = _default_row(p_name.strip(), p_team.strip(), p_pos, float(p_price))
    st.session_state.players = pd.concat([st.session_state.players, pd.DataFrame([new_row])], ignore_index=True)

# ---------------- Editing: Week Tabs ----------------
st.markdown("### Enter/edit per‑week inputs (Goals μ, Assists μ in expected counts; others as %) ")
tabs = st.tabs([f"GW{w}" for w in range(1, WEEKS + 1)])

for idx, tab in enumerate(tabs, start=1):
    with tab:
        st.caption(
            "Use **Goals μ**/**Assists μ** as expected counts (e.g., 0.55). Clean Sheet/Play90/Yellow/Action thresholds are in % (0–100).\n"
            "Poisson mode implies E[goal points] = μ_goals × position goal points; E[assist points] = μ_assists × 3."
        )
        # Prepare view for this week
        week_cols = BASE_COLS + [f"{e}_GW{idx}" for e, _ in EVENTS]
        # Column config per week
        colcfg = {
            "Team": st.column_config.TextColumn(width="medium"),
            "Position": st.column_config.SelectboxColumn(options=POSITION_OPTIONS, width="small"),
            "Price": st.column_config.NumberColumn(format="£%.1f", step=0.1, help="FPL price in £m"),
            f"Goals μ_GW{idx}": st.column_config.NumberColumn(format="%.2f", step=0.05, help="Expected goals (xG) this GW"),
            f"Assists μ_GW{idx}": st.column_config.NumberColumn(format="%.2f", step=0.05, help="Expected assists (xA) this GW"),
            f"CleanSheet%_GW{idx}": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
            f"Play90%_GW{idx}": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
            f"YellowCard%_GW{idx}": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
            f"CBI+T ≥10 % (DEF)_GW{idx}": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
            f"CBI+T+Rec ≥12 % (MID/FWD)_GW{idx}": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
        }

        edited = st.data_editor(
            st.session_state.players[week_cols],
            num_rows="dynamic",
            use_container_width=True,
            column_config=colcfg,
            hide_index=True,
            key=f"editor_gw{idx}",
        )
        # Write changes back
        st.session_state.players[week_cols] = edited[week_cols]

# ---------------- Filters & Controls ----------------
st.markdown("### Filters & Views")
colf = st.columns([1.2, 1, 1, 1, 1])
all_teams = sorted([t for t in st.session_state.players["Team"].dropna().unique() if str(t).strip() != ""])
with colf[0]:
    team_filter = st.multiselect("Team(s)", options=all_teams, default=all_teams)
with colf[1]:
    pos_filter = st.multiselect("Position(s)", options=POSITION_OPTIONS, default=POSITION_OPTIONS)
with colf[2]:
    price_min = st.number_input("Min price", 3.5, 15.0, 3.5, 0.5)
with colf[3]:
    price_max = st.number_input("Max price", 3.5, 15.0, 15.0, 0.5)
with colf[4]:
    search_text = st.text_input("Search name/team", "").strip().lower()

week_select = st.multiselect("Weeks to total", [f"GW{w}" for w in range(1, WEEKS + 1)], default=[f"GW{w}" for w in range(1, WEEKS + 1)])
show_top_n = st.number_input("Show top N (by total over selected weeks)", 1, 200, 50, 1)
min_points = st.number_input("Min expected pts (per selected total)", 0.0, 100.0, 0.0, 0.5)

# ---------------- Calculations ----------------

def to_prob(x):
    try:
        return float(x) / 100.0
    except Exception:
        return 0.0


def expected_points_for_week(row: pd.Series, week: int) -> float:
    pos = row["Position"]
    goal_pts = GOAL_POINTS.get(pos, 0)
    cs_pts = CS_POINTS.get(pos, 0)

    mu_g = float(row.get(f"Goals μ_GW{week}", 0) or 0)
    mu_a = float(row.get(f"Assists μ_GW{week}", 0) or 0)

    p_cs = to_prob(row.get(f"CleanSheet%_GW{week}", 0))
    p_90 = to_prob(row.get(f"Play90%_GW{week}", 0))
    p_yel = to_prob(row.get(f"YellowCard%_GW{week}", 0))

    p_def_actions = to_prob(row.get(f"CBI+T ≥10 % (DEF)_GW{week}", 0))
    p_att_actions = to_prob(row.get(f"CBI+T+Rec ≥12 % (MID/FWD)_GW{week}", 0))

    # Poisson expectation: E[goal points] = μ_goals * points; E[assist points] = μ_assists * 3
    exp = (
        mu_g * goal_pts +
        mu_a * ASSIST_POINTS +
        p_cs * cs_pts +
        p_90 * APPEAR_90_POINTS +
        p_yel * YELLOW_POINTS
    )

    if pos == "DEF":
        exp += p_def_actions * DEF_ACTION_THRESHOLD_POINTS
    elif pos in ("MID", "FWD"):
        exp += p_att_actions * ATT_ACTION_THRESHOLD_POINTS
    return float(np.round(exp, 2))

players = st.session_state.players.copy()
for w in range(1, WEEKS + 1):
    players[f"EP_GW{w}"] = players.apply(lambda r, ww=w: expected_points_for_week(r, ww), axis=1)

# Total over selected weeks
selected_weeks_idx = [int(w.replace("GW", "")) for w in week_select]
players["EP_Total"] = players[[f"EP_GW{i}" for i in selected_weeks_idx]].sum(axis=1).round(2)

# Apply filters
mask = (
    players["Team"].isin(team_filter) &
    players["Position"].isin(pos_filter) &
    (players["Price"].fillna(0) >= price_min) & (players["Price"].fillna(0) <= price_max)
)
if search_text:
    mask &= (
        players["Player"].str.lower().str.contains(search_text, na=False) |
        players["Team"].str.lower().str.contains(search_text, na=False)
    )

filtered = players[mask]
filtered = filtered[filtered["EP_Total"] >= min_points]
filtered_sorted = filtered.sort_values(["EP_Total", "Player"], ascending=[False, True], ignore_index=True)

# ---------------- Views ----------------
st.markdown("### Top players by total expected points (selected weeks)")
st.dataframe(
    filtered_sorted.head(int(show_top_n))[["Player", "Team", "Position", "Price"] + [f"EP_GW{i}" for i in range(1, WEEKS + 1)] + ["EP_Total"]],
    use_container_width=True,
)

with st.expander("See all columns (including per‑week inputs)"):
    st.dataframe(filtered_sorted, use_container_width=True)

# ---------------- Downloads ----------------
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button("Download ALL PLAYERS (current table)", data=players.to_csv(index=False).encode("utf-8"), file_name="fpl_players_all_poisson.csv", mime="text/csv")
with col_dl2:
    st.download_button("Download FILTERED (top view)", data=filtered_sorted.head(int(show_top_n)).to_csv(index=False).encode("utf-8"), file_name="fpl_players_filtered_poisson.csv", mime="text/csv")

st.caption(
    "Scoring: Goals (FWD/MID/DEF/GK): 4/5/6/6, Assist: 3, Clean sheet (FWD/MID/DEF/GK): 0/1/4/4, 90 minutes: 2; "
    "Yellow: -1. **Poisson mode:** E[goal points] = μ_goals × goal points; E[assist points] = μ_assists × 3. "
    "**2025 action bonuses:** DEF ≥10 CBI+T: +2; MID/FWD ≥12 CBI+T+Recoveries: +2."
)

