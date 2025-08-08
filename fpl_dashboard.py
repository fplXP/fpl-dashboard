import streamlit as st
import pandas as pd
import numpy as np
from typing import List
from itertools import combinations

st.set_page_config(page_title="FPL Prediction Dashboard", layout="wide")

st.title("⚽ FPL Prediction Dashboard — Multi-Week Expected Points (Poisson goals/assists, 2025 rules, bonus points)")

# ---------------------- CONFIG ----------------------
WEEKS = 5
POSITION_OPTIONS = ["FWD", "MID", "DEF", "GK"]
GOAL_POINTS = {"FWD": 4, "MID": 5, "DEF": 6, "GK": 6}
CS_POINTS = {"FWD": 0, "MID": 1, "DEF": 4, "GK": 4}
ASSIST_POINTS = 3
APPEAR_90_POINTS = 2
YELLOW_POINTS = -1
DEF_ACTION_THRESHOLD_POINTS = 2  # DEF ≥10 (CBI+T)
ATT_ACTION_THRESHOLD_POINTS = 2  # MID/FWD ≥12 (CBI+T+Rec)

BASE_COLS = ["Player", "Team", "Position", "Price"]
EVENTS = [
    ("Goals μ", "Expected goals (xG)"),
    ("Assists μ", "Expected assists (xA)"),
    ("CleanSheet%", "Clean sheet chance"),
    ("Play90%", "Chance of playing 90 minutes"),
    ("YellowCard%", "Yellow card chance"),
    ("CBI+T ≥10 % (DEF)", "Defenders only"),
    ("CBI+T+Rec ≥12 % (MID/FWD)", "Midfielders/Forwards only"),
    ("BonusPts", "Expected bonus points"),
]

REQUIRED_COLS: List[str] = BASE_COLS.copy()
for w in range(1, WEEKS + 1):
    for e, _ in EVENTS:
        REQUIRED_COLS.append(f"{e}_GW{w}")

# ---------------------- HELPERS ----------------------

def _default_row(player: str, team: str, pos: str, price: float) -> dict:
    row = {"Player": player, "Team": team, "Position": pos, "Price": price}
    for w in range(1, WEEKS + 1):
        row.update({
            f"Goals μ_GW{w}": 0.0, f"Assists μ_GW{w}": 0.0,
            f"CleanSheet%_GW{w}": 0, f"Play90%_GW{w}": 0, f"YellowCard%_GW{w}": 0,
            f"CBI+T ≥10 % (DEF)_GW{w}": 0, f"CBI+T+Rec ≥12 % (MID/FWD)_GW{w}": 0,
            f"BonusPts_GW{w}": 0.0,
        })
    return row


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
    bonus_pts = float(row.get(f"BonusPts_GW{week}", 0) or 0)

    exp = (
        mu_g * goal_pts +
        mu_a * ASSIST_POINTS +
        p_cs * cs_pts +
        p_90 * APPEAR_90_POINTS +
        p_yel * YELLOW_POINTS +
        bonus_pts
    )
    if pos == "DEF":
        exp += p_def_actions * DEF_ACTION_THRESHOLD_POINTS
    elif pos in ("MID", "FWD"):
        exp += p_att_actions * ATT_ACTION_THRESHOLD_POINTS
    return float(np.round(exp, 2))

# ---------------------- SESSION INIT ----------------------
if "players" not in st.session_state:
    st.session_state.players = pd.DataFrame([
        _default_row("Saka", "Arsenal", "MID", 9.0),
        _default_row("Watkins", "Aston Villa", "FWD", 9.0),
        _default_row("Saliba", "Arsenal", "DEF", 6.0),
    ])[REQUIRED_COLS]

# Ensure all required columns exist
for col in REQUIRED_COLS:
    if col not in st.session_state.players.columns:
        st.session_state.players[col] = 0
st.session_state.players = st.session_state.players[REQUIRED_COLS]

# ---------------------- SIDEBAR: DATA MGMT ----------------------
st.sidebar.header("Data management")

# Template download
if st.sidebar.button("Create CSV template"):
    template_df = pd.DataFrame(columns=REQUIRED_COLS)
    st.sidebar.download_button(
        label="Download template.csv",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="fpl_template_multiweek_poisson_bonus.csv",
        mime="text/csv",
    )

# CSV upload
uploaded = st.sidebar.file_uploader("Upload players CSV", type=["csv"])
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
            elif replace_mode == "Append rows":
                st.session_state.players = pd.concat([st.session_state.players, up], ignore_index=True)
            else:  # update by Player
                base = st.session_state.players.set_index("Player")
                inc = up.set_index("Player")
                base.update(inc)
                st.session_state.players = base.reset_index()
            st.sidebar.success("CSV processed.")
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
        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        team_map = teams.set_index("id")["name"].to_dict()
        df = pd.DataFrame({
            "Player": elements["web_name"],
            "Team": elements["team"].map(team_map),
            "Position": elements["element_type"].map(pos_map),
            "Price": elements["now_cost"].astype(float) / 10.0,
        })
        for w in range(1, WEEKS + 1):
            for e, _ in EVENTS:
                df[f"{e}_GW{w}"] = 0
        df = df[REQUIRED_COLS]
        if replace_mode == "Replace all rows":
            st.session_state.players = df.copy()
        elif replace_mode == "Append rows":
            st.session_state.players = pd.concat([st.session_state.players, df], ignore_index=True)
        else:
            base = st.session_state.players.set_index("Player")
            inc = df.set_index("Player")
            base.update(inc)
            st.session_state.players = base.reset_index()
        st.sidebar.success("Imported players and prices from FPL.")
    except Exception as e:
        st.sidebar.error(f"FPL import failed: {e}")

# Add single player form
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

# ---------------------- INPUT EDITOR ----------------------
st.markdown("### Enter/edit per-week inputs")
tabs = st.tabs([f"GW{w}" for w in range(1, WEEKS + 1)])
for idx, tab in enumerate(tabs, start=1):
    with tab:
        week_cols = BASE_COLS + [f"{e}_GW{idx}" for e, _ in EVENTS]
        colcfg = {
            "Team": st.column_config.TextColumn(width="medium"),
            "Position": st.column_config.SelectboxColumn(options=POSITION_OPTIONS, width="small"),
            "Price": st.column_config.NumberColumn(format="£%.1f", step=0.1),
        }
        edited = st.data_editor(
            st.session_state.players[week_cols],
            num_rows="dynamic",
            use_container_width=True,
            column_config=colcfg,
            hide_index=True,
            key=f"editor_gw{idx}",
        )
        st.session_state.players[week_cols] = edited[week_cols]

# ---------------------- CALCULATIONS ----------------------
players = st.session_state.players.copy()
for w in range(1, WEEKS + 1):
    players[f"EP_GW{w}"] = players.apply(lambda r, ww=w: expected_points_for_week(r, ww), axis=1)

# Controls
colf = st.columns([1.2, 1, 1, 1, 1])
all_teams = sorted([t for t in players["Team"].dropna().unique() if str(t).strip() != ""])
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
show_top_n = st.number_input("Show top N", 1, 200, 50, 1)
min_points = st.number_input("Min expected pts (total)", 0.0, 200.0, 0.0, 0.5)
hide_zero = st.checkbox("Hide players with 0 predicted points in selected week(s)", value=True)

selected_weeks_idx = [int(w.replace("GW", "")) for w in week_select]
players["EP_Total"] = players[[f"EP_GW{i}" for i in selected_weeks_idx]].sum(axis=1).round(2)

# Pts per £m for the total (also usable for one-week views if only one selected)
players["PtsPerMil"] = (players["EP_Total"] / players["Price"].replace(0, np.nan)).fillna(0).round(3)

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

filtered = players[mask].copy()
if hide_zero:
    # hide if all selected GWs are 0
    if selected_weeks_idx:
        zero_mask = filtered[[f"EP_GW{i}" for i in selected_weeks_idx]].sum(axis=1) == 0
        filtered = filtered[~zero_mask]

filtered_sorted = filtered.sort_values(["EP_Total", "Player"], ascending=[False, True], ignore_index=True)

st.markdown("### Predicted points (filtered)")
st.dataframe(
    filtered_sorted.head(int(show_top_n))["Player Team Position Price ".split() + [f"EP_GW{i}" for i in range(1, WEEKS + 1)] + ["EP_Total", "PtsPerMil"]],
    use_container_width=True,
)

with st.expander("See all columns"):
    st.dataframe(filtered_sorted, use_container_width=True)

# Downloads
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button("Download ALL", data=players.to_csv(index=False).encode("utf-8"), file_name="fpl_players_all.csv", mime="text/csv")
with col_dl2:
    st.download_button("Download FILTERED", data=filtered_sorted.head(int(show_top_n)).to_csv(index=False).encode("utf-8"), file_name="fpl_players_filtered.csv", mime="text/csv")

# ---------------------- BEST XI TOOL ----------------------
st.markdown("---")
st.header("🧠 Best XI under budget (single GW)")
budget = st.number_input("Max Budget (£m)", 60.0, 120.0, 100.0, 0.5)
selected_gw = st.selectbox("Gameweek to optimise", [f"GW{w}" for w in range(1, WEEKS + 1)], index=0)

# Use EP for the chosen week
week_idx = int(selected_gw.replace("GW", ""))
players_for_xi = players.copy()
players_for_xi["EP_THIS_GW"] = players_for_xi[f"EP_GW{week_idx}"]

# Use only players with EP > 0 if desired
use_hide_zero_for_xi = st.checkbox("Exclude zero-point players from Best XI search", value=True)
if use_hide_zero_for_xi:
    players_for_xi = players_for_xi[players_for_xi["EP_THIS_GW"] > 0]

# Keep top N per position to keep the combinatorics manageable
TOP_GK, TOP_DEF, TOP_MID, TOP_FWD = 5, 12, 12, 8
pool = {
    "GK": players_for_xi[players_for_xi.Position == "GK"].nlargest(TOP_GK, "EP_THIS_GW"),
    "DEF": players_for_xi[players_for_xi.Position == "DEF"].nlargest(TOP_DEF, "EP_THIS_GW"),
    "MID": players_for_xi[players_for_xi.Position == "MID"].nlargest(TOP_MID, "EP_THIS_GW"),
    "FWD": players_for_xi[players_for_xi.Position == "FWD"].nlargest(TOP_FWD, "EP_THIS_GW"),
}

formations = [(3,4,3), (3,5,2), (4,4,2), (4,3,3), (5,3,2), (5,4,1)]

best_team = None
best_score = -1
best_cost = None
best_formation = None

for (d_cnt, m_cnt, f_cnt) in formations:
    # choose 1 GK, d_cnt DEF, m_cnt MID, f_cnt FWD
    for gk in pool["GK"].itertuples(index=False):
        for defs in combinations(pool["DEF"].itertuples(index=False), d_cnt):
            for mids in combinations(pool["MID"].itertuples(index=False), m_cnt):
                for fwds in combinations(pool["FWD"].itertuples(index=False), f_cnt):
                    team = [gk] + list(defs) + list(mids) + list(fwds)
                    total_price = sum(p.Price for p in team)
                    if total_price > budget:
                        continue
                    total_points = sum(getattr(p, "EP_THIS_GW") for p in team)
                    if total_points > best_score:
                        best_score = total_points
                        best_cost = total_price
                        best_team = team
                        best_formation = (d_cnt, m_cnt, f_cnt)

if best_team:
    st.subheader(f"🏆 Best XI for {selected_gw} under £{budget:.1f}m — Formation {best_formation[0]}-{best_formation[1]}-{best_formation[2]}")
    best_df = pd.DataFrame([
        {"Player": p.Player, "Team": p.Team, "Pos": p.Position, "Price": p.Price, f"PredPts ({selected_gw})": getattr(p, "EP_THIS_GW")}
        for p in best_team
    ])
    best_df = best_df.sort_values(["Pos", f"PredPts ({selected_gw})"], ascending=[True, False])
    st.dataframe(best_df, use_container_width=True)
    st.markdown(f"**Total Cost:** £{best_cost:.1f}m")
    st.markdown(f"**Total Predicted Points:** {best_score:.2f}")
    st.download_button(
        "Download Best XI (CSV)",
        data=best_df.to_csv(index=False).encode("utf-8"),
        file_name=f"best_xi_{selected_gw}_under_{budget:.1f}m.csv",
        mime="text/csv",
    )
else:
    st.warning("No valid XI found within budget. Try increasing budget or ensure players have non-zero EP for the selected week.")
