"""
Airline Loyalty — Member Retention Console
Prototype for marketing managers to identify high-value at-risk members
and decide what action to take.

Run:  python -m streamlit run app.py

Expects these CSVs in the same directory:
  - member_fvs.csv               (from phase3_forward_value_score.ipynb)
  - panel_cleaned.csv            (from phase2 notebook)
  - loyalty_cleaned.csv          (from phase2 notebook)
  - labeled_features_lean.csv    (from phase3 features script)
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ---------- Page config ----------
st.set_page_config(
    page_title="Member Retention Console",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Constants ----------
SEGMENT_NAMES = {
    0: "Heavy Redeemers",
    1: "Established Loyalists",
    2: "Disengaging At-Risk",
    3: "Growing Power Flyers",
    4: "Newer Members",
}

SEGMENT_ACTIONS = {
    0: {"label": "DEPRIORITIZE", "color": "#888888",
        "channel": "No proactive outreach — monitor only",
        "timing": "Monthly review",
        "offer": "None",
        "rationale": "Heavy program usage without commensurate flying. Retention spend has low expected return; the member's expected forward value is below cohort median despite mid-tier CLV.",
        "success_criterion": "Maintain low retention spend; no expected recovery."},
    1: {"label": "MAINTAIN", "color": "#4A8B47",
        "channel": "Standard email newsletter",
        "timing": "Monthly cadence",
        "offer": "None — relationship-maintenance only",
        "rationale": "Established, stable loyalty pattern. Low churn risk and moderate forward value. Standard communication preserves the relationship without consuming retention budget.",
        "success_criterion": "Engagement (open rate) sustained at baseline."},
    2: {"label": "URGENT INTERVENE", "color": "#B22222",
        "channel": "CSR phone call",
        "timing": "Within 5 business days",
        "offer": "Tier protection guarantee for 6 months + 2x bonus points on next 3 flights",
        "rationale": "Member is declining (negative flight trend) with extended recency. They will likely disengage in the coming months if no action is taken. Mid-tier historical value is recoverable if intervention is timely.",
        "success_criterion": "At least 1 flight in the next 60 days."},
    3: {"label": "PROTECT", "color": "#1F3A5F",
        "channel": "Personalized email from account manager",
        "timing": "Quarterly check-in",
        "offer": "Tier maintenance reassurance + exclusive lounge access voucher",
        "rationale": "Growing power flyer with strong upward trajectory and highest forward value in the cohort. Protect the relationship to preserve future revenue contribution.",
        "success_criterion": "Maintain or accelerate current flight cadence."},
    4: {"label": "NURTURE", "color": "#D4A017",
        "channel": "Lifecycle email series",
        "timing": "Monthly through the first 24 months of membership",
        "offer": "Bonus points on 3rd flight + intro to redemption catalog",
        "rationale": "Newer member building flight cadence. Strong forward value trajectory if engagement deepens. Targeted nurture helps convert them into Growing Power Flyers.",
        "success_criterion": "Complete first redemption + reach 6 flights in the next 6 months."},
}

ACTION_COLOR_MAP = {info["label"]: info["color"] for info in SEGMENT_ACTIONS.values()}

# ---------- Path config (edit this if running outside the repo root) ----------
DATA_DIR = "data/processed"

# Tier colors: Star = silvery blue, Nova = mid blue, Aurora = gold
TIER_COLORS = {
    "Star":   "#A0BBD9",
    "Nova":   "#2E5C8A",
    "Aurora": "#D4A017",
}

# ---------- Data loaders (cached) ----------
@st.cache_data
def load_member_fvs():
    df = pd.read_csv("member_fvs.csv")
    df["P_churn"] = df["lgbm_proba"]
    df["VaR"] = df["fvs"] * df["P_churn"]
    df["segment_name"] = df["segment"].map(SEGMENT_NAMES)
    df["action_label"] = df["segment"].map(lambda s: SEGMENT_ACTIONS[s]["label"])
    return df

@st.cache_data
def load_loyalty():
    return pd.read_csv("loyalty_cleaned.csv", parse_dates=["enroll_date", "cancel_date"])

@st.cache_data
def load_panel():
    return pd.read_csv("panel_cleaned.csv", parse_dates=["flight_date", "enroll_date", "cancel_date"])

@st.cache_data
def load_features():
    return pd.read_csv("labeled_features_lean.csv", parse_dates=["as_of_date"])

try:
    fvs = load_member_fvs()
    loyalty = load_loyalty()
    panel = load_panel()
    features = load_features()
    test_features = features[features["as_of_date"] == pd.Timestamp("2018-07-01")].copy()
except FileNotFoundError as e:
    st.error(f"Could not load required data file: {e}")
    st.info("Make sure these files are in the same directory as app.py: "
            "member_fvs.csv, loyalty_cleaned.csv, panel_cleaned.csv, labeled_features_lean.csv")
    st.stop()


# ---------- Helpers ----------
def color_action(val):
    """Return CSS style string for an Action cell based on its value."""
    color = ACTION_COLOR_MAP.get(val, "#000")
    return f"color: {color}; font-weight: 600;"


def compute_flags(feat, test_features):
    """Compute diagnostic feature flags for the 'why flagged' panel."""
    flags = []

    rec = feat["rec_months_since_flight"]
    if rec >= 4:
        flags.append(("🔴", f"Has not flown in {rec:.1f} months",
                      "Recency is the strongest churn predictor in the model."))
    elif rec >= 2:
        flags.append(("🟡", f"Last flight was {rec:.1f} months ago",
                      "Moderate recency — within normal range but worth watching."))
    else:
        flags.append(("🟢", f"Flew very recently ({rec:.1f} months ago)",
                      "Active recency reduces churn signal."))

    ma = feat["freq_months_active_6mo"]
    if ma <= 2:
        flags.append(("🔴", f"Active only {int(ma)} of last 6 months",
                      "Low engagement consistency strongly correlates with churn."))
    elif ma <= 4:
        flags.append(("🟡", f"Active in {int(ma)} of last 6 months",
                      "Moderate engagement consistency."))
    else:
        flags.append(("🟢", f"Active in {int(ma)} of last 6 months",
                      "Consistent engagement is a strong retention signal."))

    tr = feat["trend_flights_diff"]
    if tr <= -3:
        flags.append(("🔴", f"Flight activity declining sharply ({int(tr)} fewer flights in recent 3mo)",
                      "Negative trend is a leading indicator of disengagement."))
    elif tr >= 3:
        flags.append(("🟢", f"Flight activity accelerating (+{int(tr)} flights in recent 3mo)",
                      "Positive trajectory; retention investment pays off."))
    else:
        flags.append(("🟡", f"Flight activity broadly flat ({int(tr):+d} flight differential)",
                      "No strong trend signal."))

    clv_proxy = feat["mon_clv_asof_proxy"]
    clv_75 = test_features["mon_clv_asof_proxy"].quantile(0.75)
    clv_25 = test_features["mon_clv_asof_proxy"].quantile(0.25)
    if clv_proxy >= clv_75:
        flags.append(("🟢", f"High historical value (top quartile, {clv_proxy:,.0f} points accumulated)",
                      "Worth protecting if forward value also holds."))
    elif clv_proxy <= clv_25:
        flags.append(("🟡", f"Lower historical value (bottom quartile)",
                      "Lifecycle stage matters more than absolute scale for this member."))

    return flags


# ---------- Compact header ----------
st.markdown(
    """
    <div style='margin-top: -10px; margin-bottom: 10px;'>
        <span style='font-size: 24px; font-weight: 700;'>✈ Member Retention Console</span>
        <span style='color: #888; font-size: 13px; margin-left: 14px;'>
            Data as of July 2018 · Forward Value Score (FVS) projects 6-month forward member value
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🚨 At-Risk List", "🔍 Member Detail", "📊 Segment Overview"])

# ============ TAB 1: AT-RISK LIST ============
with tab1:
    filter_col, main_col = st.columns([1, 5], gap="medium")

    # ----- Filter sidebar (left column) -----
    with filter_col:
        st.markdown("##### Filters")
        seg_filter = st.multiselect(
            "Segments",
            options=sorted(fvs["segment"].unique()),
            default=sorted(fvs["segment"].unique()),
            format_func=lambda s: SEGMENT_NAMES[s],
            label_visibility="visible",
        )
        min_fvs = st.slider(
            "Minimum FVS ($)",
            min_value=0,
            max_value=int(fvs["fvs"].max()),
            value=1500,
            step=500,
        )
        min_pchurn = st.slider(
            "Minimum P(churn)",
            min_value=0.0,
            max_value=float(fvs["P_churn"].max()),
            value=0.005,
            step=0.005,
            format="%.3f",
        )

    # ----- Main area (right column) -----
    with main_col:
        # Apply filters
        filtered = fvs[
            fvs["segment"].isin(seg_filter)
            & (fvs["fvs"] >= min_fvs)
            & (fvs["P_churn"] >= min_pchurn)
        ].sort_values("VaR", ascending=False)

        # Top row: Show-Top dropdown + 3 metric tiles
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            top_n = st.selectbox("Show top", [10, 25, 50, 100, 250, 500], index=2)
        displayed = filtered.head(int(top_n))
        with c2:
            st.metric("Total Value at Risk", f"${displayed['VaR'].sum():,.0f}")
        with c3:
            st.metric("Median FVS", f"${displayed['fvs'].median():,.0f}" if len(displayed) else "—")
        with c4:
            st.metric("Median P(churn)", f"{displayed['P_churn'].median()*100:.1f}%" if len(displayed) else "—")

        if len(displayed) == 0:
            st.warning("No members match the current filters. Loosen the thresholds in the left panel.")
        else:
            # Build 5-column display table
            display = displayed[["Loyalty Number", "segment_name", "VaR", "action_label"]].copy()
            display.columns = ["Loyalty Number", "Segment", "VaR ($)", "Action"]
            display.insert(0, "Rank", range(1, len(display) + 1))
            display["VaR ($)"] = display["VaR ($)"].apply(lambda v: f"${v:,.0f}")
            styled = display.style.map(color_action, subset=["Action"])

            # ----- Read prior selection from session state (set by table widget below) -----
            sel_state = st.session_state.get("atrisk_table")
            selected_rows = []
            if sel_state is not None:
                if hasattr(sel_state, "selection"):
                    rows = getattr(sel_state.selection, "rows", None)
                    selected_rows = list(rows) if rows else []
                elif isinstance(sel_state, dict):
                    selected_rows = sel_state.get("selection", {}).get("rows", []) or []

            # ----- Render selected-member detail ABOVE the table -----
            if selected_rows:
                sel_idx = selected_rows[0]
                if sel_idx < len(displayed):
                    sel_loyalty_no = displayed.iloc[sel_idx]["Loyalty Number"]
                    m = fvs[fvs["Loyalty Number"] == sel_loyalty_no].iloc[0]
                    feat_row = test_features[test_features["Loyalty Number"] == sel_loyalty_no].iloc[0]
                    loy_row = loyalty[loyalty["Loyalty Number"] == sel_loyalty_no].iloc[0]

                    tier = loy_row['Loyalty Card']
                    tier_color = TIER_COLORS.get(tier, "#000")

                    st.markdown(
                        f"""
                        <div style='border: 1px solid #444; border-radius: 8px; padding: 14px;
                                    background: rgba(255,255,255,0.03); margin-bottom: 12px;'>
                            <div style='font-size: 13px; color: #888; margin-bottom: 6px;'>
                                Selected member · Loyalty Number <b style='color: #ddd;'>{int(sel_loyalty_no)}</b>
                                · open the Member Detail tab for full breakdown
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    d1, d2, d3, d4, d5, d6 = st.columns(6)
                    with d1:
                        st.metric("FVS", f"${m['fvs']:,.0f}")
                    with d2:
                        st.metric("P(churn)", f"{m['P_churn']*100:.1f}%")
                    with d3:
                        st.metric("Historical CLV", f"${m['CLV']:,.0f}")
                    with d4:
                        st.markdown(
                            f"<div style='font-size: 13px; color: rgb(160,160,160);'>Tier</div>"
                            f"<div style='font-size: 24px; font-weight: 600; color: {tier_color};'>{tier}</div>",
                            unsafe_allow_html=True,
                        )
                    with d5:
                        st.metric("Tenure", f"{feat_row['demo_tenure_months']:.0f} mo")
                    with d6:
                        st.metric("Province", loy_row['Province'])

                    st.markdown("")  # small spacer
            else:
                st.caption("ℹ Click any row in the table below to see a quick detail card "
                           "(it will appear here above the table).")

            # ----- Render the table BELOW the detail card -----
            st.dataframe(
                styled,
                key="atrisk_table",
                on_select="rerun",
                selection_mode="single-row",
                hide_index=True,
                use_container_width=True,
                height=520,
            )

# ============ TAB 2: MEMBER DETAIL ============
with tab2:
    # Narrow dropdown
    dropdown_col, _ = st.columns([1, 3])
    with dropdown_col:
        member_id = st.selectbox(
            "Select a Loyalty Number",
            options=sorted(fvs["Loyalty Number"].unique()),
            index=0,
        )

    if member_id:
        m = fvs[fvs["Loyalty Number"] == member_id].iloc[0]
        feat = test_features[test_features["Loyalty Number"] == member_id]
        if len(feat) == 0:
            st.warning("This member was not in the test cohort.")
        else:
            feat = feat.iloc[0]
            loy_row = loyalty[loyalty["Loyalty Number"] == member_id].iloc[0]
            seg = int(m["segment"])
            action = SEGMENT_ACTIONS[seg]
            tier = loy_row['Loyalty Card']
            tier_color = TIER_COLORS.get(tier, "#000")

            # 2-column split
            left_col, right_col = st.columns([1, 1], gap="large")

            with left_col:
                st.markdown(
                    f"""
                    <div style='border: 1px solid #444; border-radius: 8px; padding: 18px;
                                background: rgba(255,255,255,0.02); line-height: 1.9;'>
                        <table style='width: 100%; font-size: 14px;'>
                            <tr>
                                <td style='color: #888; padding: 6px 0;'>Loyalty Number</td>
                                <td style='text-align: right; font-weight: 600;'>{int(member_id)}</td>
                            </tr>
                            <tr>
                                <td style='color: #888; padding: 6px 0;'>Tier</td>
                                <td style='text-align: right; font-weight: 700; color: {tier_color};'>{tier}</td>
                            </tr>
                            <tr>
                                <td style='color: #888; padding: 6px 0;'>Segment</td>
                                <td style='text-align: right; font-weight: 600;'>{SEGMENT_NAMES[seg]}</td>
                            </tr>
                            <tr>
                                <td style='color: #888; padding: 6px 0;'>Tenure</td>
                                <td style='text-align: right; font-weight: 600;'>{feat['demo_tenure_months']:.0f} months</td>
                            </tr>
                            <tr>
                                <td style='color: #888; padding: 6px 0;'>Province</td>
                                <td style='text-align: right; font-weight: 600;'>{loy_row['Province']}</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right_col:
                # Top: Recommended action
                st.markdown(
                    f"""
                    <div style='border: 1px solid #444; border-radius: 8px; padding: 16px;
                                background: rgba(255,255,255,0.02); margin-bottom: 12px;'>
                        <div style='color: #888; font-size: 13px;'>Recommended action</div>
                        <div style='color: {action['color']}; font-size: 22px; font-weight: 700; margin: 4px 0 12px 0;'>
                            {action['label']}
                        </div>
                        <div style='font-size: 13px; line-height: 1.8;'>
                            <b>Channel:</b> {action['channel']}<br>
                            <b>Timing:</b> {action['timing']}<br>
                            <b>Offer:</b> {action['offer']}<br>
                            <b>Success criterion:</b> {action['success_criterion']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Bottom: Why flagged
                flags = compute_flags(feat, test_features)
                flag_rows_html = ""
                for icon, headline, detail in flags:
                    flag_rows_html += (
                        f"<div style='margin-bottom: 8px;'>"
                        f"<span style='font-size: 14px;'>{icon} <b>{headline}</b></span><br>"
                        f"<span style='color: #888; font-size: 12px; margin-left: 22px;'>{detail}</span>"
                        f"</div>"
                    )
                st.markdown(
                    f"""
                    <div style='border: 1px solid #444; border-radius: 8px; padding: 16px;
                                background: rgba(255,255,255,0.02);'>
                        <div style='color: #888; font-size: 13px; margin-bottom: 10px;'>Why this member is flagged</div>
                        {flag_rows_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Metric tiles
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                st.metric("Historical CLV", f"${m['CLV']:,.0f}",
                          help="Lifetime invoice value (backward-looking)")
            with t2:
                st.metric("Forward Value Score", f"${m['fvs']:,.0f}",
                          help="Expected forward 6-month value (behavior-adjusted)")
            with t3:
                st.metric("P(churn)", f"{m['P_churn']*100:.1f}%",
                          help="Probability of zero flights in next 6 months")
            with t4:
                st.metric("Value at Risk", f"${m['VaR']:,.0f}",
                          help="FVS × P(churn) — expected forward value at risk")

            # Flight history chart
            st.markdown("##### Monthly flight activity (last 18 months)")
            asof = pd.Timestamp("2018-07-01")
            history = panel[
                (panel["Loyalty Number"] == member_id)
                & (panel["flight_date"] >= asof - pd.DateOffset(months=18))
                & (panel["flight_date"] < asof)
            ].copy()
            if len(history):
                monthly = history.groupby("flight_date")["Total Flights"].sum().reset_index()
                chart = (
                    alt.Chart(monthly)
                    .mark_bar(color="#2E5C8A")
                    .encode(
                        x=alt.X("flight_date:T", title="Month"),
                        y=alt.Y("Total Flights:Q", title="Flights booked"),
                        tooltip=["flight_date:T", "Total Flights:Q"],
                    )
                    .properties(height=180)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No flight activity recorded in the last 18 months.")

# ============ TAB 3: SEGMENT OVERVIEW ============
with tab3:
    st.markdown("##### Segment-level patterns")
    st.caption("Strategic view for monthly review. Use At-Risk List for weekly operations.")

    seg_summary = fvs.groupby("segment").agg(
        n=("Loyalty Number", "count"),
        avg_fvs=("fvs", "mean"),
        avg_clv=("CLV", "mean"),
        avg_p_churn=("P_churn", "mean"),
        total_var=("VaR", "sum"),
    ).reset_index()
    seg_summary["fvs_to_clv"] = seg_summary["avg_fvs"] / seg_summary["avg_clv"]

    seg_order = [3, 4, 1, 2, 0]  # priority order
    cols = st.columns(5)
    for col, seg_id in zip(cols, seg_order):
        row = seg_summary[seg_summary["segment"] == seg_id].iloc[0]
        action_color = SEGMENT_ACTIONS[seg_id]["color"]
        action_label = SEGMENT_ACTIONS[seg_id]["label"]
        with col:
            st.markdown(
                f"""
                <div style='border: 1px solid #444; border-radius: 8px; padding: 16px;
                            min-height: 320px; background: rgba(255,255,255,0.02);'>
                    <h3 style='margin: 0 0 14px 0; color: white; font-size: 18px; line-height: 1.25;'>
                        {SEGMENT_NAMES[seg_id]}
                    </h3>
                    <table style='width: 100%; font-size: 13px; line-height: 1.9;'>
                        <tr>
                            <td style='color: #999;'>Members</td>
                            <td style='text-align: right; font-weight: 700;'>{int(row['n']):,}</td>
                        </tr>
                        <tr>
                            <td style='color: #999;'>Avg FVS</td>
                            <td style='text-align: right; font-weight: 700;'>${row['avg_fvs']:,.0f}</td>
                        </tr>
                        <tr>
                            <td style='color: #999;'>Avg CLV</td>
                            <td style='text-align: right; font-weight: 700;'>${row['avg_clv']:,.0f}</td>
                        </tr>
                        <tr>
                            <td style='color: #999;'>FVS / CLV</td>
                            <td style='text-align: right; font-weight: 700;'>{row['fvs_to_clv']:.2f}</td>
                        </tr>
                        <tr>
                            <td style='color: #999;'>Avg P(churn)</td>
                            <td style='text-align: right; font-weight: 700;'>{row['avg_p_churn']*100:.1f}%</td>
                        </tr>
                        <tr>
                            <td style='color: #999;'>Total VaR</td>
                            <td style='text-align: right; font-weight: 700;'>${row['total_var']:,.0f}</td>
                        </tr>
                    </table>
                    <div style='margin-top: 16px; padding-top: 10px; border-top: 1px solid #444;
                                color: {action_color}; font-size: 15px; font-weight: 700;'>
                        ▌ {action_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("##### How the five segments compare")

    chart_data = seg_summary.copy()
    chart_data["segment_label"] = chart_data["segment"].map(SEGMENT_NAMES)

    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.markdown("**Average Forward Value Score by segment**")
        chart_fvs = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("avg_fvs:Q", title="Avg FVS ($)"),
                y=alt.Y("segment_label:N", sort="-x", title=""),
                color=alt.Color("segment_label:N", legend=None,
                                 scale=alt.Scale(range=["#888888", "#4A8B47", "#B22222", "#1F3A5F", "#D4A017"])),
                tooltip=["segment_label", "avg_fvs"],
            )
            .properties(height=200)
        )
        st.altair_chart(chart_fvs, use_container_width=True)

    with chart_b:
        st.markdown("**Total Value at Risk by segment**")
        chart_var = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("total_var:Q", title="Total VaR ($)"),
                y=alt.Y("segment_label:N", sort="-x", title=""),
                color=alt.Color("segment_label:N", legend=None,
                                 scale=alt.Scale(range=["#888888", "#4A8B47", "#B22222", "#1F3A5F", "#D4A017"])),
                tooltip=["segment_label", "total_var"],
            )
            .properties(height=200)
        )
        st.altair_chart(chart_var, use_container_width=True)

    st.markdown("---")
    st.markdown("##### The CLV vs FVS divergence")
    st.caption("Each dot is a member. The diagonal marks where FVS = 0.5 × CLV. "
               "Members above the line are forward-value heavy (worth investing in); "
               "members below are past-value heavy (limited retention ROI).")

    scatter_data = fvs.sample(min(2000, len(fvs)), random_state=42).copy()
    scatter_data["segment_label"] = scatter_data["segment"].map(SEGMENT_NAMES)
    scatter = (
        alt.Chart(scatter_data)
        .mark_circle(size=30, opacity=0.5)
        .encode(
            x=alt.X("CLV:Q", title="Historical CLV ($)"),
            y=alt.Y("fvs:Q", title="Forward Value Score ($)"),
            color=alt.Color("segment_label:N", title="Segment",
                             scale=alt.Scale(range=["#888888", "#4A8B47", "#B22222", "#1F3A5F", "#D4A017"])),
            tooltip=["Loyalty Number", "segment_label", "CLV", "fvs", "P_churn"],
        )
        .properties(height=400)
    )
    diag = (
        alt.Chart(pd.DataFrame({
            "CLV": [0, scatter_data["CLV"].max()],
            "fvs": [0, scatter_data["CLV"].max() * 0.5],
        }))
        .mark_line(color="gray", strokeDash=[5, 5])
        .encode(x="CLV", y="fvs")
    )
    st.altair_chart(scatter + diag, use_container_width=True)
