"""
Operational panel: maximize predicted digester biogas yield with GA + RF.

Local:
  .\\.venv\\Scripts\\streamlit.exe run streamlit_app.py
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ga_core import (
    FEATURE_META,
    FEATURE_NAMES,
    GA_DEFAULTS,
    RATIO_MAX,
    RATIO_MIN,
    bounds_table,
    digester_inventory_table,
    format_value,
    load_model,
    resolve_digester,
    run_ga,
)
from weather_service import (
    DEFAULT_LAT,
    DEFAULT_LON,
    DEFAULT_PLACE,
    get_weather_for_optimization,
)

st.set_page_config(
    page_title="Biogas yield optimization · GA",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEPS = [
    "1 · Objective",
    "2 · Weather",
    "3 · Search space",
    "4 · GA settings",
    "5 · Run",
    "6 · Results",
]


def _inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; max-width: 1180px; }
        div[data-testid="stMetric"] {
            background: linear-gradient(160deg, #f3f7f4 0%, #e8f0ea 100%);
            border: 1px solid #c5d5c8;
            border-radius: 10px;
            padding: 0.6rem 0.9rem;
        }
        .step-hint {
            color: #3d4f42;
            font-size: 0.95rem;
            margin-bottom: 0.8rem;
        }
        .source-box {
            background: #f7faf8;
            border-left: 4px solid #2f6b4f;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0 1rem 0;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        h1, h2, h3 { letter-spacing: -0.02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    defaults = {
        "step": 0,
        "weather_ctx": None,
        "ga_result": None,
        "ga_history": [],
        "model_ok": None,
        "model_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource(show_spinner=False)
def cached_model():
    return load_model()


def nav_buttons(can_next: bool = True):
    c1, c2, _ = st.columns([1, 1, 6])
    with c1:
        if st.session_state.step > 0:
            if st.button("Back", use_container_width=True):
                st.session_state.step -= 1
                st.rerun()
    with c2:
        if st.session_state.step < len(STEPS) - 1 and can_next:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.step += 1
                st.rerun()


def step_goal():
    st.header("Optimization objective")
    st.markdown(
        """
        <div class="source-box">
        <b>Problem type:</b> constrained <b>maximization</b> of predicted daily biogas production —
        not inverse control (“produce exactly X m³/d today”).<br><br>
        <b>Objective</b><br>
        maximize &nbsp; <i>ŷ</i> = RF(<b>x</b>) &nbsp; &nbsp; [biogas yield, m³/d]<br><br>
        <b>Decision variables</b> <b>x</b>: sludge flows, solids loadings, digester unit
        (four plant tanks WKF A–D), within operational min/max bounds.
        Active volume V_eff is set from the selected digester
        (A/B = 2240 m³, C/D = 2290 m³).<br>
        <b>Exogenous inputs</b>: weather lags (T<sub>amb</sub>−1 d, P<sub>sum</sub>−2 d, H<sub>avg</sub>−5 d)
        fixed for the operating day from a weather service.<br>
        <b>Constraint</b>: volumetric ratio R<sub>PS/ES</sub> = Q<sub>PS</sub>/Q<sub>ES</sub>
        must lie in [0.74, 3.60]; otherwise fitness = 0.<br><br>
        Surrogate model: Random Forest regressor trained on plant digester data.
        Search method: genetic algorithm (PyGAD).
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Surrogate model", "Random Forest")
    c2.metric("Optimizer", "Genetic algorithm")
    c3.metric("Objective", "max ŷ [m³/d]")

    st.subheader("Model input vector")
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        meta = FEATURE_META[name]
        rows.append(
            {
                "#": i,
                "Variable": meta["label"],
                "Symbol": meta["symbol"],
                "Unit": meta["unit"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    try:
        _ = cached_model()
        st.session_state.model_ok = True
        st.success("Model loaded: `model_biogaz_rf_optimized.joblib`")
    except Exception as e:
        st.session_state.model_ok = False
        st.session_state.model_error = str(e)
        st.error(f"Failed to load model: {e}")

    nav_buttons(can_next=bool(st.session_state.model_ok))


def step_weather():
    st.header("Meteorological boundary conditions")
    st.markdown(
        '<p class="step-hint">The RF model expects lagged ambient covariates: '
        "temperature lag 1 d, precipitation lag 2 d, relative humidity lag 5 d. "
        "These are exogenous (not optimized).</p>",
        unsafe_allow_html=True,
    )

    mode_label = st.radio(
        "Weather source",
        [
            "Open-Meteo API (live)",
            "Historical plant CSV (weather_daily_api.csv)",
            "Fixed scenario (15 °C / 0 mm / 60 %)",
        ],
    )
    mode_map = {
        "Open-Meteo API (live)": "api",
        "Historical plant CSV (weather_daily_api.csv)": "csv",
        "Fixed scenario (15 °C / 0 mm / 60 %)": "scenario",
    }
    mode = mode_map[mode_label]

    city = None
    lat, lon = DEFAULT_LAT, DEFAULT_LON
    place = DEFAULT_PLACE

    if mode == "api":
        loc_mode = st.selectbox(
            "Location",
            ["Warsaw (default)", "Other city (Poland)", "Manual coordinates"],
        )
        if loc_mode == "Other city (Poland)":
            city = st.text_input("City", value="Krakow")
        elif loc_mode == "Manual coordinates":
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", value=DEFAULT_LAT, format="%.4f")
            lon = c2.number_input("Longitude", value=DEFAULT_LON, format="%.4f")
            place = f"lat={lat:.4f}, lon={lon:.4f}"

    ref = None
    if mode == "csv":
        ref = st.date_input(
            "Operating day (lags computed backward)",
            value=date(2024, 6, 15),
            min_value=date(2024, 1, 6),
            max_value=date(2024, 12, 31),
        )

    if st.button("Load weather", type="primary"):
        with st.spinner("Fetching meteorological data..."):
            ctx = get_weather_for_optimization(
                mode=mode,
                latitude=lat,
                longitude=lon,
                place_label=place,
                city=city,
                reference_date=ref,
            )
            st.session_state.weather_ctx = ctx

    ctx = st.session_state.weather_ctx
    if ctx is None:
        st.info("Load weather to continue.")
        nav_buttons(can_next=False)
        return

    st.markdown(
        f'<div class="source-box"><b>Source:</b> {ctx.source}<br>'
        f"<b>Location:</b> {ctx.place_label}<br>"
        f"<b>Operating day:</b> {ctx.reference_date}</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "T_amb, lag 1 d",
        format_value(
            "temperatura_srednia_C_lag_1",
            ctx.features["temperatura_srednia_C_lag_1"],
        ),
    )
    m2.metric(
        "P_sum, lag 2 d",
        format_value("opad_suma_mm_lag_2", ctx.features["opad_suma_mm_lag_2"]),
    )
    m3.metric(
        "H_avg, lag 5 d",
        format_value(
            "wilgotnosc_srednia_pct_lag_5",
            ctx.features["wilgotnosc_srednia_pct_lag_5"],
        ),
    )

    show = ctx.features_table().drop(columns=["Raw value"], errors="ignore")
    st.dataframe(show, hide_index=True, use_container_width=True)

    if ctx.daily is not None and len(ctx.daily) > 0:
        date_col = "Date" if "Date" in ctx.daily.columns else "Data"
        plot_df = ctx.daily.tail(14).copy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_df[date_col],
                y=plot_df["temperatura_srednia_C"],
                name="Mean temperature [°C]",
                mode="lines+markers",
            )
        )
        fig.add_trace(
            go.Bar(
                x=plot_df[date_col],
                y=plot_df["opad_suma_mm"],
                name="Precipitation [mm]",
                yaxis="y2",
                opacity=0.45,
            )
        )
        fig.update_layout(
            title="Recent daily weather (inputs to lag construction)",
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(title="Temperature [°C]"),
            yaxis2=dict(title="Precipitation [mm]", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

    nav_buttons(can_next=True)


def step_bounds():
    st.header("Operational search space")
    st.markdown(
        '<p class="step-hint">Min/max bounds define the feasible gene space. '
        "Weather covariates are locked (min = max) for the selected day. "
        "The ratio R_PS/ES is derived from flows and checked as a hard constraint. "
        "Active volume V_eff is tied to the selected digester (four plant tanks).</p>",
        unsafe_allow_html=True,
    )
    ctx = st.session_state.weather_ctx
    if ctx is None:
        st.warning("Set weather in step 2 first.")
        nav_buttons(can_next=False)
        return

    st.subheader("Plant digesters (from WKF 2024 inventory)")
    inv = digester_inventory_table()
    st.dataframe(
        inv[["Digester", "Encoded ID", "Active volume V_eff"]],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Source columns: Objetość WKF A–D [m³] in `wyniki WKF-2024-v3.csv`. "
        "Encoding matches `encoder_komory.joblib` (A=0 … D=3)."
    )

    df = bounds_table(ctx.features)
    show = df.drop(columns=["_key"]).copy()
    for col in ("Min", "Max"):
        show[col] = [format_value(k, v) for k, v in zip(df["_key"], df[col])]
    st.dataframe(show, hide_index=True, use_container_width=True)

    opt = df[df["Role"] == "decision"].copy()
    opt = opt[~opt["_key"].isin(["Komora_ID"])]
    fig = go.Figure()
    for _, row in opt.iterrows():
        fig.add_trace(
            go.Bar(
                name=row["Symbol"],
                x=[f"{row['Symbol']}\n[{row['Unit']}]"],
                y=[row["Max"] - row["Min"]],
                base=[row["Min"]],
                text=[f"[{row['Min']:.2f} – {row['Max']:.2f}]"],
                textposition="outside",
            )
        )
    fig.update_layout(
        title="Decision-variable ranges (flows and loadings)",
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=80),
        yaxis_title="Value (native unit)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Hard constraint: if R_PS/ES ∉ [{RATIO_MIN:.2f}, {RATIO_MAX:.2f}], "
        "the individual receives fitness = 0 m³/d. "
        "V_eff is always set from the chosen digester (A/B → 2240 m³, C/D → 2290 m³)."
    )
    nav_buttons()


def step_ga_params():
    st.header("Genetic algorithm configuration")
    st.markdown(
        '<p class="step-hint">Defaults match the validated operational setup. '
        "Adjust only for sensitivity analysis during the presentation.</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    num_generations = c1.number_input(
        "Generations",
        min_value=5,
        max_value=100,
        value=GA_DEFAULTS["num_generations"],
    )
    sol_per_pop = c2.number_input(
        "Population size",
        min_value=5,
        max_value=50,
        value=GA_DEFAULTS["sol_per_pop"],
    )
    num_parents_mating = c3.number_input(
        "Parents mating",
        min_value=2,
        max_value=20,
        value=GA_DEFAULTS["num_parents_mating"],
    )

    c4, c5, c6 = st.columns(3)
    keep_parents = c4.number_input(
        "Elitism (keep_parents)",
        min_value=0,
        max_value=10,
        value=GA_DEFAULTS["keep_parents"],
    )
    mutation_percent_genes = c5.number_input(
        "Mutation [% of genes]",
        min_value=1,
        max_value=50,
        value=GA_DEFAULTS["mutation_percent_genes"],
    )
    seed = c6.number_input("Random seed", min_value=0, max_value=99999, value=42)

    st.session_state.ga_params_ui = {
        "num_generations": int(num_generations),
        "sol_per_pop": int(sol_per_pop),
        "num_parents_mating": int(num_parents_mating),
        "keep_parents": int(keep_parents),
        "mutation_percent_genes": int(mutation_percent_genes),
        "parent_selection_type": GA_DEFAULTS["parent_selection_type"],
        "crossover_type": GA_DEFAULTS["crossover_type"],
        "mutation_type": GA_DEFAULTS["mutation_type"],
    }
    st.session_state.ga_seed = int(seed)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Parameter": k,
                    "Value": v,
                    "Baseline": GA_DEFAULTS.get(k, "—"),
                }
                for k, v in st.session_state.ga_params_ui.items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Selection: {GA_DEFAULTS['parent_selection_type']} · "
        f"Crossover: {GA_DEFAULTS['crossover_type']} · "
        f"Mutation: {GA_DEFAULTS['mutation_type']}"
    )
    nav_buttons()


def step_run():
    st.header("Run optimization")
    ctx = st.session_state.weather_ctx
    if ctx is None:
        st.warning("Weather missing — return to step 2.")
        nav_buttons(can_next=False)
        return

    params = st.session_state.get("ga_params_ui", dict(GA_DEFAULTS))
    seed = st.session_state.get("ga_seed", 42)

    st.write(
        f"Population **{params['sol_per_pop']}**, generations **{params['num_generations']}**, "
        f"weather: **{ctx.source}**."
    )

    if st.button("Start optimization", type="primary"):
        model = cached_model()
        progress = st.progress(0.0, text="Starting...")
        chart_ph = st.empty()
        hist: list[float] = []

        def on_gen(gen_i, best):
            hist.append(best)
            progress.progress(
                min(gen_i / max(params["num_generations"], 1), 1.0),
                text=(
                    f"Generation {gen_i}/{params['num_generations']} · "
                    f"best ŷ = {best:.2f} m³/d"
                ),
            )
            fig = px.line(
                x=list(range(1, len(hist) + 1)),
                y=hist,
                labels={
                    "x": "Generation",
                    "y": "Best fitness ŷ [m³/d]",
                },
                title="GA convergence",
            )
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            chart_ph.plotly_chart(fig, use_container_width=True)

        with st.spinner("Optimizing..."):
            result = run_ga(
                model,
                ctx.features,
                ga_params=params,
                on_generation=on_gen,
                seed=seed,
            )
        st.session_state.ga_result = result
        st.session_state.ga_history = result.history
        progress.progress(1.0, text="Completed")
        st.success(
            f"Finished. Predicted biogas yield: **{result.fitness:.2f} m³/d**"
        )
        st.session_state.step = 5
        st.rerun()

    if st.session_state.ga_result is not None:
        st.caption("A result is available — continue to step 6 or re-run.")
        nav_buttons(can_next=True)
    else:
        nav_buttons(can_next=False)


def step_result():
    st.header("Recommended operating point")
    result = st.session_state.ga_result
    if result is None:
        st.warning("No result yet — run the GA in step 5.")
        nav_buttons(can_next=False)
        return

    st.metric("Predicted biogas yield ŷ", f"{result.fitness:.2f} m³/d")

    sol = result.solution
    enc_id, code, volume = resolve_digester(sol)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Q_PS", format_value(FEATURE_NAMES[0], sol[0]))
    c2.metric("Q_ES", format_value(FEATURE_NAMES[1], sol[1]))
    c3.metric("R_PS/ES", format_value(FEATURE_NAMES[2], sol[2]))
    c4.metric("Digester", f"WKF {code} (ID={enc_id})")

    c5, c6, c7 = st.columns(3)
    c5.metric("L_TS", format_value(FEATURE_NAMES[3], sol[3]))
    c6.metric("L_VS", format_value(FEATURE_NAMES[4], sol[4]))
    c7.metric("V_eff", f"{volume:.0f} m³")

    st.subheader("Full solution vector")
    out = result.as_dataframe()
    st.dataframe(
        out[["Variable", "Symbol", "Unit", "Formatted"]].rename(
            columns={"Formatted": "Value"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    if result.history:
        fig = px.line(
            x=list(range(1, len(result.history) + 1)),
            y=result.history,
            labels={"x": "Generation", "y": "Best fitness ŷ [m³/d]"},
            title="Best-fitness trajectory",
        )
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feasibility check against bounds")
    bt = bounds_table(result.weather)
    cmp = bt.merge(
        out[["_key", "Value", "Formatted"]],
        left_on="_key",
        right_on="_key",
    )
    cmp["Within bounds"] = cmp.apply(
        lambda r: "yes" if r["Min"] - 1e-9 <= r["Value"] <= r["Max"] + 1e-9 else "NO",
        axis=1,
    )
    cmp["Min"] = [format_value(k, v) for k, v in zip(cmp["_key"], cmp["Min"])]
    cmp["Max"] = [format_value(k, v) for k, v in zip(cmp["_key"], cmp["Max"])]
    st.dataframe(
        cmp[["Variable", "Unit", "Min", "Max", "Formatted", "Within bounds"]].rename(
            columns={"Formatted": "Solution"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    export = out[["Variable", "Symbol", "Unit", "Value"]].copy()
    csv = export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download solution CSV",
        data=csv,
        file_name="ga_biogas_optimum.csv",
        mime="text/csv",
    )

    if st.button("Reset wizard"):
        st.session_state.weather_ctx = None
        st.session_state.ga_result = None
        st.session_state.ga_history = []
        st.session_state.step = 0
        st.rerun()

    nav_buttons(can_next=False)


def main():
    _inject_css()
    init_state()

    st.title("Digester biogas yield optimization")
    st.caption(
        "Operational decision support · Random Forest surrogate · genetic search · Open-Meteo"
    )

    cols = st.columns(len(STEPS))
    for i, label in enumerate(STEPS):
        with cols[i]:
            if i == st.session_state.step:
                st.markdown(f"**[{label}]**")
            elif i < st.session_state.step:
                st.markdown(f"~~{label}~~")
            else:
                st.markdown(label)

    st.divider()

    with st.sidebar:
        st.header("Navigation")
        choice = st.radio("Step", STEPS, index=st.session_state.step)
        st.session_state.step = STEPS.index(choice)
        st.divider()
        st.markdown(
            """
            **Problem**
            - Maximize predicted biogas [m³/d]
            - Not a setpoint target (“produce X”)

            **Data**
            - RF surrogate model
            - Operational min/max gene bounds
            - Weather: [Open-Meteo](https://open-meteo.com/)
            """
        )
        st.caption("`streamlit run streamlit_app.py`")

    pages = [
        step_goal,
        step_weather,
        step_bounds,
        step_ga_params,
        step_run,
        step_result,
    ]
    pages[st.session_state.step]()


if __name__ == "__main__":
    main()
