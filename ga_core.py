"""
Genetic algorithm core for digester setpoint optimization.

Problem definition (maximization, not target-seeking):
  maximize  ŷ = RF(x)   [predicted biogas production, m³/d]
  subject to  x within operational gene bounds
              R_PS/ES = Q_PS / Q_ES ∈ [0.74, 3.60]
              weather lags fixed for the operating day
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import pygad

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model_biogaz_rf_optimized.joblib"

# Column names expected by the trained Random Forest (do not rename).
FEATURE_NAMES = [
    "Ilość OW [m3/d]",
    "Ilość ON [m3/d]",
    "Stosunek OW/ON",
    "Obciążenie A [kg sm/m3]",
    "Obciążenie A' [kg smo/m3]",
    "Objętość Czynna [m3]",
    "Komora_ID",
    "temperatura_srednia_C_lag_1",
    "opad_suma_mm_lag_2",
    "wilgotnosc_srednia_pct_lag_5",
]

# Display labels + SI-style units for the UI (scientific / operational).
FEATURE_META = {
    "Ilość OW [m3/d]": {
        "label": "Primary sludge flow (Q_PS)",
        "unit": "m³/d",
        "symbol": "Q_PS",
    },
    "Ilość ON [m3/d]": {
        "label": "Excess sludge flow (Q_ES)",
        "unit": "m³/d",
        "symbol": "Q_ES",
    },
    "Stosunek OW/ON": {
        "label": "PS/ES volumetric ratio (R_PS/ES)",
        "unit": "–",
        "symbol": "R_PS/ES",
    },
    "Obciążenie A [kg sm/m3]": {
        "label": "Total solids loading (L_TS)",
        "unit": "kg_TS/m³",
        "symbol": "L_TS",
    },
    "Obciążenie A' [kg smo/m3]": {
        "label": "Volatile solids loading (L_VS)",
        "unit": "kg_VS/m³",
        "symbol": "L_VS",
    },
    "Objętość Czynna [m3]": {
        "label": "Digester active volume (V_eff)",
        "unit": "m³",
        "symbol": "V_eff",
    },
    "Komora_ID": {
        "label": "Digester unit (WKF)",
        "unit": "–",
        "symbol": "Unit",
    },
    "temperatura_srednia_C_lag_1": {
        "label": "Ambient temperature, lag 1 d (T_amb)",
        "unit": "°C",
        "symbol": "T_amb⁻¹",
    },
    "opad_suma_mm_lag_2": {
        "label": "Precipitation sum, lag 2 d (P_sum)",
        "unit": "mm",
        "symbol": "P_sum⁻²",
    },
    "wilgotnosc_srednia_pct_lag_5": {
        "label": "Relative humidity, lag 5 d (H_avg)",
        "unit": "%",
        "symbol": "H_avg⁻⁵",
    },
}

GENE_BOUNDS = [
    {
        "name": FEATURE_NAMES[0],
        "low": 35.10,
        "high": 112.20,
        "optimizable": True,
        "note": "Decision variable",
    },
    {
        "name": FEATURE_NAMES[1],
        "low": 25.50,
        "high": 83.20,
        "optimizable": True,
        "note": "Decision variable",
    },
    {
        "name": FEATURE_NAMES[2],
        "low": 0.74,
        "high": 3.60,
        "optimizable": False,
        "note": "Derived: Q_PS / Q_ES; infeasible → fitness = 0",
    },
    {
        "name": FEATURE_NAMES[3],
        "low": 0.98,
        "high": 4.11,
        "optimizable": True,
        "note": "Decision variable",
    },
    {
        "name": FEATURE_NAMES[4],
        "low": 0.77,
        "high": 2.75,
        "optimizable": True,
        "note": "Decision variable",
    },
    {
        "name": FEATURE_NAMES[5],
        "low": 2240.0,
        "high": 2290.0,
        "optimizable": False,
        "note": "Derived from digester unit (A/B=2240 m³, C/D=2290 m³)",
    },
    {
        "name": FEATURE_NAMES[6],
        "low": 0.0,
        "high": 3.0,
        "optimizable": True,
        "note": "Encoded digester: 0=A, 1=B, 2=C, 3=D (4 tanks)",
    },
    {
        "name": FEATURE_NAMES[7],
        "low": 15.0,
        "high": 15.0,
        "optimizable": False,
        "note": "Exogenous (weather API / scenario)",
    },
    {
        "name": FEATURE_NAMES[8],
        "low": 0.0,
        "high": 0.0,
        "optimizable": False,
        "note": "Exogenous (weather API / scenario)",
    },
    {
        "name": FEATURE_NAMES[9],
        "low": 60.0,
        "high": 60.0,
        "optimizable": False,
        "note": "Exogenous (weather API / scenario)",
    },
]

GA_DEFAULTS = {
    "num_generations": 30,
    "num_parents_mating": 4,
    "sol_per_pop": 15,
    "parent_selection_type": "tournament",
    "keep_parents": 2,
    "crossover_type": "single_point",
    "mutation_type": "random",
    "mutation_percent_genes": 15,
}

# Optional fixed meteorological scenario (not from live API).
SCENARIO_WEATHER = {
    "temperatura_srednia_C_lag_1": 15.0,
    "opad_suma_mm_lag_2": 0.0,
    "wilgotnosc_srednia_pct_lag_5": 60.0,
}

RATIO_MIN = 0.74
RATIO_MAX = 3.60

# Plant digesters from wyniki WKF-2024-v3.csv / WKF3_AUGMENTED_LONG.csv
# LabelEncoder order: A=0, B=1, C=2, D=3 (encoder_komory.joblib)
DIGESTER_UNITS = {
    0: {"code": "A", "volume_m3": 2240.0},
    1: {"code": "B", "volume_m3": 2240.0},
    2: {"code": "C", "volume_m3": 2290.0},
    3: {"code": "D", "volume_m3": 2290.0},
}
VOLUME_MIN = min(u["volume_m3"] for u in DIGESTER_UNITS.values())
VOLUME_MAX = max(u["volume_m3"] for u in DIGESTER_UNITS.values())
DIGESTER_ID_MIN = 0
DIGESTER_ID_MAX = 3


def digester_inventory_table() -> pd.DataFrame:
    rows = []
    for enc_id, meta in DIGESTER_UNITS.items():
        rows.append(
            {
                "Digester": f"WKF {meta['code']}",
                "Encoded ID": enc_id,
                "Active volume V_eff": f"{meta['volume_m3']:.0f} m³",
                "V_eff [m³]": meta["volume_m3"],
            }
        )
    return pd.DataFrame(rows)


def resolve_digester(sol: np.ndarray) -> tuple[int, str, float]:
    """Clamp / round unit ID and return matching active volume."""
    enc_id = int(np.clip(np.rint(sol[6]), DIGESTER_ID_MIN, DIGESTER_ID_MAX))
    meta = DIGESTER_UNITS[enc_id]
    return enc_id, meta["code"], float(meta["volume_m3"])


def apply_plant_consistency(sol: np.ndarray) -> np.ndarray:
    """Force V_eff to match the selected digester (A/B vs C/D)."""
    out = np.array(sol, dtype=float, copy=True)
    enc_id, _, volume = resolve_digester(out)
    out[6] = float(enc_id)
    out[5] = volume
    return out


def feature_label(name: str, with_unit: bool = True) -> str:
    meta = FEATURE_META[name]
    if with_unit:
        return f"{meta['label']} [{meta['unit']}]"
    return meta["label"]


def format_value(name: str, value: float) -> str:
    unit = FEATURE_META[name]["unit"]
    if name == "Komora_ID":
        enc_id = int(np.clip(np.rint(value), DIGESTER_ID_MIN, DIGESTER_ID_MAX))
        code = DIGESTER_UNITS[enc_id]["code"]
        return f"{code} (ID={enc_id})"
    if unit == "–":
        return f"{value:.2f}"
    if unit == "%":
        return f"{value:.1f} %"
    if unit == "°C":
        return f"{value:.1f} °C"
    if unit == "mm":
        return f"{value:.1f} mm"
    if unit == "m³":
        return f"{value:.0f} m³"
    if unit == "m³/d":
        return f"{value:.2f} m³/d"
    if "kg" in unit:
        return f"{value:.2f} {unit}"
    return f"{value:.2f} {unit}"


def load_model(path: Path | str = MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Train the Random Forest first."
        )
    return joblib.load(path)


def build_gene_space(weather: dict[str, float]) -> list[dict]:
    t = float(weather["temperatura_srednia_C_lag_1"])
    p = float(weather["opad_suma_mm_lag_2"])
    h = float(weather["wilgotnosc_srednia_pct_lag_5"])
    return [
        {"low": 35.10, "high": 112.20},
        {"low": 25.50, "high": 83.20},
        {"low": RATIO_MIN, "high": RATIO_MAX},
        {"low": 0.98, "high": 4.11},
        {"low": 0.77, "high": 2.75},
        # V_eff is overwritten from digester ID in fitness (plant inventory).
        {"low": VOLUME_MIN, "high": VOLUME_MAX},
        {"low": DIGESTER_ID_MIN, "high": DIGESTER_ID_MAX},
        {"low": t, "high": t},
        {"low": p, "high": p},
        {"low": h, "high": h},
    ]


def bounds_table(weather: dict[str, float] | None = None) -> pd.DataFrame:
    rows = []
    for i, b in enumerate(GENE_BOUNDS):
        low, high = b["low"], b["high"]
        if weather is not None and i >= 7:
            key = FEATURE_NAMES[i]
            val = float(weather[key])
            low = high = val
        meta = FEATURE_META[b["name"]]
        rows.append(
            {
                "Variable": meta["label"],
                "Symbol": meta["symbol"],
                "Unit": meta["unit"],
                "Min": low,
                "Max": high,
                "Role": "decision" if b["optimizable"] else "fixed / derived",
                "Note": b.get("note", ""),
                "_key": b["name"],
            }
        )
    return pd.DataFrame(rows)


@dataclass
class GAResult:
    solution: np.ndarray
    fitness: float
    history: list[float] = field(default_factory=list)
    weather: dict[str, float] = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def as_dataframe(self) -> pd.DataFrame:
        sol = apply_plant_consistency(self.solution.copy())
        ow, on = sol[0], sol[1]
        sol[2] = ow / on if on > 0.1 else 0.0
        rows = []
        for name, val in zip(FEATURE_NAMES, sol):
            meta = FEATURE_META[name]
            rows.append(
                {
                    "Variable": meta["label"],
                    "Symbol": meta["symbol"],
                    "Unit": meta["unit"],
                    "Value": float(val),
                    "Formatted": format_value(name, float(val)),
                    "_key": name,
                }
            )
        return pd.DataFrame(rows)


def run_ga(
    model,
    weather: dict[str, float],
    ga_params: dict | None = None,
    on_generation: Callable[[int, float], None] | None = None,
    seed: int | None = None,
) -> GAResult:
    """
    Maximize predicted biogas [m³/d] with a genetic algorithm.
    Weather features are locked (low = high) for the selected operating day.
    """
    params = {**GA_DEFAULTS, **(ga_params or {})}
    gene_space = build_gene_space(weather)
    history: list[float] = []

    def fitness_func(ga_instance, solution, solution_idx):
        sol = apply_plant_consistency(np.array(solution, dtype=float))
        ow, on = sol[0], sol[1]
        real_ratio = 0.0 if on < 0.1 else ow / on
        if real_ratio < RATIO_MIN or real_ratio > RATIO_MAX:
            return 0.0
        sol[2] = real_ratio
        input_df = pd.DataFrame([sol], columns=FEATURE_NAMES)
        return float(model.predict(input_df)[0])

    def _on_gen(ga_instance):
        best = float(ga_instance.best_solution()[1])
        history.append(best)
        if on_generation is not None:
            on_generation(ga_instance.generations_completed, best)

    ga_kwargs = dict(
        num_generations=int(params["num_generations"]),
        num_parents_mating=int(params["num_parents_mating"]),
        fitness_func=fitness_func,
        sol_per_pop=int(params["sol_per_pop"]),
        num_genes=len(FEATURE_NAMES),
        gene_space=gene_space,
        parent_selection_type=params["parent_selection_type"],
        keep_parents=int(params["keep_parents"]),
        crossover_type=params["crossover_type"],
        mutation_type=params["mutation_type"],
        mutation_percent_genes=int(params["mutation_percent_genes"]),
        on_generation=_on_gen,
        suppress_warnings=True,
    )
    if seed is not None:
        ga_kwargs["random_seed"] = int(seed)

    ga_instance = pygad.GA(**ga_kwargs)
    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()
    solution = apply_plant_consistency(np.array(solution, dtype=float))
    ow, on = solution[0], solution[1]
    solution[2] = ow / on if on > 0.1 else 0.0

    return GAResult(
        solution=solution,
        fitness=float(solution_fitness),
        history=history,
        weather=dict(weather),
        params=params,
    )
