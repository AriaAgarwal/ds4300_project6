import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from climate_api import CLIMATE_API

ALL_YEARS_STR = [str(y) for y in range(1961, 2025)]

# Approximate "Global North" for visualization (high-income OECD-style + developed
# Asia-Pacific). Everything else is colored as Global South. Not a legal UN definition.
GLOBAL_NORTH_ISO3 = frozenset(
    {
        "USA",
        "CAN",
        "GBR",
        "IRL",
        "FRA",
        "DEU",
        "ITA",
        "ESP",
        "PRT",
        "NLD",
        "BEL",
        "LUX",
        "CHE",
        "AUT",
        "NOR",
        "SWE",
        "DNK",
        "FIN",
        "ISL",
        "GRC",
        "AUS",
        "NZL",
        "JPN",
        "KOR",
        "ISR",
        "SGP",
        "CZE",
        "SVK",
        "SVN",
        "EST",
        "LVA",
        "LTU",
        "POL",
        "HUN",
        "HRV",
        "BGR",
        "ROU",
        "CYP",
        "MLT",
    }
)


def _is_global_north(iso3):
    if not iso3 or not isinstance(iso3, str):
        return False
    return iso3.strip().upper() in GLOBAL_NORTH_ISO3


def _bubble_areas_from_counts(counts, s_min=55.0, s_max=950.0):
    """Map disaster counts to matplotlib scatter `s` (area in pt²); sqrt for readability."""
    c = np.asarray(counts, dtype=float)
    c = np.maximum(c, 0.0)
    if c.size == 0:
        return c
    if np.ptp(c) == 0:
        return np.full(c.shape, (s_min + s_max) / 2.0)
    t = np.sqrt(c + 1.0)
    t_lo, t_hi = float(t.min()), float(t.max())
    if t_hi <= t_lo:
        return np.full(c.shape, (s_min + s_max) / 2.0)
    u = (t - t_lo) / (t_hi - t_lo)
    return s_min + u * (s_max - s_min)


def load_data(api):
    api.load_climate_data()
    api.load_border_data()


def _df_records(df):
    """DataFrame rows as plain dicts; NaN -> None for JSON-like plotting."""
    if df is None or df.empty:
        return []
    out = df.replace({np.nan: None})
    return out.to_dict(orient="records")


def _temperature_change_for_countries(api, country_names):
    """Full temperature_change object per country (annual keys 1961–2024)."""
    if not country_names:
        return {}
    rows = api.aql(
        """
        FOR c IN countries
            FILTER c.country IN @names
            RETURN { country: c.country, iso3: c.iso3, temps: c.temperature_change }
        """,
        {"names": country_names},
    )
    return {r["country"]: r for r in rows}


def plot_paris_threshold_timeline(
    api,
    top_n=30,
    outfile="paris_threshold_improved.png",
    show=False,
):
    """
    Query 1 (full_threshold_crossing) for crossing metadata; supplementary AQL
    for raw temperature_change. Top N by latest_anomaly; rolling mean + YlOrRd_r
    by first crossing year; muted steelblue for not-yet-crossed.
    """
    df = api.full_threshold_crossing()
    if df.empty:
        print("plot_paris_threshold_timeline: no data from full_threshold_crossing.")
        return

    df = df.sort_values("latest_anomaly", ascending=False, na_position="last").head(top_n)
    meta_by_country = {r["country"]: r for r in _df_records(df)}
    country_order = df["country"].tolist()
    temps_by_country = _temperature_change_for_countries(api, country_order)

    results_annual = []
    for name in country_order:
        m = meta_by_country.get(name)
        doc = temps_by_country.get(name)
        if not m or not doc or not doc.get("temps"):
            continue
        results_annual.append(
            {
                "country": name,
                "iso3": doc.get("iso3"),
                "temps": doc["temps"],
                "crossed_1_5_raw": bool(m.get("crossed_1_5_raw")),
                "first_crossed_year": m.get("first_crossed_year"),
                "latest_anomaly": m.get("latest_anomaly"),
            }
        )

    if not results_annual:
        print("plot_paris_threshold_timeline: no annual series after merge.")
        return

    all_years = ALL_YEARS_STR
    fig, ax = plt.subplots(figsize=(16, 9))

    crossed_countries = []
    not_crossed = []

    for c in results_annual:
        temps_dict = c.get("temps") or {}
        temps = [temps_dict.get(y) for y in all_years]
        years_int = [int(y) for y in all_years]

        s = pd.Series(temps, index=years_int, dtype=float)
        s = s.interpolate(method="linear", limit=2)
        rolled = s.rolling(window=5, center=True, min_periods=3).mean()

        if c.get("crossed_1_5_raw"):
            crossed_countries.append((c, s, rolled))
        else:
            not_crossed.append((c, s, rolled))

    for _, _, rolled in not_crossed:
        ax.plot(
            rolled.index,
            rolled.values,
            color="steelblue",
            alpha=0.25,
            linewidth=1.0,
            zorder=1,
        )

    cmap = plt.cm.YlOrRd_r
    crossing_years = [
        int(c["first_crossed_year"])
        for c, _, _ in crossed_countries
        if c.get("first_crossed_year") not in (None, "")
    ]
    if crossing_years:
        yr_min, yr_max = min(crossing_years), max(crossing_years)
    else:
        yr_min, yr_max = None, None

    for c, s, rolled in crossed_countries:
        cross_yr = c.get("first_crossed_year")
        if cross_yr is not None and cross_yr != "" and yr_min is not None and yr_max is not None:
            cy = int(cross_yr)
            span = max(yr_max - yr_min, 1)
            norm_val = (cy - yr_min) / span
            color = cmap(norm_val * 0.85)
        else:
            color = "grey"

        ax.plot(s.index, s.values, color=color, alpha=0.15, linewidth=0.6, zorder=2)
        ax.plot(rolled.index, rolled.values, color=color, alpha=0.85, linewidth=1.8, zorder=3)

        if cross_yr is not None and cross_yr != "":
            cy = int(cross_yr)
            cross_val = rolled.get(cy)
            if cross_val is not None and not np.isnan(cross_val):
                ax.scatter(
                    cy,
                    cross_val,
                    color=color,
                    s=60,
                    zorder=5,
                    edgecolors="black",
                    linewidths=0.5,
                )

            latest = c.get("latest_anomaly")
            latest_ok = latest is not None and not pd.isna(latest) and float(latest) >= 3.0
            earliest_ok = yr_min is not None and cy <= yr_min + 5
            if earliest_ok or latest_ok:
                rd = rolled.dropna()
                if rd.empty:
                    continue
                last_val = rd.iloc[-1]
                ax.annotate(
                    c.get("iso3") or "",
                    xy=(2024, last_val),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    va="center",
                    fontweight="bold",
                )

    ax.axhline(
        y=1.5,
        color="red",
        linewidth=2.0,
        linestyle="--",
        zorder=6,
        label="1.5°C threshold (raw data baseline)",
    )
    ax.axhline(
        y=1.1,
        color="orange",
        linewidth=1.5,
        linestyle="--",
        zorder=6,
        label="Paris-adjusted threshold (~1.1°C)",
    )
    ax.axhline(y=0, color="grey", linewidth=0.5, linestyle="-", alpha=0.4, zorder=1)

    if yr_min is not None and yr_max is not None and crossing_years:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=yr_min, vmax=yr_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label("Year first crossed 1.5°C\n(darker = earlier crossing)", fontsize=9)
        cbar.ax.invert_yaxis()

    legend_elements = [
        Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="1.5°C threshold"),
        Line2D([0], [0], color="orange", linewidth=1.5, linestyle="--", label="Paris-adjusted (~1.1°C)"),
        Line2D([0], [0], color="darkred", linewidth=2, label="Crossed threshold (bold = 5yr avg)"),
        Line2D([0], [0], color="steelblue", linewidth=1, alpha=0.5, label="Not yet crossed"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_xlim(1961, 2027)
    ax.set_ylim(-2.5, 5.5)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(
        "Temperature Anomaly (°C)\nrelative to 1951–1980 baseline",
        fontsize=11,
    )
    n_show = len(results_annual)
    ax.set_title(
        "Country Temperature Anomalies vs. Paris Agreement Thresholds (1961–2024)\n"
        f"Top {n_show} countries by latest anomaly  |  Bold lines = 5-year rolling average  |  "
        "Dots = year of first threshold crossing",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    print(f"Saved {outfile}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_climate_injustice_bubble(api, outfile="climate_injustice_bubble.png"):
    """Query 2: early crossers × INFORM risk; bubble area ∝ disasters since crossing."""
    df = api.climate_injustice()
    results = _df_records(df)
    if not results:
        print("plot_climate_injustice_bubble: no rows from climate_injustice.")
        return

    rows_plot = []
    for c in results:
        fy = c.get("first_crossed_1_5")
        y = c.get("inform_risk_2022")
        if fy is None or y is None:
            continue
        try:
            dcount = float(c.get("disasters_since_crossing") or 0)
        except (TypeError, ValueError):
            dcount = 0.0
        rows_plot.append(
            {
                "x": int(fy),
                "y": float(y),
                "dcount": dcount,
                "country": c.get("country", ""),
                "iso3": c.get("iso3"),
            }
        )

    if not rows_plot:
        print("plot_climate_injustice_bubble: no plottable rows.")
        return

    counts = [r["dcount"] for r in rows_plot]
    areas = _bubble_areas_from_counts(np.array(counts))

    fig, ax = plt.subplots(figsize=(12, 8))

    color_south = "#c0392b"
    edge_south = "#7b241c"
    color_north = "#2874a6"
    edge_north = "#1b4f72"

    for r, s_area in zip(rows_plot, areas):
        north = _is_global_north(r["iso3"])
        face = color_north if north else color_south
        edge = edge_north if north else edge_south
        ax.scatter(
            r["x"],
            r["y"],
            s=s_area,
            alpha=0.65,
            color=face,
            edgecolors=edge,
            linewidths=0.8,
        )
        ax.annotate(
            r["country"],
            (r["x"], r["y"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.axvline(
        x=2015,
        color="navy",
        linestyle="--",
        linewidth=1.5,
        label="Paris Agreement signed (2015)",
    )
    ax.set_xlabel("Year first crossed 1.5°C threshold")
    ax.set_ylabel("INFORM climate risk score (2022)")
    ax.set_title(
        "Climate injustice: high-risk countries that crossed 1.5°C earliest\n"
        "(bubble area ∝ disaster event counts since crossing; sqrt-scaled for readability)"
    )

    legend_markers = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_south,
            markeredgecolor=edge_south,
            markersize=10,
            label="Global South (approx., ISO3 not in North set)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_north,
            markeredgecolor=edge_north,
            markersize=10,
            label="Global North (approx., high-income / OECD-style ISO3)",
        ),
        Line2D([0], [0], color="navy", linestyle="--", linewidth=1.5, label="Paris Agreement (2015)"),
    ]
    ax.legend(handles=legend_markers, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_threshold_watchlist(api, outfile="threshold_watchlist.png"):
    """Query 3: horizontal bars by latest anomaly; YlOrRd gradient; ETA annotation."""
    df = api.close_to_threshold()
    results = _df_records(df)
    if not results:
        print("plot_threshold_watchlist: no rows from close_to_threshold.")
        return

    countries = [c["country"] for c in results]
    anomalies = [c["latest_anomaly"] for c in results]
    eta = [c.get("est_years_to_threshold") for c in results]

    norm = mcolors.Normalize(vmin=min(anomalies), vmax=1.5)
    cmap = plt.cm.YlOrRd
    colors = [cmap(norm(a)) for a in anomalies]

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(countries, anomalies, color=colors, edgecolor="grey", linewidth=0.5)

    ax.axvline(x=1.5, color="red", linestyle="--", linewidth=2, label="1.5°C threshold")
    ax.axvline(x=1.1, color="orange", linestyle="--", linewidth=1.5, label="Paris-adjusted threshold")

    for i, (bar, e) in enumerate(zip(bars, eta)):
        if e is None or pd.isna(e):
            continue
        label = f"~{int(round(float(e)))}yr"
        ax.text(
            anomalies[i] + 0.02,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=7,
            color="navy",
        )

    ax.set_xlabel("Latest temperature anomaly (°C)")
    ax.set_title(
        "Countries approaching the 1.5°C threshold\n"
        "(annotated with estimated years until crossing at current rate)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}")


def main():
    api = CLIMATE_API()

    load_data(api)
    plot_paris_threshold_timeline(api, top_n=30)
    plot_climate_injustice_bubble(api)
    plot_threshold_watchlist(api)


if __name__ == "__main__":
    main()
