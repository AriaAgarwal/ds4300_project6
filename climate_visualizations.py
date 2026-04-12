import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from climate_api import CLIMATE_API

# approximate "Global North" for visualization (high-income + developed). everything else = Global South.
GLOBAL_NORTH = [
        "USA", "CAN", "GBR", "IRL", "FRA", "DEU", "ITA", "ESP", "PRT", "NLD",
        "BEL", "LUX", "CHE", "AUT", "NOR", "SWE", "DNK", "FIN", "ISL", "GRC",
        "AUS", "NZL", "JPN", "KOR", "ISR", "SGP", "CZE", "SVK", "SVN", "EST",
        "LVA", "LTU", "POL", "HUN", "HRV", "BGR", "ROU", "CYP", "MLT"]


def _bubble_areas_from_counts(counts, s_min=55.0, s_max=950.0):
    """
    maps disaster counts to matplotlib scatter
    - compresses counts with sqrt, then linearly stretches them so the smallest and largest counts in the data use the full bubble-size range chosen
    - s_min and s_max are the minimum and maximum bubble sizes in points²
    """
    # counts is an array of disaster counts
    c = np.asarray(counts, dtype=float)
    c = np.maximum(c, 0.0)
    # if there are no counts, return an array of the mean of s_min and s_max
    if c.size == 0:
        return c
    # if the range of counts is 0, return an array of the mean of s_min and s_max
    if np.ptp(c) == 0:
        # np.ptp is the range of the counts
        return np.full(c.shape, (s_min + s_max) / 2.0)
    # compress counts with sqrt
    t = np.sqrt(c + 1.0)
    # t_lo and t_hi are the minimum and maximum of the compressed counts
    t_lo, t_hi = float(t.min()), float(t.max())
    if t_hi <= t_lo:
        # if the range of the compressed counts is 0, return an array of the mean of s_min and s_max
        return np.full(c.shape, (s_min + s_max) / 2.0)
    # linearly stretch the compressed counts to the full bubble-size range
    u = (t - t_lo) / (t_hi - t_lo)
    return s_min + u * (s_max - s_min)


def load_data(api):
    api.load_climate_data()
    api.load_border_data()


def _df_records(df):
    """Turn a DataFrame into a list of row dicts; NaN becomes None."""
    if df is None or df.empty:
        return []
    out = df.replace({np.nan: None})
    return out.to_dict(orient="records")


def _temperature_change_for_countries(api, country_names):
    """Load each country's temperature_change dict from Arango (year keys as strings)."""
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
    outfile="paris_threshold_timeline.png",
    show=False,
):
    """
    two-panel figure: left = anomaly time series; right = latest anomaly ranking.
    """
    df = api.full_threshold_crossing()
    # sort the dataframe by latest anomaly descending and get the top n countries
    df = df.sort_values("latest_anomaly", ascending=False, na_position="last").head(top_n)
    country_names = df["country"].tolist()

    # create a dictionary of the info by name
    info_by_name = {}
    # loop through the rows and add the info to the dictionary
    for row in _df_records(df):
        info_by_name[row["country"]] = row

    # get the temperature change for the countries
    temps_by_name = _temperature_change_for_countries(api, country_names)
    # create a list of the year keys
    year_keys = [str(y) for y in range(1961, 2025)]
    # create a list of the results annual
    results_annual = []
    # loop through the country names and add the results to the list
    for name in country_names:
        # get the temperature change for the country
        temps_dict = temps_by_name[name].get("temps")
        # get the meta data for the country
        meta = info_by_name[name]
        # add the results to the list
        results_annual.append(
            {
                "country": name,
                # get the iso3 code for the country
                "iso3": temps_by_name[name].get("iso3"),
                # add the temperature change for the country
                "temps": temps_dict,
                # add if the country has crossed the 1.5°C threshold
                "crossed": bool(meta.get("crossed_1_5_raw")),
                # add the first crossing year
                "first_cross_year": meta.get("first_crossed_year"),
                # add the latest anomaly
                "latest_anomaly": meta.get("latest_anomaly"),
            }
        )

    # build the crossed and not crossed blocks
    crossed_blocks = []
    not_crossed_blocks = []
    for item in results_annual:
        # create a list of the yearly values
        yearly_vals = []
        # loop through the year keys and add the yearly values to the list
        for yk in year_keys:
            yearly_vals.append(item["temps"].get(yk))
        # create a list of the years
        x_years = list(range(1961, 2025))
        # create a pandas series of the yearly values
        annual = pd.Series(yearly_vals, index=x_years, dtype=float)
        # interpolate the yearly values
        annual = annual.interpolate(method="linear", limit=2)
        # create a rolling mean of the yearly values
        smooth = annual.rolling(window=5, center=True, min_periods=3).mean()
        # create a block
        block = {"info": item, "annual": annual, "smooth": smooth}
        # add the block to the crossed blocks if the country has crossed the 1.5°C threshold
        if item["crossed"]:
            crossed_blocks.append(block)
        # add the block to the not crossed blocks if the country has not crossed the 1.5°C threshold
        else:
            not_crossed_blocks.append(block)

    # create a color map
    cmap = plt.cm.YlOrRd_r
    crossing_years = []
    # loop through the crossed blocks and add the crossing years to the list
    for block in crossed_blocks:
        # get the first crossing year
        fy = block["info"]["first_cross_year"]
        # add the crossing year to the list if it is not None and not an empty string
        if fy is not None and str(fy).strip() != "":
            crossing_years.append(int(fy))

    # get the minimum and maximum crossing years
    yr_min = min(crossing_years)
    yr_max = max(crossing_years)

    # create a color bar if the minimum and maximum crossing years are not None
    use_colorbar = yr_min is not None and yr_max is not None

    # create a function to get the color for the crossing year
    def color_for_crossing(fy):
        """get the color for the crossing year"""
        if yr_min is None or fy is None or str(fy).strip() == "":
            return "grey"
        cy = int(fy)
        spread = max(yr_max - yr_min, 1)
        t = (cy - yr_min) / spread * 0.85
        return cmap(t)

    # create a figure
    fig = plt.figure(figsize=(28, 10))
    if use_colorbar:
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.04, 1.35], wspace=0.32)
        ax_left = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax_right = fig.add_subplot(gs[0, 2])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.32)
        ax_left = fig.add_subplot(gs[0, 0])
        cax = None
        ax_right = fig.add_subplot(gs[0, 1])

    # plot the not crossed blocks
    for block in not_crossed_blocks:
        # get the smooth values
        smooth = block["smooth"]
        # plot the smooth values
        ax_left.plot(
            smooth.index,
            smooth.values,
            color="#b8b8b8",
            alpha=0.85,
            linewidth=1.0,
            zorder=1,
        )

    # plot the crossed blocks
    for block in crossed_blocks:
        # get the info for the block
        info = block["info"]
        # get the annual values
        annual = block["annual"]
        # get the smooth values
        smooth = block["smooth"]
        # get the first crossing year
        fy = info["first_cross_year"]
        # get the color for the crossing year
        line_color = color_for_crossing(fy)
        # plot the annual values

        # plot the annual values
        ax_left.plot(annual.index, annual.values, color=line_color, alpha=0.15, linewidth=0.6, zorder=2)
        ax_left.plot(smooth.index, smooth.values, color=line_color, alpha=0.85, linewidth=1.8, zorder=3)
        # if the first crossing year is not None and not an empty string, get the value at the crossing year
        if fy is not None and str(fy).strip() != "":
            cy = int(fy)
            val_at_cross = smooth.get(cy)
            if val_at_cross is not None and not np.isnan(val_at_cross):
                ax_left.scatter(
                    cy,
                    val_at_cross,
                    color=line_color,
                    s=60,
                    zorder=5,
                    edgecolors="black",
                    linewidths=0.5,
                )

    # add a horizontal line at 1.5°C
    ax_left.axhline(1.5, color="red", linewidth=2.0, linestyle="--", zorder=6)
    # add a horizontal line at 1.1°C
    ax_left.axhline(1.1, color="orange", linewidth=1.5, linestyle="--", zorder=6)
    ax_left.axhline(0, color="grey", linewidth=0.5, linestyle="-", alpha=0.4, zorder=1)
    ax_left.set_xlim(1961, 2027)
    ax_left.set_ylim(-2.5, 5.5)
    ax_left.set_xlabel("Year", fontsize=11)
    ax_left.set_ylabel("Temperature anomaly (°C)\nrelative to 1951–1980 baseline", fontsize=11)

    if use_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=yr_min, vmax=yr_max))
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax)
        cb.set_label("Year first crossed 1.5°C\n(darker = earlier crossing)", fontsize=9)
        cb.ax.invert_yaxis()

    legend_bits = [
        Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="1.5°C threshold"),
        Line2D([0], [0], color="orange", linewidth=1.5, linestyle="--", label="Paris-adjusted (~1.1°C)"),
        Line2D([0], [0], color="darkred", linewidth=2, label="Crossed (bold = 5-yr avg)"),
        Line2D([0], [0], color="#b8b8b8", linewidth=1, alpha=0.9, label="Not yet crossed"),
    ]
    ax_left.legend(handles=legend_bits, loc="upper left", fontsize=9)

    #  sort the results by the latest anomaly
    ranked = sorted(results_annual, key=lambda r: float(r["latest_anomaly"]) if r.get("latest_anomaly") is not None and not pd.isna(r["latest_anomaly"]) else -999.0, reverse=False)
    bar_labels = []
    bar_lengths = []
    bar_colors = []
    for r in ranked:
        bar_labels.append(r["country"])
        la = r["latest_anomaly"]
        if la is None or pd.isna(la):
            bar_lengths.append(0.0)
        else:
            bar_lengths.append(float(la))
        if not r["crossed"]:
            bar_colors.append("steelblue")
        else:
            bar_colors.append(color_for_crossing(r.get("first_cross_year")))

    bars = ax_right.barh(bar_labels, bar_lengths, color=bar_colors, edgecolor="grey", linewidth=0.5)
    ax_right.axvline(1.5, color="red", linestyle="--", linewidth=2.0)
    ax_right.axvline(1.1, color="orange", linestyle="--", linewidth=1.5)
    ax_right.set_xlabel("Latest Temperature Anomaly (°C) — 2024", fontsize=11)
    ax_right.tick_params(axis="y", labelsize=8, pad=6)
    for lbl in ax_right.get_yticklabels():
        lbl.set_ha("right")

    if len(bar_lengths) > 0:
        xmax = max(bar_lengths)
    else:
        xmax = 1.5
    pad = max(0.06 * xmax, 0.1)
    for i in range(len(bars)):
        bar = bars[i]
        r = ranked[i]
        fy = r.get("first_cross_year")
        if r["crossed"] and fy is not None and str(fy).strip() != "":
            label = str(int(fy))
        else:
            label = "—"
        x_end = bar_lengths[i]
        y_mid = bar.get_y() + bar.get_height() / 2
        ax_right.text(x_end + pad, y_mid, label, va="center", fontsize=7, clip_on=False)

    ax_right.set_xlim(0, max(xmax + pad * 3.5, 1.65))

    fig.suptitle(
        "Paris Agreement Threshold Analysis: Which Countries Have Already Crossed 1.5°C?",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.93,
        "Left: Temperature anomaly trends 1961-2024 with 5-year rolling average  |  "
        "Right: 2024 anomaly ranking by country",
        fontsize=10,
        ha="center",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_climate_injustice_bubble(api, outfile="climate_injustice_bubble.png"):
    """
    x = first year crossing 1.5°C, y = INFORM risk, bubble size = disasters since then.
    Red = Global South (by ISO3 list), blue = Global North (approx.).
    """
    # run the climate injustice query
    df = api.climate_injustice()
    rows = _df_records(df)
    # build a simple list: one dict per bubble we can plot
    points = []
    for row in rows:
        year_cross = row.get("first_crossed_1_5")
        risk = row.get("inform_risk_2022")
        if year_cross is None or risk is None:
            continue
        try:
            disaster_total = float(row.get("disasters_since_crossing") or 0)
        except (TypeError, ValueError):
            disaster_total = 0.0
        points.append(
            {
                "year_cross": int(year_cross),
                "risk": float(risk),
                "disasters": disaster_total,
                "name": row.get("country", ""),
                "iso3": row.get("iso3"),
            }
        )

    # bubble sizes from disaster counts (sqrt + stretch; see helper)
    disaster_list = [p["disasters"] for p in points]
    bubble_sizes = _bubble_areas_from_counts(np.array(disaster_list))

    # draw each bubble: color by rough Global North vs South
    fig, ax = plt.subplots(figsize=(12, 8))
    color_south = "#c0392b"
    edge_south = "#7b241c"
    color_north = "#2874a6"
    edge_north = "#1b4f72"

    # draw each bubble
    for i in range(len(points)):
        p = points[i]
        size = bubble_sizes[i]
        code = p["iso3"]
        is_north = isinstance(code, str) and code.strip().upper() in GLOBAL_NORTH
        face = color_north if is_north else color_south
        edge = edge_north if is_north else edge_south

        ax.scatter(
            p["year_cross"],
            p["risk"],
            s=size,
            alpha=0.65,
            color=face,
            edgecolors=edge,
            linewidths=0.8,
        )
        ax.annotate(
            p["name"],
            (p["year_cross"], p["risk"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    # paris 2015 line, labels, legend, save
    ax.axvline(x=2015, color="navy", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Year first crossed 1.5°C threshold")
    ax.set_ylabel("INFORM climate risk score (2022)")
    ax.set_title(
        "Climate injustice: high-risk countries that crossed 1.5°C earliest\n"
        "(bubble area = disaster event counts since crossing)"
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
            label="Global South",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_north,
            markeredgecolor=edge_north,
            markersize=10,
            label="Global North",
        ),
        Line2D([0], [0], color="navy", linestyle="--", linewidth=1.5, label="Paris Agreement (2015)"),
    ]
    ax.legend(handles=legend_markers, loc="upper left", fontsize=9)

    ax.margins(x=0.12, y=0.08)
    fig.tight_layout(pad=1.4)
    fig.savefig(outfile, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_threshold_watchlist(api, outfile="threshold_watchlist.png"):
    """
    countries close to 1.5°C but not over yet; bars = latest anomaly; text = years to threshold.
    """
    # run query
    df = api.close_to_threshold()
    rows = _df_records(df)

    # pull out parallel lists (easy to loop with an index)
    country_names = []
    latest_vals = []
    years_to_go = []
    for row in rows:
        country_names.append(row["country"])
        latest_vals.append(row["latest_anomaly"])
        years_to_go.append(row.get("est_years_to_threshold"))

    # bar colors: hotter (closer to 1.5) = more red in YlOrRd
    cmap = plt.cm.YlOrRd
    lo = min(latest_vals)
    hi = 1.5
    norm = mcolors.Normalize(vmin=lo, vmax=hi)
    bar_colors = []
    for v in latest_vals:
        bar_colors.append(cmap(norm(v)))

    # horizontal bars + vertical reference lines
    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(country_names, latest_vals, color=bar_colors, edgecolor="grey", linewidth=0.5)

    ax.axvline(x=1.5, color="red", linestyle="--", linewidth=2, label="1.5°C threshold")
    ax.axvline(x=1.1, color="orange", linestyle="--", linewidth=1.5, label="Paris-adjusted threshold")

    # optional text: estimated years until crossing (use bar center y so labels line up)
    for i, bar in enumerate(bars):
        eta = years_to_go[i]
        if eta is None or pd.isna(eta):
            continue
        yrs = int(round(float(eta)))
        y_center = bar.get_y() + bar.get_height() / 2
        ax.text(
            latest_vals[i] + 0.02,
            y_center,
            f"~{yrs}yr",
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


def plot_climate_stress(api, outfile="climate_stress.png"):
    """
    Query 4: horizontal bars of composite climate stress score (sorted low → high on chart).
    """
    df = api.climate_stress()
    rows = _df_records(df)

    # sort so first row in file = lowest stress, last = highest (barh draws bottom to top)
    df_sorted = pd.DataFrame(rows).sort_values("stress_score", ascending=True)

    country_names = df_sorted["country"].tolist()
    stress_scores = df_sorted["stress_score"].tolist()

    # one bar per country
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(country_names, stress_scores, color="#c0392b", edgecolor="grey", linewidth=0.5)

    ax.set_xlabel("Climate stress score")
    ax.set_title("Top countries by composite climate stress")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)



def main():
    api = CLIMATE_API()

    load_data(api)
    plot_paris_threshold_timeline(api, top_n=30)
    plot_climate_injustice_bubble(api)
    plot_threshold_watchlist(api)
    plot_climate_stress(api)


if __name__ == "__main__":
    main()
