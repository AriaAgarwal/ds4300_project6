from arango import ArangoClient
import pandas as pd
import requests
import json
import csv
import io

class CLIMATE_API:

    def __init__(self):
        self.client = ArangoClient(hosts='http://localhost:8529')
        sys_db = self.client.db('_system', username="", password="")

       # Create database
        if not sys_db.has_database('climate_data'):
            sys_db.create_database('climate_data')

        # Connect to database
        self.db = self.client.db('climate_data', username="", password="")

    def clean_climate_data(self):
        risk_df = pd.read_csv('risk.csv')
        extreme_weather_df = pd.read_csv('extreme_weather.csv')
        temp_df = pd.read_csv('temperature_change.csv')

        # Cleaning risk data
        risk_df['COUNTRY'] = risk_df['COUNTRY'].str.split(',').str[0].str.strip()
        year_cols = [str(y) for y in range(2013, 2023)]
        risk_cols = ['COUNTRY', 'INDICATOR', 'UNIT', 'DECIMALS_DISPLAYED'] + year_cols
        risk_df = risk_df[risk_cols]
        risk_df[year_cols] = risk_df[year_cols].apply(pd.to_numeric, errors='coerce')
        risk_df['INDICATOR'] = risk_df['INDICATOR'].str.lower().str.replace(' ', '_')

        # Cleaning extreme weather data
        extreme_weather_df['Country'] = extreme_weather_df['Country'].str.split(',').str[0].str.strip()
        year_cols_disasters = [str(y) for y in range(1980, 2025)]
        disaster_cols = ['Country', 'Indicator'] + year_cols_disasters
        extreme_weather_df = extreme_weather_df[disaster_cols]

        # Splitting up indicator column
        extreme_weather_df['Indicator'] = extreme_weather_df['Indicator'].str.split(',', n=1).str[1]
        extreme_weather_df[['Metric', 'Disaster']] = extreme_weather_df['Indicator'].str.split(':', n=1, expand=True)
        extreme_weather_df['Metric'] = extreme_weather_df['Metric'].str.lower().str.strip().str.replace(' ', '_')
        extreme_weather_df['Disaster'] = extreme_weather_df['Disaster'].str.lower().str.strip().str.replace(' ','_').str.replace('-', '_')

        # Cleaning temperature data
        temp_df['Country'] = temp_df['Country'].str.split(',').str[0].str.strip()
        year_cols_temp = [str(y) for y in range(1961, 2025)]
        temp_df = temp_df[['Country', 'ISO2', 'ISO3'] + year_cols_temp]
        temp_df[year_cols_temp] = temp_df[year_cols_temp].apply(pd.to_numeric, errors='coerce')

        # Filter risk and disasters to only countries in temp_df
        valid_countries = set(temp_df['Country'].unique())
        risk_df = risk_df[risk_df['COUNTRY'].isin(valid_countries)]
        extreme_weather_df = extreme_weather_df[extreme_weather_df['Country'].isin(valid_countries)]

        return risk_df, extreme_weather_df, temp_df

    def load_climate_data(self):
        risk_df, extreme_weather_df, temp_df = self.clean_climate_data()

        if not self.db.has_collection('countries'):
            self.db.create_collection('countries')
        else:
            self.db.collection('countries').truncate()

        countries_col = self.db.collection('countries')
        countries = {}

        risk_year_cols = [str(y) for y in range(2013, 2023)]
        disaster_year_cols = [str(y) for y in range(1980, 2025)]
        temp_year_cols = [str(y) for y in range(1961, 2025)]

        # Formatting risk data
        for row in risk_df.iterrows():
            row = row[1]
            country = row['COUNTRY']
            if country not in countries:
                countries[country] = {
                    'country': country,
                    'iso2': None,
                    'iso3': None,
                    'inform_risk': {},
                    'disasters': {},
                    'temperature_change': {}
                }
            countries[country]['inform_risk'][row['INDICATOR']] = {
                year: row[year] for year in risk_year_cols
            }

        # Formatting extreme weather data
        for row in extreme_weather_df.iterrows():
            row = row[1]
            country = row['Country']
            if country not in countries:
                countries[country] = {
                    'country': country,
                    'iso2': None,
                    'iso3': None,
                    'inform_risk': {},
                    'disasters': {},
                    'temperature_change': {}
                }
            if row['Disaster'] not in countries[country]['disasters']:
                countries[country]['disasters'][row['Disaster']] = {}
            countries[country]['disasters'][row['Disaster']][row['Metric']] = {
                year: row[year] for year in disaster_year_cols
            }

        # Formatting temperature data
        for row in temp_df.iterrows():
            row = row[1]
            country = row['Country']
            if country not in countries:
                countries[country] = {
                    'country': country,
                    'iso2': row['ISO2'],
                    'iso3': row['ISO3'],
                    'inform_risk': {},
                    'disasters': {},
                    'temperature_change': {}
                }
            else:
                countries[country]['iso2'] = row['ISO2']
                countries[country]['iso3'] = row['ISO3']
            countries[country]['temperature_change'] = {
                year: row[year] for year in temp_year_cols
            }

        # Insert into ArangoDB
        docs = list(countries.values())
        for doc in docs:
            doc['_key'] = doc['country'].replace(' ', '_').replace(',', '').replace('.', '')

        docs = json.loads(pd.Series(docs).to_json(orient='records'))
        countries_col.insert_many(docs)
        print(f"Inserted {len(docs)} country documents into ArangoDB.")

    def load_border_data(self):
        response = requests.get(
            'https://raw.githubusercontent.com/geodatasource/country-borders/master/GEODATASOURCE-COUNTRY-BORDERS.CSV')
        reader = csv.DictReader(io.StringIO(response.text))

        if not self.db.has_collection('borders'):
            self.db.create_collection('borders', edge=True)
        else:
            self.db.collection('borders').truncate()

        if self.db.has_graph('climate_graph'):
            self.db.delete_graph('climate_graph')
        self.db.create_graph('climate_graph', edge_definitions=[{
            'edge_collection': 'borders',
            'from_vertex_collections': ['countries'],
            'to_vertex_collections': ['countries']
        }])

        # Build ISO2 to _key mapping
        iso2_to_key = {}
        for doc in self.db.collection('countries').all():
            if doc['iso2']:
                iso2_to_key[doc['iso2']] = doc['_key']

        # Insert edges (outside the for doc loop!)
        borders_col = self.db.collection('borders')
        edges = []
        for row in reader:
            from_code = row['country_code']
            to_code = row['country_border_code']
            if not to_code or from_code not in iso2_to_key or to_code not in iso2_to_key:
                continue
            edges.append({
                '_from': f"countries/{iso2_to_key[from_code]}",
                '_to': f"countries/{iso2_to_key[to_code]}"
            })

        borders_col.insert_many(edges)
        print(f"Inserted {len(edges)} border edges into ArangoDB.")


    def aql(self, query, bind_vars=None):
        """Run AQL and return rows as dictionaries"""
        cursor = self.db.aql.execute(query, bind_vars=bind_vars or {})
        return list(cursor)


    def full_threshold_crossing(self):
        """
        Full 1.5°C threshold crossing profile per country
        Baseline note: temperature_change values are anomalies vs 1951-1980.
        Pre-industrial to 1951-1980 baseline ≈ +0.3°C, so Paris-adjusted
        threshold in this dataset ≈ 1.1°C (conservative) to 1.2°C.
        We report both the raw 1.5°C crossing AND the Paris-adjusted 1.1°C
        crossing so readers can interpret both.
        """

        rows = self.aql(
        """           
        LET all_years = [
        "1961","1962","1963","1964","1965","1966","1967","1968","1969","1970",
        "1971","1972","1973","1974","1975","1976","1977","1978","1979","1980",
        "1981","1982","1983","1984","1985","1986","1987","1988","1989","1990",
        "1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",
        "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",
        "2011","2012","2013","2014","2015","2016","2017","2018","2019","2020",
        "2021","2022","2023","2024"
        ]

        LET decades = ["1961","1971","1981","1991","2001","2011","2021"]

        FOR country IN countries
        FILTER LENGTH(country.temperature_change) > 0

        // ── Baseline: average anomaly 1961–1990 (pre-acceleration era) ──
        LET baseline_avg = AVG(
            FOR y IN ["1961","1962","1963","1964","1965","1966","1967","1968","1969","1970",
                    "1971","1972","1973","1974","1975","1976","1977","1978","1979","1980",
                    "1981","1982","1983","1984","1985","1986","1987","1988","1989","1990"]
            LET v = country.temperature_change[y]
            FILTER IS_NUMBER(v)
            RETURN v
        )

        // ── Decade-by-decade average anomalies ──
        LET decade_avgs = (
            FOR d IN decades
            LET yr_int = TO_NUMBER(d)
            LET decade_years = (
                FOR y IN all_years
                FILTER TO_NUMBER(y) >= yr_int AND TO_NUMBER(y) < yr_int + 10
                RETURN y
            )
            LET avg_val = AVG(
                FOR y IN decade_years
                LET v = country.temperature_change[y]
                FILTER IS_NUMBER(v)
                RETURN v
            )
            RETURN {
                decade: CONCAT(d, "s"),
                avg_anomaly: IS_NUMBER(avg_val) ? ROUND(avg_val * 100) / 100 : null
            }
        )

        // ── First year crossing raw 1.5°C ──
        LET crossing_1_5 = FIRST(
            FOR y IN all_years
            LET v = country.temperature_change[y]
            FILTER IS_NUMBER(v) AND v >= 1.5
            SORT y ASC
            RETURN { year: y, anomaly: ROUND(v * 100) / 100 }
        )

        // ── First year crossing Paris-adjusted threshold (~1.1°C) ──
        LET crossing_paris = FIRST(
            FOR y IN all_years
            LET v = country.temperature_change[y]
            FILTER IS_NUMBER(v) AND v >= 1.1
            SORT y ASC
            RETURN { year: y, anomaly: ROUND(v * 100) / 100 }
        )

        // ── Sustained crossing: 5-year rolling avg >= 1.5°C ──
        // (single-year spikes can be El Nino noise; sustained matters more)
        LET sustained_crossing = FIRST(
            FOR anchor IN all_years
            LET anchor_int = TO_NUMBER(anchor)
            FILTER anchor_int >= 1965   // need at least 4 years before it
            LET slice_avg = AVG(
                FOR y IN all_years
                FILTER TO_NUMBER(y) <= anchor_int
                FILTER TO_NUMBER(y) >= anchor_int - 4
                LET v = country.temperature_change[y]
                FILTER IS_NUMBER(v)
                RETURN v
            )
            FILTER IS_NUMBER(slice_avg) AND slice_avg >= 1.5
            SORT anchor ASC
            RETURN {
                year: anchor,
                five_yr_avg: ROUND(slice_avg * 100) / 100
            }
        )

        // ── Latest anomaly and total warming since 1961 ──
        LET latest = LAST(
            FOR y IN all_years
            LET v = country.temperature_change[y]
            FILTER IS_NUMBER(v)
            RETURN { year: y, anomaly: v }
        )

        LET total_warming = IS_NUMBER(latest.anomaly) AND IS_NUMBER(baseline_avg)
            ? ROUND((latest.anomaly - baseline_avg) * 100) / 100
            : null

        // ── INFORM risk score ──
        LET risk = country.inform_risk.inform_risk_index["2022"]

        // ── Distance from 1.5°C threshold (for countries not yet crossed) ──
        LET distance_from_threshold = crossing_1_5 == null AND IS_NUMBER(latest.anomaly)
            ? ROUND((1.5 - latest.anomaly) * 100) / 100
            : null

        FILTER IS_NUMBER(latest.anomaly)
        SORT (crossing_1_5 != null ? 0 : 1) ASC,
            (crossing_1_5 != null ? crossing_1_5.year : "9999") ASC,
            latest.anomaly DESC

        RETURN {
            country:                    country.country,
            iso3:                       country.iso3,

            // Threshold crossings
            crossed_1_5_raw:            crossing_1_5 != null,
            first_crossed_year:         crossing_1_5 != null ? crossing_1_5.year : null,
            anomaly_at_crossing:        crossing_1_5 != null ? crossing_1_5.anomaly : null,
            sustained_crossing_year:    sustained_crossing != null ? sustained_crossing.year : null,
            sustained_crossing_avg:     sustained_crossing != null ? sustained_crossing.five_yr_avg : null,

            // Paris-adjusted (1951-1980 baseline offset)
            crossed_paris_adjusted:     crossing_paris != null,
            paris_crossing_year:        crossing_paris != null ? crossing_paris.year : null,

            // How close are non-crossers?
            degrees_from_threshold:     distance_from_threshold,

            // Trend over time
            baseline_avg_1961_1990:     IS_NUMBER(baseline_avg) ? ROUND(baseline_avg * 100) / 100 : null,
            latest_year:                latest.year,
            latest_anomaly:             IS_NUMBER(latest.anomaly) ? ROUND(latest.anomaly * 100) / 100 : null,
            total_warming_since_1961:   total_warming,
            decade_by_decade:           decade_avgs,

            // Vulnerability
            inform_risk_score_2022:     IS_NUMBER(risk) ? ROUND(risk * 100) / 100 : null
        }
        """
        )
        return pd.DataFrame(rows)


    def climate_injustice(self):
        """Countries that crossed earliest AND have the highest INFORM risk (most vulnerable, least responsible)"""
        rows = self.aql(
        """                   
        LET all_years = [
        "1961","1962","1963","1964","1965","1966","1967","1968","1969","1970",
        "1971","1972","1973","1974","1975","1976","1977","1978","1979","1980",
        "1981","1982","1983","1984","1985","1986","1987","1988","1989","1990",
        "1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",
        "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",
        "2011","2012","2013","2014","2015","2016","2017","2018","2019","2020",
        "2021","2022","2023","2024"
        ]

        FOR country IN countries
        FILTER LENGTH(country.inform_risk) > 0

                LET risk = country.inform_risk.inform_risk_index["2022"]
                FILTER IS_NUMBER(risk) AND risk >= 5.0   // high vulnerability

        LET crossing = FIRST(
            FOR y IN all_years
            LET v = country.temperature_change[y]
            FILTER IS_NUMBER(v) AND v >= 1.5
            SORT y ASC
            RETURN { year: y, anomaly: v }
        )
        FILTER crossing != null

        LET latest_anomaly = LAST(
            FOR y IN all_years
            LET v = country.temperature_change[y]
            FILTER IS_NUMBER(v)
            RETURN v
        )

        // Total disasters since crossing year
        LET disasters_since_crossing = SUM(
            FOR dtype IN ATTRIBUTES(country.disasters)
            LET d = country.disasters[dtype]
            RETURN SUM(
                FOR y IN all_years
                FILTER TO_NUMBER(y) >= TO_NUMBER(crossing.year)
                LET v = d.number_of_events[y]
                RETURN IS_NUMBER(v) ? v : 0
            )
        )

        SORT crossing.year ASC, risk DESC
        RETURN {
            country:                country.country,
            iso3:                   country.iso3,
            first_crossed_1_5:      crossing.year,
            anomaly_at_crossing:    ROUND(crossing.anomaly * 100) / 100,
            current_anomaly:        ROUND(latest_anomaly * 100) / 100,
            inform_risk_2022:       ROUND(risk * 100) / 100,
            disasters_since_crossing: disasters_since_crossing,
            // Years they've been above threshold — time under climate stress
            years_above_threshold:  2024 - TO_NUMBER(crossing.year)
        }
            """
        )
        return pd.DataFrame(rows)


    def close_to_threshold(self):
        """Countries closest to but not yet at 1.5°C"""
        rows = self.aql(
        """                     
        LET all_years_q3 = [
        "1961","1962","1963","1964","1965","1966","1967","1968","1969","1970",
        "1971","1972","1973","1974","1975","1976","1977","1978","1979","1980",
        "1981","1982","1983","1984","1985","1986","1987","1988","1989","1990",
        "1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",
        "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",
        "2011","2012","2013","2014","2015","2016","2017","2018","2019","2020",
        "2021","2022","2023","2024"
        ]

        FOR country IN countries
        // Confirm NOT yet crossed
        LET has_crossed = LENGTH(
            FOR y IN all_years_q3
            LET v = country.temperature_change[y]
            FILTER IS_NUMBER(v) AND v >= 1.5
            LIMIT 1
            RETURN y
        ) > 0
        FILTER !has_crossed

        LET latest_anomaly = LAST(
            FOR y IN all_years_q3
            LET v = country.temperature_change[y]
            FILTER IS_NUMBER(v)
            RETURN v
        )
        FILTER IS_NUMBER(latest_anomaly) AND latest_anomaly >= 0.8

        // Rate of warming: slope over last 20 years
        LET recent_years = ["2004","2005","2006","2007","2008","2009","2010","2011",
                            "2012","2013","2014","2015","2016","2017","2018","2019",
                            "2020","2021","2022","2023"]
        LET first_recent = country.temperature_change["2004"]
        LET last_recent  = country.temperature_change["2023"]
        LET warming_rate_per_decade = (IS_NUMBER(first_recent) AND IS_NUMBER(last_recent))
            ? ROUND(((last_recent - first_recent) / 2.0) * 100) / 100
            : null

        // Projected years until crossing at current rate
        LET years_to_threshold = (IS_NUMBER(warming_rate_per_decade) AND warming_rate_per_decade > 0)
            ? ROUND(((1.5 - latest_anomaly) / (warming_rate_per_decade / 10)) * 10) / 10
            : null

        LET risk = country.inform_risk.inform_risk_index["2022"]

        SORT latest_anomaly DESC
        LIMIT 25
        RETURN {
            country:                    country.country,
            iso3:                       country.iso3,
            latest_anomaly:             ROUND(latest_anomaly * 100) / 100,
            degrees_from_threshold:     ROUND((1.5 - latest_anomaly) * 100) / 100,
            warming_rate_per_decade:    warming_rate_per_decade,
            est_years_to_threshold:     years_to_threshold,
            inform_risk_score:          IS_NUMBER(risk) ? ROUND(risk * 100) / 100 : null
        }
        """
            )
        return pd.DataFrame(rows)

    def climate_stress(self):
        """Top countries with the highest overall climate stress"""
        rows = self.aql(
        """
        FOR c IN countries

        LET risk = TO_NUMBER(c.inform_risk["climate-driven_inform_risk_indicator"]["2022"])
        LET temp = TO_NUMBER(c.temperature_change["2022"])

        LET disaster_total = SUM(
            FOR d IN ATTRIBUTES(c.disasters, true)
                FOR m IN ATTRIBUTES(c.disasters[d], true)
                    LET val = TO_NUMBER(c.disasters[d][m]["2022"])
                    FILTER val != null
                    RETURN val
        )

        FILTER risk != null
        FILTER temp != null
        FILTER disaster_total != null AND disaster_total > 0

        LET norm_disaster = LOG(disaster_total + 1)

        LET stress_score = (risk * 0.5) + (temp * 0.3) + (norm_disaster * 0.2)

        SORT stress_score DESC
        LIMIT 20

        RETURN {
            country: c.country,
            stress_score: ROUND(stress_score * 1000) / 1000
        }
        """
            )
        return pd.DataFrame(rows)
    
    def risk_clustering(self):
        """Do the high-risk countries cluster geographically based on the bordering countries"""
        rows = self.aql(
        """
        FOR c IN countries

        LET risk = TO_NUMBER(c.inform_risk["climate-driven_inform_risk_indicator"]["2022"])
        FILTER risk != null AND risk > 0

        LET neighbors = (
            FOR n IN 1..1 OUTBOUND c borders
                LET neighbor_risk = TO_NUMBER(n.inform_risk["climate-driven_inform_risk_indicator"]["2022"])
                FILTER neighbor_risk != null AND neighbor_risk > 0
                RETURN neighbor_risk
        )

        LET avg_neighbor_risk = AVERAGE(neighbors)

        FILTER avg_neighbor_risk != null

        RETURN {
            country: c.country,
            risk: ROUND(risk * 100) / 100,
            avg_neighbor_risk: ROUND(avg_neighbor_risk * 100) / 100,
            num_neighbors: LENGTH(neighbors)
        }
    """
        )
        return pd.DataFrame(rows)
    
    def disaster_growth_high_risk(self):
        """Which disaster types are increasing the fastest in high-risk countries"""
        rows = self.aql(
            """
            FOR c IN countries

            LET risk = TO_NUMBER(c.inform_risk["climate-driven_inform_risk_indicator"]["2022"])
            FILTER risk != null AND risk >= 5

            FOR disaster_type IN ATTRIBUTES(c.disasters, true)
                FILTER disaster_type != "total"
                LET early_avg = AVERAGE(
                    FOR year IN ["2000", "2005"]
                        LET yearly_sum = SUM(
                            FOR m IN ATTRIBUTES(c.disasters[disaster_type], true)
                                LET val = TO_NUMBER(c.disasters[disaster_type][m][year])
                                FILTER val != null
                                RETURN val
                        )
                        RETURN yearly_sum
                )

                LET recent_avg = AVERAGE(
                    FOR year IN ["2018", "2022"]
                        LET yearly_sum = SUM(
                            FOR m IN ATTRIBUTES(c.disasters[disaster_type], true)
                                LET val = TO_NUMBER(c.disasters[disaster_type][m][year])
                                FILTER val != null
                                RETURN val
                        )
                        RETURN yearly_sum
                )

                FILTER early_avg != null AND recent_avg != null

                LET growth = recent_avg - early_avg

                SORT growth DESC
                LIMIT 50

                RETURN {
                    country: c.country,
                    disaster_type: disaster_type,
                    growth: ROUND(growth * 100) / 100
                }
        """
        )
        return pd.DataFrame(rows)