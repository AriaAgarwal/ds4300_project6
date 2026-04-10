from arango import ArangoClient
import pandas as pd
import json

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