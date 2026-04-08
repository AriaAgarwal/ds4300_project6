from arango import ArangoClient
import pandas as pd

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
        year_cols = [str(y) for y in range(2013, 2023)]
        risk_cols = ['COUNTRY', 'INDICATOR', 'UNIT', 'DECIMALS_DISPLAYED'] + year_cols
        risk_df = risk_df[risk_cols]
        risk_df[year_cols] = risk_df[year_cols].apply(pd.to_numeric, errors='coerce')
        risk_df['INDICATOR'] = risk_df['INDICATOR'].str.lower().str.replace(' ', '_')

        # Cleaning extreme weather data
        year_cols_disasters = [str(y) for y in range(1980, 2025)]
        disaster_cols = ['Country', 'Indicator'] + year_cols_disasters
        extreme_weather_df = extreme_weather_df[disaster_cols]

        # Splitting up indicator column
        extreme_weather_df['Indicator'] = extreme_weather_df['Indicator'].str.split(',', n=1).str[1]
        extreme_weather_df[['Metric', 'Disaster']] = extreme_weather_df['Indicator'].str.split(':', n=1, expand=True)
        extreme_weather_df['Metric'] = extreme_weather_df['Metric'].str.lower().str.strip().str.replace(' ', '_')
        extreme_weather_df['Disaster'] = extreme_weather_df['Disaster'].str.lower().str.strip().str.replace(' ','_').str.replace('-', '_')

        # Cleaning temperature data
        year_cols_temp = [str(y) for y in range(1961, 2025)]
        temp_df = temp_df[['Country', 'ISO2', 'ISO3'] + year_cols_temp]
        temp_df[year_cols_temp] = temp_df[year_cols_temp].apply(pd.to_numeric, errors='coerce')

        return risk_df, extreme_weather_df, temp_df

    def load_climate_data(self):
        risk_df, extreme_weather_df, temp_df = self.clean_climate_data()
