from climate_api import CLIMATE_API

def load_data(api):
    api.load_climate_data()

def main():
    api = CLIMATE_API()
    load_data(api)
    risk_df, extreme_weather_df, temp_df = api.clean_climate_data()




if __name__ == "__main__":
    main()