from climate_api import CLIMATE_API

def load_data(api):
    api.load_climate_data()
    api.load_border_data()

def main():
    api = CLIMATE_API()
    load_data(api)




if __name__ == "__main__":
    main()