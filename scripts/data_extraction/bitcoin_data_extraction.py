import pandas as pd
import requests
from datetime import datetime
    
    
def get_hourly_price_data(timestamp):
    
    API_key = ...
    url = 'https://min-api.cryptocompare.com/data/histohour?'
    headers = {
        "Host": 'min-api.cryptocompare.com',
    }

    params = {
        "fsym": "BTC",
        "tsym": "USD",
        "toTs": timestamp,
        "limit": 2000,
        "api_key": API_key,
    }

    response = requests.get(url, headers=headers, params=params)

    print(response.status_code)
    myjson = response.json()
    raw_price_data = myjson['Data']
    
    # convert to csv file
    hourly_price_data = pd.DataFrame.from_dict(raw_price_data)
    hourly_price_data.set_index("time", inplace=True)
    hourly_price_data.index = pd.to_datetime(hourly_price_data.index, unit='s')
    hourly_price_data['datetimes'] = hourly_price_data.index
    hourly_price_data['datetimes'] = hourly_price_data['datetimes'].dt.strftime(
        '%Y-%m-%d %H:%M')
    
    return hourly_price_data


def get_data():
    full_data = []
    date = datetime(2023, 3, 30, 0, 0)
    while date > datetime(2020, 1, 1, 0, 0):
        date_timestamp = int(date.timestamp())
        hourly_data = get_hourly_price_data(date_timestamp)
        full_data.append(hourly_data)
        date = datetime.strptime(hourly_data['datetimes'].min(), '%Y-%m-%d %H:%M')
    
    # save the file
    full_data = pd.concat(full_data, axis=0)
    full_data.to_csv('/mnt/bitcoin_hourly_prices_till_2023-03-30.csv')
    
    return full_data

if __name__ == "__main__":
    data = get_data()
    print(data.shape)
    print(data['datetimes'].min())
    print(data['datetimes'].max())