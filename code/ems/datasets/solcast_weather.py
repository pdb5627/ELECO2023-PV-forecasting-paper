import os
import glob
import pandas as pd
import sqlalchemy

from .dataset_with_forecast import DataSetWithForecast


class SolCastWeather(DataSetWithForecast):
    def __init__(self, data_dir='data/SolCast',
                 db_engine=None, db_table='solcast_weather',
                 resample_interval='30min', average_interval='1h'):
        super().__init__(db_engine, db_table, resample_interval, average_interval)
        self.data_dir = data_dir
        self.ids = [1]

    def _define_col_names(self):
        self._actual_columns = [
            #('location', sqlalchemy.Text),
            #('lat', sqlalchemy.Float),
            #('lon', sqlalchemy.Float),
            ('clouds', sqlalchemy.Float),
            ('ghi', sqlalchemy.Float),
            ('ebh', sqlalchemy.Float),
            ('dni', sqlalchemy.Float),
            ('dhi', sqlalchemy.Float)
        ]

        self._forecast_columns = [
            #('location', sqlalchemy.Text),
            #('lat', sqlalchemy.Float),
            #('lon', sqlalchemy.Float),
            ('temp', sqlalchemy.Float),
            ('clouds', sqlalchemy.Float),
            ('ghi', sqlalchemy.Float),
            ('ghi90', sqlalchemy.Float),
            ('ghi10', sqlalchemy.Float),
            ('ebh', sqlalchemy.Float),
            ('ebh90', sqlalchemy.Float),
            ('ebh10', sqlalchemy.Float),
            ('dni', sqlalchemy.Float),
            ('dni90', sqlalchemy.Float),
            ('dni10', sqlalchemy.Float),
            ('dhi', sqlalchemy.Float)
        ]

    def _read_file(self, file_name):
        df = pd.read_csv(file_name, parse_dates=['current_dt', 'dt'])
        df = df.drop(columns=['location', 'lat', 'lon'])
        return df

    def get_fx_by_date(self, site_id=0, start=None, end=None, past=True):
        df = super().get_fx_by_date(site_id, start, end, past)
        # Resample to average_interval using mean
        # fx_type = df['type'].unique()[0]
        df = df.drop(columns=['type'])
        if len(df) > 0 and self.average_interval is not None:
            df2 = df.resample(self.average_interval).mean()
        else:
            df2 = df
        return df2
