import os
import glob
import pandas as pd
import sqlalchemy

from .dataset_with_forecast import DataSetWithForecast


class MeteogramForecast(DataSetWithForecast):
    def __init__(self, data_dir='data/MGM/Meteogram/Ankara_Cankaya_old',
                 db_engine=None, db_table='meteogram',
                 resample_interval='1h', average_interval=None):
        super().__init__(db_engine, db_table, resample_interval, average_interval)
        self.data_dir = data_dir
        self.ids = [1]

    def _define_col_names(self):
        self._actual_columns = []

        self._forecast_columns = [
            #('location', sqlalchemy.Text),
            ('clouds', sqlalchemy.Float),
            ('temperature', sqlalchemy.Float),
            ('rain', sqlalchemy.Float)
        ]

    def _read_file(self, file_name):
        df = pd.read_csv(file_name, parse_dates=['current_dt', 'dt'])
        df = df.drop(columns=['location'])
        df['type'] = 'hourly'
        if df['current_dt'].dt.tz is not None:
            df['current_dt'] = df['current_dt'].dt.tz_convert(None)
        if df['dt'].dt.tz is not None:
            df['dt'] = df['dt'].dt.tz_convert(None)
        return df

    def _rebuild_processed_data(self):
        """ Override raw data processing to skip this step. """
        return
