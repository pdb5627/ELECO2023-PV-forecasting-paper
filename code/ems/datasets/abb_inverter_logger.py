import glob
import os
import json
import pandas as pd
import sqlalchemy
from .base import DataSet, extract_complete_days

"""
This dataset is what was logged from the EEE department rooftop solar system 
beginning in March 2021. The data is extracted from the datalogger included in the
ABB inverter.
"""


class ABBInverterDataSet(DataSet):
    _df_cache = None

    def __init__(self, data_dir='data/ABB_inverter', time_zone='Europe/Istanbul',
                 db_engine=None, db_table='abb_inverter',
                 resample_interval='5min', average_interval='1h', scale_factor=27.8):
        super().__init__(db_engine, db_table, resample_interval, average_interval)
        self.data_dir = data_dir
        self.time_zone = time_zone
        self.scale_factor = scale_factor


        if self._df_cache is None and db_engine is None:
            self.df = self._load_df()
            self._df_cache = self.df
        elif db_engine is None:
            self.df = self._df_cache
        self.ids = [1]

    @property
    def file_names(self):
        """ Get list of file names in the data directory. """
        return sorted(glob.glob(os.path.join(self.data_dir, '*.json')))

    def _define_table_data(self):
        self.data_table = sqlalchemy.Table(
            self.db_table_data,
            self.meta,
            sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True),
            sqlalchemy.Column('dt', sqlalchemy.DateTime, index=True),
            sqlalchemy.Column(self.raw_col_name, sqlalchemy.Float),
            sqlalchemy.Column(self.scaled_col_name, sqlalchemy.Float)
        )
        return self.data_table

    def _define_col_names(self):
        self.raw_col_name = 'value'
        self.scaled_col_name = 'P_out'

    def _load_df(self, complete_days=False):
        df_list = []
        for file_name in self.file_names:
            df = self._read_file(file_name)
            df_list.append(df)
        df = pd.concat(df_list, copy=False).reset_index(drop=False)

        df = df.drop_duplicates(subset='dt')
        df = df.set_index('dt')
        df = df.sort_index()

        #self.scale_factor = df[self.raw_col_name].max()
        df[self.scaled_col_name] = df[self.raw_col_name]/self.scale_factor

        if complete_days:
            df = extract_complete_days(df)
        # First resample to 5min interval to fill in gaps with zeros
        # TODO: FIX ME
        df = df.resample('5min').mean()
        df = df.interpolate(method='slinear', limit=5, limit_area='inside')
        df = df.resample('1H').mean()
        return df

    def _read_file(self, file_name):
        with open(file_name, 'r') as f:
            data = f.read()
        data = json.loads(data)
        [data] = data['feeds'].values()
        df = pd.json_normalize(data, record_path=['datastreams',
                                                  'm103_1_W',
                                                  'data'])
        df['dt'] = pd.to_datetime(df['timestamp'])
        df.set_index(df['dt'], inplace=True)

        # Keep only value column and index
        df = df['value'].to_frame()
        # Rename raw column to specified raw column name
        df.columns = [self.raw_col_name]
        # Sort by dt index. Reverse order first since it comes in oldest to newest
        df = df[::-1]
        df = df.sort_index()

        # Fix time zone. It appears the logger returns a timestamp that says it is UTC but actually is localized
        # This is probably a local configuration error.
        df = df.tz_localize(None)
        df = df.tz_localize(self.time_zone)
        df = df.tz_convert('UTC')
        df = df.tz_localize(None)

        return df

    def _addl_postprocess(self, df):
        """ Add column for scaled output. """
        if self.scale_factor is not None:
            df[self.scaled_col_name] = df[self.raw_col_name] / self.scale_factor
        else:
            df[self.scaled_col_name] = df[self.raw_col_name]
        return df

    def get_data_by_date(self, site_id=0, start=None, end=None):
        """ Returns data beginning on start and going to end (not inclusive).
        start and end may be dates or datetimes. """
        if self.db_engine is None:
            idx = pd.date_range(start=start, end=end, freq='1H',
                                inclusive='left')
            return self.df.loc[idx]
        else:
            if start is not None:
                start = pd.to_datetime(start)
            if end is not None:
                end = pd.to_datetime(end) - pd.to_timedelta(self.resample_interval)/10  # Open interval on end
            return self._load_dt_range(self.data_table, start, end)
