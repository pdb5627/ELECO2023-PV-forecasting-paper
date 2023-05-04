import pandas as pd
import sqlalchemy
from .base import DataSet
from ems.solar_model import clearsky

"""
This "dataset" is imply a way to cache the pvlib modeled output for a given system since these calculations are
somewhat computationally intensive and are reused many times in the PV forecasts in the ems.forecast sub-package.
When reading from the database, does not verify that the current location matches the location of previous calculations.
"""


class ClearskyModel(DataSet):
    _df_cache = None

    def __init__(self, location, db_engine=None, db_table='clearsky_model',
                 resample_interval='1min', average_interval='1h'):
        """ location: dict with location information. Should have members
            'lat', 'lon', 'name', 'elevation', 'tilt', 'azimuth', 'nominal_max_output'."""
        super().__init__(db_engine, db_table, resample_interval, average_interval)
        self.location = location

        self.ids = [1]

    def _define_table_files(self):
        self.files_table = None
        return self.files_table

    def _define_col_names(self):
        self.raw_col_name = 'modeled_output'

    def _define_table_data_raw(self):
        self.data_table_raw = sqlalchemy.Table(
            self.db_table_data_raw,
            self.meta,
            sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True),
            sqlalchemy.Column('dt', sqlalchemy.DateTime, index=True),
            sqlalchemy.Column(self.raw_col_name, sqlalchemy.Float),
            sqlalchemy.Column('effective_irradiance', sqlalchemy.Float),
            sqlalchemy.Column('modeled_ghi', sqlalchemy.Float)
        )
        return self.data_table_raw

    def _define_table_data(self):
        self.data_table = sqlalchemy.Table(
            self.db_table_data,
            self.meta,
            sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True),
            sqlalchemy.Column('dt', sqlalchemy.DateTime, index=True),
            sqlalchemy.Column(self.raw_col_name, sqlalchemy.Float),
            sqlalchemy.Column('effective_irradiance', sqlalchemy.Float),
            sqlalchemy.Column('modeled_ghi', sqlalchemy.Float)
        )
        return self.data_table

    def import_all_data(self, start=None, end=None):
        """ Import all available data into the database. Any existing data is dropped from the database first.
        Generates the clearsky model output from start (inclusive) to end (exclusive). If start and/or end
        are not provided, a window of 6 weeks is provided, centered on the current date if neither is given. """
        self.meta.drop_all(self.db_engine)
        self.import_new_data(start, end)

    def import_new_data(self, start=None, end=None):
        """ Import available data that has not already been imported. Existing data is kept.
        Generates the clearsky model output from start (inclusive) to end (exclusive). If start and/or end
        are not provided, a window of 6 weeks is provided, centered on start of the current date if neither is given. """
        # Set default start and end if none given
        default_window = pd.to_timedelta('6w')
        if start is None and end is None:
            cur_day = pd.Timestamp.today().floor('D')
            start =  cur_day - default_window/2
            end = cur_day + default_window
        elif start is None:
            end = pd.to_datetime(end)
            start = end - default_window
        elif end is None:
            start = pd.to_datetime(start)
            end = start + default_window

        self.meta.create_all(self.db_engine)
        intervals_to_add = self._new_intervals_to_add([(start, end)])
        for new_start, new_end in intervals_to_add:
            index = pd.date_range(start=new_start, end=new_end, freq=self.resample_interval, inclusive='left')
            df_in = pd.DataFrame(index=index)
            df_in.index.name = 'dt'
            # Clearsky model is calculated without temperature being provided
            df = clearsky(self.location, df_in)
            self._import_df_to_db(df)
        if intervals_to_add:
            self._rebuild_processed_data()

    def get_data_by_date(self, site_id=0, start=None, end=None):
        """ Returns data beginning on start and going to end (not inclusive).
        start and end may be dates or datetimes. """
        # If a closed range is given, make sure it is loaded into the dataset if it isn't already
        if start is not None and end is not None:
            self.import_new_data(start, end)
        if start is not None:
            start = pd.to_datetime(start)
        if end is not None:
            end = pd.to_datetime(end) - pd.to_timedelta(self.resample_interval)/10  # Open interval on end

        return self._load_dt_range(self.data_table, start, end)
