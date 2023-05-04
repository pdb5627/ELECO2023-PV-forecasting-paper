from datetime import timedelta
from copy import deepcopy
import sqlalchemy
import pandas as pd
import numpy as np

from .base import DataSet

class DataSetWithForecast(DataSet):
    """ This is a generic base class for datasets that include forecast values in addition to
        actual values. It is assumed that the forecast is generated for some future time period(s)
        and that de-duplication of the forecast data is not needed as it is for actuals.
    """
    def __init__(self, db_engine=None, db_table_prefix='generic_dataset',
                 resample_interval='5min', average_interval='1h'):
        super().__init__(db_engine, db_table_prefix, resample_interval, average_interval)

    def _define_table_names(self):
        super()._define_table_names()
        self.db_table_data_fx = self.db_table_prefix + '_data_fx'

    def _define_tables(self):
        """ Create table definitions in object metadata member.
        Does not connect to database or create tables in the database itself."""
        super()._define_tables()
        self._define_table_data_fx()

    def _define_col_names(self):
        # Almost certainly need to override this method in child classes.
        self._actual_columns = [
            ('value', sqlalchemy.Float)
        ]

        self._forecast_columns = [
            ('value', sqlalchemy.Float)
        ]

    def _define_table_data_raw(self):
        if self._actual_columns:
            self.data_table_raw = sqlalchemy.Table(
                self.db_table_data_raw,
                self.meta,
                sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True),
                sqlalchemy.Column('dt', sqlalchemy.DateTime, index=True),
                *[sqlalchemy.Column(*c) for c in self._actual_columns]
            )
        else:
            self.data_table_raw = None
        return self.data_table_raw

    def _define_table_data(self):
        if self._actual_columns:
            self.data_table = sqlalchemy.Table(
                self.db_table_data,
                self.meta,
                sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True),
                sqlalchemy.Column('dt', sqlalchemy.DateTime, index=True),
                *[sqlalchemy.Column(*c) for c in self._actual_columns]
            )
        else:
            self.data_table = None
        return self.data_table

    def _define_table_data_fx(self):
        if self._forecast_columns:
            self.data_table_fx = sqlalchemy.Table(
                self.db_table_data_fx,
                self.meta,
                sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True),
                sqlalchemy.Column('current_dt', sqlalchemy.DateTime, index=True),
                sqlalchemy.Column('dt', sqlalchemy.DateTime, index=True),
                sqlalchemy.Column('type', sqlalchemy.Text),
                *[sqlalchemy.Column(*c) for c in self._forecast_columns]
            )
        else:
            self.data_table_fx = None
        return self.data_table_fx

    @staticmethod
    def _split_actual_fx(df: pd.DataFrame):
        """ Given an input DataFrame with combined actual and forecast data, splits
            the data to return two DataFrames: one for actual, and one for forecast
            data. The input DataFrame is assumed to have a column named 'type' which
            takes the value 0 or '0' for actual values and 'daily', 'hourly', etc. for
            forecasts. The columns 'current_dt' and 'type' are dropped from the DataFrame
            of actual values. Any columns with all NA values are also dropped from both
            DataFrames.
        """
        idx = (df['type'] == 0) | (df['type'] == '0')

        df_actual = df.loc[idx]
        df_actual = df_actual.drop(columns=['current_dt', 'type'])
        df_actual = df_actual.set_index('dt', drop=True)
        df_actual = df_actual.dropna(axis=1, how='all')
        df_actual = df_actual.sort_index()

        df_fx = df.loc[~idx]
        df_fx = df_fx.dropna(axis=1, how='all')

        return df_actual, df_fx

    def _import_df_to_db(self, df, file_name=None, callback=None):
        """ Add data from DataFrame into database. """
        df_actual, df_fx = self._split_actual_fx(df)

        # Create callback function to add forecast to db
        def f(conn):
            # Add forecasts
            df_fx.to_sql(self.db_table_data_fx, conn, index=False, if_exists='append')
            if callback is not None:
                callback(conn)

        # Parent function will add actuals and track filenames and intervals
        # Callback will be called to add the forecasts.
        super()._import_df_to_db(df_actual, file_name, f)

    def _remove_duplicate_rows_all_tables(self):
        for t in [self.data_table_raw, self.data_table_fx]:
            if t is not None:
                self._remove_duplicate_rows(t)

    def get_fx_by_date(self, site_id=0, start=None, end=None, past=True):
        """ Returns data beginning on start and going to end (not inclusive).
        start and end may be dates or datetimes.
        If past is True, then weather forecasts will not be generated after the beginning of the forecast period.
        """
        if start is not None:
            start = pd.to_datetime(start)
        if end is not None:
            end = pd.to_datetime(end)

        table = self.data_table_fx
        cond = []
        if start is not None:
            cond.append(table.c.dt >= start)
        if start is not None:
            cond.append(table.c.dt < end)
        if past and start is not None:
            cond.append(table.c.current_dt < start)

        # Pandas "query" code for getting most recent forecast:
        # fc_irr = fc_irr.loc[fc_irr.groupby(['dt'])['current_dt'].idxmax()]
        idxmax = sqlalchemy.select(table.c.dt, sqlalchemy.sql.func.max(table.c.current_dt).label('max_dt'))\
            .where(*cond).group_by(table.c.dt).cte()
        stmt = sqlalchemy.select(table).\
            select_from(table.join(idxmax,
                                   sqlalchemy.and_(idxmax.c.max_dt == table.c.current_dt,
                                                   idxmax.c.dt == table.c.dt)))\
            .order_by(table.c.dt)
        df = pd.read_sql(stmt, self.db_engine, index_col='dt')
        df = df.drop(columns='id')
        # Convert columns from sqlalchemy quoted_name to str. Otherwise sklearn issues a warning.
        df.columns = [str(c) for c in df.columns]
        return df
