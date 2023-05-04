import os
import glob
from datetime import timedelta
import sqlalchemy
import pandas as pd
import numpy as np


class DataSet:
    def __init__(self, db_engine=None, db_table_prefix='generic_dataset',
                 resample_interval='5min', average_interval='1h'):
        self.start = None
        self.end = None
        self.db_engine = db_engine
        self.db_table_prefix = db_table_prefix
        self.resample_interval = resample_interval
        self.average_interval = average_interval
        self._define_col_names()

        if db_engine is not None:
            self.meta = sqlalchemy.MetaData()
            self._define_tables()

    def _define_table_names(self):
        # Define table names
        self.db_table_data_raw = self.db_table_prefix + '_data_raw'
        self.db_table_data = self.db_table_prefix + '_data'
        self.db_table_files = self.db_table_prefix + '_files'
        self.db_table_intervals = self.db_table_prefix + '_intervals'

    def _define_col_names(self):
        self.raw_col_name = 'value'
        self.scaled_col_name = 'P_out'

    def _define_tables(self):
        """ Create table definitions in object metadata member.
        Does not connect to database or create tables in the database itself."""
        self._define_table_names()
        self._define_table_data_raw()
        self._define_table_data()
        self._define_table_files()
        self._define_table_intervals()

    def _define_table_data_raw(self):
        self.data_table_raw = sqlalchemy.Table(
            self.db_table_data_raw,
            self.meta,
            sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True),
            sqlalchemy.Column('dt', sqlalchemy.DateTime, index=True),
            sqlalchemy.Column(self.raw_col_name, sqlalchemy.Float)
        )
        return self.data_table_raw

    def _define_table_data(self):
        self.data_table = sqlalchemy.Table(
            self.db_table_data,
            self.meta,
            sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True),
            sqlalchemy.Column('dt', sqlalchemy.DateTime, index=True),
            sqlalchemy.Column(self.raw_col_name, sqlalchemy.Float)
        )
        return self.data_table

    def _define_table_files(self):
        self.files_table = sqlalchemy.Table(
            self.db_table_files,
            self.meta,
            sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True),
            sqlalchemy.Column('fname', sqlalchemy.String)
        )
        return self.files_table

    def _define_table_intervals(self):
        self.intervals_table = sqlalchemy.Table(
            self.db_table_intervals,
            self.meta,
            sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True),
            sqlalchemy.Column('start', sqlalchemy.DateTime),
            sqlalchemy.Column('end', sqlalchemy.DateTime)
        )
        return self.intervals_table

    def import_all_data(self):
        """ Import all available data into the database. Any existing data is dropped from the database first. """
        self.meta.drop_all(self.db_engine)
        self.import_new_data()

    def import_new_data(self):
        """ Import available data that has not already been imported. Existing data is kept. """
        self.meta.create_all(self.db_engine)
        with self.db_engine.connect() as conn:
            existing_file_names = set(conn.execute(sqlalchemy.select(self.files_table.c.fname)).scalars())
        for file_name in self.file_names:
            if file_name in existing_file_names:
                continue
            df = self._read_file(file_name)
            self._import_df_to_db(df, file_name)
        # Remove duplicates
        self._remove_duplicate_rows_all_tables()
        self._rebuild_processed_data()

    def _import_df_to_db(self, df, file_name=None, callback=None):
        """ Add data from DataFrame into database. """
        if not df.empty:
            # Make sure df is sorted
            df = df.sort_index()

            intervals_to_add = [(df.index.min(), df.index.max())]
            intervals_to_add = self._new_intervals_to_add(intervals_to_add)
        else:
            intervals_to_add = []

        with self.db_engine.begin() as conn:
            for new_start, new_end in intervals_to_add:
                idx = (df.index >= new_start) & (df.index <= new_end)
                # Skip intervals with no datapoints in them
                if idx.sum() == 0:
                    continue
                # Add data to data table
                df.loc[idx].to_sql(self.db_table_data_raw, conn, if_exists='append')
                # Add interval to interval table
                conn.execute(sqlalchemy.insert(self.intervals_table).values(start=new_start, end=new_end))

            if callback is not None:
                callback(conn)

            # Add filename to file table
            if file_name is not None:
                conn.execute(sqlalchemy.insert(self.files_table).values(fname=file_name))

    def _new_intervals_to_add(self, intervals_to_add):
        Δt = pd.to_timedelta(self.resample_interval) / 10
        stmt = sqlalchemy.select(self.intervals_table.c.start, self.intervals_table.c.end).where(
            sqlalchemy.and_(intervals_to_add[0][1] >= self.intervals_table.c.start,
                            intervals_to_add[0][0] <= self.intervals_table.c.end)
        )
        with self.db_engine.connect() as conn:
            db_intervals = list(conn.execute(stmt))
        for existing_start, existing_end in db_intervals:
            next_intervals = []
            while intervals_to_add:
                new_start, new_end = intervals_to_add.pop()
                # Check for overlap. Maybe not needed if existing interval query works?
                if new_end < existing_start or new_start > existing_end:
                    next_intervals.append((new_start, new_end))
                    continue
                if new_start < existing_start:
                    next_intervals.append((new_start, existing_start - Δt))
                if new_end > existing_end:
                    next_intervals.append((existing_end + Δt, new_end))
            intervals_to_add = next_intervals
        return intervals_to_add

    def _rebuild_processed_data(self):
        """ Rebuilds table of processed data. Currently reprocessed *all* data.
        Missing data is saved as NaN. """
        # Interpolation step. Initially fill the entire period, even the gaps.
        # Interpolated data is saved in a DataFrame in memory rather than to the db.
        df = self._load_dt_range(self.data_table_raw)
        index = pd.date_range(
            start=df.index[0].ceil(self.resample_interval),
            end=df.index[-1].floor(self.resample_interval),
            freq=self.resample_interval,
            name='dt'
        )
        df2 = interpolate_to_index(df, index)

        # Fill "large" gaps with NaN
        with self.db_engine.connect() as conn:
            db_intervals = list(conn.execute(sqlalchemy.select(self.intervals_table.c.start,
                                                               self.intervals_table.c.end)
                                             .order_by(self.intervals_table.c.start)))

        # Set large gaps between intervals to NaN.
        Δt = index[1] - index[0]
        for (start1, end1), (start2, end2) in zip(db_intervals[:-1], db_intervals[1:]):
            if start2 - end1 > 2*Δt:
                df2.loc[(end1+Δt/10):(start1-Δt/10)] = np.nan

        # Resampling step. Use avg.
        if self.average_interval is not None:
            df3 = df2.resample(self.average_interval).mean()
        else:
            df3 = df2

        # Additional post-processing
        df4 = self._addl_postprocess(df3)

        # Save to database
        with self.db_engine.begin() as conn:
            self.data_table.drop(conn)
            self.data_table.create(conn)
            df4.to_sql(self.db_table_data, conn, if_exists='append')

        return df4

    def _addl_postprocess(self, df):
        return df

    def _load_dt_range(self, table, start=None, end=None):
        """ Return DataFrame with data from specified database table.
        Start and end parameters should be datetime compatible. Start and end
        times are inclusive. (Add or subtract a small Δ to make one or both exclusive. """
        cond = []
        if start is not None:
            cond.append(table.c.dt >= start)
        if start is not None:
            cond.append(table.c.dt <= end)
        stmt = sqlalchemy.select(table)
        if cond:
            stmt = stmt.where(*cond)
        stmt = stmt.order_by(table.c.dt)
        df = pd.read_sql(stmt, self.db_engine, index_col='dt')
        df = df.drop(columns='id')
        # Convert columns from sqlalchemy quoted_name to str. Otherwise sklearn issues a warning.
        df.columns = [str(c) for c in df.columns]
        return df

    @property
    def file_names(self):
        """ Get list of file names in the data directory. """
        return sorted(glob.glob(os.path.join(self.data_dir, '*.csv')))

    def _read_file(self, file_name):
        """ Read an individual data file and return a DataFrame of the data. """
        # Override in child classes.
        raise NotImplementedError

    def _remove_duplicate_rows(self, table):
        cols = [v for k, v in table.columns.items() if k != 'id']
        """ SQL Code
        DELETE from meteogram_data_fx
        WHERE id NOT IN
        (
            SELECT MAX(id) FROM meteogram_data_fx GROUP BY current_dt, dt
        )
        """
        ids_to_keep = sqlalchemy.select(sqlalchemy.sql.func.max(table.c.id).label('max_id')).group_by(*cols)
        stmt = sqlalchemy.delete(table).where(table.c.id.not_in(ids_to_keep))
        with self.db_engine.begin() as conn:
            conn.execute(stmt)

    def _remove_duplicate_rows_all_tables(self):
        for t in [self.data_table_raw]:
            self._remove_duplicate_rows(t)

    @property
    def date_range(self):
        """ Returns the start and end dates of the data set."""
        return self.start, self.end

    def get_data_by_date(self, site_id=0, start=None, end=None):
        """ Returns data beginning on start and going to end (not inclusive).
        start and end may be dates or datetimes. """
        if start is not None:
            start = pd.to_datetime(start)
        if end is not None:
            end = pd.to_datetime(end) - pd.to_timedelta(self.resample_interval)/10  # Open interval on end
        return self._load_dt_range(self.data_table, start, end)

    def get_data_batches(self, site_id, ndays, start=None, end=None,
                         incomplete=True):
        """ Generator that returns chunks of ndays of data. Data is retrieved
        sequentially. Rows for days with missing data are filled with NA.
        The incomplete parameter determines whether the last, incomplete batch
        should be returned or not."""
        if start is None:
            start = self.start
        if end is None:
            end = self.end
        days_available = (end - start).days + 1
        for n in range(days_available//ndays):
            yield self.get_data_by_date(site_id,
                                        start + n*timedelta(days=ndays),
                                        start + (n+1)*timedelta(days=ndays) - timedelta(days=1))
        if incomplete and (start + (days_available//ndays)*timedelta(days=ndays) <= end):
            yield self.get_data_by_date(site_id,
                                        start + (days_available//ndays)*timedelta(days=ndays),
                                        end)


def extract_complete_days(df, expected_interval=None):
    """ Returns a new DataFrame containing only the rows of df  corresponding to
    days without any missing samples."""
    index = df.index
    samples = pd.DataFrame(df.index.to_series().resample('1D').count())
    samples.columns = ['samples']
    samples['Date'] = pd.to_datetime(samples.index.date)
    df['Date'] = pd.to_datetime(df.index.date)
    df2 = pd.merge(df, samples, on='Date')
    df2.set_index(index, inplace=True)
    if expected_interval is None:
        expected_interval = index[1] - index[0]
    expected_samples = timedelta(days=1) // expected_interval
    complete_days = (df2['samples'] == expected_samples)
    df3 = df2.where(complete_days).dropna()
    return df3


def interpolate_to_index(df, index):
    idx_after = np.searchsorted(df.index, index)
    after = df.loc[df.index[idx_after], :].to_numpy()
    before = df.loc[df.index[idx_after - 1], :].to_numpy()
    after_time = df.index[idx_after].to_numpy()
    before_time = df.index[idx_after - 1].to_numpy()
    span = after_time - before_time
    after_weight = (after_time - index.to_numpy()) / span
    before_weight = (index.to_numpy() - before_time) / span
    interpolated_data = (after.T * before_weight + before.T * after_weight).T
    rtn = pd.DataFrame(interpolated_data, index=index, columns=df.columns)
    return rtn
