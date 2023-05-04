"""
This file includes functions that are basic and useful to the various forecasting functions.

"""
import pandas as pd
import numpy as np
import sqlalchemy
from typing import Union
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from ems.datasets import ABBInverterDataSet, ClearskyModel, SolCastWeather, MeteogramForecast

# TODO: Refactor this into a function or object or something
engine = sqlalchemy.create_engine("sqlite+pysqlite:///../data/datasets.sqlite", echo=False)


class ForecastError(Exception):
    """ A problem occurred with generating the forecast and no forecast could be created. """


def refresh_data():
    ABBInverterDataSet(db_engine=engine).import_new_data()
    SolCastWeather(db_engine=engine).import_new_data()
    MeteogramForecast(db_engine=engine).import_new_data()


def get_start_end(start=None, end=None, duration=None):
    if start is None and end and duration:
        start = pd.to_datetime(end) - pd.to_timedelta(duration)
        end = pd.to_datetime(end)
    elif end is None and start and duration:
        start = pd.to_datetime(start)
        end = pd.to_datetime(start) + pd.to_timedelta(duration)
    return start, end


def load_hist(location, start=None, end=None, duration=None, ensure_cols=None):
    """
    Load the historical values for the various data sources.
    Two out of the three date-specifying parameters should be given.
    :param location: Location dict
    :param start: Starting date in format suitable for Pandas to_datetime
    :param end: Ending date in format suitable for Pandas to_datetime.
        Note that dates without times will have the time set to midnight (i.e.
        beginning of the day)
    :param duration: Duration of range in format suitable for Pandas to_timedelta
    :param ensure_cols: Optional list of column names to ensure are in the returned DataFrame. Load from available
        forecasts if necessary.
    :return: DataFrame with joined historical data, resampled to common 1h interval
    """
    start, end = get_start_end(start, end, duration)

    # Load SolCast estimated actual data
    ds = SolCastWeather(db_engine=engine)
    actual = ds.get_data_by_date(start=start, end=end)

    # Load actual output (using database)
    ds = ABBInverterDataSet(db_engine=engine)
    actual_output = ds.get_data_by_date(start=start, end=end)

    # Load "clearsky" model results
    cs_ds = ClearskyModel(location, db_engine=engine)
    clearsky = cs_ds.get_data_by_date(start=start, end=end)

    # Merge dataframes
    hist = pd.merge(actual, actual_output, left_index=True, right_index=True, how='outer')
    hist = pd.merge(hist, clearsky, left_index=True, right_index=True, how='outer')

    if ensure_cols is not None:
        actuals_fx = None
        for col in ensure_cols:
            # If the column is not in the historical database, use the latest forecast
            if col not in hist.columns:
                if actuals_fx is None:
                    actuals_fx = load_forecast(location, start=start, end=end, past=False)
                hist[col] = actuals_fx[col]

    return hist


def load_forecast(location, start=None, end=None, duration=None, past=True):
    """
    Load the most recent forecast values for the various data sources.
    Two out of the three date-specifying parameters should be given.
    :param location: Location dict
    :param start: Starting date in format suitable for Pandas to_datetime
    :param end: Ending date in format suitable for Pandas to_datetime.
        Note that dates without times will have the time set to midnight (i.e.
        beginning of the day)
    :param duration: Duration of range in format suitable for Pandas to_timedelta
    :param past: True if weather forecast should not be from after the beginning of the forecast period
    :return: DataFrame with joined historical data, resampled to common 1h interval
    """
    start, end = get_start_end(start, end, duration)

    # Load SolCast forecast data
    ds = SolCastWeather(db_engine=engine)
    fc_irr = ds.get_fx_by_date(start=start, end=end)

    # Load "clearsky" model results
    cs_ds = ClearskyModel(location, db_engine=engine)
    clearsky = cs_ds.get_data_by_date(start=start, end=end)

    # Load Meteogram fx data
    ds = MeteogramForecast(db_engine=engine)
    # Get any forecast available, even if generated later than the start time.
    df_meteogram = ds.get_fx_by_date(start=start, end=end, past=past)
    df_meteogram = df_meteogram.drop(columns=['current_dt', 'type'])
    df_meteogram = df_meteogram.rename(columns={'clouds': 'meteogram_clouds',
                        'temperature': 'meteogram_temperature',
                        'rain': 'meteogram_rain'})
    df_meteogram['meteogram_clouds'] *= 100

    # Merge dataframes
    forecast = pd.merge(fc_irr, clearsky, left_index=True, right_index=True, how='outer')
    forecast = pd.merge(forecast, df_meteogram, left_index=True, right_index=True, how='outer')

    return forecast


all_dayahead_fx = None


def save_dayahead_fx(fx_info: dict, include_actual=True):
    global all_dayahead_fx
    # For now save all the columns. If it turns out to be a memory problem, we can get just the columns we need.
    fx_df = pd.merge(fx_info['forecast_df'], fx_info['fx_weather'], left_index=True, right_index=True)

    if include_actual:
        # Load actuals if they aren't already provided
        if 'actuals' in fx_info:
            actuals = fx_info['actuals']
        else:
            actuals = load_hist(fx_info['location'], start=fx_info['fc_start'], duration=fx_info['lookahead'],
                                ensure_cols=fx_info['pred_cols'])
        fx_df['actual'] = actuals['P_out'].fillna(0)
        fx_df['actual_csratio'] = calc_csratio(fx_df['actual'], fx_df['time_max'])
        fx_df['actual_ghi'] = actuals['ghi']

    fx_df = fx_df.reset_index()
    fx_df['type'] = 'hourly'
    fx_df['current_dt'] = fx_info['fc_start']

    if all_dayahead_fx is None:
        all_dayahead_fx = fx_df
    else:
        all_dayahead_fx = pd.concat([all_dayahead_fx, fx_df], axis=0, ignore_index=True)


def load_dayahead_fx(location, fc_start, start=None, end=None, duration=None):
    """
    Load the previously calculated day-ahead forecast
    Two out of the three date-specifying parameters should be given.
    :param location: Location dict
    :param start: Starting date in format suitable for Pandas to_datetime
    :param end: Ending date in format suitable for Pandas to_datetime.
        Note that dates without times will have the time set to midnight (i.e.
        beginning of the day)
    :param duration: Duration of range in format suitable for Pandas to_timedelta
    :return: DataFrame with joined historical data, resampled to common 1h interval
    """
    # TODO: Load dayahead forecasts from a database or something and filter on location
    global all_dayahead_fx
    if all_dayahead_fx is None:
        raise ForecastError('No day-ahead forecast data is available.')

    start, end = get_start_end(start, end, duration)

    # I assume that day-ahead forecasts are generated every 24h at a consistent time of day
    # The forecast for the first period must have been generated prior to the start time, up to 24h ahead
    #da_period = pd.to_timedelta('24h')

    # Load SolCast estimated actual data
    df = all_dayahead_fx

    # Extract latest day-ahead forecast
    # Get rows in desired date range
    df = df.loc[(df['dt'] >= start) & (df['dt'] < end)]
    # Forecast should not be from after the beginning of the forecast period
    df = df.loc[df['current_dt'] <= fc_start]
    # Get latest available forecast (maximum current_dt)
    df = df.loc[df.groupby(['dt'])['current_dt'].idxmax()]

    df = df.set_index('dt')

    return df


def calc_csratio(P_out, clearsky, eps_pu=0.01, eps_ratio=0.5):
    eps = eps_pu * (clearsky.max() - clearsky.min())
    return (P_out + eps_ratio*eps)/(clearsky + eps)


def inv_csratio(csratio, clearsky, eps_pu=0.01, eps_ratio=0.5):
    eps = eps_pu * (clearsky.max() - clearsky.min())
    return csratio * (clearsky + eps) - eps_ratio*eps


def aggregate_by_time(series: pd.Series, aggfunc='max'):
    """ Create a pivot table of a series with a datetime index on a regular interval using the specified
    aggregation function. """
    df = pd.DataFrame({series.name : series, 'HourMinute': 100*series.index.hour + series.index.minute}, index=series.index)
    table = df.pivot_table(index='HourMinute', aggfunc=aggfunc).squeeze()
    return table


def apply_time_mapping(time_index: Union[pd.DatetimeIndex, pd.DataFrame, pd.Series],
                       aggregated_table: pd.Series):
    """ Apply the aggregated data table from `aggregate_by_time` to a DatetimeIndex.
    Example usages:
        fx_weather['time_max'] = apply_time_mapping(fx_idx, time_max)
        fx_weather['time_max'] = apply_time_mapping(fx_weather, time_max)
    """
    if not isinstance(time_index, pd.DatetimeIndex) and hasattr(time_index, 'index'):
        time_index = time_index.index
    if not isinstance(time_index, pd.DatetimeIndex):
        raise TypeError('time_index should be a DatetimeIndex or have an index member that is one.')
    return pd.Series(100*time_index.hour + time_index.minute, index=time_index).map(aggregated_table)


def label_by_weather(grouping_col: pd.Series, hist: pd.DataFrame, breakpoints=(0.85,), labels=('cloudy', 'clear'),
                     norm_by_time_max=True, group_by_day=False, fig=None):
    if norm_by_time_max:
        time_max = aggregate_by_time(grouping_col, 'max')
        time_max = pd.Series(hist.index.time, index=hist.index, name='Time').map(time_max)
        grouping_col = grouping_col / time_max

    # For temporary data exploration purposes, do both daily and separate hourly classifications
    # Daily classification
    date_group = grouping_col.groupby(grouping_col.index.date)
    date_mean = date_group.aggregate('mean')

    # Hourly classification
    time_group = grouping_col.groupby(grouping_col.index.time)

    # Optionally create plot for examining the distribution
    if fig is not None:
        fig.clear()
        num_plots = len(time_group) + 1
        fig.subplots((num_plots - 1) // 3 + 1, 3)
        fig.tight_layout()

        n_ax = 0
        ax = fig.axes[n_ax]
        normed_txt = ('not ' if not norm_by_time_max else '') + 'normed by time max'
        date_mean.plot(ax=ax, kind='hist', title=f'Daily Mean {grouping_col.name}, {normed_txt}')

        for time, data in time_group:
            n_ax += 1
            ax = fig.axes[n_ax]
            data.plot(ax=ax, kind='hist', title=f'Hourly {grouping_col.name} at {time}, {normed_txt}')

    cuts = [float('-inf'), *breakpoints, float('inf')]

    if group_by_day:
        category = pd.cut(date_mean, cuts, labels=labels)
        weather_label = pd.Series(grouping_col.index.date, index=grouping_col.index, name='Time').map(category)
    else:
        weather_label = pd.cut(grouping_col, cuts, labels=labels)

    return (weather_label, fig) if fig else weather_label


def mean_bias_error(y_pred, y_true, *, sample_weight=None):
    output_errors = np.average(y_pred - y_true, weights=sample_weight, axis=0)
    return np.average(output_errors)


def root_mean_square_error(y_pred, y_true):
    return mean_squared_error(y_true, y_pred, squared=False)

def skewness_plot(series, ax, dist_txt=None):
    """
    Plots a series using the histogram plot with kde and plots, mean, mode, and median lines.
    *** Dependencies ***
    Series must be a pandas.Series
    Seaborn must be imported as sns
    matplotlib.pyplot must be imported as plt
    """
    ## handy multi-plot function for showing mode, median, and mean lines in a distplot
    ## (but using histplot since distplot is deprecated)
    ## Author - Abram Flansburg
    ## Intended for use in Jupyter / iPython
    ## https://gist.github.com/aflansburg/f576d29dd510e20b5fa421ddad638136
    #sns.set_style("whitegrid", {'axes.grid' : False})

    #plt.title(series.name)


    # Implementing own kde plot because I want to be able to evaluate it
    # np.nanmax() and np.nanmin() ignores the missing values
    sample_range = np.nanmax(series) - np.nanmin(series)
    ind = np.linspace(
        np.nanmin(series) - 0.5 * sample_range,
        np.nanmax(series) + 0.5 * sample_range,
        1000,
    )
    gkde = gaussian_kde(series, bw_method=None)
    y = gkde.evaluate(ind)
    [ln] = ax.plot(ind, y)

    #series.plot(ax=ax, kind='density')
    marks_at = series.mean()
    mark_ht = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15
    mark_ymid = gkde.evaluate(marks_at)
    mark_ymin = mark_ymid - mark_ht / 2
    mark_ymax = mark_ymid + mark_ht / 2
    mean_ln = ax.vlines(marks_at, mark_ymin, mark_ymax, linestyles='dotted', color=ln.get_color())

    marks_at = series.quantile([0.5, 0.25, 0.75])
    mark_ymid = gkde.evaluate(marks_at)
    mark_ymin = mark_ymid - mark_ht/2
    mark_ymax = mark_ymid + mark_ht/2
    median_lns = ax.vlines(marks_at, mark_ymin, mark_ymax, linestyles='dashed', color=ln.get_color())

    """
    ax.axvline(series.mean(), color='red')
    ax.axvline(series.median(), color='green', linestyle="dotted")

    q1_values = series.quantile([0.25, 0.75])
    ax.axvline(q1_values.iloc[0], color='green', linestyle="dashed")
    ax.axvline(q1_values.iloc[1], color='green', linestyle="dashed")
    """
    # Make room for legend below axes
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
    # Check for existing legend (e.g. from another plot)
    existing_legend = ax.get_legend()
    plt.legend([ln, mean_ln, median_lns],
        ["Distribution" if dist_txt is None else dist_txt,
                f"Mean = {series.mean():.3g}",
                f"Median = {marks_at.iloc[0]:.3g}\n"
                f"Q1 = {marks_at.iloc[1]:.3g}\n"
                f"Q3 = {marks_at.iloc[2]:.3g}\n"],
               loc='upper left' if existing_legend is None else 'upper right',
               bbox_to_anchor=(0.0, -0.05) if existing_legend is None else (1.0, -0.05))
    if existing_legend is not None:
        ax.add_artist(existing_legend)
