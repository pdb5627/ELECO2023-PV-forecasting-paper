import sys
from datetime import timedelta
import ems.datasets as datasets
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def pivot_df(df):
    """ Pivot dataframe around hours """

    df = df.resample('1H').mean()
    df['Date'] = df.index.date
    df['Hour'] = df.index.hour
    pivoted = df.pivot('Date', 'Hour')
    return pivoted

def forecast(df):
    """
    Forecasts the following 24 hours based on the history passed in.

    :param df: Historical output on an hourly basis.
    :return: DataFrame with hourly forecast, max, and min bands.
    """
    # Use only P_out column, delete all the rest
    df = df['P_out'].to_frame()
    pivoted = pivot_df(df)
    pivoted.columns = pivoted.columns.droplevel()
    forecast_df = pd.DataFrame()
    nroll = -(df.index[-1].hour + 1) % 24
    forecast_df['forecast'] = np.roll(pivoted.median().transpose(), nroll)
    stddev = pivoted.std().transpose()
    forecast_df['min'] = np.roll(pivoted.quantile(0.1).transpose(), nroll)
    forecast_df['max'] = np.roll(pivoted.max().transpose(), nroll)
    forecast_df.index = pd.date_range(start=df.index[-1] + timedelta(hours=1),
                                      periods=24,
                                      freq='1H')

    return forecast_df


def forecast_intraday(df):
    """
    Forecasts the remaining hours of the day based on the history passed in.

    :param df: Historical output on an hourly basis.
    :return: DataFrame with hourly forecast, max, and min bands.
    """
    # Use only P_out column, delete all the rest
    df = df['P_out'].to_frame()
    pivoted = pivot_df(df)
    pivoted.columns = pivoted.columns.droplevel()
    prev_hours = ~pivoted.iloc[-1].isna()
    forecast_hours = ~prev_hours
    rank = pivoted.loc[:, prev_hours].sum(axis=1).rank()[-1]
    percentile = rank / (prev_hours.sum() - 1)

    forecast_df = pd.DataFrame()
    forecast_df['forecast'] = pivoted.quantile(percentile).transpose()[forecast_hours]
    stddev = pivoted.std().transpose()
    forecast_df['min'] = pivoted.quantile(max(0, percentile-0.25)).transpose()[forecast_hours]
    forecast_df['max'] = pivoted.quantile(min(1, percentile+0.25)).transpose()[forecast_hours]
    forecast_df.index = pd.date_range(start=df.index[-1] + timedelta(hours=1),
                                      periods=forecast_hours.sum(),
                                      freq='1H')
    return forecast_df


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    ds = datasets.SolCastPVDataSet()
    df = ds.get_data_by_date(site_id=1)
    return 0

    # Daily Forecast
    ds = datasets.DBlockDataSet()
    model_df = ds.get_data_by_date(start='2018-08-17', end='2018-08-31')
    forecast_df = forecast(model_df)
    actual = ds.get_data_by_date(start='2018-08-31', end='2018-09-01')
    plt.figure(1)
    forecast_df.plot(ax=plt.gca())
    actual['P_out'].plot(ax=plt.gca(), legend='Actual')
    plt.show()

    # Intra-day forecast
    model_df = ds.get_data_by_date(start='2018-08-17', end='2018-08-31 09:00')
    forecast_df = forecast_intraday(model_df)
    actual = ds.get_data_by_date(start='2018-08-31 09:00', end='2018-09-01')
    plt.figure(2)
    forecast_df.plot(ax=plt.gca())
    actual['P_out'].plot(ax=plt.gca(), legend='Actual')
    plt.show()

    df = list(ds.get_data_batches(1, 14, incomplete=False))
    print(df)

    return 0


if __name__ == '__main__':
    sys.exit(main())
