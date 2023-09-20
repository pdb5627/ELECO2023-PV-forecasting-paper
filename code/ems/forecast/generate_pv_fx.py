import logging
import pandas as pd
from ems.modeling.modeling_window import ModelingWindow
from .dayahead import cs_ratio_forecast, irr_forecast, persistence_fx, generic_fx
from .intraday import intraday_update
from .utils import load_dayahead_fx


logger = logging.getLogger(__name__)


def generate_pv_fx(location, fx_window: ModelingWindow, dayahead_hours=(6,), scaling_factor=1.0, dayahead_method='meteogram',
                   intraday_method='fx_output') -> pd.Series:
    """
    Generate actual PV forecast from historical output and weather forecast.
    Historical actuals are loaded from the database as is clearsky.
    Any forecast method that is called is strictly based on historical actual or weather forecast data as of the start
    time of the forecast.
    Intraday methods are used to update the first day's forecast if the starting time has significant clearsky output.

    :param location: Location dict
    :param fx_window: ModelingWindow with start and end of forecast period and localization info.
    :param dayahead_hours: Hours of the day at which dayahead forecast should be generated. (default=(6,))
    :param dayahead_method: String indicating the day-ahead forecast method to be used. Possible values are as follows:
        'meteogram': Day-ahead forecasts are based on the MGM Meteogram cloudiness forecast. (default)
        'irradiance': Day-ahead forecasts are based on the SolCast irradiance forecast.
        'solcast_clouds': Day-ahead forecasts are based on the SolCast cloudiness forecast.
        'load_from_file': Day-ahead forecasts are pulled from all_dayahead_fx using load_dayahead_fx.
        'persistence': Day-ahead forecasts are a repeat of the previous day.
    :param intraday_method: String indicating the day-ahead forecast method to be used.
        See the `kind` parameter of `ems.forecast.intraday.intraday_update()`.
        Default is 'fx_output'.
    :return: Series with forecast PV values
    """
    start = fx_window.start
    end = fx_window.end
    lookback = '28d'
    if dayahead_method != 'load_from_file' and fx_window.start_localized.hour in dayahead_hours:
        # Generate a new day-ahead forecast
        lookahead = pd.to_datetime(end) - pd.to_datetime(start)
        if dayahead_method == 'meteogram':
            fx_info = cs_ratio_forecast(location, start, lookback=lookback, lookahead=lookahead)
        elif dayahead_method == 'irradiance':
            fx_info = irr_forecast(location, start, lookback=lookback, lookahead=lookahead)
        elif dayahead_method == 'solcast_clouds':
            fx_info = generic_fx(location, 'fx_csratio2', ['clouds'], start, lookback, lookahead, 2, True)
        elif dayahead_method == 'persistence':
            fx_info = persistence_fx(location, start, lookback, '24h', lookahead)
        else:
            raise NotImplementedError(f'Forecast method "{dayahead_method}" not implemented.')
        fx_df = fx_info['forecast_df']['forecast']
        time_max = fx_info['fx_weather']['time_max']  # needed for clearsky
    else:
        # Load previous day-ahead forecast.
        # If generated at least daily and going until the end of whole days, the entire forecast period should have a
        # previously generated forecast.
        fx_df = load_dayahead_fx(None, start, start, end)
        fx_df, time_max = fx_df['forecast'], fx_df['time_max']

        # Apply intraday update to first day if the first hour has significant clearsky output
        during_day = time_max.iloc[0] > 0.1*time_max.max()
        if during_day:
            next_dark_hour = time_max[time_max < 0.01].index[0]
            try:
                fx_intraday = intraday_update(location, start, lookback=lookback, lookahead=next_dark_hour - start,
                                              kind=intraday_method)
                fx_df.update(fx_intraday['forecast'])
            except Exception as e:
                logger.info(f'Intraday update at {start} not applied. {e.args}')
        else:
            logger.info(f'Intraday update at {start} not applied because it is night.')

    return fx_df*scaling_factor
