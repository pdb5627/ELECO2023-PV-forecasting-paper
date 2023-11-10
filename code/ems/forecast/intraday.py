"""
Intra-day forecast update routines

"""
import pandas as pd
import numpy as np
import statsmodels
import statsmodels.api as sm
from .utils import load_dayahead_fx, load_hist, load_forecast, calc_csratio, inv_csratio, ForecastError
import logging

logger = logging.getLogger(__name__)


def intraday_update(location, fc_start, lookback='2w', lookahead='24h', kind='fx_output'):
    # Ensure we have datetimes
    fc_start = pd.to_datetime(fc_start)
    lookback = pd.to_timedelta(lookback)
    lookahead = pd.to_timedelta(lookahead)

    # Load day-ahead forecasts
    # I assume that the day-ahead forecasts were saved with include_actuals=True so I don't have to load those here
    hist_dayahead_fx = load_dayahead_fx(location, fc_start, end=fc_start, duration=lookback)
    if hist_dayahead_fx.empty:
        raise ForecastError('No historical day-ahead forecasts available')
    end_of_hist = hist_dayahead_fx.index[-1]
    # TODO: Add the time resolution to the series so it doesn't have to be hardcoded here.
    delta_t = pd.to_timedelta('1h')
    if fc_start - end_of_hist > delta_t:
        raise ForecastError(f'Historical data is too old. Last actuals are from {end_of_hist}')
    dayahead_fx = load_dayahead_fx(location, fc_start, start=fc_start, duration=lookahead)
    if dayahead_fx.empty:
        raise ForecastError('No current day-ahead forecast available')
    end_of_fx = dayahead_fx.index[-1]
    if fc_start + lookahead - end_of_fx > delta_t:
        logger.warning(f'Day-ahead forecast does not cover the desired period. Last forecast is for {end_of_fx}')

    updated_fx = pd.DataFrame(index=dayahead_fx.index, columns=['forecast', 'csratio'], dtype=float)

    # Persistence method
    if kind == 'persistence':
        updated_fx.loc[:, 'csratio'] = hist_dayahead_fx.at[end_of_hist, 'actual_csratio']
        updated_fx['forecast'] = inv_csratio(updated_fx['csratio'], dayahead_fx['time_max'])

    # Exogenous variable method
    elif kind =='exog':
        lags = 2
        if len(hist_dayahead_fx) <= lags + 4:
            raise ForecastError('Insufficient previous dayahead forecast data for intraday update.')
        hist_dayahead_fx = hist_dayahead_fx.asfreq('1h')

        p, d, q, P, D, Q, exog = (lags, 0, 0, 0, 0, 0, hist_dayahead_fx['fx_csratio'])
        model = sm.tsa.statespace.SARIMAX(hist_dayahead_fx['actual_csratio'], exog=exog,
                                          order=(p, d, q),
                                          seasonal_order=(P, D, Q, 24), enforce_invertibility=False).fit(disp=False)

        updated_fx['csratio'] = model.forecast(dayahead_fx.index[-1], exog=dayahead_fx['fx_csratio'])
        updated_fx['forecast'] = inv_csratio(updated_fx['csratio'], dayahead_fx['time_max'])

    elif kind == 'sarimax':
        p, d, q, P, D, Q, exog = (1, 0, 2, 0, 0, 0, True)
        if len(hist_dayahead_fx) <= p + 4:
            raise ForecastError('Insufficient previous dayahead forecast data for intraday update.')
        hist_dayahead_fx = hist_dayahead_fx.asfreq('1h')
        model = sm.tsa.statespace.SARIMAX(hist_dayahead_fx['actual'], exog=hist_dayahead_fx['forecast'] if exog else None,
                                          order=(p, d, q),
                                          seasonal_order=(P, D, Q, 24), enforce_invertibility=False).fit(disp=False)
        updated_fx['forecast'] = model.forecast(dayahead_fx.index[-1], exog=dayahead_fx['forecast'] if exog else None).clip(0)
        updated_fx['csratio'] = calc_csratio(updated_fx['forecast'], dayahead_fx['time_max'])

    elif kind == 'scaling':
        scale_factor = hist_dayahead_fx.loc[hist_dayahead_fx.index[-1], 'actual']/(hist_dayahead_fx.loc[hist_dayahead_fx.index[-1], 'forecast'] + 1e-3)
        updated_fx['forecast'] = scale_factor*dayahead_fx['forecast']
        updated_fx['csratio'] = calc_csratio(updated_fx['forecast'], dayahead_fx['time_max'])

    # Calculate residuals
    elif kind in {'fx_output', 'fx_csratio'}:
        if kind == 'fx_output':
            resid = (hist_dayahead_fx['actual'] - hist_dayahead_fx['forecast']) #.dropna()
        elif kind == 'fx_csratio':
            resid = (hist_dayahead_fx['actual_csratio'] - hist_dayahead_fx['fx_csratio']) #.dropna()


        # Fit a model to past residuals
        lags = 2
        if len(resid) <= lags + 3:
            raise ForecastError('Insufficient previous dayahead forecast data for intraday update')
        resid = resid.asfreq('1h')
        p, d, q, P, D, Q, exog = (lags, 0, 0, 0, 0, 0, False)
        model = sm.tsa.statespace.SARIMAX(resid,
                                          order=(p, d, q),
                                          seasonal_order=(P, D, Q, 24), enforce_invertibility=False).fit(disp=False)

        # Apply model to current forecast
        correction_AR = model.forecast(fc_start+lookahead)

        # Add correction to day-ahead forecast
        if kind == 'fx_output':
            updated_fx['forecast'] = (dayahead_fx['forecast'] + correction_AR).clip(0)
            updated_fx['csratio'] = calc_csratio(updated_fx['forecast'], dayahead_fx['time_max'])
        elif kind == 'fx_csratio':
            updated_fx['csratio'] = (dayahead_fx['fx_csratio'] + correction_AR).clip(0)
            updated_fx['forecast'] = inv_csratio(updated_fx['csratio'], dayahead_fx['time_max'])

    else:
        raise ValueError(f"Unknown kind value '{kind}'")

    # Force all forecasts to be between 0 and time_max
    updated_fx['forecast'] = updated_fx['forecast'].clip(0, dayahead_fx['time_max'])
    updated_fx['csratio'] = calc_csratio(updated_fx['forecast'], dayahead_fx['time_max'])

    return updated_fx
