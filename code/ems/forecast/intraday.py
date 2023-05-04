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
    # I assume that the day-aheaad forecasts were saved with include_actuals=True so I don't have to load those here
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
        model = statsmodels.tsa.ar_model.AutoReg(hist_dayahead_fx['actual_csratio'], lags=lags, exog=hist_dayahead_fx['fx_csratio'],
                                                 old_names=False).fit()
        updated_fx['csratio'] = model.forecast(dayahead_fx.index[-1], exog=dayahead_fx['fx_csratio'])
        updated_fx['forecast'] = inv_csratio(updated_fx['csratio'], dayahead_fx['time_max'])

    elif kind == 'sarimax':
        """
        # Temporary code to test lots of models:
        ps = [0, 1, 2, 3]
        ds = [0, 1]
        qs = [0, 1, 2, 3]
        Ps = [0, 1, 2, 3]
        Ds = [0, 1]
        Qs = [0, 1, 2, 3]
        exogs = [False, True]
        model_list = []
        #max_models = 5
        # Winning model parameters: (1, 0, 2, 0, 0, 0, True)
        for p, d, q, P, D, Q, exog in itertools.product(ps, ds, qs, Ps, Ds, Qs, exogs):
            model = sm.tsa.statespace.SARIMAX(hist['P_out'].fillna(0), exog=hist_dayahead_fx['forecast'] if exog else None,
                                              order=(p, d, q),
                                            seasonal_order=(P, D, Q, 24), enforce_invertibility=False).fit(disp=False)
            aic = model.aic
            print((p, d, q, P, D, Q, exog), aic)
            #if len(model_list) < max_models or (len(model_list) >= max_models and (not model_list or aic < model_list[-1])):
            model_list.append(((p, d, q, P, D, Q, exog), aic))
        model_list.sort(key=lambda m: m[1])
        #model_list = model_list[:max_models]
        print('='*60)
        print('Best models:')
        print(model_list[:30])
        """
        # Run the best one
        p, d, q, P, D, Q, exog = (1, 0, 2, 0, 0, 0, True) #model_list[0][0]
        if len(hist_dayahead_fx) <= p + 4:
            raise ForecastError('Insufficient previous dayahead forecast data for intraday update.')
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
            resid = (hist_dayahead_fx['actual'] - hist_dayahead_fx['forecast']).dropna()
        elif kind == 'fx_csratio':
            resid = (hist_dayahead_fx['actual_csratio'] - hist_dayahead_fx['fx_csratio']).dropna()


        # Fit a model to past residuals
        lags = 2
        if len(resid) <= lags + 3:
            raise ForecastError('Insufficient previous dayahead forecast data for intraday update')
        resid_model = statsmodels.tsa.ar_model.AutoReg(resid, lags=lags, old_names=False).fit()

        # Apply model to current forecast
        correction_AR = resid_model.forecast(fc_start+lookahead)

        # Add correction to day-ahead forecast
        # TODO: Confirm if correction_AR should be ADDED or SUBTRACTED from the forecast???
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


def load_forecast_w_update(location, fc_start, pred_cols, lookback='2w', lookahead='24h'):
    """
    Creates a model on the forecast error and updates the forecast based on recent residuals.

    NOTE: I believe that this function may have been abandoned, so I am not sure that it works at all any more!!

    :param location: Location dict
    :param pred_cols: Columns for which to create an updated forecast
    :param fc_start: Start time of forecast
    :param lookback: Duration of historical data to use
    :param lookahead: Duration of forecast to generate
    :return: DataFrame with updated forecast
    """
    fc_start = pd.to_datetime(fc_start)
    lookback = pd.to_timedelta(lookback)
    lookahead = pd.to_timedelta(lookahead)

    # Load historical actuals and forecasts
    hist = load_hist(location, end=fc_start, duration=lookback).fillna(0)
    fx_hist = load_forecast(location, end=fc_start, duration=lookback, past=False).fillna(0)

    # Load actuals and forecasts for forecast period
    fx_weather = load_forecast(location, start=fc_start, duration=lookahead)
    fx_weather_actual = load_hist(location, start=fc_start, duration=lookahead)

    rtn = pd.DataFrame()

    #pred_cols =  ('ghi',)
    for col in pred_cols:
        if col == 'ghi':
            # Ratio to modeled (specifically for ghi)
            modeled_col = hist[f'modeled_{col}']
            eps = 0.01*(modeled_col.max() - modeled_col.min())
            hist_resid = (hist[col] + eps)/(modeled_col + eps) - (fx_hist[col] + eps)/(modeled_col + eps)
            modeled_col = fx_weather[f'modeled_{col}']
            fx_ratio = (fx_weather[col] + eps) / (modeled_col + eps)
            fx_resid = (fx_weather_actual[col] + eps) / (modeled_col + eps) - fx_ratio

            # Need to concat previous data so that lags are available
            resids = pd.concat([hist[col], fx_weather_actual[col]]) - pd.concat([fx_hist[col], fx_weather[col]])
        else:
            hist_resid = hist[col] - fx_hist[col]
            fx_resid = fx_weather_actual[col] - fx_weather[col]
        # Need to concat previous data so that lags are available
        resids = pd.concat([hist_resid, fx_resid])

        # Build the model based on past data
        lags = 2
        resid_AR = statsmodels.tsa.ar_model.AutoReg(hist_resid, lags=lags, old_names=False).fit()

        # Apply the model to future data
        model = statsmodels.tsa.ar_model.AutoReg(resids, lags=lags, old_names=False)
        correction_AR = model.predict(resid_AR.params, start=fx_weather.index[0], end=fx_weather.index[-1])

        if col == 'ghi':
            fx_ratio2 = fx_ratio + correction_AR
            rtn[col] = fx_ratio2*(modeled_col + eps) - eps
        else:
            rtn[col] = fx_weather[col] + correction_AR

    return rtn
