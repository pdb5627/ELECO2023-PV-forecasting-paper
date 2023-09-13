"""
Day-ahead forecast routines

"""
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
from sklearn.preprocessing import PolynomialFeatures  # StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import pandas as pd
from typing import Optional
from functools import partial
import itertools

# from .bayes_model import LinearModel, TransformedTargetPYMCRegressor
from .utils import load_hist, load_forecast, calc_csratio, inv_csratio, save_dayahead_fx, \
    aggregate_by_time, apply_time_mapping, label_by_weather, root_mean_square_error


def irr_forecast(location, fc_start, lookback='2w', lookahead='24h'):
    return generic_fx(location, 'fx_output', ['ghi'], fc_start, lookback, lookahead, 1, False)


def cs_ratio_forecast(location, fc_start, lookback='2w', lookahead='24h', cs_kind='model'):
    """
    Clear-sky ratio-based forecast. Uses meteogram-cloudiness as the weather forecast.
    :param location: Location dict
    :param fc_start: Start time of forecast
    :param lookback: Duration of historical data to use
    :param lookahead: Duration of forecast to generate
    :param cs_kind: Two options:
            'model': Uses a pvlib model for the clearsky reference
            'max': Uses the max in that time period during the lookback window as a clearsky reference
    :return: dict of local variables from forecasting function. See generic_fx for more information.
    """
    if cs_kind == 'model':
        return generic_fx(location, 'fx_csratio', ['meteogram_clouds'], fc_start, lookback, lookahead, 2, True)
    elif cs_kind == 'max':
        return generic_fx(location, 'fx_csratio2', ['meteogram_clouds'], fc_start, lookback, lookahead, 2, True)
    else:
        raise NotImplementedError(f'Clearsky kind (cs_kind) parameter value "{cs_kind}" not implemented.')


def generic_fx(location, kind, pred_cols, fc_start, lookback='2w', lookahead='24h', model_order=1, fit_intercept=True):
    """
    Create hourly forecast using SolCast irradiance forecast
    :param location: Location dict
    :param kind: Kind of forecast calculation to perform.
        'fx_output': regress input prediction columns  to output PV
        'fx_csratio': regress input prediction columns to clear sky ratio (clear sky from pvlib model)
        'fx_csratio2': regress input prediction columns to clear sky ratio (clear sky from max at given time interval)
        'analytical': load irradiation and temperature data to analytical model and then regress to output PV
                      (not yet implemented)
    :param pred_cols: columns to use for regression & forecast
    :param fc_start: Start time of forecast
    :param lookback: Duration of historical data to use
    :param lookahead: Duration of forecast to generate
    :param model_order: Order of polynomial regression to apply. (default = 1)
    :param fit_intercept: Whether to fit constant term (i.e. intercept) of regression model. (default = True)
    :return: dict of local variables from forecasting function.
    """
    hist = load_hist(location, end=fc_start, duration=lookback)

    # If the pred_col is not in the historical database, load the most recent available forecasts
    hist_fx = None
    for pred_col in pred_cols:
        if pred_col not in hist.columns:
            if hist_fx is None:
                hist_fx = load_forecast(location, end=fc_start, duration=lookback, past=False)
            hist[pred_col] = hist_fx[pred_col]

    # Regression model will not cope with NaN. Replace with 0's.
    # This is trouble if NA means no data rather than 0
    hist['P_out'] = hist['P_out'].fillna(0)
    hist[pred_cols] = hist[pred_cols].fillna(0)
    time_max = aggregate_by_time(hist['P_out'])
    hist['time_max'] = apply_time_mapping(hist, time_max)
    if kind == 'fx_csratio':
        cs_col = 'modeled_output'
    else:
        cs_col = 'time_max'
    hist['csratio'] = calc_csratio(hist['P_out'], hist[cs_col])
    # First keep only rows with "significant" sunlight
    # Use csratio to filter out snowcover or other problems
    hist = hist.loc[(hist['ghi'] >= 0.01 * 800) & (hist['P_out'] >= 0.03) & (hist['csratio'] >= 0.1)]

    # Fit regression to lookback data
    model = Pipeline([('poly', PolynomialFeatures(degree=model_order, include_bias=False)),
                      ('linear', LinearRegression(fit_intercept=fit_intercept))])

    # Bayesian linear regression for experimentation
    prior_params = {
        "intercept": {"loc": 0, "scale": 2},
        "slope": {"loc": 0, "scale": 2},
        "obs_error": 1,
    }
    sampler_config = {
        "draws": 1_000,
        "tune": 1_000,
        "chains": 3,
        "target_accept": 0.95,
    }
    # model2 = Pipeline([
    #     ('input_scaling', StandardScaler()),
    #     ('linear_model', TransformedTargetPYMCRegressor(LinearModel(prior_params, sampler_config),
    #                                                     transformer=StandardScaler()))])

    if kind == 'fx_output':
        model.fit(hist[pred_cols], hist['P_out'])
        with pd.option_context('mode.chained_assignment', None):
            hist['P_out_pred'] = model.predict(hist[pred_cols])
            hist['P_out_resid'] = hist['P_out_pred'] - hist['P_out']
        regression_r2 = r2_score(hist["P_out"], hist["P_out_pred"])

        # model2.fit(hist[pred_cols], hist['P_out'])

    elif kind in {'fx_csratio', 'fx_csratio2'}:
        model.fit(hist[pred_cols], hist['csratio'])
        with pd.option_context('mode.chained_assignment', None):
            hist['csratio_pred'] = model.predict(hist[pred_cols])
            hist['csratio_resid'] = hist['csratio_pred'] - hist['csratio']
            hist['P_out_pred'] = inv_csratio(hist['csratio_pred'], hist[cs_col])
            hist['P_out_resid'] = hist['P_out_pred'] - hist['P_out']
        regression_r2 = r2_score(hist["csratio"], hist["csratio_pred"])

        # model2.fit(hist[pred_cols[0]], hist['csratio'])

    elif kind == 'analytical':
        raise NotImplementedError(f'Kind value ''{kind}'' not yet implemented')
    else:
        raise ValueError(f'Unknown kind value ''{kind}''')

    # Load weather fx_weather data with updates
    fx_weather = load_forecast(location, start=fc_start, duration=lookahead)
    fx_weather['time_max'] = apply_time_mapping(fx_weather, time_max)
    # TODO: How to handle situation if forecast data does not cover the desired duration?? What about interpolation?

    # Apply regression model to forecast weather
    forecast_df = pd.DataFrame(index=fx_weather.index, columns=['forecast', 'fx_csratio'], dtype=float)

    if kind == 'fx_output':
        forecast_df['forecast'] = model.predict(fx_weather[pred_cols])
        # Force all forecasts to be between 0 and time_max
        forecast_df['forecast'] = forecast_df['forecast'].clip(0, fx_weather['time_max'])
        forecast_df['fx_csratio'] = calc_csratio(forecast_df['forecast'], fx_weather[cs_col])

        # forecast_df['forecast_point'] = model2.predict(fx_weather[pred_cols])
        # forecast_df['forecast_point'] = forecast_df['forecast_point'].clip(0, fx_weather['time_max'])

        # forecast_samples = model2.predict_proba(fx_weather[pred_cols])
    elif kind in {'fx_csratio', 'fx_csratio2'}:
        forecast_df['fx_csratio'] = model.predict(fx_weather[pred_cols]).clip(0, 1)
        forecast_df['forecast'] = inv_csratio(forecast_df['fx_csratio'], fx_weather[cs_col])

        # forecast_point = model2.predict(fx_weather[pred_cols])
        # forecast_samples = model2.predict_proba(fx_weather[pred_cols])

    elif kind == 'analytical':
        raise NotImplementedError(f'Kind value ''{kind}'' not yet implemented')

    # Add distribution of residuals to the forecast to represent the uncertainty
    # Residual at each time period is divided into two weather groupings and the residuals of the matching
    # group are returned as the distribution
    norm_by_time_max = (pred_cols[0] == 'ghi')
    hist_weather_label = label_by_weather(hist[pred_cols[0]], hist, norm_by_time_max=norm_by_time_max,
                                          group_by_day=True)
    fx_weather_label = label_by_weather(fx_weather[pred_cols[0]], hist, norm_by_time_max=norm_by_time_max,
                                        group_by_day=True)

    grouped_hist = hist['P_out_resid'].groupby([hist.index.time, hist_weather_label])
    agg_args = {f'q{q:0.2f}': partial(np.quantile, q=q) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}
    residual_quantiles = grouped_hist.agg(**agg_args)
    fx_time_label_df = pd.DataFrame({'time': fx_weather.index.time, 'label': fx_weather_label}, index=fx_weather.index)
    fx_residual_quantiles = pd.merge(fx_time_label_df, residual_quantiles, how="left", left_on=['time', 'label'],
                                     right_index=True)

    # Save variables for plotting (or other use)
    forecast_info = locals()

    save_dayahead_fx(forecast_info)

    return forecast_info #pd.merge(forecast_df, fx_weather, left_index=True, right_index=True)


def persistence_fx(location, fc_start, lookback_for_max='4w', lookback_for_mean='24h', lookahead='24h'):
    """
    Create hourly forecast using average of historical data for each time period.
    :param location: Location dict
    :param fc_start: Start time of forecast
    :param lookback_for_max: Duration of historical data to use for computing max output at each time of day
    :param lookback_for_mean: Duration of historical data to use for computing mean output at each time of day
    :param lookahead: Duration of forecast to generate
    :return: dict of local variables from forecasting function.
    """

    hist = load_hist(location, end=fc_start, duration=lookback_for_max)
    hist['P_out'] = hist['P_out'].fillna(0)
    time_max = aggregate_by_time(hist['P_out'])

    if lookback_for_max != lookback_for_mean:
        hist = load_hist(location, end=fc_start, duration=lookback_for_mean)
        hist['P_out'] = hist['P_out'].fillna(0)
    time_mean = aggregate_by_time(hist['P_out'], 'mean')

    fx_idx = pd.date_range(fc_start, periods=pd.to_timedelta(lookahead) / pd.to_timedelta('1h'), freq="H")
    fx_weather = pd.DataFrame({'dt': fx_idx}, index=fx_idx)
    fx_weather['time_max'] = apply_time_mapping(fx_weather, time_max)
    fx_weather['time_mean'] = apply_time_mapping(fx_weather, time_mean)

    forecast_df = pd.DataFrame(index=fx_weather.index)

    forecast_df['forecast'] = fx_weather['time_mean']
    forecast_df['fx_csratio'] = calc_csratio(forecast_df['forecast'], fx_weather['time_max'])

    # Save variables for plotting (or other use)
    kind = 'persistence'
    pred_cols = []
    lookback = lookback_for_max
    model_order, fit_intercept, model, regression_r2 = 4*[None]
    forecast_info = locals()

    save_dayahead_fx(forecast_info)

    return forecast_info


def dayahead_fx_plot(fig: Optional[matplotlib.figure.Figure] = None, plot_ahead=pd.to_timedelta('24h'), **kwargs):
    """
    Plot day ahead forecast. Creates a new A4-size plot with 2x2 subplots.
    Beware: Also adds some columns to the forecast_df and fx_weather dataframes. TODO: Fix this

    :param fig: Optional figure object on which to generate the plots. Available because matplotlib seems to leak memory
        when making and closing a large number of plots.
    :param plot_ahead: Optional Pandas timedelta for how far out to plot. This allows generating plots for 1 day ahead
        while computing 2 day ahead forecasts for other purposes.
    :param kwargs: Dict with info input and output from the day ahead forecast function. This includes most of
        local variables of the generic_fx function. Field are as follows:
        location: Location dict used for the forecast
        hist: DataFrame of historical data loaded for the lookback
        forecast_df: DataFrame with the forecast
        fx_weather: DataFrame with the forecast weather information
        model: Regression model fit to the prediction data
        All the parameters sent to the generic_fx function are included. See that function for documentation.
        location
        kind
        pred_cols
        fc_start
        lookahead
        lookback
        model_order
        fit_intercept
    :return: None
    """

    location, kind, pred_cols, fc_start, lookahead, lookback, model_order, fit_intercept = itemgetter(
        'location', 'kind', 'pred_cols', 'fc_start', 'lookahead', 'lookback', 'model_order', 'fit_intercept')(kwargs)
    hist, forecast_df, fx_weather, model, regression_r2 = itemgetter('hist', 'forecast_df', 'fx_weather', 'model',
                                                                     'regression_r2')(kwargs)

    # Load actuals if they aren't already provided
    actuals = load_hist(location, start=fc_start, duration=plot_ahead, ensure_cols=pred_cols)

    # Trim dataframes to just the period to be plotted
    forecast_df = forecast_df.loc[(forecast_df.index < fc_start + plot_ahead)].copy()
    fx_weather = fx_weather.loc[(fx_weather.index < fc_start + plot_ahead)].copy()

    forecast_df['actual'] = actuals['P_out'].fillna(0)
    if kind == 'fx_csratio':
        cs_col = 'modeled_output'
    else:
        cs_col = 'time_max'
    forecast_df['actual_csratio'] = calc_csratio(forecast_df['actual'], fx_weather[cs_col])

    actuals['P_out_resid'] = forecast_df['actual'] - forecast_df['forecast']

    # Compare to actuals
    forecast_RMSE = root_mean_square_error(forecast_df['actual'], forecast_df['forecast'])

    # Use passed in figure object if there is one, otherwise create a new A4 page
    if fig is None:
        fig = plt.figure(figsize=(11.69, 8.27))
    else:
        plt.figure(fig.number)
    # Reuse axes if there are the right number of them, otherwise make a new set
    if len(fig.axes) != 4:
        fig.clear()
        fig.subplots(2, 2)
    ax = fig.axes[0]  # plt.subplot(2, 2, 1)
    ax.clear()
    ax.set_title('PV output')
    forecast_df[['forecast', 'actual']].plot(ax=ax)
    ax.grid(True, which='both', axis='both')
    ax.set_ylim(top=0.9)
    ax.set_xlabel(None)
    ax.text(0.02, 0.98, f'Forecast RMSE: {forecast_RMSE:.3g}',
            ha='left', va='top', transform=ax.transAxes)

    ax = fig.axes[2]
    ax.clear()
    if pred_cols:
        ax.set_title('Weather forecast')
        fx_weather['actual_' + pred_cols[0]] = actuals[pred_cols[0]]
        fx_weather[[pred_cols[0], 'actual_' + pred_cols[0]]].plot(ax=ax)
        # fx_weather2['ghi'].plot(ax=ax, label='updated_ghi_fx')

        irr_fx_RMSE = root_mean_square_error(fx_weather['actual_' + pred_cols[0]], fx_weather[pred_cols[0]])
        # irr_fx2_RMSE = root_mean_square_error(fx_weather['actual_ghi'], fx_weather2['ghi'])
        ax.grid(True, which='both', axis='both')
        ylim_lu = {'ghi': 1000, 'clouds': 100, 'meteogram_clouds': 100}
        try:
            ax.set_ylim(top=ylim_lu[pred_cols[0]])
        except ValueError:
            pass
        ax.set_xlabel(None)
        ax.text(0.02, 0.98, f'Forecast RMSE: {irr_fx_RMSE:.3g}', ha='left', va='top', transform=ax.transAxes)
        ax.legend()
    # plt.show()

    ax = fig.axes[1]
    ax.clear()
    if pred_cols:
        if kind == 'fx_output':
            ax.plot(hist[pred_cols[0]], hist['P_out'], '.', alpha=0.5)
            pred_plot = hist[[pred_cols[0], 'P_out_pred']].sort_values(by=pred_cols[0])
            ax.plot(pred_plot[pred_cols[0]], pred_plot['P_out_pred'])
            ax.plot(fx_weather['actual_' + pred_cols[0]], forecast_df['actual'])
        elif kind in {'fx_csratio', 'fx_csratio2'}:
            ax.plot(hist[pred_cols[0]], hist['csratio'], '.', alpha=0.5)
            pred_plot = hist[[pred_cols[0], 'csratio_pred']].sort_values(by=pred_cols[0])
            ax.plot(pred_plot[pred_cols[0]], pred_plot['csratio_pred'])
            ax.plot(fx_weather['actual_' + pred_cols[0]], forecast_df['actual_csratio'])
        try:
            ax.set_xlim(right=ylim_lu[pred_cols[0]])
        except ValueError:
            pass
        ax.set_ylim(top=0.9)
        ax.grid(True, which='both', axis='both')
        ax.set_xlabel(pred_cols[0])
        if kind == 'fx_output':
            ax.set_ylabel('PV Output')
        elif kind == 'fx_csratio':
            ax.set_ylabel('Clearsky Ratio')
        ax.set_title('Regression')
        ax.legend(['Historical data', 'Regression', 'Actual (est.)'])
        ax.text(0.95, 0.05, f'Regression R2: {regression_r2:.2f}', ha='right', va='bottom', transform=ax.transAxes)

    ax = fig.axes[3]
    ax.clear()
    forecast_df['fx_csratio'].plot(ax=ax, label='forecast')
    forecast_df['actual_csratio'].plot(ax=ax, label='actual')
    ax.set_ylim(0, 1.2)
    ax.set_xlabel(None)
    ax.grid(True, which='both', axis='both')
    ax.legend()
    ax.set_title('Clearsky Ratio')

    return fig


def dayahead_fx_residual_plot(fig: Optional[matplotlib.figure.Figure] = None, group_by_day=False, **kwargs):
    hist, pred_cols = itemgetter('hist', 'pred_cols')(kwargs)

    norm_by_time_max = (pred_cols[0] == 'ghi')
    hist_weather_label = label_by_weather(hist[pred_cols[0]], hist, norm_by_time_max=norm_by_time_max,
                                          group_by_day=group_by_day)

    grouped_hist = hist['P_out_resid'].groupby([hist.index.time, hist_weather_label])

    # Use passed in figure object if there is one, otherwise create a new page
    if fig is None:
        fig = plt.figure(figsize=(11.69, 2*8.27))
    else:
        plt.figure(fig.number)

    # How many subplots do we need? No need for 24 since many hours will have P_out = P_out_resid = 0
    times = pd.unique(hist_weather_label.index.time)
    times.sort()
    num_plots = len(times)*2

    # Reuse axes if there are the right number of them, otherwise make a new set
    if not fig.axes or len(fig.axes) != num_plots:
        fig.clear()
        fig.subplots((num_plots - 1) // 2 + 1, 2)
        # fig.set_layout_engine('tight')
        fig.set_tight_layout(True)

    if group_by_day:
        fig.suptitle('Weather labeling for whole day')
    else:
        fig.suptitle('Weather labeling for each period of day')

    n_ax = 0

    # To get unaggregated residuals use something like the following
    for time, label in itertools.product(times, ['cloudy', 'clear']):
        try:
            residuals = grouped_hist.get_group((time, label))
        except KeyError:
            # If nothing for the group was found, clear the axis
            fig.axes[n_ax].clear()
            n_ax += 1
            continue

        ax = fig.axes[n_ax]
        ax.clear()
        residuals.plot(ax=ax, kind='hist', title=f'Historical residuals at {time}, weather label={label}')
        n_ax += 1
