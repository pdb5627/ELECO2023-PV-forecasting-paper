#! /usr/bin/env python
"""
This script is for comparison of day-ahead forecast methods.

"""

import sys
import warnings
from datetime import datetime
from pathlib import Path
import logging
import logging.config

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error
import statsmodels
import statsmodels.api as sm
from ems.datasets import ClearskyModel
from ems.forecast.utils import (load_hist, skewness_plot, engine, mean_bias_error, root_mean_square_error,
                                load_dayahead_fx)
import ems.forecast.utils
import ems.forecast.dayahead
from ems.forecast.dayahead import irr_forecast, cs_ratio_forecast, dayahead_fx_plot, dayahead_fx_residual_plot

program_name = 'dayahead_method_comparison'
program_runtime = datetime.now()
results_dir = Path('../results') / program_name
results_dir.mkdir(parents=True, exist_ok=True)

logfile = results_dir / f'{program_name}.log'

logging_config = {
    'version': 1,
    'formatters': {
        'file': {'format': '%(asctime)s %(levelname)-8s %(name)s: %(message)s'},
    },
    'handlers': {
        'file': {'class': 'logging.FileHandler',
                 'filename': logfile,
                 'formatter': 'file',
                 'level': 'INFO'},
        'console': {'class': 'logging.StreamHandler',
                    'formatter': 'file',
                    'level': 'DEBUG'}
    },
    'loggers': {
        '': {'handlers': ['file', 'console'],
             'level': 'WARNING'},
        'ems': {'level': 'DEBUG'},
        program_name: {'level': 'DEBUG'},
    }
}
logging.config.dictConfig(logging_config)
logging.captureWarnings(True)
logger = logging.getLogger(program_name)

import matplotlib
import matplotlib.ticker as ticker
if __name__ == '__main__':
    # matplotlib.use('TkAgg')
    matplotlib.use('PDF')

    plt.ion() # Set matplotlib to interactive so plots don't block
matplotlib.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'Linux Libertine O'],
                        'legend.framealpha': 0.8,
                        'legend.frameon': True})
plt.rc('font', size=6)


def main(argv=None):
    # Load location info
    locations = pd.read_csv('../data/locations.csv')
    location = locations[locations['name'] == 'EEE B Block'].iloc[0].to_dict()

    # data_start = pd.to_datetime('3/30/2021 2:00')  # Beginning after the snow cover days
    data_start = pd.to_datetime('2/21/2021 2:00')  # Beginning of actuals
    data_end = pd.to_datetime('7/1/2021 2:00')
    # Following two lines for testing with missing data
    data_start = pd.to_datetime('2/21/2021 2:00')  # Beginning of actuals
    data_end = pd.to_datetime('10/1/2023 2:00')
    lookback = pd.to_timedelta('28d')
    lookahead = pd.to_timedelta('24h')

    # For the day-ahead metrics, begin after the snow cover days.
    # Only the most recent forecast will be used (i.e. when it is one day ahead, not two days ahead).
    metrics_range = [pd.to_datetime('3/30/2021 2:00'), data_end]  # End was pd.to_datetime('7/1/2021 2:00')

    logger.info(f'Doing day-ahead method comparison')
    logger.info(f'{data_start=}')
    logger.info(f'{data_end=}')
    logger.info(f'{lookback=}')
    logger.info(f'{lookahead=}')

    # Refresh data (comment out when no new data is expected)
    # ems.forecast.utils.refresh_data()

    # Save dict of all the day-ahead forecasts for after-the-fact analysis
    all_dayahead_fx = dict()
    metrics_list = [root_mean_square_error, mean_absolute_error, mean_bias_error]
    results_metrics = pd.DataFrame(columns=['RMSE', 'MAE', 'MBE'])
    results_metrics_daily_total = pd.DataFrame(columns=['RMSE', 'MAE', 'MBE'])
    results_rmse_by_month = pd.DataFrame(columns=range(1, 13))

    # All FX methods available:
    # ['Persistence', 'Persistence 3 Days Avg', 'Persistence 28 Days Avg', 'SolCast Cloudiness',
    #  'SolCast GHI', 'MGM Meteogram1', 'MGM Meteogram']
    # To save time, not running the methods that aren't reported in the paper.
    for fx_method in ['Persistence', 'SolCast Cloudiness', 'SolCast GHI', 'MGM Meteogram']:

        logger.info('='*80)
        logger.info(f'Forecast Method: {fx_method}')

        # Reset any previous day-ahead forecasts
        ems.forecast.utils.all_dayahead_fx = None

        if fx_method == 'SolCast GHI':
            pred_col = 'ghi'
            actual_col = 'actual_ghi'
            col_txt = 'irradiation'
            pdf_fname = results_dir / 'SolCastGHIForecasts.pdf'
            save_fx_fname = results_dir / 'all_dayahead_fx_ghi.parquet'

        elif fx_method == 'SolCast Cloudiness':
            pred_col = 'clouds'
            actual_col = 'clouds'
            col_txt = 'Solcast clouds'
            pdf_fname = results_dir / 'SolcastCloudsForecasts.pdf'
            save_fx_fname = results_dir / 'all_dayahead_fx_solcast_clouds.parquet'

            # Pre-load clearsky calcs
            cs_ds = ClearskyModel(location, db_engine=engine)
            cs_ds.import_new_data(start=(data_start - lookback), end=(data_end + lookahead))

        elif fx_method == 'MGM Meteogram1':
            pred_col = 'meteogram_clouds'
            actual_col = 'meteogram_clouds'
            col_txt = 'Meteogram clouds'
            pdf_fname = results_dir / 'MeteogramForecasts_csratio_model.pdf'
            save_fx_fname = results_dir / 'all_dayahead_fx_csratio_model.parquet'

            # Pre-load clearsky calcs
            cs_ds = ClearskyModel(location, db_engine=engine)
            cs_ds.import_new_data(start=(data_start - lookback), end=(data_end + lookahead))

        elif fx_method == 'MGM Meteogram':
            pred_col = 'meteogram_clouds'
            actual_col = 'meteogram_clouds'
            col_txt = 'Meteogram clouds'
            pdf_fname = results_dir / 'MeteogramForecasts_csratio_max.pdf'
            save_fx_fname = results_dir / 'all_dayahead_fx_csratio_max.parquet'

        elif fx_method == 'Persistence':
            pred_col = None
            pdf_fname = results_dir / 'PersistenceForecasts.pdf'
            save_fx_fname = results_dir / 'all_dayahead_persistence.parquet'

        elif fx_method == 'Persistence 3 Days Avg':
            pred_col = None
            pdf_fname = results_dir / 'Persistence3DaysForecasts.pdf'
            save_fx_fname = results_dir / 'all_dayahead_persistence3days.parquet'

        elif fx_method == 'Persistence 28 Days Avg':
            pred_col = None
            pdf_fname = results_dir / 'Persistence28DaysForecasts.pdf'
            save_fx_fname = results_dir / 'all_dayahead_persistence28days.parquet'

        else:
            raise NotImplementedError(f'Forecast method {fx_method} not implemented.')

        # Generate day-ahead forecasts & plots
        with PdfPages(pdf_fname) as pdf:
            fig = plt.figure(figsize=(11.69, 8.27))
            # fig2 = plt.figure(figsize=(11.69, 8.27*2))
            for fc_start in pd.date_range(start=data_start + lookback,
                                          end=data_end - lookahead,
                                          freq='1d', tz=None):
                logger.info(f'Forecasting {fc_start}')
                try:
                    if fx_method == 'MGM Meteogram1':
                        forecast_info = cs_ratio_forecast(location, fc_start, lookback, lookahead, cs_kind='model')
                    elif fx_method == 'MGM Meteogram':
                        forecast_info = cs_ratio_forecast(location, fc_start, lookback, lookahead, cs_kind='max')
                    elif fx_method == 'SolCast GHI':
                        forecast_info = irr_forecast(location, fc_start, lookback, lookahead)
                    elif fx_method == 'SolCast Cloudiness':
                        forecast_info = ems.forecast.dayahead.generic_fx(location, 'fx_csratio2', ['clouds'], fc_start,
                                                                         lookback, lookahead, 2, True)
                    elif fx_method == 'Persistence':
                        forecast_info = ems.forecast.dayahead.persistence_fx(location, fc_start,
                                                                             lookback, '24h', lookahead)
                    elif fx_method == 'Persistence 3 Days Avg':
                        forecast_info = ems.forecast.dayahead.persistence_fx(location, fc_start,
                                                                             lookback, '3d', lookahead)
                    elif fx_method == 'Persistence 28 Days Avg':
                        forecast_info = ems.forecast.dayahead.persistence_fx(location, fc_start,
                                                                             lookback, '28d', lookahead)
                    else:
                        raise NotImplementedError(f'Forecast method {fx_method} not implemented.')
                    dayahead_fx_plot(fig=fig, **forecast_info)
                    pdf.savefig(fig)

                    # norm_by_time_max = (forecast_info['pred_col'] == 'ghi')
                    # grouping_col = forecast_info['hist'][forecast_info['pred_col']]
                    # weather_label, fig = ems.forecast.utils.label_by_weather(grouping_col, forecast_info['hist'],
                    #     norm_by_time_max=norm_by_time_max, group_by_day=False)  #, fig=fig)
                    # pdf.savefig(fig)
                    # fig.clear()

                    # for group_by_day in [False, True]:
                    #     dayahead_fx_residual_plot(fig=fig2, group_by_day=group_by_day, **forecast_info)
                    #     pdf.savefig(fig2)

                except ValueError:
                    # TODO: Need to handle missing weather forecast data better.
                    logger.info(f'Missing weather forecast data. Skipping forecast for {fc_start}')

            # Save forecasts to disk
            all_dayahead_fx[fx_method] = ems.forecast.utils.all_dayahead_fx
            all_dayahead_fx[fx_method].to_parquet(save_fx_fname, compression='gzip', index=False)

            fx_all = load_dayahead_fx(None, metrics_range[1], metrics_range[0], metrics_range[1])
            summary_cols = ['actual', 'forecast'] + ([pred_col, actual_col] if pred_col else [])
            fx_day_summary = fx_all[summary_cols].resample('1D').sum()

            results_metrics.loc[fx_method] = [m(fx_all["actual"], fx_all["forecast"]) for m in metrics_list]
            results_metrics_daily_total.loc[fx_method] = [m(fx_day_summary["actual"], fx_day_summary["forecast"])
                                                          for m in metrics_list]
            df = all_dayahead_fx[fx_method].set_index('dt')
            # Drop rows with very low actual PV output. Could be missing data.
            df = df[pd.Series(df.index.date, index=df.index).map(df['actual'].groupby(df.index.date).sum() >= 1)]
            results_rmse_by_month.loc[fx_method] = df.groupby(df.index.month).apply(lambda x: root_mean_square_error(x['actual'], x['forecast']))

            logger.info('='*60)
            logger.info(f'Overall metrics: {results_metrics.loc[fx_method]}, ({metrics_range[0]} - {metrics_range[1]})')
            logger.info('')
            logger.info('Actual P_out - Forecast P_out Daily Sums')
            logger.info((fx_day_summary['actual'] - fx_day_summary['forecast']).describe())
            logger.info('')
            # if actual_col is not None:
            #     logger.info('')
            #     logger.info(f'Actual {col_txt} - forecast {col_txt}')
            #     logger.info((fx_day_summary[actual_col] - fx_day_summary[pred_col]).describe())

            plt.figure(figsize=(11.69, 8.27))
            ax = plt.subplot(2, 2, 1)
            fx_day_summary[['forecast', 'actual']].plot(ax=plt.gca())
            plt.title('PV forecast')
            plt.xlabel(None)
            # Make room for xlabel
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

            ax = plt.subplot(2, 2, 2)
            skewness_plot(fx_day_summary['actual'] - fx_day_summary['forecast'], ax,
                          'forecast residual')
            plt.title('PV forecast residual')

            ax = plt.subplot(2, 2, 3)
            if pred_col:
                # Predictor as average rather than sum
                fx_day_summary[pred_col] /= 24
                fx_day_summary[actual_col] /= 24

                fx_day_summary[[pred_col, actual_col]].plot(ax=plt.gca())
                plt.title(f'{col_txt} forecast')
                plt.xlabel(None)
                # Make room for xlabel
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

            # ax = plt.subplot(2, 2, 4)
            # skewness_plot(fx_day_summary[actual_col] - fx_day_summary[pred_col], plt.gca(), f'{col_txt} residual')
            # (fx_day_summary['actual_ghi'] - fx_day_summary['ghi']).plot(ax=plt.gca(), kind='density')
            # (fx_day_summary['actual_ghi'] - fx_day_summary['ghi']).plot(ax=plt.gca(), kind='box')
            # plt.title(f'Solcast {col_txt} forecast residual')
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(11.69, 8.27))
            ax = plt.subplot(2, 1, 1)
            sm.graphics.tsa.plot_acf(fx_all['actual'] - fx_all['forecast'], ax=ax, lags=48)
            ax = plt.subplot(2, 1, 2)
            sm.tsa.graphics.plot_pacf(fx_all['actual'] - fx_all['forecast'], ax=ax, lags=48, method='ywm')
            pdf.savefig()
            plt.close()

    logger.info('Mean-square errors of different methods forecasting the hourly output')
    logger.info(results_metrics)
    results_metrics.style.to_latex(results_dir / 'error_comparison_table.tex')
    results_metrics.to_csv(results_dir / 'error_comparison_table.csv')

    logger.info('Mean-square errors of different methods forecasting the daily total energy')
    logger.info(results_metrics_daily_total)
    results_metrics_daily_total.style.to_latex(results_dir / 'error_comparison_table_daily_total.tex')
    results_metrics_daily_total.to_csv(results_dir / 'error_comparison_table_daily_total.csv')

    logger.info('Mean-square errors of different methods by month')
    logger.info(results_rmse_by_month)
    results_rmse_by_month.style.to_latex(results_dir / 'error_comparison_table_rmse_by_month.tex')
    results_rmse_by_month.to_csv(results_dir / 'error_comparison_table_rmse_by_month.csv')


    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    results_rmse_by_month.T.plot(ax=ax, use_index=False)
    ax.set_xlim(-0.5, 11.5)
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(12)))
    ax.set_xticklabels(results_rmse_by_month.columns)
    # ax.set_xticks(range(1, 13))
    ax.set_xlabel('Month')
    ax.set_ylabel('RMSE')
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / 'error_comparison_rmse_by_month.pdf')
    plt.close()

    return 0


if __name__ == '__main__':
    with warnings.catch_warnings():
        # warnings.simplefilter("error")  # Uncomment to check for warnings while running the program.
        sys.exit(main())
