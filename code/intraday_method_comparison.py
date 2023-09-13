#! /usr/bin/env python
"""
This script is for comparing different methods of doing intra-day forecast updates

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
from ems.forecast.utils import load_hist, root_mean_square_error, mean_bias_error, load_dayahead_fx
import ems.forecast.utils
from ems.forecast.intraday import intraday_update
from ems.forecast.generate_pv_fx import generate_pv_fx
from ems.modeling.modeling_window import ModelingWindow

program_name = 'intraday_method_comparison'
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
        'py.warnings': {'level': 'ERROR'},  # TODO: Better to fix these warnings than just not show them
        'ems': {'level': 'DEBUG'},
        program_name: {'level': 'DEBUG'},
    }
}
logging.config.dictConfig(logging_config)
logging.captureWarnings(True)
logger = logging.getLogger(program_name)

#plt.ioff() # Set matplotlib to interactive so plots don't block
import matplotlib
#matplotlib.use('TkAgg')  # Use TkAgg when interactive plotting is desired. Still responds in PyCharm debugging mode, unlike QtAgg.
matplotlib.use('PDF')  # PDF is quicker and lighter when the plots are just being saved.


def intraday_update_demo(location):
    """
    Demonstrates intraday updates by using pre-calculated day-ahead forecasts (should be saved in ems.forecast.utils)
    and updating the forecast within the day. The main purpose of this function is to generate plots for troubleshooting
    or publication.
    :param location: Location dict.
    :return:
    """
    dayahead_fx_file = Path('../results/dayahead_method_comparison/') / 'all_dayahead_fx_solcast_clouds.parquet'
    all_dayahead_fx = pd.read_parquet(dayahead_fx_file)
    ems.forecast.utils.all_dayahead_fx = all_dayahead_fx

    logger.info(f'Doing intraday method comparison (plots)')
    logger.info(f'dayahead_updates archive loaded from {dayahead_fx_file}')
    logger.info(f'dayahead_updates archive goes from {all_dayahead_fx["dt"].iloc[0]} '
                f'to {all_dayahead_fx["dt"].iloc[-1]}')

    fig = plt.figure(figsize=(11.69 / 2, 8.27))
    fig.subplots(2, 1)
    ax_output = fig.axes[0]
    ax_csratio = fig.axes[1]
    fig.set_tight_layout(True)
    for kind in ['persistence', 'fx_output', 'fx_csratio', 'exog', 'sarimax', 'scaling']:

        logger.info('=' * 80)
        logger.info(f'Forecast Update Method: {kind}')

        pdf_fname = results_dir / f'IntradayUpdate-{kind}.pdf'

        with PdfPages(pdf_fname) as pdf:

            for fc_start_day in ['2021-03-30', '2021-04-02', '2021-04-07', '2021-04-16', '2021-04-18', '2021-04-22',
                                 '2021-04-26']:  # ['2021-03-30', '2021-04-02', '2021-04-07']:
                ax_output.clear()
                # ax_output.legend(handles=[])
                ax_csratio.clear()
                # ax_csratio.legend(handles=[])

                fc_start = pd.to_datetime(fc_start_day + ' 02:00')
                fc_end = pd.to_datetime(fc_start_day + ' 22:00')
                dayahead_fx = load_dayahead_fx(None, fc_start, fc_start, fc_end)
                dayahead_fx['forecast'].plot(ax=ax_output, label='Dayahead forecast')
                dayahead_fx['fx_csratio'].plot(ax=ax_csratio, label='Dayahead forecast')

                time_max = dayahead_fx['time_max'].iloc[6:]  # At least 6 hours after the start of the fx
                next_dark_hour = time_max[time_max < 0.01].index[0]
                for fc_start_time in [' 06:00', ' 09:00', ' 12:00']:
                    fc_start = pd.to_datetime(fc_start_day + fc_start_time)
                    lookahead = next_dark_hour - fc_start

                    logger.info(f'Forecasting {fc_start}')
                    try:
                        updated_fx = intraday_update(location, fc_start, lookback='28d', lookahead=lookahead, kind=kind)
                    except ems.forecast.utils.ForecastError as e:
                        logger.warning(f'Forecast error detected! {e}')
                        continue
                    updated_fx['forecast'].plot(ax=ax_output, label=f'Forecast update ({fc_start:%H:%M})')
                    updated_fx['csratio'].plot(ax=ax_csratio, label=f'Forecast update ({fc_start:%H:%M})')

                dayahead_fx['actual'].plot(ax=ax_output, label='Actual output')
                fig.suptitle(f'Intraday Forecasts, {kind} method, for {fc_start:%Y-%m-%d}')
                ax_output.set_xlabel(None)
                ax_output.set_ylabel('PV Output (pu)')
                ax_output.set_ylim(top=0.9)
                ax_output.grid(True, which='both', axis='both')
                ax_output.legend()

                dayahead_fx['actual_csratio'].plot(ax=ax_csratio, label='Actual output')
                ax_csratio.set_xlabel(None)
                ax_csratio.set_ylabel('Clearsky Ratio (pu)')
                ax_csratio.grid(True, which='both', axis='both')
                ax_csratio.legend()

                pdf.savefig()

def intraday_comparison(location):
    """
    Generates intraday updates at every hour whenever possible. Dayahead forecasts are assumed to have been previously
    generated. Intraday updates are only calculated during daylight when sufficient history is available.
    Updates are not calculated for the same time that a new dayahead forecast was generated.
    :param location:
    :return:
    """
    dayahead_fx_file = Path('../results/dayahead_method_comparison/') / 'all_dayahead_fx_solcast_clouds.parquet'
    all_dayahead_fx = pd.read_parquet(dayahead_fx_file)
    ems.forecast.utils.all_dayahead_fx = all_dayahead_fx

    tz = 'Europe/Istanbul'
    data_start = pd.to_datetime('3/30/2021 2:00') # Beginning after the snow cover days
    data_end = pd.to_datetime('7/1/2021 2:00')

    lookback = pd.to_timedelta('28d')
    lookahead = pd.to_timedelta('1d')

    stride = pd.to_timedelta('1h')  # Stride must be an integer number of hours
    day_start_offset = pd.Timedelta('6h')  # Used for water usage aggregation, so not relevant to this program

    intraday_methods = ['persistence', 'fx_output', 'fx_csratio', 'exog', 'sarimax', 'scaling']
    fx_offsets = [0, 1, 2, 6]

    logger.info(f'Doing intraday method comparison (metrics)')
    logger.info(f'{data_start=}')
    logger.info(f'{data_end=}')
    logger.info(f'{lookback=}')
    logger.info(f'{lookahead=}')
    logger.info(f'{stride=}')
    logger.info(f'{tz=}')
    logger.info(f'{day_start_offset=}')
    logger.info(f'dayahead_updates archive loaded from {dayahead_fx_file}')
    logger.info(f'dayahead_updates archive goes from {all_dayahead_fx["dt"].iloc[0]} '
                f'to {all_dayahead_fx["dt"].iloc[-1]}')

    metrics_list = [root_mean_square_error, mean_absolute_error, mean_bias_error]
    results_metrics = pd.DataFrame(index=pd.MultiIndex.from_product([fx_offsets, intraday_methods],
                                                                    names=['Offset', 'Method']),
                                   columns=['RMSE', 'MAE', 'MBE'], dtype=float)
    results_metrics_daily_total = results_metrics.copy()

    dayahead_method = 'load_from_file'

    for intraday_method in intraday_methods:
        logger.info(f'Generating intraday updates using method={intraday_method}')

        # Like a for loop here but with the possibility of changing stride within the loop
        start_time = data_start + lookback
        end_time = data_end - lookahead

        fx_record = pd.DataFrame(index=pd.date_range(start_time, end=end_time, freq=stride),
                                 columns=fx_offsets, dtype=float)

        # Like a for loop here but with the possibility of changing stride within the loop
        while start_time <= end_time:
            logger.info(f'Generating forecast for {start_time}')

            fx_window = ModelingWindow(start=start_time,
                                       delta_t=stride,
                                       whole_days=1,
                                       day_start_offset=day_start_offset,
                                       tz=tz)
            pv_fx = generate_pv_fx(location, fx_window, (3,), 1.0, dayahead_method, intraday_method)
            # Save forecasts at selected offsets
            fx_record.loc[start_time] = pv_fx[fx_offsets].values

            start_time += stride

        # Compute error metrics for forecasts using actuals
        hist = load_hist(location, start=data_start, end=data_end)
        # Shift so all forecasts for the same time period are in the same row
        fx_record = pd.DataFrame({h: fx_record[h].shift(h) for h in fx_offsets})

        fx_day_summary = fx_record.resample('1D').sum()
        hist_day_summary = hist.loc[fx_record.index, 'P_out'].resample('1D').sum()

        for h in fx_offsets:
            fx = fx_record[h].dropna()
            results_metrics.loc[(h, intraday_method), :] = [m(hist.loc[fx.index, 'P_out'], fx)
                                                            for m in metrics_list]
            fx = fx_day_summary[h].dropna()
            results_metrics_daily_total.loc[(h, intraday_method), :] = [m(hist_day_summary.loc[fx.index], fx)
                                                                        for m in metrics_list]
        logger.info('=' * 60)
        logger.info(f'Overall metrics: {results_metrics.loc[(slice(None), intraday_method), :]}')
        logger.info('')

    logger.info('Mean-square errors of different methods forecasting the hourly output')
    logger.info(results_metrics)
    results_metrics.style.to_latex(results_dir / 'error_comparison_table.tex')

    logger.info('Mean-square errors of different methods forecasting the daily total energy')
    logger.info(results_metrics_daily_total)
    results_metrics_daily_total.style.to_latex(results_dir / 'error_comparison_table_daily_total.tex')


def main(argv=None):

    # Load location info
    locations = pd.read_csv('../data/locations.csv')
    location = locations[locations['name'] == 'EEE B Block'].iloc[0].to_dict()

    to_do = {'examples', 'metrics'}

    # The dataset for intraday_update_demo should go back to 2/21/2021.
    if 'examples' in to_do:
        intraday_update_demo(location)

    # The dataset for comparison should go back to 3/30/2021 and should be at least 30h ahead forecast.
    if 'metrics' in to_do:
        intraday_comparison(location)

    return 0


if __name__ == '__main__':
    sys.exit(main())
