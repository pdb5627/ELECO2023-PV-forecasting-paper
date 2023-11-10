from pathlib import Path
import os
import random
from datetime import datetime
import logging.config

program_name = 'mpc_demo'
program_runtime = datetime.now()
results_dir = Path('../results') / program_name
results_dir.mkdir(parents=True, exist_ok=True)

logfile = results_dir / f'{program_name}.log'
logfile_debug = results_dir / f'{program_name}_debug.log'


class LogOnlyLevel(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, log_record):
        return log_record.levelno == self.__level


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
        'file_debug': {'class': 'logging.FileHandler',
                       'filename': logfile_debug,
                       'formatter': 'file',
                       'level': 'DEBUG'},
        'console': {'class': 'logging.StreamHandler',
                    'formatter': 'file',
                    'level': 'DEBUG'}
    },
    'loggers': {
        '': {'handlers': ['file', 'console', 'file_debug'],
             'level': 'WARNING'},
        'ems': {'level': 'DEBUG'},
        'ems.pyomo_utils': {'level': 'INFO'},
        # 'ems.optimization_two_pumps': {'level': 'WARNING'},
        'pyomo': {'level': 'CRITICAL'},
        program_name: {'level': 'DEBUG'},
    },
    'filters': {
        'debug_only': {
            '()': LogOnlyLevel,
            'level': logging.DEBUG
        }
    }
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger(program_name)

from types import SimpleNamespace
from typing import Optional, Mapping
from itertools import groupby
import dataclasses
import dill as pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import sqlalchemy
from ems.optimization_two_pumps import TwoPumpModel, TwoPumpModelParameters
from ems.datasets import ABBInverterDataSet, ClearskyModel
from ems.modeling.modeling_window import ModelingWindow
from ems.forecast.generate_pv_fx import generate_pv_fx
from ems.forecast.utils import aggregate_by_time, apply_time_mapping


if __name__ == '__main__':
    # matplotlib.use('TkAgg')
    matplotlib.use('PDF')

    plt.ion() # Set matplotlib to interactive so plots don't block
    matplotlib.rcParams['font.family'] = 'serif'
    plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'Linux Libertine O'],
                         'legend.framealpha': 0.8,
                         'legend.frameon': True})
    plt.rc('font', size=6)

# Define db connection and load ABB inverter dataset
engine = sqlalchemy.create_engine("sqlite+pysqlite:///../data/datasets.sqlite", echo=False)
ds_abb = ABBInverterDataSet(db_engine=engine)
#ds_abb.import_new_data()
ds_clearsky = ClearskyModel(location={'lat': 39.890893,
                                      'lon': 32.782331,
                                      'elevation': 800,
                                      'tilt': 19.8,
                                      'azimuth': 180},
                            db_engine=engine)


def time_of_use_rate(h):
    rate = {6: 0.39,
            17: 0.64,
            22: 0.95,
            24: 0.39}

    for hmax, r in rate.items():
        if h < hmax:
            return r
    return r


def time_of_day_water_efficiency(h: pd.Series):
    eta_w = np.array([[ 0, 1.0],
                      [ 6, 1.0],
                      [10, 0.8],
                      [14, 0.5],
                      [18, 0.7],
                      [22, 1.0],
                      [24, 1.0]]).T

    eta_wt = np.interp(h, eta_w[0], eta_w[1])
    return eta_wt


def  get_base_model_params() -> dict:
    """
    Returns a set of parameters for the model with common model parameters filled.
    (See code for details)
    Does NOT fill the following parameters:
        Scenario-defining parameters:
            Pload
            P_PVavail
            Vuse_desired
        State values:
            E_BSS0_pu
            sinv0
            Vw1_0
            Vw2_0
        Depend on time of day:
            lookahead
            D
            Cgrid
            eta_w
    """
    # Model metaparameters
    days = 3
    lookahead = 72

    p = {

        'lookahead': lookahead,
        'delta_t': 1,
        'D': None,

        # Model data
        'Ppump1_max': 15000.,
        'Qw1': 50.,

        # Parameters based on Grundfos SP45S-7 Pump at 41.1m head
        'Ppump2_max': 2300.,
        'Ppump2_min': 890.,
        'Qpump2_min': 0.803,
        'Qpump2_max': 13.7,

        'E_BSS_max': 9600.,
        'E_BSS_lower_pu': 0.1,
        'E_BSS_upper_pu': 0.95,

        'P_BSS_ch_max_pu': 1 / 10,
        'eta_BSS': 0.95,
        'E_absorb_pu': 0.8,

        'Vw1_min': 5.,
        'Vw1_max': 120.,

        'Vw2_min': 5.,
        'Vw2_max': 50.,

        'Quse1_max': 50.,
        'Quse2_max': 50.,

        'C_BSS': 0.01,
        'C_BSS_switching': 1.,
        'C_pump_switching': 1.,

        'Cw_short': 100.
    }

    return p


class ActualsGenerator:
    """
    A callable class to return actuals for PV_avail, Pload, and Vuse_desired.
    Implemented as a class so that long sequence of randomly generated values can be kept between calls rather than
    regenerated every time. Values are cached for a single seed value only. If called with a different seed, new
    series of random values are generated.
    """
    def __init__(self):
        # Date range for generating random sequences. Must be log enough to include all start and end dates.
        # Changing the end date will not change the sequence except to make it longer.
        # Changing the start date WILL change the sequence, even for the same seed value.
        self.start_date = pd.to_datetime('1/1/2010')
        self.end_date = pd.to_datetime('12/31/2025')
        self._delta_t = pd.to_timedelta('1h')  # The PV actuals are only on a one-hour interval, so this cannot be changed
        self._seed = 143
        self.Pload : Optional[pd.DataFrame] = None
        self.Vuse_desired : Optional[pd.DataFrame] = None


    def seed(self, seed):
        self._seed = seed
        random.seed(seed)
        self.Pload_seed = random.randint(0, 10 ** 7)
        self.Vuse_seed = random.randint(0, 10 ** 7)
        self.generate_Pload()
        self.generate_Vuse_desired()

    def generate_Pload(self):
        # Generate Pload sequence using random numbers
        random.seed(self.Pload_seed, version=2)
        # Uniform distribution
        Pload_max = 1e3
        # Avoid use of numpy.random since it is not guaranteed to be consistent across versions. Use Python instead.
        num_points = (self.end_date - self.start_date) // self._delta_t
        Pload_values = [random.uniform(0, Pload_max) for _ in range(num_points)]
        Pload = pd.Series(Pload_values,
                          index=pd.date_range(start=self.start_date, end=self.end_date, freq='1h', inclusive='left', tz=None))
        self.Pload = Pload

    def generate_Vuse_desired(self):
        # Generate Vuse_desired sequence using random numbers
        random.seed(self.Vuse_seed, version=2)
        # Normal distribution with given mean and standard deviation
        # TODO: This is problematic because it can generate negative numbers. Currently clipping to 0.
        Vuse_mean = 70
        Vuse_std = 30
        # Avoid use of numpy.random since it is not guaranteed to be consistent across versions. Use Python instead.
        num_points = int((self.end_date - self.start_date) / pd.to_timedelta('1d'))
        Vuse_values = [random.gauss(Vuse_mean, Vuse_std) for _ in range(num_points)]
        Vuse = pd.Series(Vuse_values,
                         index=pd.date_range(start=self.start_date, end=self.end_date, freq='1D', inclusive='left', tz=None))
        self.Vuse_desired = Vuse.clip(0)

    def __call__(self, modeling_window: ModelingWindow, seed=143) -> SimpleNamespace:
        """
        :param modeling_window: ModelingWindow object representing the window of time for which to generate data
        :param seed: All randomly generated data can be reproduced by calling with the same seed.
        :return: Namespace with fields for available_pv, Vuse_desired, and Pload.
        """
        if seed != self._seed:
            self.seed(seed)

        data = SimpleNamespace()
        data.seed = seed

        # Generate PV_avail sequence using ABB inverter log database
        data.available_pv = ds_abb.get_data_by_date(start=modeling_window.start,
                                                    end=modeling_window.end)['P_out'] * 7.5e3
        data.available_pv = data.available_pv.clip(0)  # Ensure PV output is positive for this model
        # Extract Pload and Vuse_desired from pre-generated random sequences
        Δ = self._delta_t / 10
        data.Pload = self.Pload[modeling_window.start:modeling_window.end - Δ].copy()
        data.Vuse_desired = self.Vuse_desired[modeling_window.Vuse_start:modeling_window.Vuse_end].copy()

        return data


generate_actuals = ActualsGenerator()


def generate_random_pv_fx(start, end, std, seed=143) -> pd.Series:
    """
    Generate random PV forecast from actuals using a seed value for reproducibility.
    Actuals are loaded from the database as is clearsky.
    The forecast is calculated as
        fx = actual + gaussian(0, std)
    Nighttime fx is set to zero and fx is not allowed to be negative.
    The point in time at which the forecast is generated does not affect the forecast.

    :param start: Pandas-compatible starting datetime or string
    :param end: Pandas-compatible ending datetime or string
    :param std: Standard-deviation of the error to be added to the actuals in each period, expressed in per-unit of
                nominal PV rating.
    :param seed:
    :return: Series with forecast PV values
    """
    df = ds_abb.get_data_by_date(start=start, end=end)
    actual = df['P_out']
    df = ds_clearsky.get_data_by_date(start=start, end=end)
    clearsky = df['modeled_output']

    random.seed(hash((seed, int((start - pd.to_datetime('1/1/2010')) / pd.to_timedelta('1h')))), 2)
    num_points = int((end - start) / pd.to_timedelta('1h'))
    error = [random.gauss(0, std) for _ in range(num_points)]
    fx = pd.Series(error,
                   index=pd.date_range(start=start, end=end, freq='1h', inclusive='left', tz=None))
    # If clearsky is very small, set error to zero
    fx[clearsky < 0.001] = 0
    # Add actual and error
    fx = fx + actual
    fx[fx < 0] = 0
    fx *= 7.5e3
    return fx


# Load location info
locations = pd.read_csv('../data/locations.csv')
location = locations[locations['name'] == 'EEE B Block'].iloc[0].to_dict()


def build_model(modeling_window: ModelingWindow, scenario_data: SimpleNamespace, state_data: SimpleNamespace,
                initialize_model=True) -> TwoPumpModel:
    """
    Combine base model with scenario and state data to produce a complete model
    :param modeling_window: ModelingWindow object representing the window of time to be modeled.
    :param scenario_data: namespace with available_pv, Pload, and Vuse_desired. Vuse_desired is assumed to already be
            adjusted in the first day for actual water use already completed for that day.
    :param state_data: namespace with E_BSS0_pu or E_BSS0, sinv0, Vw1_0 and Vw2_0
    :param initialize_model: Set to True to initialize and build the Pyomo model. False if no Pyomo model needed or
            if initialization & building will be done elsewhere.
    :return:

    It is assumed that the modeling window includes any logic needed for whole day boundaries.
    """
    tz = modeling_window.tz

    # Base model params
    p = get_base_model_params()

    # Localize the time for following calculations
    local_dt_idx = scenario_data.available_pv.tz_localize('UTC').tz_convert(tz).index
    local_hour = pd.Series(local_dt_idx.hour, index=scenario_data.available_pv.index)

    # Electric rates by time of day.
    electric_rates = local_hour.apply(time_of_use_rate)
    electric_rates /= 7.0  # Convert from TL to USD based on approximate exchange rate at the time

    p.update({'lookahead': modeling_window.num_points,
              'index': scenario_data.available_pv.index,
              'modeling_window': modeling_window,
              'Pload': scenario_data.Pload.values,
              'P_PVavail': scenario_data.available_pv.values,
              'Vuse_desired': scenario_data.Vuse_desired.values,
              'E_BSS0_pu': state_data.E_BSS0_pu,
              'sinv0': state_data.sinv0,
              'Vw1_0': state_data.Vw1_0,
              'Vw2_0': state_data.Vw2_0,
              'Cgrid': electric_rates.values,
              'eta_w': time_of_day_water_efficiency(local_hour)})

    p = TwoPumpModelParameters(**p)

    m = TwoPumpModel(p)
    # Monkey patch on the scenario data.
    m.scenario_data = scenario_data

    if initialize_model:
        m.initialize()
        m.build_model()

    return m


def optimize_and_simulate_from_actuals(seed=42, dayahead_method='solcast_clouds', intraday_method='sarimax',
                                       precomputed_optimization_models: Optional[Mapping[ModelingWindow, TwoPumpModel]] = None,
                                       precomputed_simulation_models: Optional[Mapping[ModelingWindow, TwoPumpModel]] = None,
                                       period_plots=True):
    """
    In progress work to start making the desired information flow.
    """
    one_hr = pd.to_timedelta('1h')
    tz = 'Europe/Istanbul'
    data_start = pd.to_datetime('3/30/2021 3:00')  # Beginning after the snow cover days. Must start at time for generating a day-ahead fx
    data_end = pd.to_datetime('7/1/2021 3:00')

    lookback = pd.to_timedelta('28d')
    lookahead = pd.to_timedelta('1d')

    stride = one_hr # Stride must be an integer number of hours
    day_start_offset = pd.Timedelta('6h')
    dayahead_hours = (6,)  # Run day-ahead fx at 6am localized time

    path = results_dir / f'{seed=},{dayahead_method=},{intraday_method=}'
    if not os.path.exists(path):
        os.makedirs(path)

    start_time = data_start + lookback
    end_time = data_end - lookahead

    logger.info(f'Doing MPC simulation comparison')
    logger.info(f'{data_start=}')
    logger.info(f'{data_end=}')
    logger.info(f'{lookback=}')
    logger.info(f'{lookahead=}')
    logger.info('Stride is adaptive -- optimization is run only during daylight hours')
    logger.info(f'{tz=}')
    logger.info(f'{day_start_offset=}')

    # Initial values for state
    # TODO: Make a class for this to define the interface
    state_data = SimpleNamespace()
    state_data.E_BSS0_pu = 0.3
    state_data.sinv0 = 0
    state_data.Vw1_0 = 50.
    state_data.Vw2_0 = 25.

    Vuse_prev = {'day': None, 'Vuse': 0.}

    optimization_models = dict()
    optimization_results = dict()
    simulation_models = dict()
    simulation_results = dict()

    # Reusable figure and axis object to avoid memory leakage issue
    comparison_fig = plt.figure(figsize=(5, 3))
    comparison_ax = plt.axes()

    while start_time < end_time:

        ##### Generate forecast

        optimization_window = ModelingWindow(start=start_time,
                                             delta_t=one_hr,
                                             whole_days=1,
                                             day_start_offset=day_start_offset,
                                             tz=tz)

        logger.info(f'Starting forecasting for period {optimization_window.start} - {optimization_window.end}')
        # Actuals is only used for the Vuse_desired. Otherwise not used.
        actuals = generate_actuals(optimization_window, seed)
        if dayahead_method == 'oracle':
            pv_fx = actuals.available_pv
        else:
            pv_fx = generate_pv_fx(location, optimization_window, dayahead_hours, 7.5e3, dayahead_method, intraday_method)
            pv_fx = pv_fx.clip(0)  # Make sure the forecast is positive for optimization model. (Actuals go negative)
            assert len(pv_fx) == len(actuals.available_pv)
        fx = SimpleNamespace()
        fx.seed = seed
        fx.dayahead_method = dayahead_method
        fx.intraday_method = intraday_method
        # Forecast PV uses forecast method
        fx.available_pv = pv_fx
        # Set stride to run next optimization after some PV is expected to be available.
        # Without available PV, there isn't really anything to optimize.
        day_pu_threshold = 0.01
        during_day = pv_fx > day_pu_threshold * pv_fx.max()
        if during_day.iloc[0]:
            stride = one_hr
        else:
            next_daylight_hour = pv_fx[during_day].index[0]
            tz_offset = optimization_window.end_localized - optimization_window.end.tz_localize(optimization_window.tz)
            # Maybe not the most efficient way to calculate the stride, but handles len(dayahead_hours) > 1
            while not((start_time + stride + tz_offset).hour in dayahead_hours or
                      (start_time + stride >= next_daylight_hour)):
                stride += one_hr


        # Forecast Pload is just a constant load value
        fx.Pload = pd.Series(500, index=pv_fx.index)
        # Reduce Vuse_desired by previous water use, the previous day matches the current one
        if Vuse_prev['day'] == actuals.Vuse_desired.index[0]:
            if Vuse_prev['Vuse'] > actuals.Vuse_desired[0]:
                actuals.Vuse_desired[0] = 0
            else:
                actuals.Vuse_desired[0] -= Vuse_prev['Vuse']
        # Forecast Vuse_desired is the same as actual.
        fx.Vuse_desired = actuals.Vuse_desired

        if precomputed_optimization_models and optimization_window in precomputed_optimization_models:
            logger.info(f'Using precomputed optimization model for period {optimization_window.start} - {optimization_window.end}')
            fx_model = precomputed_optimization_models[optimization_window]
            if fx_model.optimal_operation is None and fx_model.initialized_operation is not None:
                fx_model.optimal_operation = fx_model.initialized_operation
        else:
            fx_model = build_model(optimization_window, fx, state_data)

            logger.info(f'Starting optimization for period {optimization_window.start} - {optimization_window.end}')

            try:
                fx_model.optimize(time_limit=60)
                fx_model.get_optimal()

            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                logger.error(f'Optimization failed. Exception raised: {e}', stack_info=True)
                logger.info('Attempting to recover from failed optimization by using initialization values instead')
                fx_model.initialize()
                fx_model.optimal_operation = fx_model.initialized_operation

        decision_variables = fx_model.optimal_operation[fx_model.decision_variables]
        optimization_models[optimization_window] = fx_model
        optimization_results[optimization_window] = fx_model.optimal_operation

        ##### Simulation using actuals

        simulation_window = ModelingWindow(start=start_time,
                                           end=start_time + stride,
                                           delta_t=one_hr,
                                           day_start_offset=day_start_offset,
                                           tz=tz)

        actuals_for_simulation = generate_actuals(simulation_window, seed)
        actuals_for_simulation.dayahead_method = dayahead_method
        actuals_for_simulation.intraday_method = intraday_method

        if precomputed_simulation_models and simulation_window in precomputed_simulation_models:
            logger.info(f'Using precomputed simulation model for period {simulation_window.start} - {simulation_window.end}')
            simulation_model = precomputed_simulation_models[simulation_window]
        else:
            logger.info(f'Starting simulation for period {simulation_window.start} - {simulation_window.end}')

            # No need to modify Vuse_desired since it is not used in the simulation.
            simulation_model = build_model(simulation_window, actuals_for_simulation, state_data, initialize_model=False)
            simulation_model.simulate_operation(decision_variables, stride//one_hr)

            if period_plots:
            # Generate and save plots
                with PdfPages(path / f'{start_time:%Y%m%d_%H%M}_optimization.pdf') as pdf:
                    fig = fx_model.plot_optimization_results('optimization')
                    pdf.savefig(fig)
                    plt.clf()
                    plt.close('all')

                with PdfPages(path / f'{start_time:%Y%m%d_%H%M}_optimize_vs_simulate_comparisons.pdf') as pdf:
                    # Plot PV forecast separately to show full length of forecast vs actual even though simulation may be
                    # for a shorter time period
                    results_dict = {'Forecast': pd.DataFrame({'P_PVavail': fx.available_pv}),
                                    'Actual': pd.DataFrame({'P_PVavail': actuals.available_pv})}
                    comparison_ax = plot_comparison(results_dict, 'P_PVavail', True, ax=comparison_ax)
                    pdf.savefig(comparison_ax.figure)
                    comparison_ax.clear()

                    results_dict = {'Forecast/Optimized': fx_model.optimal_operation,
                                    'Actual/Simulated': simulation_model.simulation_results}
                    params_dict = {'Forecast/Optimized': fx_model.params,
                                   'Actual/Simulated': simulation_model.params}
                    for col in ['P_PV', 'Ppump1', 'Pload', 'Pgrid', 'E_BSS', 'Qpump1', 'Qpump2', 'Quse1', 'Quse2',
                                'Quse_eff', 'Vw1', 'Vw2']:
                        comparison_ax = plot_comparison(results_dict, col, True, params_dict, comparison_ax)
                        pdf.savefig(comparison_ax.figure)
                        comparison_ax.clear()

        sim_results = simulation_model.simulation_results

        # Save simulation model and results
        simulation_models[simulation_window] = simulation_model
        simulation_results[simulation_window] = simulation_model.simulation_results



        ##### Extract starting state for following iteration
        state_data = SimpleNamespace()

        final_idx = sim_results.index[-1]
        state_data.E_BSS0_pu = sim_results.at[final_idx, 'E_BSS'] / simulation_model.params.E_BSS_max
        state_data.sinv0 = sim_results.at[final_idx, 'sinv']
        state_data.Vw1_0 = sim_results.at[final_idx, 'Vw1']
        state_data.Vw2_0 = sim_results.at[final_idx, 'Vw2']

        # If previous final day of simulation is the same as this one, add to the accumulated Vuse_prev
        final_idx = sim_results.index[len(actuals_for_simulation.Vuse_desired) - 1]
        if Vuse_prev['day'] == actuals_for_simulation.Vuse_desired.index[-1]:
            Vuse_prev['Vuse'] += sim_results.at[final_idx, 'Vuse']
        else:
            Vuse_prev = {'day': simulation_window.Vuse_day(actuals_for_simulation.Vuse_desired.index[-1]),
                         'Vuse': sim_results.at[final_idx, 'Vuse']}

        # Don't need Pyomo model objects any more, so clear those out to free some memory
        fx_model.model = None
        simulation_model.model = None

        start_time += stride

    return optimization_models, simulation_models


def summarize_and_plot(optimization_models, simulation_models):
    first_model = next(iter(simulation_models.values()))
    seed = first_model.scenario_data.seed
    dayahead_method = first_model.scenario_data.dayahead_method
    intraday_method = first_model.scenario_data.intraday_method
    path = results_dir / f'{seed=},{dayahead_method=},{intraday_method=}'
    if not os.path.exists(path):
        os.makedirs(path)

    logger.info(f'Generating summary and plot for {seed=}, {dayahead_method=}, and {intraday_method=}.')
    logger.info(f'{path=}')

    ##### Generate plots and evaluate overall objective function cost
    model_with_combined_results = concatenate_simulation_windows(simulation_models, optimization_models)

    m = model_with_combined_results
    decision_variables = m.simulation_results[m.decision_variables].copy()
    prev_simulation = m.simulation_results.copy()
    decision_variables.at['2021-04-30 23:00:00', 'spump1'] = True
    m.simulate_operation(decision_variables, len(decision_variables))

    logger.info(model_with_combined_results.total_costs)
    model_with_combined_results.total_costs.to_latex(path / 'simulated_total_cost.tex')

    if model_with_combined_results.optimal_operation is not None:
        with PdfPages(path / 'optimized_operation.pdf') as pdf:
            fig = model_with_combined_results.plot_optimization_results('optimization')
            pdf.savefig(fig)
            plt.close(fig)

    if model_with_combined_results.simulation_results is not None:
        with PdfPages(path / 'simulated_operation.pdf') as pdf:
            fig = model_with_combined_results.plot_optimization_results('simulation')
            pdf.savefig(fig)
            plt.close(fig)

    if (model_with_combined_results.optimal_operation is not None and
            model_with_combined_results.simulation_results is not None):
        comparison_fig = plt.figure(figsize=(5, 3))
        comparison_ax = plt.axes()

        with PdfPages(path / 'optimize_vs_simulate_comparisons.pdf') as pdf:
            results_dict = {'Forecast/Optimized': model_with_combined_results.optimal_operation,
                            'Actual/Simulated': model_with_combined_results.simulation_results}
            params_dict = {'Forecast/Optimized': next(iter(optimization_models.values())).params,
                            'Actual/Simulated': next(iter(simulation_models.values())).params}
            for col in ['P_PVavail', 'P_PV', 'Ppump1', 'Pload', 'Pgrid', 'E_BSS', 'Qpump1', 'Qpump2', 'Quse1', 'Quse2',
                        'Quse_eff', 'Vw1', 'Vw2']:
                comparison_ax.clear()
                comparison_ax = plot_comparison(results_dict, col, True, params_dict, comparison_ax)
                pdf.savefig(comparison_ax.figure)

    return model_with_combined_results


def concatenate_simulation_windows(simulation_models: Mapping[ModelingWindow, TwoPumpModel],
                                   optimization_models: Optional[Mapping[ModelingWindow, TwoPumpModel]] = None):
    """
    Creates a new model with simulation results and optionally optimization results concatenated over the consecutive
    simulation modeling windows.
    :param simulation_models: dict with key that is the ModelingWindow and value that is the model with simulation
            results
    :param optimization_models: dict with key that is the ModelingWindow and value that is the model with optimization
            results. (Optional)
    :return: New model object with simulation and optionally optimization results filled in.
    """
    # Create new model object
    simulation_windows = list(simulation_models.keys())
    first_model = simulation_models[simulation_windows[0]]
    combined_modeling_window = ModelingWindow(start=simulation_windows[0].start,
                                              end=simulation_windows[-1].end,
                                              delta_t=simulation_windows[0].delta_t,
                                              day_start_offset=simulation_windows[0].day_start_offset,
                                              tz=simulation_windows[0].tz)
    state_data = SimpleNamespace()
    state_data.E_BSS0_pu = first_model.params.E_BSS0_pu
    state_data.sinv0 = first_model.params.sinv0
    state_data.Vw1_0 = first_model.params.Vw1_0
    state_data.Vw2_0 = first_model.params.Vw2_0
    actuals = generate_actuals(combined_modeling_window, first_model.scenario_data.seed)
    m = build_model(combined_modeling_window, actuals, state_data, initialize_model=False)
    m.simulation_results = pd.concat(m.simulation_results for m in simulation_models.values())
    m.params.index = m.simulation_results.index

    # Compute objective function components. They are just monkey-patched onto the object for now.
    m.per_period_costs = m.simulation_objective_function(m.simulation_results)
    m.total_costs = m.per_period_costs.sum()

    if optimization_models:
        m.optimal_operation = pd.concat([m.optimal_operation.loc[:w.end - w.delta_t/10, ~m.optimal_operation.columns.duplicated()] for m, w in
                                         zip(optimization_models.values(), simulation_windows)])
    return m


@dataclasses.dataclass
class ColumnPlotInfo:
    legend_txt: str
    axis_label: str
    scale: float = 1
    accumulate_from: Optional[str] = None


columns_info = {'P_PVavail': ColumnPlotInfo('PV available', 'P (kW)', 1/1000),
                'P_PV': ColumnPlotInfo('PV utilized', 'P (kW)', 1/1000),
                'P_BSSdisch': ColumnPlotInfo('BSS (discharge)', 'P (kW)', -1/1000),
                'P_BSSch': ColumnPlotInfo('BSS (charge)', 'P (kW)', 1/1000),
                'P_BSS': ColumnPlotInfo('BSS Power', 'P (kW)', 1/1000),
                'Pload': ColumnPlotInfo('Load', 'P (kW)', 1/1000),
                'Ppump1': ColumnPlotInfo('Pump 1', 'P (kW)', 1/1000),
                'Ppump2': ColumnPlotInfo('Pump 2', 'P (kW)', 1/1000),
                'Pgrid': ColumnPlotInfo('Grid', 'P (kW)', 1/1000),
                'E_BSS': ColumnPlotInfo('BSS Energy', 'E (kW-h)', 1/1000, 'E_BSS0'),
                'Vw1': ColumnPlotInfo('Reservoir 1', 'Water Volume (m^3)', 1, 'Vw1_0'),
                'Vw2': ColumnPlotInfo('Reservoir 2', 'Water Volume (m^3)', 1, 'Vw2_0'),
                'Qpump1': ColumnPlotInfo('Pump 1 Water', 'Water Flow (m^3/h)', 1),
                'Qpump2': ColumnPlotInfo('Pump 2 Water', 'Water Flow (m^3/h)', 1),
                'Quse1': ColumnPlotInfo('Res. 1 Water Use', 'Water Flow (m^3/h)', 1),
                'Quse2': ColumnPlotInfo('Res. 2 Water Use', 'Water Flow (m^3/h)', 1),
                'Quse_eff': ColumnPlotInfo('Total Effective Water Use', 'Water Flow (m^3/h)', 1)}


def plot_comparison(results_dict, col, include_title=True, params_dict: Mapping[str, TwoPumpModelParameters]=None, ax: Optional[plt.Axes]=None):
    ci = columns_info[col]
    if ax is None:
        fig = plt.figure(figsize=(5, 3))
        ax = plt.axes()
    else:
        fig = ax.figure
    plot_data = pd.DataFrame()
    for k, r in results_dict.items():
        legend_txt = k if include_title else f'{k} {ci.legend_txt}'
        plot_data[legend_txt] = r[col] * ci.scale

    if ci.accumulate_from:
        start_time = plot_data.index[0]
        plot_data.index = plot_data.index + (plot_data.index[1] - plot_data.index[0])
        if params_dict:
            for k in results_dict:
                legend_txt = k if include_title else f'{k} {ci.legend_txt}'
                plot_data.loc[start_time, legend_txt] = params_dict[k].__getattribute__(ci.accumulate_from)*ci.scale
        plot_data = plot_data.sort_index()
    plot_args = dict()
    if not ci.accumulate_from:
        plot_args['drawstyle'] = 'steps-post'
    plot_data.plot(ax=ax, **plot_args)
    ax.set_xlabel('')
    ax.set_ylabel(ci.axis_label)
    if include_title:
        ax.set_title(ci.legend_txt)
    return ax


if __name__ == '__main__':
    seed = 42
    for dayahead_method, intraday_method in [('persistence', 'persistence'),
                                             ('solcast_clouds', 'sarimax'),
                                             ('solcast_clouds', None),
                                             ('oracle', None),
                                             ]:

        precomputed_optimization_models, precomputed_simulation_models = None, None
        optimization_models, simulation_models = optimize_and_simulate_from_actuals(seed,
                                                                                    dayahead_method,
                                                                                    intraday_method,
                                                                                    precomputed_optimization_models,
                                                                                    precomputed_simulation_models)

        summarize_and_plot(optimization_models, simulation_models)
