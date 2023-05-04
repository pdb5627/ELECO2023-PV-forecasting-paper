"""
Operational optimization problem formulation
Built using Pyomo for modeling
"""
import logging
from pyomo.environ import Var, Expression, Constraint, quicksum
from pyomo.core.expr.current import evaluate_expression
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds
from pyomo.common.tee import capture_output
from typing import Callable, Optional, Sequence, Union
from types import SimpleNamespace
from dataclasses import dataclass

from ems.pyomo_utils import *
from ems.modeling.modeling_window import ModelingWindow

import numpy as np
import numpy.ma as ma
import pandas as pd
from matplotlib import pyplot as plt
logger = logging.getLogger(__name__)

#pyomo.util.infeasible.logger.setLevel(logging.DEBUG)


def num_switches_old(model_var):
    idx = list(model_var.index_set())
    return quicksum(abs(model_var[t] - model_var[t - 1]) for t in idx[1:])


def num_switches(model_var):
    idx = list(model_var.index_set())
    # TODO: This does not include switching in the first period. Consider passing var0 as an optional parameter
    return quicksum(linear_abs(model_var[t] - model_var[t - 1]) for t in idx[1:])


def initializer(data_init, converter: Callable = float):
    if isinstance(data_init, (pd.DataFrame, pd.Series)):
        return lambda m, t: converter(data_init.iloc[t])
    else:
        return lambda m, t: converter(data_init[t])


class PiecewiseFunction:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __call__(self, x):
        return np.interp(x, self.xs, self.ys)


@dataclass
class TwoPumpModelParameters:
    # Model metaparameters
    lookahead: int = 24  # number of periods
    delta_t: float = 1  # hr
    index: Optional[pd.DatetimeIndex] = None  # Used for only for plotting & information

    # Model data
    Pload: Optional[np.array] = None  # W
    P_PVavail: Optional[np.array] = None  # W

    Ppump1_max: float = 2200.  # W
    Qw1: float = 50.  # m^3 / hr

    Ppump2_max: float = 2200.  # W
    Ppump2_min_pu: float = 0.3  # pu of Ppump2_max
    Ppump2_min: Optional[float] = None  # W
    # The following four parameters are used by the default linear Qw2 and Ppump2 functions.
    # Constant efficiency is assumed in the default functions.
    eta_pump: float = 0.4  # pu
    head: float = 44.  # m
    rho: float = 1000.  # kg / m^3
    g: float = 9.81  # m / s^2
    # If the following two parameters are defined, a linear interpolation function will be used rather than
    # the default constant-efficiency function.
    Qpump2_min: Optional[float] = None  # m^3 / hr
    Qpump2_max: Optional[float] = None  # m^3 / hr

    Qw2: Optional[Callable] = None  # Function taking Ppump [W] returning Qpump [m^3 / hr]
    Ppump2: Optional[Callable] = None  # Function taking Qpump [m^3 / hr] returning Ppump [W]

    E_BSS_max: float = 9600.  # W-h
    E_BSS_lower_pu: float = 0.1  # pu of E_BSS_max
    E_BSS_lower: Optional[float] = None  # W-h
    E_BSS_upper_pu: float = 0.95  # pu of E_BSS_max
    E_BSS_upper: Optional[float] = None  # W-h
    E_BSS0_pu: float = 0.3  # pu of E_BSS_max
    E_BSS0: Optional[float] = None  # W-h
    sinv0: int = 0  # Binary. 0 = BSS/PV supplies load. 1 = Grid supplies load
    P_BSS_ch_max_pu: float = 1 / 10  # pu of E_BSS_max / 1hr
    P_BSS_ch_max: Optional[float] = None  # W
    eta_BSS: float = 0.95  # unitless
    E_absorb_pu: float = 0.8  # pu of E_BSS_max
    K_BSS: Optional[float] = None  # unitless, in range 0 to 1

    Vw1_min: float = 5.  # m^3
    Vw1_max: float = 120.  # m^3
    Vw1_0: float = 50.  # m^3
    Vw2_min: float = 5.  # m^3
    Vw2_max: float = 50.  # m^3
    Vw2_0: float = 25.  # m^3

    Vuse_desired: Union[float, Sequence[float]] = 150.  # m^3
    modeling_window: Optional[ModelingWindow] = None  # Not used for the model but helpful to hang on to
    D: Optional[Sequence[Sequence[int]]] = None
    Quse1_max: float = 50.  # m^3 / hr
    Quse2_max: float = 50.  # m^3 / hr

    Cgrid: Optional[np.array] = None  # Cost / W-h
    C_BSS: float = 0.01  # Cost / W-h
    C_BSS_switching: float = 1.  # Cost per switch
    C_pump_switching: float = 1.  # Cost per switch

    Cw_short: float = 100.  # Cost / m^3

    eta_w: Optional[np.array] = None  # unitless

    # eqn:pump-flow-relation
    # Constant efficiency
    def Qw2_default(self, Ppump2):
        return Ppump2 * 3600. * self.eta_pump / (self.head * self.rho * self.g)

    def Ppump2_default(self, Qw2):
        return Qw2 * self.head * self.rho * self.g / (3600. * self.eta_pump)

    # Linear relation based on interpolating between endpoints
    def Qw2_linear(self, Ppump2):
        if Ppump2 >= self.Ppump2_min:
            return self.Qpump2_min + (self.Qpump2_max - self.Qpump2_min) / (self.Ppump2_max - self.Ppump2_min) * (Ppump2 - self.Ppump2_min)
        else:
            return 0

    def Ppump2_linear(self, Qw2):
        if Qw2 >= self.Qpump2_min:
            return self.Ppump2_min + (self.Ppump2_max - self.Ppump2_min) / (self.Qpump2_max - self.Qpump2_min) * (Qw2 - self.Qpump2_min)
        else:
            return 0

    def __post_init__(self):
        # Set defaults for any optional parameters that require creation of new objects or calculations that depend on
        # other parameters

        if self.Pload is None:
            self.Pload = np.random.rand(self.lookahead) * 1e3
        if self.P_PVavail is None:
            self.P_PVavail = np.array([0, 0, 0, 0, 0, 0, 0.4, 2.8, 10.6, 15.7, 19.1, 21.2,
                                       21.7, 20.7, 18.7, 14.8, 9.2, 3.0, 0.3, 0, 0, 0, 0, 0]) * 6 / 22 * 1000.
        assert all(self.P_PVavail >= 0)
        if self.Ppump2_min is None:
            self.Ppump2_min = self.Ppump2_min_pu * self.Ppump2_max

        if self.Qw2 is None and self.Ppump2 is None:
            if self.Qpump2_min is None or self.Qpump2_max is None:
                self.Qw2 = self.Qw2_default
                self.Ppump2 = self.Ppump2_default
                self.Qpump2_min = self.Qw2(self.Ppump2_min)
                self.Qpump2_max = self.Qw2(self.Ppump2_max)
            else:
                P_points = [0, 0.999*self.Ppump2_min, self.Ppump2_min, self.Ppump2_max]
                Q_points = [0, 0.001*self.Qpump2_min, self.Qpump2_min, self.Qpump2_max]
                self.Qw2 = PiecewiseFunction(P_points, Q_points)
                self.Ppump2 = PiecewiseFunction(Q_points, P_points)

        if self.E_BSS_lower is None:
            self.E_BSS_lower = self.E_BSS_lower_pu * self.E_BSS_max
        if self.E_BSS_upper is None:
            self.E_BSS_upper = self.E_BSS_upper_pu * self.E_BSS_max
        if self.E_BSS0 is None:
            self.E_BSS0 = self.E_BSS0_pu * self.E_BSS_max

        if self.P_BSS_ch_max is None:
            self.P_BSS_ch_max = self.E_BSS_max / 10
        # P_BSS_disch is not used to constrain anything, but it is used in the MILP formulation
        self._P_BSS_disch_max = 2*np.max(self.Pload)
        if self.K_BSS is None:
            self.K_BSS = self.P_BSS_ch_max * self.eta_BSS * self.delta_t / ((1 - self.E_absorb_pu) * self.E_BSS_max)
            self.K_BSS = min(self.K_BSS, 1.)

        if self.D is None:
            if self.modeling_window:
                self.D = self.modeling_window.D
            else:
                self.D = (tuple(range(self.lookahead)),)
        # If Vuse_desired is passed as a single value rather than a Sequence, convert it to a tuple
        if not isinstance(self.Vuse_desired, (Sequence, np.ndarray)):
            self.Vuse_desired = tuple(self.Vuse_desired for _ in self.D)

        assert len(self.Vuse_desired) == len(self.D)

        if self.Cgrid is None:
            self.Cgrid = 0.15 * np.ones(self.lookahead)

        if self.eta_w is None:
            self.eta_w = 1.0 * np.ones(self.lookahead)


class TwoPumpModel:
    def __init__(self, params: TwoPumpModelParameters):
        self.params = params
        self._init = None
        self.model = None
        self.solver_output = None
        self.initialized_operation = None
        self.initialized_operation_scalars = None
        self.optimal_operation = None
        self.optimal_operation_scalars = None
        self.simulation_results = None
        self.simulation_scalar_results = None
        self.decision_variables = ['spump1', 'Ppump2', 'Quse1', 'Quse2']

    def initialize(self, init_pumping_method='cheapest', quiet=False):
        """
        Initializes a model with using a greedy pumping heuristic.
        :param init_pumping_method: 'first' for first possible pumping period for Pump 1,
            'cheapest' for running Pump 1 in the cheapest possible pumping period in each day that has inadequate water
        :return: None
        """
        logger.debug('Beginning model initialization.')
        logger.debug(f'Using pumping initialization method: {init_pumping_method}')
        p = self.params
        lookahead = p.lookahead

        eps = 1e-4  # Some constraints may be very slightly exceeded due to numerical issues in the optimization solver

        # The pandas DataFrame throws spurious warnings about trying to set a value on a copy of a slice.
        # I believe this is due to creating the DataFrame one column at a time.
        chained_assignment_setting = pd.get_option('mode.chained_assignment')
        pd.set_option('mode.chained_assignment', None)

        i = pd.DataFrame()
        # Initialization Routine
        # On the first pass, assign all opportunistic pumping periods based on P_PVavail and Pload
        i['sinv'] = np.zeros(lookahead, dtype=np.bool)
        i['sBSS'] = np.ones(lookahead, dtype=np.bool)
        i['spump1'] = np.zeros(lookahead, dtype=np.bool)
        i['E_BSS'] = np.ones(lookahead) * p.E_BSS0
        i['P_PV'] = np.array(p.P_PVavail)
        i['Pavail'] = np.array(p.P_PVavail)
        i['Ppump1'] = np.zeros(lookahead)
        i['Qpump1'] = np.zeros(lookahead)
        i['Ppump2'] = np.zeros(lookahead)
        i['Qpump2'] = np.zeros(lookahead)
        i['P_BSSch'] = np.zeros(lookahead)
        i['P_BSSdisch'] = np.zeros(lookahead)
        i['Quse1'] = np.zeros(lookahead)
        i['Quse2'] = np.zeros(lookahead)
        i['Vw1'] = np.ones(lookahead) * p.Vw1_0
        i['Vw2'] = np.ones(lookahead) * p.Vw2_0
        i['Vuse'] = pd.Series(np.zeros(len(p.D)))

        for t in range(lookahead):
            sinv_prev = i.sinv[t - 1] if t > 0 else p.sinv0
            E_BSS_prev = i.E_BSS[t - 1] if t > 0 else p.E_BSS0
            Vw2_prev = i.Vw2[t - 1] if t > 0 else p.Vw2_0

            # Inverter mode logic copied from model
            below_upper_thresh = E_BSS_prev <= p.E_BSS_upper - eps
            stay_on_utility = sinv_prev and below_upper_thresh
            switch_to_utility = E_BSS_prev <= p.E_BSS_lower - eps
            i.sinv[t] = stay_on_utility or switch_to_utility

            if ~i.sinv[t] and p.P_PVavail[t] < p.Pload[t]:
                # Use battery capacity to supply load
                i.sBSS[t] = 0
                i.P_BSSdisch[t] = min(p.Pload[t] - p.P_PVavail[t], E_BSS_prev * p.eta_BSS / p.delta_t)

            i.Pavail[t] = max(p.P_PVavail[t] - ~i.sinv[t] * p.Pload[t], 0)
            # Greedy water use
            # Check how much water is still needed for the day
            for dn, d in enumerate(p.D):
                if t in d:
                    Vuse = sum(i.Quse2[t2] + i.Quse1[t2] for t2 in d)*p.delta_t
                    break
            else:
                raise ValueError(f'Time period {t=} not in any day ({p.D=})')
            Quse2_max = min(p.Quse2_max, (p.Vuse_desired[dn] - Vuse)/p.delta_t)

            # Opportunistic pumping
            i.Ppump2[t] = min(i.Pavail[t],
                              p.Ppump2((p.Vw2_max - Vw2_prev)/p.delta_t + Quse2_max),
                              p.Ppump2_max)
            if i.Ppump2[t] < p.Ppump2_min - eps:
                i.Ppump2[t] = 0
            i.Quse2[t] = min(Quse2_max, (Vw2_prev - p.Vw2_min)/p.delta_t + p.Qw2(i.Ppump2[t]))
            i.Qpump2[t] = p.Qw2(i.Ppump2[t])
            i.Vw2[t] = Vw2_prev + (i.Qpump2[t] - i.Quse2[t])*p.delta_t
            assert i.Vw2[t] <= p.Vw2_max
            Vuse += p.eta_w[t]*i.Quse2[t]*p.delta_t
            # Any remaining PV available stored in BSS
            if i.Pavail[t] - i.Ppump2[t] > 0:
                i.P_BSSch[t] = min(i.Pavail[t] - i.Ppump2[t],
                                   p.K_BSS * (p.E_BSS_max - E_BSS_prev) / (p.eta_BSS * p.delta_t),
                                   p.P_BSS_ch_max)
            i.P_PV[t] = ~i.sinv[t] * p.Pload[t] + i.Ppump2[t] + i.P_BSSch[t] - i.P_BSSdisch[t]
            #if i.P_BSSch[t] > 0:
            #    i.sBSS[t] = 1
            i.E_BSS[t] = E_BSS_prev \
                         + i.P_BSSch[t] * p.eta_BSS * p.delta_t \
                         - i.P_BSSdisch[t] / p.eta_BSS * p.delta_t

        # After any pumping based on available PV is done, check if water is sufficient
        # This can be found by formulating an LP problem to find the maximum effective water use possible
        # given the assigned pumping schedule.
        infeasible_pumping = np.zeros(lookahead, dtype=np.bool)
        t_pump = None
        while not np.all(i.spump1):
            water_use_model = self.build_water_use_model(i)

            opt = pyo.SolverFactory('glpk')
            #log_infeasible_constraints(water_use_model)
            #log_infeasible_bounds(water_use_model)
            with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
                results = opt.solve(water_use_model, tee=True)

            if results.Solver[0]['Termination condition'] in ('infeasible', 'other'):
                # log_infeasible_constraints(water_use_model)
                # log_infeasible_bounds(water_use_model)
                # log_close_to_bounds(water_use_model)
                if t_pump is None:
                    logger.debug(f'Initialization infeasible even before any pumping is assigned.')
                    break
                else:
                    logger.debug(f'Initialization detected infeasible pumping assignment at t={t_pump}. Backing out.')
                    infeasible_pumping[t_pump] = True
                    i.spump1[t_pump] = False
                    water_use_model = self.build_water_use_model(i)
            else:
                if t_pump is not None:
                    logger.debug(f'Initialization found feasible pumping assignment at t={t_pump}.')
                    infeasible_pumping = np.zeros(lookahead, dtype=np.bool)

            # water_use_model.obj.display()
            # water_use_model.display()
            # water_use_model.pprint()
            Vuse_init_max = water_use_model.obj()
            _, optimal_water_use = vars_to_df(water_use_model)
            i.Quse1 = optimal_water_use['Quse1'].values
            i.Vw1 = optimal_water_use['Vw1'].values
            i.Quse2 = optimal_water_use['Quse2'].values
            i.Vw2 = optimal_water_use['Vw2'].values
            if Vuse_init_max >= sum(p.Vuse_desired)*0.99:
                # Break when water usage is as much as desired. A small fudge factor is used since the
                # numerical results may be slightly less than the threshold.
                break

            # Add more pumping! Do pumping from grid in period with minimum cost in the first day in which there is a
            # deficit of water.
            can_pump = ~i.spump1 & ~infeasible_pumping & ((i.Vw1 + p.Qw1) < p.Vw1_max)
            if init_pumping_method == 'cheapest':
                for dn, d in enumerate(p.D):
                    # Check if pumping is needed on this day. If not, skip ahead
                    if optimal_water_use['Vuse'][dn] >= 0.99*p.Vuse_desired[dn]:
                        continue

                    Cgrid_avail = ma.array(p.Cgrid, mask=~can_pump)
                    # Mask all days except the one being checked
                    for d_other in p.D:
                        if d_other is d:
                            continue
                        for t in d_other:
                            Cgrid_avail[t] = ma.masked
                    # If no available hours, skip to the next day
                    if np.all(Cgrid_avail.mask):
                        continue
                    # Assign pumping to the cheapest hour available
                    t_pump = np.argmin(Cgrid_avail)
                    i.spump1[t_pump] = True
                    break
                else:
                    # If no time was found to assign pumping to, exit the loop. Initialization problem may be infeasible.
                    break
            elif init_pumping_method == 'first':
                Cgrid_avail = ma.array(p.Cgrid, mask=~can_pump)
                t_pump = np.nonzero(Cgrid_avail)[0][0]
                i.spump1[t_pump] = True
            else:
                raise ValueError(f'Invalid init_pumping_method value "{init_pumping_method}"')

        i['Ppump1'] = i.spump1 * p.Ppump1_max
        i['Qpump1'] = i.spump1 * p.Qw1
        i['Pgrid_inverter'] = i.sinv * p.Pload
        i['Pgrid'] = i.Pgrid_inverter + i.Ppump1
        i['P_PV_inverter'] = i.P_PV - i.Ppump2

        i['Quse_eff'] = p.eta_w * (i.Quse1 + i.Quse2)
        i_scalar = self.simulation_objective_function(i).sum()

        i['Vuse_desired'] = pd.Series(p.Vuse_desired).copy()
        i['Pload'] = pd.Series(p.Pload).copy()
        i['P_PVavail'] = pd.Series(p.P_PVavail).copy()

        # Set index to time values rather than integer indices, similar to get_data()
        if self.params.index is not None:
            i.index = self.params.index
        else:
            i.index = i.index * p.delta_t

        # Done with spurious chained assignment warnings, so put setting back where it was
        pd.set_option('mode.chained_assignment', chained_assignment_setting)

        # Assign at end of function to ensure that the data is complete before being assigned
        self._init = i

    def build_water_use_model(self, i):
        p = self.params
        water_use_model = pyo.ConcreteModel()
        water_use_model.t = pyo.RangeSet(0, p.lookahead - 1)
        water_use_model.d = pyo.RangeSet(0, len(p.D) - 1)
        water_use_model.Quse1 = Var(water_use_model.t, bounds=(0, p.Quse1_max), initialize=initializer(i.Quse1))
        water_use_model.Quse2 = Var(water_use_model.t, bounds=(0, p.Quse2_max), initialize=initializer(i.Quse2))
        water_use_model.Vw1 = Expression(water_use_model.t, rule=lambda m, t: p.Vw1_0 + quicksum(
            i.spump1[k] * p.Qw1 - m.Quse1[k] for k in range(t + 1)) * p.delta_t)
        water_use_model.Vw2 = Expression(water_use_model.t, rule=lambda m, t: p.Vw2_0 + quicksum(
            p.Qw2(i.Ppump2[k]) - m.Quse2[k] for k in range(t + 1)) * p.delta_t)

        water_use_model.Quse_eff = Expression(water_use_model.t, rule=lambda m, t:
            p.eta_w[t] * (water_use_model.Quse1[t] + water_use_model.Quse2[t]))
        water_use_model.Vuse = Expression(water_use_model.d, rule=lambda m, d:
            pyo.quicksum(water_use_model.Quse_eff[t] * p.delta_t for t in p.D[d]))
        water_use_model.Vuse_total = pyo.sum_product(water_use_model.Vuse)

        water_use_model.obj = pyo.Objective(expr=water_use_model.Vuse_total, sense=pyo.maximize)
        water_use_model.Vw1_min = Constraint(water_use_model.t, rule=lambda m, t: p.Vw1_min <= m.Vw1[t])
        water_use_model.Vw1_max = Constraint(water_use_model.t, rule=lambda m, t: m.Vw1[t] <= p.Vw1_max)
        water_use_model.Vw2_min = Constraint(water_use_model.t, rule=lambda m, t: p.Vw2_min <= m.Vw2[t])
        water_use_model.Vw2_max = Constraint(water_use_model.t, rule=lambda m, t: m.Vw2[t] <= p.Vw2_max)

        water_use_model.Vuse_limit = Constraint(water_use_model.d, rule=lambda m, d: m.Vuse[d] <= p.Vuse_desired[d])

        return water_use_model

    def build_model(self):
        logger.debug('Building model for optimization')
        model = pyo.ConcreteModel()
        p = self.params
        i = self._init

        # Model index
        lookahead = p.lookahead
        model.t = pyo.RangeSet(0, lookahead - 1)
        model.d = pyo.RangeSet(0, len(p.D) - 1)

        # Decision Variables
        model.spump1 = Var(model.t, domain=pyo.Binary, initialize=initializer(i.spump1))
        if p.Ppump2_min > 0:
            model.Ppump2 = semicontinuous_var(model.t, bounds=(p.Ppump2_min, p.Ppump2_max),
                                              initialize=initializer(i.Ppump2))
        else:
            # Must be == 0
            model.Ppump2 = Var(model.t, bounds=(0, p.Ppump2_max),
                               initialize=initializer(i.Ppump2))
        model.Quse1 = Var(model.t, bounds=(0, p.Quse1_max), initialize=initializer(i.Quse1))  # eqn:var-limit
        model.Quse2 = Var(model.t, bounds=(0, p.Quse2_max), initialize=initializer(i.Quse2))  # eqn:var-limit

        # Intermediate Variables
        model.E_BSS = Var(model.t, bounds=(0, p.E_BSS_max), initialize=initializer(i.E_BSS))

        # eqn:pump1-power
        model.Ppump1 = Expression(model.t, rule=lambda m, t: p.Ppump1_max * m.spump1[t])

        # eqn:pump1-flow
        model.Qpump1 = Expression(model.t, rule=lambda m, t: p.Qw1 * m.spump1[t])

        # eqn:pump2-flow
        model.Qpump2 = Var(model.t, bounds=(0, p.Qpump2_max), initialize=initializer(p.Qw2(i.Ppump2)))
        if hasattr(p.Qw2, 'xs') and hasattr(p.Qw2, 'ys'):
            model.Ppump2v = Var(model.t, bounds=(0, p.Ppump2_max), initialize=initializer(i.Ppump2))
            model.Ppump2v_eq = Constraint(model.t, rule=lambda m, t: m.Ppump2[t] == m.Ppump2v[t])
            model.Qpump2_eq = pyo.Piecewise(model.t, model.Qpump2, model.Ppump2v,
                                            pw_pts=p.Qw2.xs, f_rule=p.Qw2.ys,
                                            pw_constr_type='EQ', pw_repn='INC')  # SOS2 not supported for CBC solver
        else:
            model.Qpump2_eq = Constraint(model.t, rule=lambda m, t: m.Qpump2[t] == p.Qw2(m.Ppump2[t]))

        # eqn:pv-limit
        model.P_PV = Var(model.t, bounds=lambda m, t: (0, float(p.P_PVavail[t])), initialize=initializer(i.P_PV))

        # eqn:power-balance-PV
        model.P_PV_inverter = Expression(model.t, rule=lambda m, t: m.P_PV[t] - m.Ppump2[t])

        # eqn:BSS-mode-charging2
        model.P_BSSch = Var(model.t, bounds=(0, p.P_BSS_ch_max), initialize=initializer(i.P_BSSch))

        model.P_BSSdisch = Var(model.t, bounds=(0, p._P_BSS_disch_max), initialize=initializer(i.P_BSSdisch))

        # eqn:inverter-mode
        def sinv_rule(m, t):
            if t >= 1:
                sinv_prev = m.sinv[t - 1]
                E_BSS_prev = m.E_BSS[t - 1]
                below_upper_thresh = linear_le_zero(E_BSS_prev - p.E_BSS_upper)
                stay_on_utility = linear_and(sinv_prev, below_upper_thresh)
                switch_to_utility = linear_le_zero(E_BSS_prev - p.E_BSS_lower)
                sinv_curr = linear_or(stay_on_utility, switch_to_utility)
            else:
                # In first period, previous values are constants, not expressions
                sinv_prev = p.sinv0
                E_BSS_prev = p.E_BSS0
                below_upper_thresh = (E_BSS_prev - p.E_BSS_upper <= 0)
                stay_on_utility = sinv_prev * below_upper_thresh
                switch_to_utility = 1. if E_BSS_prev <= p.E_BSS_lower else 0.
                sinv_curr = 1. if stay_on_utility or switch_to_utility else 0.
            return sinv_curr

        model.sinv = Expression(model.t, rule=sinv_rule)

        model.P_avail_to_charge = Expression(model.t, rule=lambda m, t:
            float(p.P_PVavail[t]) - m.Ppump2[t] - (1 - m.sinv[t]) * float(p.Pload[t]))
        model.sBSS = Expression(model.t, rule=lambda m, t: linear_ge_zero(m.P_avail_to_charge[t]))

        # eqn:BSS-mode-charging2
        def P_BSS_ch_rule(m, t):
            E_BSS_prev = m.E_BSS[t - 1] if t >= 1 else p.E_BSS0
            absorption_mode_limit = p.K_BSS * (p.E_BSS_max - E_BSS_prev) / (p.eta_BSS * p.delta_t)
            return absorption_mode_limit

        model._BSS_mode_charging1 = Expression(model.t, rule=P_BSS_ch_rule)

        # eqn:BSS-mode-charging3
        model._BSS_mode_charging3 = Expression(model.t, rule=lambda m, t: linear_maxn(m.P_avail_to_charge[t], 0))

        # eqn:????
        model.BSS_mode_charging4 = Constraint(model.t,
                                              rule=lambda m, t: m.P_BSSch[t] == linear_minn(m._BSS_mode_charging1[t],
                                                                                            m._BSS_mode_charging3[t],
                                                                                            m.sBSS[t] * p.P_BSS_ch_max))

        # eqn:BSS-mode-discharging
        model.BSS_mode_discharging = Constraint(model.t, rule=lambda m, t:
            m.P_BSSdisch[t] <= (1 - m.sBSS[t]) * p._P_BSS_disch_max)

        # eqn:power-balance-inverter-grid
        model.Pgrid_inverter = Expression(model.t,
                                          rule=lambda m, t: m.sinv[t] * float(p.Pload[t]))

        # eqn:power-balance-grid
        model.Pgrid = Expression(model.t,
                                 rule=lambda m, t: m.Pgrid_inverter[t] + m.Ppump1[t])

        # eqn:BSS-balance
        model.E_BSS_eq = Constraint(model.t, rule=lambda m, t: m.E_BSS[t] == p.E_BSS0 + quicksum(
            m.P_BSSch[k] * p.eta_BSS - m.P_BSSdisch[k] / p.eta_BSS
            for k in range(t + 1)) * p.delta_t)

        # eqn:water-balance-1
        model.Vw1 = Expression(model.t, rule=lambda m, t: p.Vw1_0 + quicksum(
            m.Qpump1[k] - m.Quse1[k] for k in range(t + 1)) * p.delta_t)
        # eqn:water-balance-2
        model.Vw2 = Expression(model.t, rule=lambda m, t: p.Vw2_0 + quicksum(
            m.Qpump2[k] - m.Quse2[k] for k in range(t + 1)) * p.delta_t)

        # eqn:total-water
        model.Quse_eff = Expression(model.t, rule=lambda m, t:
            p.eta_w[t] * (model.Quse1[t] + model.Quse2[t]))
        model.Vuse = Expression(model.d, rule=lambda m, d:
            pyo.quicksum(model.Quse_eff[t] * p.delta_t for t in p.D[d]))

        # eqn:power-balance-inverter
        model.power_balance_pv = Constraint(model.t,
                                            rule=lambda m, t: m.P_PV_inverter[t] + m.P_BSSdisch[t] - m.P_BSSch[t]
                                                              - (1 - m.sinv[t]) * float(p.Pload[t]) == 0)
        # eqn:grid-inverter-positive
        model.pgrid_inverter_pos = Constraint(model.t, rule=lambda m, t: m.Pgrid_inverter[t] >= 0)

        # eqn:pv-inverter-positive
        model.p_PV_inverter_pos = Constraint(model.t, rule=lambda m, t: m.P_PV_inverter[t] >= 0)

        # eqn:var-limit
        model.Vw1_min = Constraint(model.t,
                                   rule=lambda m, t: p.Vw1_min <= m.Vw1[t])
        model.Vw1_max = Constraint(model.t,
                                   rule=lambda m, t: m.Vw1[t] <= p.Vw1_max)
        model.Vw2_min = Constraint(model.t,
                                   rule=lambda m, t: p.Vw2_min <= m.Vw2[t])
        model.Vw2_max = Constraint(model.t,
                                   rule=lambda m, t: m.Vw2[t] <= p.Vw2_max)

        # Objective Function
        # eqn:objective-function
        model.grid_energy_cost = Expression(expr=quicksum(p.Cgrid[t] * model.Pgrid[t] * p.delta_t for t in model.t))
        model.battery_use_cost = Expression(
            expr=quicksum(p.C_BSS * (model.P_BSSch[t] + model.P_BSSdisch[t]) * p.delta_t for t in model.t))
        model.inadequate_water = Expression(expr=quicksum(linear_maxn(p.Vuse_desired[d] - model.Vuse[d], 0) for d in model.d))
        model.inadequate_water_cost = Expression(expr=p.Cw_short * model.inadequate_water)
        model.BSS_mode_switching_cost = Expression(expr=p.C_BSS_switching * num_switches(model.sBSS))
        model.pump_switching_cost = Expression(expr=p.C_pump_switching * num_switches(model.spump1))
        model.TOTAL_COST = Expression(expr=model.grid_energy_cost
                                           + model.battery_use_cost
                                           + model.inadequate_water_cost
                                           + model.BSS_mode_switching_cost
                                           + model.pump_switching_cost)

        model.obj = pyo.Objective(expr=model.TOTAL_COST)

        # Assign at end of function to ensure that the data is complete before being assigned
        # Optimization model was built
        self.model = model
        self.initialized_operation_scalars, self.initialized_operation = self.get_values()
        self.optimal_operation_scalars, self.optimal_operation = None, None

    def optimize(self, time_limit=5 * 60, solver_select='scip'):
        model = self.model
        p = self.params
        self.solver_output = None

        pre_optimization_obj_value = evaluate_expression(model.obj)

        pre_optimization = [f'Starting solution costs:',
                            f'    Grid energy cost: {evaluate_expression(model.grid_energy_cost):.1f}',
                            f'    Battery use cost: {evaluate_expression(model.battery_use_cost):.1f}',
                            f'    Inadequate water cost: {evaluate_expression(model.inadequate_water_cost):.1f}'
                                f' ({evaluate_expression(model.inadequate_water) / sum(p.Vuse_desired) * 100:.1f}% of desired water not supplied)',
                            f'    Battery mode switching cost: {evaluate_expression(model.BSS_mode_switching_cost):.1f}',
                            f'    Pump switching cost: {evaluate_expression(model.pump_switching_cost):.1f}',
                            f'    TOTAL: {pre_optimization_obj_value:.1f}']

        for l in pre_optimization:
            logger.info(l)

        # Check if starting solution is feasible
        #log_infeasible_constraints(model)  # Spews lots of junk if there are uninitialized values

        # Solve
        logger.info(f'Solving with solver_select={solver_select}')

        results = run_solver(model, time_limit=time_limit, solver_select=solver_select)

        if results.Solver[0]['Termination condition'] == 'infeasible':
            log_infeasible_constraints(model)
            log_infeasible_bounds(model)
        #     log_close_to_bounds(model)

        # model.obj.display()
        # model.display()
        # model.pprint()

        self.solver_output = results

        post_optimization_obj_value = evaluate_expression(model.obj)
        post_optimization = ['Optimal solution costs:',
                             f'    Grid energy cost: {evaluate_expression(model.grid_energy_cost):.1f}',
                             f'    Battery use cost: {evaluate_expression(model.battery_use_cost):.1f}',
                             f'    Inadequate water cost: {evaluate_expression(model.inadequate_water_cost):.1f}'
                                f' ({evaluate_expression(model.inadequate_water) / sum(p.Vuse_desired) * 100:.1f}% of desired water not supplied)',
                             f'    Battery mode switching cost: {evaluate_expression(model.BSS_mode_switching_cost):.1f}',
                             f'    Pump switching cost: {evaluate_expression(model.pump_switching_cost):.1f}',
                             f'    TOTAL: {post_optimization_obj_value:.1f}']

        for l in itertools.chain(post_optimization):
            logger.info(l)
        logger.info(f'Objective value change: {pre_optimization_obj_value:.1f} -> {post_optimization_obj_value:.1f}')
        logger.info(f'Ratio of optimal / initial: {post_optimization_obj_value/pre_optimization_obj_value:.2f}')

    def get_values(self, model=None):
        if model is None:
            model = self.model
        scalar_vars, array_vars = vars_to_df(model)
        array_vars = array_vars[:self.params.lookahead]
        # Convert binary columns to boolean values
        # astype converting from float to bool sets anything non-zero to True, even small values like 1e-10.
        for col in ['spump1', 'sinv', 'sBSS']:
            array_vars[col] = (array_vars[col] > 0.5)
        # Add some other columns of interest
        array_vars['Vuse_desired'] = pd.Series(self.params.Vuse_desired)
        array_vars['Pload'] = pd.Series(self.params.Pload)
        array_vars['P_PVavail'] = pd.Series(self.params.P_PVavail)
        if self.params.index is not None:
            array_vars.index = self.params.index
        else:
            array_vars.index = array_vars.index*self.params.delta_t

        return scalar_vars, array_vars

    def get_optimal(self):
        self.optimal_operation_scalars, self.optimal_operation = self.get_values()
        return self.optimal_operation

    def plot_optimal(self):
        o = self.optimal_operation
        o[['P_BSSch', 'P_BSSdisch', 'Ppump1', 'Ppump2', 'Pload', 'P_PV']].plot(drawstyle='steps')
        o[['E_BSS']].plot()
        o[['Qpump1', 'Quse1', 'Qpump2', 'Quse2']].plot(drawstyle='steps')
        o[['Vw1', 'Vw2']].plot()
        o[['sinv', 'sBSS', 'spump1']].plot(include_bool=True, drawstyle='steps', subplots=True)
        return

    def plot_optimization_results(self, which_results='optimization', manual_results=None):
        """
        :param which_results: str to represent which results to plot. Possible values are
                'optimization': Optimized operation (default)
                'initialization': Initialized operation
                'simulation': Simulated operation
                'manual': DataFrame of results passed in
        :param manual_results: DataFrame of results to plot
        :return: figure object with subplots.
        """
        if which_results == 'optimization':
            o = self.optimal_operation
        elif which_results == 'initialization':
            o = self.initialized_operation
        elif which_results == 'simulation':
            o = self.simulation_results
        elif which_results == 'manual':
            o = manual_results
        else:
            raise ValueError(f'Invalid value for which_results provided: {which_results}')

        fig = plt.figure(figsize=(10.64, 4.40))

        # Manually setting the axis positions so that I can tweak each plot size independently
        # without messing up the position of everything on the page.
        l_margin = 0.04
        r_margin = 0.015
        t_margin = 0.04
        b_margin = 0.08
        ax_width = 1 / 3 - l_margin - r_margin
        ax_height = 1 / 2 - t_margin - b_margin
        ax = fig.add_axes([l_margin, 0.5 + b_margin, ax_width, ax_height])  # Position: [left, bottom, width, height]
        ax.set_title('PV-side Power')
        plot_data = pd.DataFrame()
        plot_data['PV available'] = o['P_PVavail']
        plot_data['PV utilized'] = o['P_PV']
        plot_data['BSS (discharge)'] = o['P_BSSdisch']
        plot_data['BSS (charge)'] = -1 * o['P_BSSch']
        plot_data['Load'] = -1 * o['Pload'] * (1 - o['sinv'])
        plot_data['Pump 2'] = -1 * o['Ppump2']
        plot_data /= 1000
        ax = plot_data.plot(ax=ax, drawstyle='steps-post')
        ax.xaxis.grid(True, which='minor')
        # plt.ylim(top=0.9)
        ax.set_xlabel(None)
        ax.set_ylabel('Power (kW)')
        ax.legend(loc="upper right")

        ht_factor = 0.7
        ax = fig.add_axes([l_margin, b_margin + ax_height * (1 - 0.7), ax_width, ax_height * 0.7])
        ax.set_title('Grid-side Power')
        plot_data = pd.DataFrame()
        plot_data['Grid'] = o['Pgrid']
        plot_data['Load'] = -1 * o['Pload'] * o['sinv']
        plot_data['Pump 1'] = -1 * o['Ppump1']
        plot_data /= 1000
        ax = plot_data.plot(ax=ax, drawstyle='steps-post')
        ax.set_xlabel(None)
        ax.set_ylabel('Power (kW)')
        ax.legend(loc="upper right")

        ht_factor = 0.7
        ax = fig.add_axes(
            [1 / 3 + l_margin, 0.5 + b_margin + ax_height * (1 - ht_factor), ax_width, ax_height * ht_factor])
        ax.set_title('BSS Energy')
        plot_data = pd.DataFrame()
        plot_data['BSS Energy'] = o['E_BSS']
        start_time = plot_data.index[0]
        plot_data.index = plot_data.index + (plot_data.index[1] - plot_data.index[0])
        plot_data.loc[start_time, 'BSS Energy'] = self.params.E_BSS0
        plot_data = plot_data.sort_index()
        plot_data /= 1000

        ax = plot_data.plot(ax=ax)

        ax.set_xlabel(None)
        ax.set_ylabel('Energy (kW-h)')
        ax.legend(loc="upper right")

        ax = fig.add_axes([1 / 3 + l_margin, b_margin, ax_width, ax_height])
        ax.set_title('BSS Power')
        plot_data = pd.DataFrame()
        plot_data['Charging'] = o['P_BSSch']
        plot_data['Discharging'] = -1 * o['P_BSSdisch']
        plot_data /= 1000

        ax = plot_data.plot(ax=ax, drawstyle='steps-post')

        ax.set_xlabel(None)
        ax.set_ylabel('Power (kW)')
        ax.legend(loc="upper right")

        table_ht = 0.3
        ax_height = (1. - table_ht) / 2 - b_margin - t_margin
        ax = fig.add_axes([2 / 3 + l_margin, table_ht + 2 * b_margin + ax_height + t_margin, ax_width, ax_height])
        ax.set_title('Reservoir Water Volumes')
        plot_data = pd.DataFrame()
        plot_data['Res. 1'] = o['Vw1']
        plot_data['Res. 2'] = o['Vw2']

        start_time = plot_data.index[0]
        plot_data.index = plot_data.index + (plot_data.index[1] - plot_data.index[0])
        plot_data.loc[start_time, 'Res. 1'] = self.params.Vw1_0
        plot_data.loc[start_time, 'Res. 2'] = self.params.Vw2_0
        plot_data = plot_data.sort_index()

        ax = plot_data.plot(ax=ax)

        ax.set_xlabel(None)
        ax.set_ylabel('Water Volume (m^3)')

        ax.legend(loc="upper right")

        ax = fig.add_axes([2 / 3 + l_margin, table_ht + b_margin, ax_width, ax_height])
        ax.set_title('Water Usage')
        plot_data = pd.DataFrame()
        plot_data['Res. 1'] = o['Quse1']
        plot_data['Res. 2'] = o['Quse2']

        ax = plot_data.plot(ax=ax, drawstyle='steps-post')

        ax.set_xlabel(None)
        ax.set_ylabel('Water Use (m^3/h)')
        ax.legend(loc="upper right")

        ax = fig.add_axes([2 / 3 + l_margin, 0.03, ax_width, table_ht - 0.03])
        ax.set_axis_off()

        if which_results == 'initialization':
            ax.text(0.75, 0.5, 'Plotting initialized values',
                    ha='left', va='top', transform=ax.transAxes)
        elif which_results == 'optimization':
            if self.initialized_operation_scalars is not None and self.optimal_operation_scalars is not None:
                table_data = pd.DataFrame({'Initialized': self.initialized_operation_scalars,
                                           'Optimized': self.optimal_operation_scalars}).round(1)
                table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                                 rowLabels=table_data.index, colWidths=[0.25, 0.25], loc='center right')
            else:
                ax.text(0.75, 0.5, 'Plotting optimized values',
                        ha='left', va='top', transform=ax.transAxes)
        elif which_results == 'simulation':
            ax.text(0.75, 0.5, 'Plotting simulated values',
                    ha='left', va='top', transform=ax.transAxes)
        elif which_results == 'manual':
            ax.text(0.75, 0.5, 'Plotting manually provided values',
                    ha='left', va='top', transform=ax.transAxes)

        # plt.show(block=True)
        return fig

    def do_all(self, optimize=True, plots=False):
        self.initialize()
        self.build_model()
        if optimize:
            self.optimize(solver_select='scip')
        self.get_optimal()
        if plots:
            self.plot_optimal()

    def log_infeasible_constraints(self):
        log_infeasible_constraints(self.model)
        log_infeasible_bounds(self.model)

    def simulate_operation(self, decision_variables: pd.DataFrame, n_steps : int):
        p = self.params

        eps = 1e-4  # Some constraints may be very slightly exceeded due to numerical issues in the optimization solver

        # The pandas DataFrame throws spurious warnings about trying to set a value on a copy of a slice.
        # I believe this is due to creating the DataFrame one column at a time.
        chained_assignment_setting = pd.get_option('mode.chained_assignment')
        pd.set_option('mode.chained_assignment', None)

        i = pd.DataFrame()
        # Initialization Routine
        # On the first pass, assign all opportunistic pumping periods based on P_PVavail and Pload
        i['sinv'] = np.zeros(n_steps, dtype=np.bool)
        i['sBSS'] = np.ones(n_steps, dtype=np.bool)
        i['spump1'] = np.zeros(n_steps, dtype=np.bool)
        i['E_BSS'] = np.ones(n_steps) * p.E_BSS0
        i['P_PV'] = np.array(p.P_PVavail[:n_steps])
        i['P_avail_to_charge'] = np.array(p.P_PVavail[:n_steps])
        i['Ppump1'] = np.zeros(n_steps)
        i['Qpump1'] = np.zeros(n_steps)
        i['Ppump2'] = np.zeros(n_steps)
        i['Qpump2'] = np.zeros(n_steps)
        i['P_BSSch'] = np.zeros(n_steps)
        i['P_BSSdisch'] = np.zeros(n_steps)
        i['Quse1'] = np.zeros(n_steps)
        i['Quse2'] = np.zeros(n_steps)
        i['Vw1'] = np.ones(n_steps) * p.Vw1_0
        i['Vw2'] = np.ones(n_steps) * p.Vw2_0
        i['Vuse'] = pd.Series(np.zeros(len(p.D)))

        for t in range(n_steps):
            sinv_prev = i.sinv[t - 1] if t > 0 else p.sinv0
            E_BSS_prev = i.E_BSS[t - 1] if t > 0 else p.E_BSS0
            Vw1_prev = i.Vw1[t - 1] if t > 0 else p.Vw1_0
            Vw2_prev = i.Vw2[t - 1] if t > 0 else p.Vw2_0

            # Inverter mode logic copied from model
            below_upper_thresh = E_BSS_prev <= p.E_BSS_upper - eps
            stay_on_utility = sinv_prev and below_upper_thresh
            switch_to_utility = E_BSS_prev <= p.E_BSS_lower - eps
            i.sinv[t] = stay_on_utility or switch_to_utility

            # Pump 2 and Quse2 according to decision_variables
            i.Ppump2[t] = min(decision_variables.Ppump2.iloc[t], p.P_PVavail[t], p.Ppump2_max)
            if i.Ppump2[t] < p.Ppump2_min - eps:
                i.Ppump2[t] = 0
            i.Quse2[t] = min(decision_variables.Quse2.iloc[t], p.Quse2_max,
                             (Vw2_prev - p.Vw2_min)/p.delta_t + p.Qw2(i.Ppump2[t]))

            i.Ppump2[t] = min(i.Ppump2[t], p.Ppump2((p.Vw2_max - Vw2_prev)/p.delta_t + i.Quse2[t]))
            if i.Ppump2[t] < p.Ppump2_min:
                i.Ppump2[t] = 0
                i.Qpump2[t] = 0
            else:
                i.Qpump2[t] = p.Qw2(i.Ppump2[t])
            i.Quse2[t] = min(decision_variables.Quse2.iloc[t], p.Quse2_max,
                             (Vw2_prev - p.Vw2_min) / p.delta_t + i.Qpump2[t])

            i.Vw2[t] = Vw2_prev + (i.Qpump2[t] - i.Quse2[t])*p.delta_t
            assert 0.999*p.Vw2_min <= i.Vw2[t] <= p.Vw2_max*1.001

            # Remaining PV power goes to load
            i.P_avail_to_charge[t] = p.P_PVavail[t]- i.Ppump2[t] - ~i.sinv[t] * p.Pload[t]

            if ~i.sinv[t] and i.P_avail_to_charge[t] < 0:
                # Use battery capacity to supply load
                i.sBSS[t] = False
                # It is possible that there is not enough battery energy and some load may not be supplied.
                # E_BSS_min should be set to avoid this situation, but this makes sure E_BSS won't go negative.
                i.P_BSSdisch[t] = min(-i.P_avail_to_charge[t], E_BSS_prev * p.eta_BSS / p.delta_t)

            # Any remaining PV available stored in BSS
            if i.sBSS[t] > 0:
                i.P_BSSch[t] = min(i.P_avail_to_charge[t],
                                   p.K_BSS * (p.E_BSS_max - E_BSS_prev) / (p.eta_BSS * p.delta_t),
                                   p.P_BSS_ch_max)

            i.P_PV[t] = ~i.sinv[t] * p.Pload[t] + i.Ppump2[t] + i.P_BSSch[t] - i.P_BSSdisch[t]
            i.E_BSS[t] = E_BSS_prev \
                         + i.P_BSSch[t] * p.eta_BSS * p.delta_t \
                         - i.P_BSSdisch[t] / p.eta_BSS * p.delta_t

            # Run Pump 1 according to schedule
            prospective_Vw1 = Vw1_prev + (p.Qw1 - decision_variables.Quse1.iloc[t])*p.delta_t
            if decision_variables.spump1.iloc[t] and prospective_Vw1 <= p.Vw1_max + eps:
                i.Ppump1[t] = p.Ppump1_max
                i.Qpump1[t] = p.Qw1
                i.spump1[t] = True
            else:
                i.Ppump1[t] = 0.
                i.Qpump1[t] = 0.
                i.spump1[t] = False
            i.Quse1[t] = min(decision_variables.Quse1.iloc[t], (Vw1_prev - p.Vw1_min)/p.delta_t + i.Qpump1[t])
            i.Vw1[t] = Vw1_prev + (i.Qpump1[t] - i.Quse1[t])*p.delta_t

        i['Pgrid_inverter'] = i.sinv * p.Pload
        i['Pgrid'] = i.Pgrid_inverter + i.Ppump1
        i['P_PV_inverter'] = i.P_PV - i.Ppump2

        i['Quse_eff'] = p.eta_w * (i.Quse1 + i.Quse2)
        i_scalar = self.simulation_objective_function(i).sum()

        i['Vuse_desired'] = pd.Series(p.Vuse_desired).copy()
        i['Pload'] = pd.Series(p.Pload).copy()
        i['P_PVavail'] = pd.Series(p.P_PVavail).copy()

        # Set index to time values rather than integer indices, similar to get_data()
        if p.index is not None:
            i.index = p.index
        else:
            i.index = i.index * p.delta_t

        # Done with spurious chained assignment warnings, so put setting back where it was
        pd.set_option('mode.chained_assignment', chained_assignment_setting)

        # Assign at end of function to ensure that the data is complete before being assigned
        self.simulation_results = i
        self.simulation_scalar_results = i_scalar
        return i

    def simulation_objective_function(self, simulation_results):
        i = simulation_results
        p = self.params

        # For-loop calculation of Vuse
        for d in range(len(p.D)):
            i.Vuse.iat[d] = sum(i.Quse_eff[t] for t in p.D[d]) * p.delta_t

        # Calculate the objective function values
        obj_fn_periods = pd.DataFrame(columns=['grid_energy_cost', 'battery_use_cost', 'inadequate_water',
                                               'inadequate_water_cost', 'BSS_mode_switching_cost',
                                               'pump_switching_cost', 'TOTAL_COST'],
                                      index=i.index, dtype=np.float)
        obj_fn_periods['grid_energy_cost'] = p.Cgrid * i.Pgrid * p.delta_t
        obj_fn_periods['battery_use_cost'] = p.C_BSS * (i.P_BSSch + i.P_BSSdisch) * p.delta_t
        # Inadequate water may have problems with index not matching???
        obj_fn_periods['inadequate_water'] = np.maximum(p.Vuse_desired - i.Vuse.iloc[:len(p.D)], 0)
        obj_fn_periods['inadequate_water_cost'] = p.Cw_short * obj_fn_periods.inadequate_water
        # TODO: Cost of BSS mode switching in the first period is not included here or in pyomo model
        col_idx = obj_fn_periods.columns.get_loc('BSS_mode_switching_cost')
        obj_fn_periods.iat[0, col_idx] = 0.
        obj_fn_periods.iloc[1:, col_idx] = p.C_BSS_switching * np.abs(np.diff(i.sBSS))
        # TODO: Cost of pump switching in the first period is not included here or in pyomo model
        col_idx = obj_fn_periods.columns.get_loc('pump_switching_cost')
        obj_fn_periods.iat[0, col_idx] = 0.
        obj_fn_periods.iloc[1:, col_idx] = p.C_pump_switching * np.abs(np.diff(i.spump1))
        # Fill any NA values with 0
        obj_fn_periods = obj_fn_periods.fillna(0.)
        obj_fn_periods['TOTAL_COST'] = (obj_fn_periods.grid_energy_cost
                                      + obj_fn_periods.battery_use_cost
                                      + obj_fn_periods.inadequate_water_cost
                                      + obj_fn_periods.BSS_mode_switching_cost
                                      + obj_fn_periods.pump_switching_cost)
        return obj_fn_periods


