import logging
import pyomo.environ as pyo
from pyomo.common.tee import capture_output
import pandas as pd
import itertools
logger = logging.getLogger(__name__)

M = 100000
neos_email_address = 'neos@mailinator.com'


def get_model(*e):
    """ Utility function to get the model associated with an expression.
    """
    for e1 in e:
        if hasattr(e1, 'model'):
            return e1.model()
        if hasattr(e1, 'linear_vars'):
            for v in e1.linear_vars:
                m = get_model(v)
                if m is not None:
                    return m
        if hasattr(e1, 'args'):
            for arg in e1.args:
                m = get_model(arg)
                if m is not None:
                    return m

    return None


def linear_max(e1, e2):
    """
    Suppose the return value z = max(e1, e2)
    The following constraints apply:
      e1 <= z <= e2 + M*u
      e2 <= z <= e1 + M(1 - u)
      u: {0, 1}. u = 1 when z == e1, u = 0 when z == e2
    """

    m = get_model(e1, e2)

    if not hasattr(m, '_max_binary_vars'):
        m._max_binary_vars = pyo.VarList(domain=pyo.Binary)
        m._max_output_vars = pyo.VarList()
        m._max_constraints = pyo.ConstraintList()
    u = m._max_binary_vars.add()
    z = m._max_output_vars.add()
    m._max_constraints.add(expr=e1 <= z)
    m._max_constraints.add(expr=z <= e2 + M * u)
    m._max_constraints.add(expr=e2 <= z)
    m._max_constraints.add(expr=z <= e1 + M * (1 - u))
    # Attempt initialization. Pyomo raises ValueError for uninitialized variables
    try:
        e1_val, e2_val = pyo.value(e1), pyo.value(e2)
        z.value = max(e1_val, e2_val)
        u.value = int(e1_val >= e2_val)
    except ValueError as exc:
        logger.debug(exc)
    return z



def linear_maxn(*e):
    """
    Suppose the return value z = max(e1, e2)
    The following constraints apply:
      e1 <= z <= e2 + M*u
      e2 <= z <= e1 + M(1 - u)
      u: {0, 1}. u = 1 when z == e1, u = 0 when z == e2
    """

    m = get_model(*e)
    n = len(e)

    if not hasattr(m, '_max_binary_vars'):
        m._max_binary_vars = pyo.VarList(domain=pyo.Binary)
        m._max_output_vars = pyo.VarList()
        m._max_constraints = pyo.ConstraintList()
    z = m._max_output_vars.add()
    u_list = []
    for e1 in e:
        u = m._max_binary_vars.add()  # Equals 1 for expression that is max.
        u_list.append(u)
        m._max_constraints.add(expr=e1 <= z)
        m._max_constraints.add(expr=z <= e1 + M * (1 - u))
    m._max_constraints.add(expr=pyo.quicksum(u_list) == 1)
    # Attempt initialization. Pyomo raises ValueError for uninitialized variables
    try:
        e_vals = [pyo.value(e1) for e1 in e]
        z.value = max(e_vals)
        for u in u_list:
            u.value = 0
        for e1, e1_val, u in zip(e, e_vals, u_list):
            if e1_val == z.value:
                u.value = 1
                break
    except ValueError as exc:
        logger.debug(exc)
    return z


def linear_min(e1, e2):
    return -1 * linear_max(-1 * e1, -1 * e2)


def linear_minn(*e):
    return -1 * linear_maxn(*[-1*e1 for e1 in e])


def clip(model, x, x_min, x_max):
    return linear_max(linear_min(x, x_max), x_min)


def linear_abs(e):
    """ returns new variable z such that z = |e|, adding linear constraints to the model to enforce this."""
    m = get_model(e)
    if not hasattr(m, '_linear_abs_output_vars'):
        m._linear_abs_binary_vars = pyo.VarList(domain=pyo.Binary)
        m._linear_abs_output_vars = pyo.VarList()
        m._linear_abs_constraints = pyo.ConstraintList()
    z = m._linear_abs_output_vars.add()
    u = m._linear_abs_binary_vars.add()
    m._linear_abs_constraints.add(expr=z <= M*(1 - u) + e)
    m._linear_abs_constraints.add(expr=z >= -M*(1 - u) + e)
    m._linear_abs_constraints.add(expr=z >= -M*u - e)
    m._linear_abs_constraints.add(expr=z <= M*u - e)
    m._linear_abs_constraints.add(expr=z >= 0)
    # Initialize if possible
    try:
        z.value = abs(pyo.value(e))
        u.value = 1 if pyo.value(e) >= 0 else 0
    except ValueError as exc:
        logger.debug(exc)
    return z


def linear_and(e1, e2):
    """ returns new variable z such that z = e1 and e2, adding linear constraints to the model to enforce this."""
    # Linearized formulation from
    # https://yalmip.github.io/tutorial/logicprogramming
    m = get_model(e1, e2)
    if not hasattr(m, '_linear_and_output_vars'):
        m._linear_and_output_vars = pyo.VarList(domain=pyo.Binary)
        m._linear_and_constraints = pyo.ConstraintList()
    z = m._linear_and_output_vars.add()
    m._linear_and_constraints.add(expr=z >= e1 + e2 - 1)
    m._linear_and_constraints.add(expr=z <= e1)
    m._linear_and_constraints.add(expr=z <= e2)
    # Initialize if possible
    try:
        z.value = pyo.value(e1) * pyo.value(e2)
    except ValueError as exc:
        logger.debug(exc)
    return z


def linear_or(e1, e2):
    """ returns new variable z such that z = e1 or e2, adding linear constraints to the model to enforce this."""
    # Linearized formulation from
    # https://yalmip.github.io/tutorial/logicprogramming
    m = get_model(e1, e2)
    if not hasattr(m, '_linear_or_output_vars'):
        m._linear_or_output_vars = pyo.VarList(domain=pyo.Binary)
        m._linear_or_constraints = pyo.ConstraintList()
    z = m._linear_or_output_vars.add()
    m._linear_or_constraints.add(expr=z <= e1 + e2)
    m._linear_or_constraints.add(expr=z >= e1)
    m._linear_or_constraints.add(expr=z >= e2)
    # Initialize if possible
    try:
        z.value = 1 if pyo.value(e1) + pyo.value(e2) > 0 else 0
    except ValueError as exc:
        logger.debug(exc)

    return z


def linear_le_zero(e1):
    """ returns new variable z such that z = 1 if e1 <= 0 otherwise z = 0,
    adding linear constraints to the model to enforce this. Uses Big M."""
    # Idea for linearized formulation from
    # https://yalmip.github.io/tutorial/logicprogramming
    # but then simplified by eliminating extra variables
    m = get_model(e1)
    if not hasattr(m, '_linear_le_zero_output_vars'):
        m._linear_le_zero_output_vars = pyo.VarList(domain=pyo.Binary)
        m._linear_le_zero_constraints = pyo.ConstraintList()
    z = m._linear_le_zero_output_vars.add()
    m._linear_le_zero_constraints.add(expr=e1 <= M*(1 - z))
    m._linear_le_zero_constraints.add(expr=e1 >= -M * z)

    # Initialize if possible
    try:
        z.value = 1 if pyo.value(e1) <= 0 else 0
    except ValueError as exc:
        logger.debug(exc)

    return z


def linear_ge_zero(e1):
    """ returns new variable z such that z = 1 if e1 >= 0 otherwise z = 0,
    adding linear constraints to the model to enforce this. Uses Big M."""
    return linear_le_zero(-1*e1)



def linear_mult_sX_incomplete(s : pyo.Var, X : pyo.Var):
    """ Linearize multiplication of a binary variably and a continuous one.
    X is assumed to have a lower bound >= 0.
    Method described at
    https://www.leandro-coelho.com/linearization-product-variables/"""
    #assert (X.parent_block() is s.model())

    # Temporarily override this function:
    return s*X

    m = X.parent_block()
    if not hasattr(m, '_linear_mult_sX_vars'):
        m._linear_mult_sX_vars = pyo.VarList(domain=pyo.NonNegativeReals)
        m._linear_mult_sX_constraints = pyo.ConstraintList()

    s_val, X_val = [evaluate_expression(v) for v in [s, X]]
    sX = m._linear_mult_sX_vars.add()
    sX.value = s_val*X_val

    m._linear_mult_sX_constraints.add(expr=sX <= s*M)
    m._linear_mult_sX_constraints.add(expr=sX <= X)
    m._linear_mult_sX_constraints.add(expr=sX >= X - (1-s)*M)
    m._linear_mult_sX_constraints.add(expr=sX >= 0)

    return sX


def semicontinuous_var(*indexes, block=None, name=None, bounds=None, initialize=None):
    """
    Returns an expression representing a semi-continuous variable, taking a value of either 0 or between the bounds.
    The expression is formulated using the method shown in 07-MILP-I_handhout.pdf:
    Benoı̂t Chachuat <benoit@mcmaster.ca>
    McMaster University Department of Chemical Engineering
    ChE 4G03: Optimization in Chemical Engineering

    Two auxiliary continuous variables in range (0, 1) and a binary variable are introduced for each index element.
    """
    if block is not None:
        m = block
    elif indexes:
        m = get_model(*indexes)
    else:
        raise ValueError('Either a reference to the block/model must be provided or some indices from which to infer a block/model.')

    no_indices = ((None,),)
    if not indexes:
        indexes = no_indices

    if not hasattr(m, '_semicontinuous_vars'):
        m._semicontinuous_vars = pyo.VarList(domain=pyo.NonNegativeReals)
        m._semicontinuous_binary_vars = pyo.VarList(domain=pyo.Binary)
        m._semicontinous_constraints = pyo.ConstraintList()
    assert bounds[0] > 0
    assert bounds[1] > bounds[0]

    x_l = dict()
    x_u = dict()
    y = dict()
    for i in itertools.product(*indexes):
        x_l[i] = m._semicontinuous_vars.add()
        x_u[i] = m._semicontinuous_vars.add()
        y[i] = m._semicontinuous_binary_vars.add()
        m._semicontinous_constraints.add(expr=y[i] + x_l[i] + x_u[i] == 1)
        # Attempt initialization
        if initialize is not None:
            z_init = initialize(m, *i)
            if z_init == 0:
                y[i].value = 1
                x_l[i].value = 0
                x_u[i].value = 0
            else:
                y[i].value = 0
                x_u[i].value = (z_init - bounds[0])/(bounds[1] - bounds[0])
                x_l[i].value = 1 - x_u[i].value
    if indexes is no_indices:
        z = pyo.Expression(name=name, expr=bounds[0]*x_l[i] + bounds[1]*x_u[i])
    else:
        z = pyo.Expression(*indexes, name=name, rule=lambda m, *i: bounds[0]*x_l[i] + bounds[1]*x_u[i])
    return z


def var_to_df(var):
    return pd.DataFrame.from_records([(v[0], v[1].value) for v in var.items()],
                                     index=var._index.name, columns=[var._index.name, var.name])


def vars_to_df(model):
    # Code from https://stackoverflow.com/questions/67491499/how-to-extract-indexed-variable-information-in-pyomo-model-and-build-pandas-data
    # get all the variables (assuming the fuller model will have constraints, params, etc.)
    model_vars = model.component_objects(ctype=pyo.Var)
    model_exprs = model.component_objects(ctype=pyo.Expression)

    serieses = []   # collection to hold the converted "serieses"
    scalar_vars = dict()

    # Same code seems to work for Vars or Expressions
    for v in itertools.chain(model_vars, model_exprs):
        k = v.name

        # Skip scalar variables and expressions.
        if not v.is_indexed():
            scalar_vars[k] = pyo.value(v)
            continue

        # make a pd.Series from each
        try:
            s = pd.Series([pyo.value(v[i]) for i in v.index_set()],
                          index=v.index_set())
        except ValueError as exc:
            logger.debug(exc)

        # if the series is multi-indexed we need to unstack it...
        if type(s.index[0]) == tuple:  # it is multi-indexed
            s = s.unstack(level=1)
        else:
            s = pd.DataFrame(s)         # force transition from Series -> df
        #print(s)

        # multi-index the columns if length greater than one
        if len(s.columns) > 1:
            s.columns = pd.MultiIndex.from_tuples([(k, t) for t in s.columns])
        else:
            s.columns = [k]

        serieses.append(s)

    df = pd.concat(serieses, axis=1, copy=True)
    df_scalars =  pd.Series(scalar_vars, dtype='float64')
    return df_scalars, df


# From https://github.com/fx-kirin/py-stdlogging/blob/master/stdlogging.py
# but heavily modified.
class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO, default_output=None):
        self.logger = logger
        self.log_level = log_level
        self.default_output = default_output
        self.linebuf = ''

    def write(self, buf):
        if self.default_output:
            self.default_output.write(buf)
        if buf != '':
            self.linebuf += buf
        *lines, self.linebuf = self.linebuf.split('\n')
        for l in lines:
            if l != '':
                self.logger.log(self.log_level, l)

    def flush(self):
        if self.default_output:
            self.default_output.flush()
        for buf in self.linebuf.split('\n'):
            if buf != '':
                self.logger.log(self.log_level, buf.rstrip())
        self.linebuf = ''

def run_solver(model, time_limit=5 * 60, solver_select='scip'):
    if solver_select == 'bonmin':
        # conda install -c conda-forge cyipopt
        # conda install -c conda-forge glpk
        # conda install -c conda-forge coinbonmin

        solver = pyo.SolverFactory('bonmin')
        solver.options['bonmin.algorithm'] = 'B-BB'
        solver.options['bonmin.time_limit'] = time_limit
        with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
            results = solver.solve(model, tee=True)

    elif solver_select == 'mindtpy':
        solver = pyo.SolverFactory('mindtpy')
        solver.options = dict()
        solver.options['mip_solver'] = 'glpk'
        solver.options['nlp_solver'] = 'ipopt'
        solver.options['strategy'] = 'OA'
        solver.options['init_strategy'] = 'rNLP'
        with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
            results = solver.solve(model, tee=True, **solver.options)

    elif solver_select == 'neos-cplex':
        import os
        os.environ['NEOS_EMAIL'] = neos_email_address
        solver_manager = pyo.SolverManagerFactory('neos')
        with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
            results = solver_manager.solve(model, opt='cplex', keepfiles=True, tee=True)

    elif solver_select == 'neos-cbc':
        import os
        os.environ['NEOS_EMAIL'] = neos_email_address
        solver_manager = pyo.SolverManagerFactory('neos')
        with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
            results = solver_manager.solve(model, opt='cbc', keepfiles=True, tee=True)

    elif solver_select == 'neos-minto':
        import os
        os.environ['NEOS_EMAIL'] = neos_email_address
        solver_manager = pyo.SolverManagerFactory('neos')
        with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
            results = solver_manager.solve(model, opt='minto', keepfiles=True, tee=True)

    elif solver_select == 'py':
        solver = pyo.SolverFactory('py')
        results = solver.solve(model)

    elif solver_select == 'cbc':
        # Print linear terms until a nonlinear one is found and raises and exception that we can look into
        # for t in pyomo.core.expr.current._decompose_linear_terms(model.grid_energy_cost):
        #    t.pprint()
        # model_linearized = pyo.TransformationFactory('contrib.induced_linearity').create_using(model)
        solver = pyo.SolverFactory('cbc')
        solver.options['Seconds'] = time_limit
        solver.options['RatioGap'] = 0.01
        with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
            results = solver.solve(model, tee=True, keepfiles=True)

    elif solver_select == 'scip':
        # Print linear terms until a nonlinear one is found and raises and exception that we can look into
        # for t in pyomo.core.expr.current._decompose_linear_terms(model.grid_energy_cost):
        #    t.pprint()
        # model_linearized = pyo.TransformationFactory('contrib.induced_linearity').create_using(model)
        solver = pyo.SolverFactory('scip')
        solver.options['limits/time'] = time_limit
        solver.options['limits/gap'] = 0
        with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
            results = solver.solve(model, tee=True, keepfiles=True)

    elif solver_select == 'glpk':
        # Print linear terms until a nonlinear one is found and raises and exception that we can look into
        # for t in pyomo.core.expr.current._decompose_linear_terms(model.grid_energy_cost):
        #    t.pprint()
        # model_linearized = pyo.TransformationFactory('contrib.induced_linearity').create_using(model)
        solver = pyo.SolverFactory('glpk')
        solver.options['tmlim'] = time_limit
        with capture_output(StreamToLogger(logger, logging.DEBUG)) as solver_output:
            results = solver.solve(model, tee=True, keepfiles=True)
    else:
        raise ValueError(f'Invalid solver selection: {solver_select}')
    # SolverFactory('gdpopt').solve(model, strategy='LOA')

    return results
