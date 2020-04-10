import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import sympy
import sympy.printing as printing
import sys,os

# function used to generate the data
def function(x):
    return 3*x**(3.5) + 2

def _pow(x1,x2):
    with np.errstate(over='ignore'):
        return _protected_exponent(x2*_protected_log(x1))

def _protected_exponent(x):
    with np.errstate(over='ignore'):
        return np.where(x < 100, np.exp(x), 2e20)

def _protected_log(x):
    with np.errstate(over='ignore'):
        return np.where(x > 1e-5, np.log(x), -100.0)

exp = make_function(function=_protected_exponent, name='exp', arity=1)
log = make_function(function=_protected_log,      name='log', arity=1)
pow = make_function(function=_pow,                name='pow', arity=2)

################################### INPUT ###########################################
points_training = 1000
points_test     = 150

# symbolic regressor parameters
population_size       = 10000
generations           = 30
tournament_size       = 100
function_set          = ('add', 'sub', 'mul', 'div', exp, log, pow)
metric                = 'mse'
init_depth            = (2, 8)
n_jobs                = 1
verbose               = 1
parsimony_coefficient = 0.0005
random_state          = None

# evolution parameters
# p_crossover + p_subtree_mutation + p_hoist_mutation + p_point_mutation <= 1.0
p_crossover        = 0.3
p_subtree_mutation = 0.25
p_hoist_mutation   = 0.1
p_point_mutation   = 0.25
p_point_replace    = 0.3
#####################################################################################

# get training set
x_train = np.linspace(1,5,points_training)
y_train = np.log10(function(x_train))
x_train = np.reshape(x_train, (-1,1))
y_train = np.reshape(y_train, (-1,))

# get test set
x_test = np.linspace(1,5,points_test)
y_test = np.log10(function(x_test))
x_test = np.reshape(x_test, (-1,1))
y_test = np.reshape(y_test, (-1,))

# define model and hyperparameters
model = SymbolicRegressor(population_size=population_size, generations=generations,
                          tournament_size=tournament_size, function_set=function_set,
                          metric=metric, init_depth=init_depth,
                          verbose=verbose, parsimony_coefficient=parsimony_coefficient,
                          p_crossover=p_crossover, random_state=random_state,
                          p_subtree_mutation=p_subtree_mutation,
                          p_hoist_mutation=p_hoist_mutation,
                          p_point_mutation=p_point_mutation)

# fit model
model.fit(x_train, y_train)

# use model to make predictions
y_pred = model.predict(x_test)
mse    = np.mean((y_test - y_pred)**2)
print('mse = %.3e'%mse)

# print best program
program = model._program
print(program)

# save results to file
x_test = np.reshape(x_test, (-1,))
np.savetxt('results.txt', np.transpose([x_test, y_test, y_pred]))



####### SYMPY #######
converter = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    'neg': lambda x    : -x,
    'pow': lambda x, y : x**y,
}

# simplify expression
program_simp = sympy.sympify(program, locals=converter)

# print simplified result
print(program_simp)
print('Latex formula:')
print(sympy.latex(program_simp))
