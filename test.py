from dolfin import *
from mshr import *
# from fenics import *
from ufl import nabla_grad
from ufl import nabla_div

# E and \nu are RB parameters
E = 10.0
nu = 0.3
lambda_1 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
lambda_2 = E / (2.0 * (1.0 + nu))
C11 = C22 = C33 = lambda_1 + 2 * lambda_2
C12 = lambda_1
C66 = lambda_2
C = as_matrix([[C11, C12, 0.], [C12, C22, 0.], [0., 0., C66]])
# S = inv(C)

assemble(C)
print(C)