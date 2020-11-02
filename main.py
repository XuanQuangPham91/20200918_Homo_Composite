""" The cell problem - 2D thermal diffusion """

import matplotlib.pyplot as plt
# %matplotlib.pyplot.jet()
from dolfin import *
import numpy as np

from ufl import nabla_grad
from ufl import nabla_div

exec(open("./generate_data.py").read())


# Read mesh from file and create function space
mesh = Mesh("./data/unitsquare.xml.gz")
# mesh = UnitSquareMesh.create(24, 24, CellType.Type.quadrilateral)
# (bottom limited for Quad mesh is 24 element for side)
V = FunctionSpace(mesh, "Lagrange", 1)

eps = pow(2, -4)

# Define Dirichlet boundary (x = 0 or x = 1)

# w_1


def boundary_i1(x):
    # return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
    return near(x[0], 0.0, DOLFIN_EPS) or near(x[0], 1.0, DOLFIN_EPS)


# w_2
def boundary_i2(x):
    # return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS
    return near(x[1], 0.0) or near(x[1], 1.0)


# Define boundary condition
u0 = Constant(0.0)
bc1 = DirichletBC(V, u0, boundary_i1)
bc2 = DirichletBC(V, u0, boundary_i2)


# Define conductivity components as MeshFunctions
c00 = MeshFunction("double", mesh, "./data/unitsquare_c00.xml.gz")
c01 = MeshFunction("double", mesh, "./data/unitsquare_c01.xml.gz")
c10 = MeshFunction("double", mesh, "./data/unitsquare_c10.xml.gz")
c11 = MeshFunction("double", mesh, "./data/unitsquare_c11.xml.gz")

# Code for C++ evaluation of conductivity
conductivity_code = """

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>

class Conductivity : public dolfin::Expression
{
public:

// Create expression with 4 components
Conductivity() : dolfin::Expression(4) {}

// Function for evaluating expression on each cell
void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const override
{
const uint cell_index = cell.index;
values[0] = (*c00)[cell_index];
values[1] = (*c01)[cell_index];
values[2] = (*c10)[cell_index];
values[3] = (*c11)[cell_index];
}

// The data stored in mesh functions
std::shared_ptr<dolfin::MeshFunction<double>> c00;
std::shared_ptr<dolfin::MeshFunction<double>> c01;
std::shared_ptr<dolfin::MeshFunction<double>> c10;
std::shared_ptr<dolfin::MeshFunction<double>> c11;

};

PYBIND11_MODULE(SIGNATURE, m)
{
py::class_<Conductivity, std::shared_ptr<Conductivity>, dolfin::Expression>
(m, "Conductivity")
.def(py::init<>())
.def_readwrite("c00", &Conductivity::c00)
.def_readwrite("c01", &Conductivity::c01)
.def_readwrite("c10", &Conductivity::c10)
.def_readwrite("c11", &Conductivity::c11);
}

"""

c = CompiledExpression(compile_cpp_code(conductivity_code).Conductivity(),
                       c00=c00, c01=c01, c10=c10, c11=c11, degree=0)

C = as_matrix(((c[0], c[1]), (c[2], c[3])))


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

A_y = Expression(
    '10 + 9*sin( 2 * pi * x[0] * x[1] * (1-x[0]) * (1-x[1]) )', degree=1, pi=np.pi)
# A_y = Expression('10 + sin(2*pi*x[0]* eps *x[1] * eps *(1-x[0]* eps)*(1-x[1]* eps))', eps=eps, degree=1)


e1 = as_vector([1., 0.])
e2 = as_vector([0., 1.])

a = inner(A_y*nabla_grad(u), nabla_grad(v))*dx
# L1 = - A_y*nabla_grad(u)[0]*dx  # other expression
L1 = -inner(A_y*e1, nabla_grad(u))*dx
L2 = -inner(A_y*e2, nabla_grad(u))*dx


# Compute solution
u1 = Function(V)
solve(a == L1, u1, bc1)

# A1, f1 = assemble_system(a, L1, bc1)
# print('A1:\n', A1.array())
# print('f1:\n', f1.get_local())
# w1 = np.array(u1.vector())
# print('w1:\n', w1)

u2 = Function(V)
solve(a == L2, u2, bc2)

# A2, f2 = assemble_system(a, L2, bc2)
# print('A2:\n', A2.array())
# print('f2:\n', f2.get_local())
# w2 = np.array(u2.vector())
# print('w2:\n', w2)


# Save solution in VTK format
file = File("./results/cell_problem_w1.pvd")
u1.rename("w1", '')
file << u1
file = File("./results/cell_problem_w2.pvd")
u2.rename("w2", '')
file << u2
file = File("./results/cell_problem_mesh.pvd")
mesh.rename("mesh", '')
file << mesh

# ----------------------------------------------------------------------------
""" Calculate numerical integration of Homogenized tensor A*_ij """

# Compute gradient

V_g = VectorFunctionSpace(mesh, 'Lagrange', 1)
v_g = TestFunction(V_g)
w_g = TrialFunction(V_g)

a = inner(w_g, v_g)*dx

# for w1 (or u1)
L_w1 = inner(grad(u1), v_g)*dx
grad_u1 = Function(V_g)
solve(a == L_w1, grad_u1)
grad_u1_y1, grad_u1_y2 = grad_u1.split(deepcopy=True)  # extract components

# Homogenized tensor A11
A11_expression = Expression(
    'A_y * ( 1 + grad_u1_y1 )', A_y=A_y, grad_u1_y1=grad_u1_y1, degree=1)
A11 = assemble(project(A11_expression, V)*dx)

# for w2 (or u2)
L_w2 = inner(grad(u2), v_g)*dx
grad_u2 = Function(V_g)
solve(a == L_w2, grad_u2)
grad_u2_y1, grad_u2_y2 = grad_u2.split(deepcopy=True)  # extract components

# Homogenized tensor A22
A22_expression = Expression(
    'A_y * ( 1 + grad_u2_y2 )', A_y=A_y, grad_u2_y2=grad_u2_y2, degree=1)
A22 = assemble(project(A22_expression, V)*dx)
# print(assemble(A22*dx))

# Homogenized tensor A12
A12_expression = Expression(
    'A_y * ( grad_u2_y1 )', A_y=A_y, grad_u2_y1=grad_u2_y1, degree=1)
A12 = assemble(project(A12_expression, V)*dx)

# Homogenized tensor A21
A21_expression = Expression(
    'A_y * ( grad_u1_y2 )', A_y=A_y, grad_u1_y2=grad_u1_y2, degree=1)
A21 = assemble(project(A21_expression, V)*dx)


A_ij = np.array([[A11, A12], [ A21, A22]])
print('The homogenized effective coefficient matrix: \n', A_ij)
print('The homogenized effective coefficient component A11 = ', A11)
print('The homogenized effective coefficient component A12 = ', A12)
print('The homogenized effective coefficient component A21 = ', A21)
print('The homogenized effective coefficient component A22 = ', A22)
# print(A11-A22)


# ----------------------------------------------------------------------------
""" Compute homogenized problem """


# Define boundary condition
bcs_hom = [
    bc1,
    bc2
]

# Define variational problem
u_hom = TrialFunction(V)
A_hom = C
# * 0.25
# f_hom = Expression('x[0]* eps *(1-x[0]* eps)*x[1]* eps*(1-x[1]* eps)', eps=eps, degree=1)
f_hom = Expression('x[0]*(1-x[0])*x[1]*(1-x[1])', degree=1)

a_hom = inner(A_hom*nabla_grad(u_hom), nabla_grad(v))*dx
L_hom = f_hom*v*dx

# Compute solution
u_hom = Function(V)
solve(a_hom == L_hom, u_hom, bcs_hom)


# epsilon setting
# a_hom_eps = inner(A_y*nabla_grad(u), nabla_grad(v))*dx
# L_hom_eps = f_hom*v*dx
# u_hom_eps = Function(V)
# solve(a_hom_eps == L_hom_eps, u_hom_eps, bcs_hom)


# A_assem_hom, f_assem_hom = assemble_system(a_hom, L_hom, bcs_hom)
# print('A_assem_hom:\n', A_assem_hom.array())
# print('f_assem_hom:\n', f_assem_hom.get_local())
# u1 = np.array(u1.vector())
# print('w1:\n', w1)

# ----------------------------------------------------------------------------
""" Plot solution """
# from mshr import *

# from matplotlib import colors, ticker, cm
# from matplotlib.mlab import bivariate_normal
# from matplotlib.colors import LogNorm
# from vedo.dolfin import plot


plt.jet()

plt.figure(1)
P1 = plot(u1, title='The cell solution $w_1$')
plt.colorbar(P1)
# plt.grid(True)
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.savefig('results/result_w1.eps', format='eps')

plt.figure(2)
P2 = plot(u2, title='The cell solution $w_2$')
plt.colorbar(P2)
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.savefig('results/result_w2.eps', format='eps')

plt.figure(3)
P3 = plot(mesh, title='Mesh')
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.savefig('results/result_mesh.eps', format='eps')

plt.figure(4)
P4 = plot(grad_u1_y1, title='grad_u1_y1')
plt.colorbar(P4)
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.savefig('results/grad_u1_y1.eps', format='eps')

plt.figure(5)
P5 = plot(grad_u1_y2, title='grad_u1_y2')
plt.colorbar(P5)
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.savefig('results/grad_u1_y2.eps', format='eps')

plt.figure(6)
P6 = plot(grad_u2_y1, title='grad_u2_y1')
plt.colorbar(P6)
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.savefig('results/grad_u2_y1.eps', format='eps')

plt.figure(7)
P7 = plot(grad_u2_y2, title='grad_u2_y2')
plt.colorbar(P7)
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.savefig('results/grad_u2_y2.eps', format='eps')

plt.figure(8)
P8 = plot(u_hom, title='The cell solution $u_{hom}$')
# plt.semilogx()
# plt.semilogy()
plt.colorbar(P8)
# plt.grid(True)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.savefig('results/result_u_hom.eps', format='eps')

plt.show()


# ----------------------------------------------------------------------------
""" Utility codes """

# # Import Soya3D
# try:
#     import soya
#     from soya.sphere import Sphere
#     from soya.label3d import Label3D
#     from soya.sdlconst import QUIT
#     _soya_imported = True
# except ImportError:
#     _soya_imported = False


# plt.figure(2)
# element = FiniteElement("BDM", tetrahedron, 3)
# plot(element)
# P9 = plot(u_hom_eps, title='The cell solution $u_{hom\_eps}$')
# plt.colorbar(P9)
# # plt.grid(True)
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.savefig('results/result_u_hom_eps.eps', format='eps')
