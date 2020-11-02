"""This demo program solves Poisson's equation

    - div C grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = 0  for x = 0 or x = 1
du/dn(x, y) = 0  for y = 0 or y = 1

The conductivity C is a symmetric 2 x 2 matrix which
varies throughout the domain. In the left part of the
domain, the conductivity is

    C = ((1, 0.3), (0.3, 2))

and in the right part it is

    C = ((3, 0.5), (0.5, 4))

The data files where these values are stored are generated
by the program generate_data.py

This demo is dedicated to BF and Marius... ;-)
"""

# Copyright (C) 2009-2011 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2009-12-16
# Last changed: 2011-06-28
# Begin demo

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np

from ufl import nabla_grad
from ufl import nabla_div

exec(open("./generate_data.py").read())

# Sub domain for Periodic boundary condition


class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]


# Read mesh from file and create function space
mesh = Mesh("../unitsquare_32_32.xml.gz")
V = FunctionSpace(mesh, "Lagrange", 1)
# V = FunctionSpace(mesh, "Lagrange", 1, constrained_domain=PeriodicBoundary())
# V = VectorFunctionSpace(mesh, "Lagrange", 1,
#                         constrained_domain=PeriodicBoundary())
# Define Dirichlet boundary (x = 0 or x = 1)

# w_1


def boundary_i1(x):
    return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

# w_2


def boundary_i2(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS


def Top(x, on_boundary):
    return on_boundary and near(x[1], 1.)


def Bottom(x, on_boundary):
    return on_boundary and near(x[1], 0.)


# class Pinpoint(SubDomain):
#     TOL = 1e-3

#     def __init__(self, coords):
#         self.coords = np.array(coords)
#         SubDomain.__init__(self)

#     def move(self, coords):
#         self.coords[:] = np.array(coords)

#     def inside(self, x, on_boundary):
#         return np.linalg.norm(x - self.coords) < DOLFIN_EPS

# Pinpoint boundary conditions
# pinpoint1 = Pinpoint([0.5, 0.5])  # center1


class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((x[1] < DOLFIN_EPS or x[1] > (1.0 - DOLFIN_EPS))
                    and on_boundary)


# boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries = MeshFunction("size_t", mesh, 2)
boundaries.set_all(0)
top = AutoSubDomain(Top)
top.mark(boundaries, 1)
bottom = AutoSubDomain(Bottom)
bottom.mark(boundaries, 2)


# Define boundary condition
u0 = Constant(0.0)
# u0 = Constant((0.0, 0.0))
# bc = DirichletBC(V, u0, boundary)
bc1 = DirichletBC(V, u0, boundary_i1)
bc2 = DirichletBC(V, u0, boundary_i2)
# bc3 = DirichletBC(V, u0, pinpoint1, 'pointwise')
bc4 = DirichletBoundary()

# bcs = [
#     bc,
#     # bc1,
#     # bc2,
#     # bc3,
#     # bc4,
# ]

# Define conductivity components as MeshFunctions
c00 = MeshFunction("double", mesh, "../unitsquare_32_32_c00.xml.gz")
c01 = MeshFunction("double", mesh, "../unitsquare_32_32_c01.xml.gz")
c11 = MeshFunction("double", mesh, "../unitsquare_32_32_c11.xml.gz")

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

  // Create expression with 3 components
  Conductivity() : dolfin::Expression(3) {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const override
  {
    const uint cell_index = cell.index;
    values[0] = (*c00)[cell_index];
    values[1] = (*c01)[cell_index];
    values[2] = (*c11)[cell_index];
  }

  // The data stored in mesh functions
  std::shared_ptr<dolfin::MeshFunction<double>> c00;
  std::shared_ptr<dolfin::MeshFunction<double>> c01;
  std::shared_ptr<dolfin::MeshFunction<double>> c11;

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Conductivity, std::shared_ptr<Conductivity>, dolfin::Expression>
    (m, "Conductivity")
    .def(py::init<>())
    .def_readwrite("c00", &Conductivity::c00)
    .def_readwrite("c01", &Conductivity::c01)
    .def_readwrite("c11", &Conductivity::c11);
}

"""

c = CompiledExpression(compile_cpp_code(conductivity_code).Conductivity(),
                       c00=c00, c01=c01, c11=c11, degree=0)

C = as_matrix(((c[0], c[1]), (c[1], c[2])))
C1 = as_vector([c[0], 0.])
C2 = as_vector([0., c[2]])
# C1 = Constant((0.5, 0.))
# C1 = as_matrix([c[0],], [c[2],])

# print('C1:\n', C1)
# print(type(C1))


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
# f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
c000 = Constant(0.5)
c111 = Constant(2.)
g = Constant(0.)
a = inner(C*nabla_grad(u), nabla_grad(v))*dx
# a = inner(grad(u), grad(v))*dx

# L1 = inner(C1, grad(v))*dx
# L1 = 0.5*grad(v)[0]*dx
L1 = c000*nabla_grad(v)[0]*dx
L2 = c111*nabla_grad(v)[1]*dx
# L1 = dot(C1, v)*dx

# print('v', type(v))
# print('grad_v', type(nabla_grad(v)))

# Compute solution
u1 = Function(V)
solve(a == L1, u1, bc1)

A1, f1 = assemble_system(a, L1, bc1)
print('A1:\n', A1.array())
print('f1:\n', f1.get_local())
w1 = np.array(u1.vector())
print('w1:\n', w1)

u2 = Function(V)
solve(a == L2, u2, bc2)

A2, f2 = assemble_system(a, L2, bc2)
print('A2:\n', A2.array())
print('f2:\n', f2.get_local())
w2 = np.array(u2.vector())
print('w2:\n', w2)


# Save solution in VTK format
file = File("poisson1.pvd")
file << u1
file = File("poisson2.pvd")
file << u2


# Plot solution
plt.figure(1)
plot(u1, title='w1')
plt.colorbar(plot(u1, title='w1'))


plt.figure(2)
plot(u2, title='w2')
plt.colorbar(plot(u2, title='w2'))

plt.show()
