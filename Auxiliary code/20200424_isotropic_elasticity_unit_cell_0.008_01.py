from dolfin import *
from dolfin_utils import meshconvert
from fenics import *
from mshr import *
from ufl import nabla_div
import meshio
import matplotlib.pyplot as plt
import numpy as np
import math

R = 0.0027
n = 10
X = 8
Y = 6
Z = 3
# mesh = BoxMesh(0, 0, 0, 8, 6, 3)
mesh = BoxMesh(dolfin.Point(0, 0, 0), dolfin.Point(X, Y, Z), n, n, n)
# mesh = Mesh("data/Job-Elasticity_shear_test_01.xml")
V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)

# E_f = 80e9
# nu_f = 0.2
# E_m = 3.35e9
# nu_m = 0.35
#
#
# class young_modulus_E(UserExpression):
#     def set_E_value(self, E_0, E_1):
#         self.E_0, self.E_1 = E_0, E_1
#
#     def eval(self, value, x):
#         if math.sqrt(x[0] ** 2 + x[1] ** 2) <= R + DOLFIN_EPS:
#             value[0] = self.E_0
#         else:
#             value[0] = self.E_1
#
#     def value_shape(self):
#         # return (1,)
#         return ()
#
#
# E = young_modulus_E(degree=0)
# E.set_E_value(E_f, E_m)
#
#
# class Nu(UserExpression):
#     def set_nu_value(self, nu_0, nu_1):
#         self.nu_0, self.nu_1 = nu_0, nu_1
#
#     def eval(self, value, x):
#         if math.sqrt(x[0] ** 2 + x[1] ** 2) <= R + DOLFIN_EPS:
#             value[0] = self.nu_0
#         else:
#             value[0] = self.nu_1
#
#     def value_shape(self):
#         # return (1,)
#         return ()
#
#
# nu = Nu(degree=0)
# nu.set_nu_value(nu_f, nu_m)

E = 3.35e9
nu = 0.35
# mu = E / (2 * (1 + nu))
# lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
C11 = C22 = C33 = E * (1 - nu) / ((1 - nu * 2) * (1 + nu))
C13 = C12 = E * nu / ((1 + nu) * (1 - 2 * nu))
C44 = C55 = C66 = E / (2 * (1 + nu))
C = as_matrix([[C11, C12, C13, 0., 0., 0.],
               [C12, C22, C13, 0., 0., 0.],
               [C13, C13, C33, 0., 0., 0.],
               [0., 0., 0., C44, 0., 0.],
               [0., 0., 0., 0., C55, 0.],
               [0., 0., 0., 0., 0., C66]])


def epsilon(v):
    return sym(grad(v))


def strain2voigt(e):
    return as_vector([e[0, 0], e[1, 1], e[2, 2], 2 * e[2, 1], 2 * e[2, 0], 2 * e[1, 0]])


def voigt2stress(s):
    return as_tensor([[s[0], s[5], s[4]],
                      [s[5], s[1], s[3]],
                      [s[4], s[3], s[2]]])


def sigma(v):
    return voigt2stress(dot(C, strain2voigt(epsilon(v))))


# def epsilon(v):
#     # return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
#     return sym(grad(v))
# def sigma(u):
#     return lmbda * nabla_div(u) * Identity(d) + 2 * mu * epsilon(u)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        # return near(x[1], 0.003) and between(x[0], (-0.004, 0.004)) and between(x[2], (0.0, 0.003)) and on_boundary
        return on_boundary and near(x[1], 6.0, DOLFIN_EPS) and between(x[2], (0.0, 3.0))


class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        # return near(x[1], -0.003) and between(x[0], (-0.004, 0.004)) and between(x[2], (0.0, 0.003)) and on_boundary
        return on_boundary and near(x[1], 0.0, DOLFIN_EPS) and between(x[2], (0.0, 3.0))


class Left(SubDomain):
    def inside(self, x, on_boundary):
        # return near(x[0], -0.004) and between(x[1], (-0.003, 0.003)) and between(x[2], (0.0, 0.003)) and on_boundary
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS) and between(x[1], (0.0, 6.0)) and between(x[2], (0.0, 3.0))


class Right(SubDomain):
    def inside(self, x, on_boundary):
        # return near(x[0], 0.004) and between(x[1], (-0.003, 0.003)) and between(x[2], (0.0, 0.003)) and on_boundary
        return on_boundary and near(x[0], 8.000, DOLFIN_EPS) and between(x[1], (0.0, 6.0)) and between(x[2], (0.0, 3.0))


class Center0(SubDomain):
    def inside(self, x, on_boundary):
        # return near(x[0], 0.0) and near(x[1], 0.0) and near(x[2], 0.0) and on_boundary
        return on_boundary and abs(x[0]) < DOLFIN_EPS and abs(x[1]) < DOLFIN_EPS and near(x[2], 3.0)


# class Center1(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and abs(x[0]) < DOLFIN_EPS and abs(x[1]) < DOLFIN_EPS and near(x[2], 0.003/2)


class Center2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < DOLFIN_EPS and abs(x[1]) < DOLFIN_EPS and near(x[2], 0.0)


class Pinpoint(SubDomain):
    TOL = 1e-3

    def __init__(self, coords):
        self.coords = np.array(coords)
        SubDomain.__init__(self)

    def move(self, coords):
        self.coords[:] = np.array(coords)

    def inside(self, x, on_boundary):
        return np.linalg.norm(x - self.coords) < DOLFIN_EPS


pinpoint1 = Pinpoint([0., 0., 0.])
pinpoint2 = Pinpoint([0., 0., 3.0])

boundaries = MeshFunction("size_t", mesh, 2)
boundaries.set_all(0)
top = Top()
top.mark(boundaries, 1)
bottom = Bottom()
bottom.mark(boundaries, 2)
left = Left()
left.mark(boundaries, 3)
right = Right()
right.mark(boundaries, 4)
center0 = Center0()
center0.mark(boundaries, 5)
# # center1 = Center1()
# center1.mark(boundaries, 6)
center2 = Center2()
center2.mark(boundaries, 7)

# bc = [DirichletBC(V, Constant((0., 0., 0.)), center0, method="pointwise"),
#       DirichletBC(V, Constant((0., 0., 0.)), center1, method="pointwise")]
bc = [
    DirichletBC(V.sub(0), 0.0, pinpoint1, 'pointwise'),
    DirichletBC(V.sub(0), 0.0, pinpoint2, 'pointwise'),
    # DirichletBC(V, Constant((0., 0., 0.)), center0, method="pointwise"),
    # DirichletBC(V, Constant((0., 0., 0.)), center1, method="pointwise"),
    # DirichletBC(V, Constant((0., 0., 0.)), center2, method="pointwise")
]

# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension()  # space dimension
v = TestFunction(V)
q = 100
f1 = Constant((q, 0, 0))
f2 = Constant((-q, 0, 0))
f3 = Constant((0, -q, 0))
f4 = Constant((0, q, 0))
# dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)
# T = Constant((0, 0, 0))
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f1, v) * ds(4)

# Compute solution
u = Function(V)
solve(a == L, u,
      bc
      )

vtkfile = File('elasticity_results/displacement_01.pvd')
vtkfile << u
# vtkfile = File('test/mesh_trelis_w_boundaries.pvd')
# vtkfile << mesh

# p = plot(u, title="FEniCS solution", mode='displacement')
# p = plot(mesh, title='mesh')
# p = plot(mesh, title='mesh', mode='displacement')
# plt.colorbar(p)
# plt.show()
