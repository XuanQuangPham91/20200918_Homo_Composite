""" 2D Linear elastic with isotropic material """
# %%
from IPython.display import Image, HTML
import matplotlib.pyplot as plt
from dolfin import *
from mshr import *
# from fenics import *
from ufl import nabla_grad
from ufl import nabla_div
import numpy as np
import math

# %%
# Create mesh ----------------------------------------------------------------
R = 0.25  # fiber radius
L = 1.  # Length of the unit cell or representative volume element (RVE)
N = 84  # mesh density
rectangle = Rectangle(Point(0., 0.), Point(L, L))
# domain = rectangle
# domain.set_subdomain(1, rectangle)
circle = Circle(Point(0.5, 0.5), R, segments=2*N)
domain = rectangle
domain.set_subdomain(1, rectangle - circle)
domain.set_subdomain(2, circle)


mesh = generate_mesh(domain, N)
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())

# Periodic boundary condition definition
# Sub domain for Periodic boundary condition
# class PeriodicBoundary(SubDomain):

#     # Left boundary is "target domain" G
#     def inside(self, x, on_boundary):
#         return bool((x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS) or (x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS) and on_boundary)

#     # Map right boundary (H) to left boundary (G)
#     def map(self, x, y):
#         y[0] = x[0] - 1.0
#         y[1] = x[1]


#---------------#
# class PeriodicBoundary(SubDomain):

#     def inside(self, x, on_boundary):
#         # Left boundary is "target domain" L
#         return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)
#         # # Bottom boundary is "target domain" B
#         # return bool(x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS and on_boundary)

#     # Map right boundary (R) to left boundary (L)
#     def map(self, x, y):
#         y[0] = x[0] - 1.0
#         y[1] = x[1]

#     # Map top boundary (T) to bottom boundary (B)
#     # def map(self, x, y):
#     #     y[0] = x[0]
#     #     y[1] = x[1] - 1.0
#---------------#


class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], 1)) or
                          (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
        elif near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.
#
#
# -------------------------------------------------------------------------------


# E and \nu
E_f = 80e9
nu_f = 0.2
E_m = 4.2e9
nu_m = 0.34


class young_modulus_E(UserExpression):
    def set_E_value(self, E_0, E_1):
        self.E_0, self.E_1 = E_0, E_1

    def eval(self, value, x):
        if math.sqrt((x[0]-0.5) ** 2 + (x[1]-0.5) ** 2) <= R + DOLFIN_EPS:
            value[0] = self.E_0
        else:
            value[0] = self.E_1

    def value_shape(self):
        # return (1,)
        return ()


E = young_modulus_E(degree=0)
E.set_E_value(E_f, E_m)


class Nu(UserExpression):
    def set_nu_value(self, nu_0, nu_1):
        self.nu_0, self.nu_1 = nu_0, nu_1

    def eval(self, value, x):
        if math.sqrt(x[0] ** 2 + x[1] ** 2) <= R + DOLFIN_EPS:
            value[0] = self.nu_0
        else:
            value[0] = self.nu_1

    def value_shape(self):
        # return (1,)
        return ()


nu = Nu(degree=0)
nu.set_nu_value(nu_f, nu_m)

# E = 80e9
# nu = 0.3
lambda_1 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
lambda_2 = E / (2.0 * (1.0 + nu))
C11 = C22 = C33 = lambda_1 + 2 * lambda_2
C12 = lambda_1
C66 = lambda_2
C = as_matrix([[C11, C12, 0.], [C12, C22, 0.], [0., 0., C66]])
# S = inv(C)


def eps(v):
    return sym(grad(v))


def strain2voigt(e):
    return as_vector([e[0, 0], e[1, 1], 2*e[0, 1]])


def voigt2stress(s):
    return as_tensor([[s[0], s[2]], [s[2], s[1]]])


def sigma(C, v):
    return voigt2stress(dot(C, strain2voigt(eps(v))))


class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], L) and on_boundary


class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary


class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0) and on_boundary


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary


# exterior boundaries MeshFunction
boundaries = MeshFunction("size_t", mesh, 1)
# boundaries.set_all(0)
Top().mark(boundaries, 1)
Left().mark(boundaries, 2)
Bottom().mark(boundaries, 3)
Right().mark(boundaries, 4)
# ds = Measure('ds')[boundaries]
dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)


# Define function space
# V = VectorFunctionSpace(mesh, 'Lagrange', 2)
V = VectorFunctionSpace(mesh, "Lagrange", 2,
                        constrained_domain=PeriodicBoundary())

# -------------------------------------------------------------------------------
# Auxiliary code for pointwise

# Sub domain for Dirichlet boundary condition -


class Pinpoint(SubDomain):
    TOL = 1e-3

    def __init__(self, coords):
        self.coords = np.array(coords)
        SubDomain.__init__(self)

    def move(self, coords):
        self.coords[:] = np.array(coords)

    def inside(self, x, on_boundary):
        return np.linalg.norm(x - self.coords) < DOLFIN_EPS


pinpoint_center = Pinpoint([0.5, 0.5])  # center
pinpoint_00 = Pinpoint([0., 0.])  # corner
pinpoint_10 = Pinpoint([1., 0.])  # corner
pinpoint_01 = Pinpoint([0., 1.])  # corner
pinpoint_11 = Pinpoint([1., 1.])  # corner

pinpoint_slip_05_00 = Pinpoint([0.5, 0.])  # slip
pinpoint_slip_05_10 = Pinpoint([0.5, 1.0])  # slip
pinpoint_slip_00_05 = Pinpoint([0., 0.5])  # slip
pinpoint_slip_10_05 = Pinpoint([1., 0.5])  # slip

bc_Center = DirichletBC(V, Constant((0, 0)), pinpoint_center, 'pointwise')
bc_4corner_00 = DirichletBC(V, Constant((0, 0)), pinpoint_00, 'pointwise')
bc_4corner_10 = DirichletBC(V, Constant((0, 0)), pinpoint_10, 'pointwise')
bc_4corner_01 = DirichletBC(V, Constant((0, 0)), pinpoint_01, 'pointwise')
bc_4corner_11 = DirichletBC(V, Constant((0, 0)), pinpoint_11, 'pointwise')

bc_slip_05_00 = DirichletBC(V.sub(1), 0.0, pinpoint_slip_05_00, 'pointwise')
bc_slip_05_10 = DirichletBC(V.sub(1), 0.0, pinpoint_slip_05_10, 'pointwise')
bc_slip_00_05 = DirichletBC(V.sub(0), 0.0, pinpoint_slip_00_05, 'pointwise')
bc_slip_10_05 = DirichletBC(V.sub(0), 0.0, pinpoint_slip_10_05, 'pointwise')


u0 = Constant((0., 0.))
# dbc = DirichletBoundary()
# bc0 = DirichletBC(V, u0, dbc)
bcs = [
    # bc0,
    # bc_Center,
    bc_4corner_00,
    bc_4corner_01,
    bc_4corner_10,
    bc_4corner_11,
]
# symmetry boundary conditions
bc0 = [
    # bc_Center,
    # bc_slip_05_00,
    # bc_slip_05_10,
    # DirichletBC(V, Constant((0.0, 0.0)), boundaries, 2),
    # DirichletBC(V.sub(1), 0.0, boundaries, 1),
    # DirichletBC(V.sub(1), 0.0, boundaries, 3),
]
bc1 = [
    # bc_Center,
    # bc_slip_00_05,
    # bc_slip_10_05,
    # DirichletBC(V.sub(0), 0.0, boundaries, 2),
    # DirichletBC(V.sub(0), 0.0, boundaries, 4),
]
bc2 = [
    # DirichletBC(V.sub(0), 0.0, boundaries, 2),
    # DirichletBC(V.sub(0), 0.0, boundaries, 4),
    # DirichletBC(V.sub(1), 0.0, boundaries, 1),
    # DirichletBC(V.sub(1), 0.0, boundaries, 3),
    # bc_Center,
]
# -------------------------------------------------------------------------------


# Define variational problem
# -----------------------
# e_ij = as_matrix([[1., 0., 0.],
#                   [0., 1., 0.],
#                   [0., 0., 1.]])
# e1 = as_vector([1., 0.])
# ------------------------
u = TrialFunction(V)
v = TestFunction(V)
# u = Function(V, name='Displacement')
u0 = Function(V, name='Displacement')
u1 = Function(V, name='Displacement')
u2 = Function(V, name='Displacement')
d = u.geometric_dimension()
# print(d)  # - chieu cua hinh hoc 2D
# ------------------------
a = inner(sigma((C), u), eps(v))*dx


# ------------------------
# l1 = - A_y*nabla_grad(u)[0]*dx  # other expression

# test for load cases i = 1,2,3

# test
# In the assembly these act like (vx, 0), (0, vy) but vx, vy are scalars
# vx, vy = v[0], v[1]
# dofs_x = V.sub(0).dofmap().dofs()
# dofs_y = V.sub(1).dofmap().dofs()
# f0 = Constant((-1.0e10, -1.0e10))
# l0 = dot(f0, v)*dx
l0 = dot(C, strain2voigt(eps(v)))[0]*dx
l1 = dot(C, strain2voigt(eps(v)))[1]*dx
l2 = dot(C, strain2voigt(eps(v)))[2]*dx
# ------------------------


# solve(a == l, u, bc)
# ------------------------
solve(a == l0, u0, bc0)
solve(a == l1, u1, bc1)
solve(a == l2, u2, bc2)
# ------------------------
# %%
# mesh = UnitCubeMesh(10, 10, 10)
# HTML(X3DOM.html(mesh))
HTML(X3DOM.html(u2.cpp_object()))
# print(type(u0))
# print(type(u0.cpp_object()))



# %%
plt.jet()
plt.figure(1)
plt.colorbar(plot(u0, title='u0', mode='displacement'))
# plt.loglog()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.savefig('results/result_grad_e0.eps', format='eps')
# np.savetxt("results/periodic_u0.txt", np.array(u0.vector()), fmt="%s")
# file = File("results/periodic_u0.xml")
# file << u0

plt.figure(2)
plt.colorbar(plot(u1, title='u1', mode='displacement'))
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.savefig('results/result_grad_e1.eps', format='eps')

plt.figure(3)
plt.colorbar(plot(u2, title='u2', mode='displacement'))
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.savefig('results/result_grad_e2.eps', format='eps')

# plt.figure(4, dpi=256)
# p4 = plot(mesh, title='Mesh')
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/mesh.eps', format='eps')

# Save solution to file in VTK format
# vtkfile = File('test/isotropic_elasticity_beam.pvd')
# vtkfile << u

# plt.hold(True)
plt.show()



