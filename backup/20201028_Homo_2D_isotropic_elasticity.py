""" 2D Linear elastic with isotropic material """

from dolfin import *
from mshr import *
# from fenics import *
from ufl import nabla_grad
from ufl import nabla_div
import numpy as np

L = 1.
N = 16 # mesh density
rectangle = Rectangle(Point(0.,0.), Point(L, L))
domain = rectangle
domain.set_subdomain(1, rectangle)
mesh = generate_mesh(domain, N)
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())

## Periodic boundary condition definition
# # Source term
# class Source(UserExpression):
#     def eval(self, values, x):
#         dx = x[0] - 0.5
#         dy = x[1] - 0.5
#         values[0] = x[0]*sin(5.0*DOLFIN_PI*x[1]) \
#                     + 1.0*exp(-(dx*dx + dy*dy)/0.02)

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((x[1] < DOLFIN_EPS or x[1] > (1.0 - DOLFIN_EPS)) \
                    and on_boundary)

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]

    # Map top boundary (T) to bottom boundary (B)
    def map(self, x, y):
        y[1] = x[1] - 1.0
        y[0] = x[0]
#-------------------------------------------------------------------------------

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

def eps(v):
    return sym(grad(v))
def strain2voigt(e):
    return as_vector([e[0,0], e[1,1], 2*e[0,1]])
def voigt2stress(s):
    return as_tensor([[s[0],s[2]],[s[2],s[1]]])
def sigma(C, v):
    return voigt2stress(dot(C, strain2voigt(eps(v))))

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],L) and on_boundary
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0) and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0) and on_boundary
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],L) and on_boundary

# exterior boundaries MeshFunction
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(0)
Top().mark(boundaries, 1)
Left().mark(boundaries, 2)
Bottom().mark(boundaries, 3)
Right().mark(boundaries, 4)
# ds = Measure('ds')[boundaries]
dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)



# Define function space
# V = VectorFunctionSpace(mesh, 'Lagrange', 2)
V = VectorFunctionSpace(mesh, "Lagrange", 2, constrained_domain=PeriodicBoundary())

#-------------------------------------------------------------------------------
# Auxiliary code for pointwise
class Pinpoint(SubDomain):
    TOL = 1e-3

    def __init__(self, coords):
        self.coords = np.array(coords)
        SubDomain.__init__(self)

    def move(self, coords):
        self.coords[:] = np.array(coords)

    def inside(self, x, on_boundary):
        return np.linalg.norm(x - self.coords) < DOLFIN_EPS

pinpoint1 = Pinpoint([0., 0.])  # center
bc_Center = DirichletBC(V, Constant((0,0)), pinpoint1, 'pointwise')

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((x[1] < DOLFIN_EPS or x[1] > (1.0 - DOLFIN_EPS)) \
                    and on_boundary)
u0 = Constant((0., 0.))
dbc = DirichletBoundary()
bc0 = DirichletBC(V, u0, dbc)
bcs = [ 
    # bc0,
    # bc_Center,
    ]
#-------------------------------------------------------------------------------


# Define variational problem
#-----------------------
e_ij = as_matrix([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])
e1 = as_vector([1., 0.])
#------------------------
du = TrialFunction(V)
v = TestFunction(V)
# u = Function(V, name='Displacement')
u0 = Function(V, name='Displacement')
u1 = Function(V, name='Displacement')
u2 = Function(V, name='Displacement')
d = du.geometric_dimension()
# print(d) - chieu cua hinh hoc 2D
#------------------------
a = inner(sigma((C*e_ij), du), eps(v))*dx

A = C*e_ij
# print(A)
#------------------------

# uniform traction on top boundary
f1 = Constant((0, 1))
f2 = Constant((1, 0))
l = dot(f1, v)*ds(1) + dot(f2, v)*ds(4)

#------------------------
# l1 = - A_y*nabla_grad(u)[0]*dx  # other expression
# e1 = as_vector([1., 0., 0.])
# print(shape(sigma(du)))

### test for load cases i = 1,2,3
## test 1
# l1 = - sigma(C, v)[0]*dx
## test 2
vx, vy = v[0], v[1]  # In the assembly these act like (vx, 0), (0, vy) but vx, vy are scalars
# dofs_x = V.sub(0).dofmap().dofs()
# dofs_y = V.sub(1).dofmap().dofs()
# l1 = - dot(div(sigma(C, u)),v)*dx
l0 = dot(C, strain2voigt(eps(v)))[0]*dx
l1 = dot(C, strain2voigt(eps(v)))[1]*dx
l2 = dot(C, strain2voigt(eps(v)))[2]*dx
# shape(eps(v))

# shape(assemble(l1))
#------------------------

# symmetry boundary conditions
bc0 = [
    # DirichletBC(V, Constant((0., 0.)), boundaries, 3),
    DirichletBC(V, Constant((0., 0.)), boundaries, 4),
    DirichletBC(V, Constant((0., 0.)), boundaries, 2),
    # DirichletBC(V, Constant((0., 0.)), boundaries, 1),
    ]
bc1 = [
    DirichletBC(V, Constant((0., 0.)), boundaries, 3),
    # DirichletBC(V, Constant((0., 0.)), boundaries, 4),
    # DirichletBC(V, Constant((0., 0.)), boundaries, 2),
    DirichletBC(V, Constant((0., 0.)), boundaries, 1),
    ]
bc2 = [
    # DirichletBC(V, Constant((0., 0.)), boundaries, 3),
    # DirichletBC(V, Constant((0., 0.)), boundaries, 4),
    # DirichletBC(V, Constant((0., 0.)), boundaries, 2),
    # DirichletBC(V, Constant((0., 0.)), boundaries, 1),
    bc_Center,
    ]
# solve(a == l, u, bc)
#------------------------
solve(a == l0, u0, bcs)
solve(a == l1, u1, bcs)
solve(a == l2, u2, bcs)
#------------------------

import matplotlib.pyplot as plt
# p = plot(u, title='Displacement of Linear elastic problem with isotropic material', mode='displacement')

plt.jet()
# mesh_plot = plot(mesh, mode='color')

plt.figure(1)
p0 = plot(u0, title='u0', mode='displacement')
plt.colorbar(p0)

plt.figure(2)
p1 = plot(u1, title='u1', mode='displacement')
plt.colorbar(p1)

plt.figure(3)
p2 = plot(u2, title='u2', mode='displacement')
plt.colorbar(p2)


# Save solution to file in VTK format
# vtkfile = File('test/isotropic_elasticity_beam.pvd')
# vtkfile << u

# plt.hold(True)
plt.show()
