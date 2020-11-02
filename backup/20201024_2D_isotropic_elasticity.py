""" 2D Linear elastic with isotropic material """

from dolfin import *
from mshr import *

L = 1.
N = 2 # mesh density
rectangle = Rectangle(Point(0.,0.), Point(L, L))
domain = rectangle
domain.set_subdomain(1, rectangle)
mesh = generate_mesh(domain, N)
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())

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
def sigma(v):
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
V = VectorFunctionSpace(mesh, 'Lagrange', 2)

# Define variational problem
du = TrialFunction(V)
v = TestFunction(V)
u = Function(V, name='Displacement')
d = du.geometric_dimension()
# print(d) - chieu cua hinh hoc 2D
a = RB_mu * inner(sigma(du), eps(v))*dx

# uniform traction on top boundary
f1 = Constant((0, 1))
f2 = Constant((1, 0))
l = dot(f1, v)*ds(1) + dot(f2, v)*ds(4)

# symmetry boundary conditions
bc = [DirichletBC(V, Constant((0., 0.)), boundaries, 3),
      DirichletBC(V, Constant((0., 0.)), boundaries, 2),]
solve(a == l, u, bc)

import matplotlib.pyplot as plt
p = plot(u, title='Displacement of Linear elastic problem with isotropic material', mode='displacement')
# mesh_plot = plot(mesh, mode='color')
plt.colorbar(p)

# Save solution to file in VTK format
vtkfile = File('test/isotropic_elasticity_beam.pvd')
vtkfile << u

# plt.hold(True)
plt.show()
