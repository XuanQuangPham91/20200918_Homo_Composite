## https://fenicsproject.discourse.group/t/pure-periodic-boundary-conditions/1940


from fenics import *
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend("WebAgg")

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)

# mesh = UnitSquareMesh(20, 20)
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), 50, 50)

# Initialize sub-domain instances
left = Left()
top = Top()
right = Right()
bottom = Bottom()

# Initialize mesh function for boundary domains
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 3)
bottom.mark(boundaries, 4)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)


# Define boundaries
B = "near(x[1], 0)"
T = "near(x[1], 1)"
L = "near(x[0], 0)"
R = "near(x[0], 1)"

class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]

pbc = PeriodicBoundary()

V = VectorFunctionSpace(mesh, 'P', 2, dim=2, constrained_domain=pbc)

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

pinpoint1 = Pinpoint([0., 0.])  # center1


# Boundary conditions
bc_L = DirichletBC(V, Constant((0,0)), L)
bc_R = DirichletBC(V, Constant((1,0)), R)
bc_T = DirichletBC(V, Constant((1,0)), T)
bc_B = DirichletBC(V, Constant((0,0)), B)
bc_Center = DirichletBC(V, Constant((0,0)), pinpoint1, 'pointwise')
bcs = [ 
    bc_T,
    # bc_Center,
    ]





Psi = interpolate(Constant((1,0)), V)
w = TestFunction(V)

F = dot(grad(Psi[0]), grad(w[0]))*dx - Psi[0]*w[0]*dx + (Psi[0]**2+Psi[1]**2)*Psi[0]*w[0]*dx \
   +dot(grad(Psi[1]), grad(w[1]))*dx - Psi[1]*w[1]*dx + (Psi[0]**2+Psi[1]**2)*Psi[1]*w[1]*dx \
   + 0.4*w[0]*ds

solve(F == 0, Psi, bcs)


p = plot(Psi, title="FEniCS solution", mode='displacement')
plt.colorbar(p)
plt.show()