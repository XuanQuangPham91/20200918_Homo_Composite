## https://fenicsproject.org/qa/3203/periodic-bcs/


from dolfin import *
# Sub domain for Periodic boundary condition
class PeriodicBoundaryLR(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
          return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
          y[0] = x[0] - 1.0
          y[1] = x[1]

# Sub domain for Periodic boundary condition
class PeriodicBoundaryBT(SubDomain):

    # Bottom boundary is "target domain" G
    def inside(self, x, on_boundary):
          return bool(x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS and on_boundary)

    # Map bottom boundary (H) to top boundary (G)
    def map(self, x, y):
          y[0] = x[0]
          y[1] = x[1] - 1.0

# Mesh
nx = ny = 100
mesh = UnitSquareMesh(nx,ny)

# Periodic BCs
pbcLR = PeriodicBoundaryLR()
pbcBT = PeriodicBoundaryBT()

bcs = [pbcLR,pbcBT]

V = FunctionSpace(mesh,'Lagrange',1,constrained_domain=bcs)