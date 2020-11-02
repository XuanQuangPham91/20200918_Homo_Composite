from dolfin import *

mesh = UnitCubeMesh(4,4,4)
F = FunctionSpace(mesh, "CG", 2)
V = VectorFunctionSpace(mesh, "CG", 2, dim=3)

u = interpolate(Expression("x[0]*x[1]*x[2]*sin(x[0]*x[1]*x[2])",degree=1),F )
grad_u = project(grad(u), V)

# Create subdomain (x0 = 1)
class Plane(SubDomain):
  def inside(self, x, on_boundary):
    return x[0] > 1.0 - DOLFIN_EPS

# Mark facets
facets = MeshFunction("size_t", mesh, 1)
Plane().mark(facets, 1)
ds = Measure("ds")[facets]

### First method ###
# Define facet normal vector (built-in method)
n = FacetNormal(mesh)
flux_1 = assemble(dot(grad_u, n)*ds(1))

### Second method ###
# Manually define the normal vector
n = Constant((1.0,0.0,0.0))
flux_2 = assemble(dot(grad_u, n)*ds(1))

print ("flux 1: ", flux_1)
print ("flux 2: ", flux_2)