""" 2D Linear elastic with isotropic material """
# %%
import matplotlib.pyplot as plt
from dolfin import *
from mshr import *
# from ufl import nabla_grad
# from ufl import nabla_div
import numpy as np
import math
import time

# %%
# Define initial paramemters --------------------------------------------------
V_f = 0.62 / 2  # volume of fiber
# R = math.sqrt(V_f / math.pi)  # 0.4442  # fiber radius
R = 0.25
L = 1.  # Length of the unit cell or representative volume element (RVE)
N = 84  # mesh density

## Create mesh ----------------------------------------------------------------
rectangle = Rectangle(Point(0., 0.), Point(L, L))
# domain = rectangle
# domain.set_subdomain(1, rectangle)
circle = Circle(Point(0.5, 0.5), R, segments=2 * N)
domain = rectangle
domain.set_subdomain(1, rectangle - circle)
domain.set_subdomain(2, circle)

mesh = generate_mesh(domain, N)
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())


## Periodic boundary condition definition
# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0))
                    and (not ((near(x[0], 0) and near(x[1], 1)) or
                              (near(x[0], 1) and near(x[1], 0))))
                    and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
        elif near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
        else:  # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.


### Define elasticity problems -------------------------------------------------
## Define material properties --------------------------------------------------
E_f = 80e9
nu_f = 0.2
E_m = 3.35e9
nu_m = 0.35


class young_modulus_E(UserExpression):
    def set_E_value(self, E_0, E_1):
        self.E_0, self.E_1 = E_0, E_1

    def eval(self, value, x):
        # if math.sqrt((x[0] - 0.5)**2 + (x[1] - 0.5)**2) <= R + DOLFIN_EPS:
        if (x[0] - 0.5)**2 + (x[1] - 0.5)**2 <= R**2 + DOLFIN_EPS:
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
        if (x[0] - 0.5)**2 + (x[1] - 0.5)**2 <= R**2 + DOLFIN_EPS:
            value[0] = self.nu_0
        else:
            value[0] = self.nu_1

    def value_shape(self):
        # return (1,)
        return ()


nu = Nu(degree=0)
nu.set_nu_value(nu_f, nu_m)

# E = 80e9
# nu = 0.2
lambda_1 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
lambda_2 = E / (2.0 * (1.0 + nu))
C1111 = C2222 = lambda_1 + 2 * lambda_2
C1122 = C2211 = lambda_1
C1212 = lambda_2
C = as_matrix([
    [C1111, C1122, 0.],  #
    [C2211, C2222, 0.],  #
    [0., 0., C1212]
])

# S = inv(C)


## Define elasticity tensor ----------------------------------------------------
def eps(v):
    return sym(grad(v))


def strain2voigt(e):
    return as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])


def voigt2stress(s):
    return as_tensor([[s[0], s[2]], [s[2], s[1]]])


def sigma(C, v):
    return voigt2stress(dot(C, strain2voigt(eps(v))))


## Dirichlet boundary conditions
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

bcs = []

### Define variational problem -------------------------------------------------
## Define function space -------------------------------------------------------
V = VectorFunctionSpace(mesh,
                        "Lagrange",
                        2,
                        constrained_domain=PeriodicBoundary())
u = TrialFunction(V)
v = TestFunction(V)

u11 = Function(V, name='Displacement-u11')
# u11.rename('u11', 'solution field u11 of the cell problem')
u22 = Function(V, name='Displacement-u22')
# u.rename('u22', 'solution field u22 of the cell problem')
u12 = Function(V, name='Displacement-u12')
# u.rename('u12', 'solution field u12 of the cell problem')
# d = u.geometric_dimension()  # print(d)  # - chieu cua hinh hoc 2D

## Bilinear form
a = inner(sigma((C), u), eps(v)) * dx

## Linear form for the three load cases
l11 = dot(C, strain2voigt(eps(v)))[0] * dx
l22 = dot(C, strain2voigt(eps(v)))[1] * dx
l12 = dot(C, strain2voigt(eps(v)))[2] * dx

## Call solver
solve(a == l11, u11, bcs)
solve(a == l22, u22, bcs)
solve(a == l12, u12, bcs)

# %%
''' Numerical integration of Homogenized elastic tensor (efficient tensor) '''
''' Compute gradient of u_kl '''

V_g = TensorFunctionSpace(
    mesh,
    'CG',
    1,
    # shape=(2, 2),
    # symmetry=True,
    constrained_domain=PeriodicBoundary(),
)
V_scalar = FunctionSpace(
    mesh,
    'CG',
    1,
    constrained_domain=PeriodicBoundary(),
)
v_g = TestFunction(V_g)
w_g = TrialFunction(V_g)

a = inner(w_g, v_g) * dx

### ----------------------------------------------------------------------------
## for w11 (or u11)
L_w11 = inner(-grad(u11), v_g) * dx
# L_w11 = inner(u11, v_g) * dx
grad_u11 = Function(V_g)
solve(a == L_w11, grad_u11)
grad_u11_1_y1, grad_u11_1_y2, grad_u11_2_y1, grad_u11_2_y2 = grad_u11.split(
    deepcopy=True)  # extract

## for w22 (or u22)
L_w22 = inner(-grad(u22), v_g) * dx
grad_u22 = Function(V_g)
solve(a == L_w22, grad_u22)
grad_u22_1_y1, grad_u22_1_y2, grad_u22_2_y1, grad_u22_2_y2 = grad_u22.split(
    deepcopy=True)

## for w12 (or u12)
L_w12 = inner(-grad(u12), v_g) * dx
grad_u12 = Function(V_g)
solve(a == L_w12, grad_u12)
grad_u12_1_y1, grad_u12_1_y2, grad_u12_2_y1, grad_u12_2_y2 = grad_u12.split(
    deepcopy=True)

###-----------------------------------------------------------------------------
''' Subtitute into equations of homogenized tensor components '''

## k = l = 1
A1111_projected = project(
    C1111 - C1111 * grad_u11_1_y1 - C1122 * grad_u11_2_y2, V_scalar)
A1111_assembled = assemble(A1111_projected * dx)

A2211_projected = project(
    C2211 - C2211 * grad_u11_1_y1 - C2222 * grad_u11_2_y2, V_scalar)
A2211_assembled = assemble(A2211_projected * dx)

# print('The homogenized coefficient A1111: ', A1111_assembled / 1e9)
# print('The homogenized coefficient A2211: ', A2211_assembled / 1e9)

## k = l = 2
A1122_projected = project(
    C1122 - C1111 * grad_u22_1_y1 - C1122 * grad_u22_2_y2, V_scalar)
A1122_assembled = assemble(A1122_projected * dx)

A2222_projected = project(
    C2222 - C2211 * grad_u22_1_y1 - C2222 * grad_u22_2_y2, V_scalar)
A2222_assembled = assemble(A2222_projected * dx)
# print('The homogenized coefficient A1122: ', A1122_assembled / 1e9)
# print('The homogenized coefficient A2222: ', A2222_assembled / 1e9)

## k =1, l = 2
A1212_projected = project(C1212 * (1 - grad_u12_1_y2 - grad_u12_2_y1),
                          V_scalar)
A1212_assembled = assemble(A1212_projected * dx)
# print('The homogenized coefficient A1212: ', A1212_assembled / 1e9)

## The homogenized tensor
A_ij = np.array([[A1111_assembled, A1122_assembled, 0],
                 [A2211_assembled, A2222_assembled, 0],
                 [0, 0, A1212_assembled]])
print('The homogenized effective coefficient tensor (GPa): \n', A_ij / 1e9)

lamda_hom = A_ij[0, 1]
mu_hom = A_ij[2, 2]
E_hom = mu_hom * (3 * lamda_hom + 2 * mu_hom) / (lamda_hom + mu_hom)
nu_hom = lamda_hom / (2 * (lamda_hom + mu_hom))

print('E_hom (GPa): ', E_hom / 1e9)
print('nu_hom: ', nu_hom)

# %%
''' Visulization '''

plt.jet()

plt.figure(1)
plt.colorbar(plot(u11, title='u11', mode='displacement'))
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.savefig('results/result_grad_e11.eps', format='eps')
# np.savetxt("results/periodic_u11.txt", np.array(u11.vector()), fmt="%s")
# file = File("results/periodic_u11.xml")
# file << u11

plt.figure(2)
plt.colorbar(plot(u22, title='u22', mode='displacement'))
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.savefig('results/result_grad_e22.eps', format='eps')

plt.figure(3)
plt.colorbar(plot(u12, title='u12', mode='displacement'))
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.savefig('results/result_grad_e12.eps', format='eps')

# plt.figure(10)
# p4 = plot(mesh, title='Mesh')
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/mesh.eps', format='eps')

## -----------------------------------------------------------------------------

# plt.figure(5)
# plt.colorbar(plot(grad_u11_1_y1, title='grad_u11_1_y1'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_u11_1_y1.eps', format='eps')

# plt.figure(6)
# plt.colorbar(plot(grad_u11_1_y2, title='grad_u11_1_y2'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_u11_1_y2.eps', format='eps')

# plt.figure(7)
# plt.colorbar(plot(grad_u11_2_y1, title='grad_u11_2_y1'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_u11_2_y1.eps', format='eps')

# plt.figure(8)
# plt.colorbar(plot(grad_u11_2_y2, title='grad_u11_2_y2'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_u11_2_y2.eps', format='eps')

# plt.figure(9)
# plt.colorbar(plot(grad_u22_1_y1, title='grad_u22_1_y1'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_u22_1_y1.eps', format='eps')

# plt.figure(10)
# plt.colorbar(plot(grad_u22_1_y2, title='grad_u22_1_y2'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_u22_1_y2.eps', format='eps')

# plt.figure(11)
# plt.colorbar(plot(grad_u22_2_y1, title='grad_u22_2_y1'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_u22_2_y1.eps', format='eps')

# plt.figure(12)
# plt.colorbar(plot(grad_u22_2_y2, title='grad_u22_2_y2'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_u22_2_y2.eps', format='eps')

# plt.figure(13)
# plt.colorbar(plot(grad_u12_1_y1, title='grad_u12_1_y1'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_u12_1_y1.eps', format='eps')

# plt.figure(14)
# plt.colorbar(plot(grad_u12_1_y2, title='grad_u12_1_y2'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_u12_1_y2.eps', format='eps')

# plt.figure(15)
# plt.colorbar(plot(grad_u12_2_y1, title='grad_u12_2_y1'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_u12_2_y1.eps', format='eps')

# plt.figure(16)
# plt.colorbar(plot(grad_u12_2_y2, title='grad_u12_2_y2'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_u12_2_y2.eps', format='eps')

# plt.figure(17)
# plot(mesh, title='Mesh')
# plt.savefig('results/Mesh.eps', format='eps')

# ------------------------------------------------------------------------------

# plt.figure(1)
# plt.colorbar(plot(grad(u11)[0, 0], V_scalar))
# plt.figure(2)
# plt.colorbar(plot(grad(u11)[1, 0], V_scalar))
# plt.figure(3)
# plt.colorbar(plot(grad(u11)[0, 1], V_scalar))
# plt.figure(4)
# plt.colorbar(plot(grad(u11)[1, 1], V_scalar))

# ------------------------------------------------------------------------------
# Save solution to file in VTK format
# vtkfile = File('test/Homo_composite_2D.pvd')
# vtkfile << u11
# vtkfile << u22
# vtkfile << u12

# ------------------------------------------------------------------------------
# plt.show(
# block=False
# )

# time.sleep(3)