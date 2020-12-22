''' 2D Linear elastic with isotropic material '''
''' Note: 2020.11.23
* Test and reference to
  * Macedo2018. Configuration: 1 fiber in the middle and 4 corners
  * Oliveira2009. Configuration: 1 fiber in 4 corners
* Each cofiguration was prepared under comment form. Can be used by decomment.
* Pictures will be saved under eps type.
* Saving Paraview file (.pvd) was prepared under comments and be considered as option.
'''
# %%
import matplotlib.pyplot as plt
from dolfin import *
from mshr import *
import numpy as np
import math
import time

# %%
# Define initial paramemters --------------------------------------------------
# V_f = 0.47  # volume of fiber - J.A Oliveira
V_f = 0.62 / 2  # volume of fiber - Macedo
R = math.sqrt(V_f / math.pi)  # fiber radius
# R = 0.2
L = 1.  # Length of the unit cell or representative volume element (RVE)
N = 120  # mesh density

## Create mesh ----------------------------------------------------------------
rectangle = Rectangle(Point(0., 0.), Point(L, L))
circle = Circle(Point(0.5, 0.5), R, segments=3 * N)
circle_corner_LeftBottom = Circle(Point(0, 0), R, segments=3 * N)
circle_corner_RightBottom = Circle(Point(1, 0), R, segments=3 * N)
circle_corner_LeftTop = Circle(Point(0, 1), R, segments=3 * N)
circle_corner_RightTop = Circle(Point(1, 1), R, segments=3 * N)

domain = rectangle
# domain.set_subdomain(1, rectangle - circle)
# domain.set_subdomain(2, circle)

## Macedo configuration
domain.set_subdomain(
    1, rectangle -
    (circle + circle_corner_RightBottom + circle_corner_LeftBottom +
     circle_corner_LeftTop + circle_corner_RightTop))
domain.set_subdomain(
    2, (circle + circle_corner_RightBottom + circle_corner_LeftBottom +
        circle_corner_LeftTop + circle_corner_RightTop))

# ## Oliveira configuration
# domain.set_subdomain(
#     1, rectangle - (circle_corner_RightBottom + circle_corner_LeftBottom +
#                     circle_corner_LeftTop + circle_corner_RightTop))
# domain.set_subdomain(2, (circle_corner_RightBottom + circle_corner_LeftBottom +
#                          circle_corner_LeftTop + circle_corner_RightTop))

# mesh generated
mesh = generate_mesh(domain, N)
subdomains = MeshFunction("size_t", mesh,
                          mesh.topology().dim(), mesh.domains())

plt.figure()
plot(subdomains)
plt.savefig('results/subdomain.eps', format='eps')
plt.show(block=False)


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

# # Reference J.Pinho-da-Cruz
# E_f = 72e3 # MPa
# nu_f = 0.3
# E_m = 3.5e3
# nu_m = 0.35

# Reference https://comet-fenics.readthedocs.io/en/latest/demo/periodic_homog_elas/periodic_homog_elas.html
# E_f = 210e3
# nu_f = 0.3
# E_m = 50e3
# nu_m = 0.2

## Reference Macedo configuration ---------------------------------------------
E_f = 80e3  # MPa
nu_f = 0.2
E_m = 3.35e3
nu_m = 0.35


class young_modulus_E(UserExpression):
    def set_E_value(self, E_0, E_1):
        self.E_0, self.E_1 = E_0, E_1

    def eval(self, value, x):
        # if (x[0] - 0.5)**2 + (x[1] - 0.5)**2 <= R**2 + DOLFIN_EPS:
        if (x[0] - 0.5)**2 + (x[1] - 0.5)**2 <= R**2 or \
            (x[0])**2 + (x[1])**2 <= R**2 + DOLFIN_EPS or \
            (x[0] - 1.0)**2 + (x[1])**2 <= R**2 + DOLFIN_EPS or \
            (x[0])**2 + (x[1] - 1.0)**2 <= R**2 + DOLFIN_EPS or \
            (x[0] - 1.0)**2 + (x[1] - 1.0)**2 <= R**2 + DOLFIN_EPS:
            value[0] = self.E_0
        else:
            value[0] = self.E_1

    def value_shape(self):
        # return (1,)
        return ()


class Nu(UserExpression):
    def set_nu_value(self, nu_0, nu_1):
        self.nu_0, self.nu_1 = nu_0, nu_1

    def eval(self, value, x):
        # if (x[0] - 0.5)**2 + (x[1] - 0.5)**2 <= R**2 + DOLFIN_EPS:
        if (x[0] - 0.5)**2 + (x[1] - 0.5)**2 <= R**2 or \
            (x[0])**2 + (x[1])**2 <= R**2 + DOLFIN_EPS or \
            (x[0] - 1.0)**2 + (x[1])**2 <= R**2 + DOLFIN_EPS or \
            (x[0])**2 + (x[1] - 1.0)**2 <= R**2 + DOLFIN_EPS or \
            (x[0] - 1.0)**2 + (x[1] - 1.0)**2 <= R**2 + DOLFIN_EPS:
            value[0] = self.nu_0
        else:
            value[0] = self.nu_1

    def value_shape(self):
        # return (1,)
        return ()


### Reference Oliveira configuration -------------------------------------------
# Aluminium-boron compotesite
# E_f = 379.3e3  # MPa
# nu_f = 0.1
# E_m = 68.3e3
# nu_m = 0.3

# class young_modulus_E(UserExpression):
#     def set_E_value(self, E_0, E_1):
#         self.E_0, self.E_1 = E_0, E_1

#     def eval(self, value, x):
#         if (x[0])**2 + (x[1])**2 <= R**2 + DOLFIN_EPS or \
#             (x[0] - 1.0)**2 + (x[1])**2 <= R**2 + DOLFIN_EPS or \
#             (x[0])**2 + (x[1] - 1.0)**2 <= R**2 + DOLFIN_EPS or \
#             (x[0] - 1.0)**2 + (x[1] - 1.0)**2 <= R**2 + DOLFIN_EPS:
#             value[0] = self.E_0
#         else:
#             value[0] = self.E_1

#     def value_shape(self):
#         # return (1,)
#         return ()

# class Nu(UserExpression):
#     def set_nu_value(self, nu_0, nu_1):
#         self.nu_0, self.nu_1 = nu_0, nu_1

#     def eval(self, value, x):
#         # if (x[0] - 0.5)**2 + (x[1] - 0.5)**2 <= R**2 + DOLFIN_EPS:
#         if (x[0])**2 + (x[1])**2 <= R**2 + DOLFIN_EPS or \
#             (x[0] - 1.0)**2 + (x[1])**2 <= R**2 + DOLFIN_EPS or \
#             (x[0])**2 + (x[1] - 1.0)**2 <= R**2 + DOLFIN_EPS or \
#             (x[0] - 1.0)**2 + (x[1] - 1.0)**2 <= R**2 + DOLFIN_EPS:
#             value[0] = self.nu_0
#         else:
#             value[0] = self.nu_1

#     def value_shape(self):
#         # return (1,)
#         return ()

## set E & nu
E = young_modulus_E(degree=0)
E.set_E_value(E_0=E_f, E_1=E_m)
nu = Nu(degree=0)
nu.set_nu_value(nu_0=nu_f, nu_1=nu_m)

# Lame's constants
lambda_1 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
lambda_2 = E / (2.0 * (1.0 + nu))

# model = "plane_strain"
# if model == "plane_stress":
#     lambda_1 = 2 * lambda_2 * lambda_1 / (lambda_1 + 2 * lambda_2)

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
dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
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
u22 = Function(V, name='Displacement-u22')
u12 = Function(V, name='Displacement-u12')

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
    2,
    # shape=(2, 2),
    # symmetry=True,
    constrained_domain=PeriodicBoundary(),
)
V_scalar = FunctionSpace(
    mesh,
    'CG',
    2,
    constrained_domain=PeriodicBoundary(),
)
v_g = TestFunction(V_g)
w_g = TrialFunction(V_g)

a = inner(w_g, v_g) * dx

### ----------------------------------------------------------------------------
## for w11 (or u11)
# L_w11 = inner(grad(u11), v_g) * dx
# grad_u11 = Function(V_g)
# solve(a == L_w11, grad_u11)
grad_u11 = project(grad(u11), V_g)
grad_u11_1_y1, grad_u11_1_y2, grad_u11_2_y1, grad_u11_2_y2 = grad_u11.split(
    deepcopy=True)  # extract

## for w22 (or u22)
# L_w22 = inner(grad(u22), v_g) * dx
# grad_u22 = Function(V_g)
# solve(a == L_w22, grad_u22)
grad_u22 = project(grad(u22), V_g)
grad_u22_1_y1, grad_u22_1_y2, grad_u22_2_y1, grad_u22_2_y2 = grad_u22.split(
    deepcopy=True)

## for w12 (or u12)
# L_w12 = inner(grad(u12), v_g) * dx
# grad_u12 = Function(V_g)
# solve(a == L_w12, grad_u12)
grad_u12 = project(grad(u12), V_g)
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
print('The homogenized effective coefficient tensor (GPa): \n', A_ij / 1e3)

lamda_hom = A_ij[0, 1]
mu_hom = A_ij[2, 2]
E_hom = mu_hom * (3 * lamda_hom + 2 * mu_hom) / (lamda_hom + mu_hom)
nu_hom = lamda_hom / (2 * (lamda_hom + mu_hom))

print('E_hom (GPa): ', E_hom / 1e3)
print('nu_hom: ', nu_hom)

#%%
# saving files
coor = mesh.coordinates()
u11_text = []
u_array = u11.vector().get_local()
print(len(u_array))

# for i in range(len(u_array)):
#     u11_temp = (coor[i][0], coor[i][1], u_array[i])
#     print("u(%8g,%8g) = %g" % (coor[i][0], coor[i][1], u_array[i]))
#     u11_text.append(coor[i][0])
np.savetxt(
    "results/periodic_u11.txt",
    np.array(u11.vector().get_local()),
    #    fmt="%s",
)
np.savetxt("results/periodic_u11_coordinate.txt", np.array(coor))
# file = File("results/periodic_u11.xml")
# file << u11

# %%
''' Visulization '''

plt.jet()

# plt.figure(1)
# plt.colorbar(plot(u11, title='u11', mode='displacement'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_e11.eps', format='eps')

# plt.figure(2)
# plt.colorbar(plot(u22, title='u22', mode='displacement'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_e22.eps', format='eps')

# plt.figure(3)
# plt.colorbar(plot(u12, title='u12', mode='displacement'))
# plt.xlabel('$y_1$')
# plt.ylabel('$y_2$')
# plt.savefig('results/result_grad_e12.eps', format='eps')

# plt.figure(4)
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

# ------------------------------------------------------------------------------
# Save solution to file in VTK format
# vtkfile = File('test/Homo_composite_2D.pvd')
# vtkfile << u11
# vtkfile << u22
# vtkfile << u12

# ------------------------------------------------------------------------------
plt.show()
# time.sleep(3)
# %%
