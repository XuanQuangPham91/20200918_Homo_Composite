#!/usr/bin/env python
# coding: utf-8

# # Periodic homogenization of linear elastic materials
# ## Introduction


# In[1]:


from __future__ import print_function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

# Because line magics are only supported by the IPython command line not by Python cl,
# use: 'exec(%matplotlib inline)' instead of %matplotlib inline
'exec(%matplotlib inline)'
# get_ipython().run_line_magic('matplotlib', 'inline') -> wrong syntax for IPython
# get_ipython().magic(u'matplotlib notebook') -> wrong syntax for IPython

a = 1.  # unit cell width
b = sqrt(3.) / 2.  # unit cell height
c = 0.5  # horizontal offset of top boundary
R = 0.2  # inclusion radius
vol = a * b  # unit cell volume
# we define the unit cell vertices coordinates for later use
vertices = np.array([[0, 0.],
                     [a, 0.],
                     [a + c, b],
                     [c, b]])

fname = "hexag_incl"
mesh = Mesh(fname + ".xml")
subdomains = MeshFunction("size_t", mesh, fname + "_physical_region.xml")
facets = MeshFunction("size_t", mesh, fname + "_facet_region.xml")
plt.figure()
plot(subdomains)
plt.show()

# In[2]:


# class used to define the periodic boundary map
class PeriodicBoundary(SubDomain):
    def __init__(self, vertices, tolerance=DOLFIN_EPS):
        """ vertices stores the coordinates of the 4 unit cell corners"""
        SubDomain.__init__(self, tolerance)
        self.tol = tolerance
        self.vv = vertices
        self.a1 = self.vv[1, :] - self.vv[0, :]  # first vector generating periodicity
        self.a2 = self.vv[3, :] - self.vv[0, :]  # second vector generating periodicity
        # check if UC vertices form indeed a parallelogram
        assert np.linalg.norm(self.vv[2, :] - self.vv[3, :] - self.a1) <= self.tol
        assert np.linalg.norm(self.vv[2, :] - self.vv[1, :] - self.a2) <= self.tol

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the 
        # bottom-right or top-left vertices
        return bool((near(x[0], self.vv[0, 0] + x[1] * self.a2[0] / self.vv[3, 1], self.tol) or
                     near(x[1], self.vv[0, 1] + x[0] * self.a1[1] / self.vv[1, 0], self.tol)) and
                    (not ((near(x[0], self.vv[1, 0], self.tol) and near(x[1], self.vv[1, 1], self.tol)) or
                          (near(x[0], self.vv[3, 0], self.tol) and near(x[1], self.vv[3, 1],
                                                                        self.tol)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], self.vv[2, 0], self.tol) and near(x[1], self.vv[2, 1], self.tol):  # if on top-right corner
            y[0] = x[0] - (self.a1[0] + self.a2[0])
            y[1] = x[1] - (self.a1[1] + self.a2[1])
        elif near(x[0], self.vv[1, 0] + x[1] * self.a2[0] / self.vv[2, 1], self.tol):  # if on right boundary
            y[0] = x[0] - self.a1[0]
            y[1] = x[1] - self.a1[1]
        else:  # should be on top boundary
            y[0] = x[0] - self.a2[0]
            y[1] = x[1] - self.a2[1]


# We now define the constitutive law for both phases:


# In[3]:


Em = 50e3
num = 0.2
Er = 210e3
nur = 0.3
material_parameters = [(Em, num), (Er, nur)]
nphases = len(material_parameters)


def eps(v):
    return sym(grad(v))


def sigma(v, i, Eps):
    E, nu = material_parameters[i]
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    return lmbda * tr(eps(v) + Eps) * Identity(2) + 2 * mu * (eps(v) + Eps)


# ## Variational formulation
#
# The previous problem is very similar to a standard linear elasticity problem, except for the periodicity constraint which has now been included in the FunctionSpace definition and for the presence of an eigenstrain term $\boldsymbol{E}$. It can easily be shown that the variational formulation of the previous problem reads as: Find $\boldsymbol{v}\in V$ such that:
#
# This readily translates into the following FEniCS code:

# In[4]:


Ve = VectorElement("CG", mesh.ufl_cell(), 2)
Re = VectorElement("R", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, MixedElement([Ve, Re]), constrained_domain=PeriodicBoundary(vertices, tolerance=1e-10))
V = FunctionSpace(mesh, Ve)

v_, lamb_ = TestFunctions(W)
dv, dlamb = TrialFunctions(W)
w = Function(W)
dx = Measure('dx')(subdomain_data=subdomains)

Eps = Constant(((0, 0), (0, 0)))
F = sum([inner(sigma(dv, i, Eps), eps(v_)) * dx(i) for i in range(nphases)])
a, L = lhs(F), rhs(F)
a += dot(lamb_, dv) * dx + dot(dlamb, v_) * dx


# ## Resolution

# In[5]:


def macro_strain(i):
    """returns the macroscopic strain for the 3 elementary load cases"""
    Eps_Voigt = np.zeros((3,))
    Eps_Voigt[i] = 1
    return np.array([[Eps_Voigt[0], Eps_Voigt[2] / 2.],
                     [Eps_Voigt[2] / 2., Eps_Voigt[1]]])


def stress2Voigt(s):
    return as_vector([s[0, 0], s[1, 1], s[0, 1]])


Chom = np.zeros((3, 3))
for (j, case) in enumerate(["Exx", "Eyy", "Exy"]):
    print("Solving {} case...".format(case))
    Eps.assign(Constant(macro_strain(j)))
    solve(a == L, w, [], solver_parameters={"linear_solver": "cg"})
    (v, lamb) = split(w)
    Sigma = np.zeros((3,))
    for k in range(3):
        Sigma[k] = assemble(sum([stress2Voigt(sigma(v, i, Eps))[k] * dx(i) for i in range(nphases)])) / vol
    Chom[j, :] = Sigma

print('Array chom')
print(np.array_str(Chom, precision=2))


# In[6]:


lmbda_hom = Chom[0, 1]
mu_hom = Chom[2, 2]
print('Chom')
print(Chom[0, 0], lmbda_hom + 2 * mu_hom)


# In[7]:


E_hom = mu_hom * (3 * lmbda_hom + 2 * mu_hom) / (lmbda_hom + mu_hom)
nu_hom = lmbda_hom / (lmbda_hom + mu_hom) / 2
print("Apparent Young modulus:", E_hom)
print("Apparent Poisson ratio:", nu_hom)

# In[8]:


# plotting deformed unit cell with total displacement u = Eps*y + v
y = SpatialCoordinate(mesh)
plt.figure()
p = plot(0.5 * (dot(Eps, y) + v), mode="displacement", title=case)
plt.colorbar(p)
plt.show()
