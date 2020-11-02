## https://fenicsproject.org/qa/1025/periodic-boundary-conditions-for-poisson-problem/

# In[1]
# Periodic Boundary Class
class PeriodicBC(SubDomain):
    def __init__(self, tolerance=DOLFIN_EPS, length = 1., length_scaling = 1.):
        SubDomain.__init__(self)
        self.tol = tolerance
        self.length = length
        self.length_scaling = length_scaling

# Left boundary is "target domain" G
def inside(self, x, on_boundary):
    # return True if on left or bottom boundary AND NOT on one of the two corners (0, L) and (L, 0)
    return bool((near(x[0], 0) or near(x[1], 0)) and 
            (not ((near(x[0], 0) and near(x[1], self.length/self.length_scaling)) or 
                    (near(x[0], self.length/self.length_scaling) and near(x[1], 0)))) and on_boundary)

def map(self, x, y):
    L = self.length/self.length_scaling
    if near(x[0], L) and near(x[1], L):
        y[0] = x[0] - L
        y[1] = x[1] - L
    elif near(x[0], L):
        y[0] = x[0] - L
        y[1] = x[1]
    else:   # near(x[1], L)
        y[0] = x[0]
        y[1] = x[1] - L

