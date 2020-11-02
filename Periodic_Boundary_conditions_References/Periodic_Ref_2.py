## https://fenicsproject.org/qa/262/possible-specify-more-than-one-periodic-boundary-condition/

Lx = 2.*pi
Ly = 2.
Lz = pi
Nx = 257
Ny = 193
Nz = 193
mesh = BoxMesh(0., -Ly/2., -Lz/2., Lx, Ly/2., Lz/2., Nx, Ny, Nz)

class PeriodicDomain(SubDomain):

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool((near(x[0], 0) or near(x[2], -Lz/2.)) and 
            (not ((near(x[0], Lx) and near(x[2], -Lz/2.)) or 
                  (near(x[0], 0) and near(x[2], Lz/2.)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], Lx) and near(x[2], Lz/2.):
            y[0] = x[0] - Lx
            y[1] = x[1] 
            y[2] = x[2] - Lz
        elif near(x[0], Lx):
            y[0] = x[0] - Lx
            y[1] = x[1]
            y[2] = x[2]
        elif near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - Lz
        else:
            y[0] = -1000
            y[1] = -1000
            y[2] = -1000

# Sub domain for Periodic boundary condition
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