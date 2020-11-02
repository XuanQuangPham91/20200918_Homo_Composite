""" The cell problem - 2D thermal diffusion """
"""This program is used to generate the coefficients c00, c01 and c11
used in the cell problem."""


# Create mesh
from dolfin import *
mesh = UnitSquareMesh(64, 64)
# mesh = UnitSquareMesh.create(2, 2, CellType.Type.quadrilateral)
# mesh = UnitSquareMesh.create(10, 10, CellType.Type.quadrilateral)

# Create mesh functions for c00, c01, c11
c00 = MeshFunction("double", mesh, 1)
c01 = MeshFunction("double", mesh, 1)
c10 = MeshFunction("double", mesh, 1)
c11 = MeshFunction("double", mesh, 1)

# Iterate over mesh and set values
for cell in cells(mesh):
    c00[cell] = 10.170925434463658
    # c00[cell] = 101.
    # c00[cell] = 11.51

    c01[cell] = 1.4757455287016668e-05

    c10[cell] = 1.4757455286990401e-05

    c11[cell] = 10.170925434463655
    # c11[cell] = 101.
    # c11[cell] = 11.51

    # if cell.midpoint().x() < 0.5:
    #     c00[cell] = 1.0
    #     c01[cell] = 0.0
    #     c11[cell] = 2.0
    # else:
    #     c00[cell] = 13.0
    #     c01[cell] = 0.0
    #     c11[cell] = 14.0

# Store to file
mesh_file = File("./data/unitsquare.xml.gz")
c00_file = File("./data/unitsquare_c00.xml.gz")
c01_file = File("./data/unitsquare_c01.xml.gz")
c10_file = File("./data/unitsquare_c10.xml.gz")
c11_file = File("./data/unitsquare_c11.xml.gz")

mesh_file << mesh
c00_file << c00
c01_file << c01
c10_file << c10
c11_file << c11

# Plot mesh functions
# plot(c00, title="C00")
# plot(c01, title="C01")
# plot(c10, title="C10")
# plot(c11, title="C11")
