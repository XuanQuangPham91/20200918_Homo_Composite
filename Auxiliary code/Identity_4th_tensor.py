from dolfin import *
from fenics import *
from ufl import *
from numpy import array

mesh = UnitSquareMesh(2,2)
i,j,k,l = indices(4)

# Tensor from StackExchange thread:
d = mesh.geometry().dim()
delta = Identity(d)
I = as_tensor(0.5*(delta[i,k]*delta[j,l] + delta[i,l]*delta[j,k]),(i,j,k,l))

# Kludgy way to change components:  Fill up a nested list with the components,
# change elements of the list, then use as_tensor to turn the list back
# into a tensor.
componentList = []
for i in range(0,d):
    componentList += [[],]
    for j in range(0,d):
        componentList[i] += [[],]
        for k in range(0,d):
            componentList[i][j] += [[],]
            for l in range(0,d):
                componentList[i][j][k] += [I[i,j,k,l],]

# Allowed to assign items in a list:
componentList[0][1][1][1] = 2.0

# Can convert lists into tensors:
I = as_tensor(componentList)

# a = as_matrix(I.tolist())
# print(assemble(inner(a,a)*dx(domain=UnitIntervalMesh(1))))

print(shape(I))
print(I)
