import numpy as np

Lx = 10.0
Ly = 10.0

Nx = 11
Ny = 11

dx = Lx/(Nx-1)
dy = Ly/(Ny-1)

x, y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))

# = np.logical_and(np.abs(x-y) < 2, np)
#rint(a)

a = np.logical_and.reduce((np.logical_and(x>=4.0, x<=6.0), np.logical_and(y>=4.0, y<=6.0)))

print(a)
