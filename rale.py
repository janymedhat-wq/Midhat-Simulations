import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Simulation Parameters
# -----------------------------
nx, ny = 60, 60       # Grid size (smaller for faster animation)
dx = dy = 1.0
dt = 0.1
nu = 0.1              # Viscosity

# Initialize velocity and pressure
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
p = np.zeros((nx, ny))
b = np.zeros((nx, ny))

# Initialize density / height for 3D visualization
density = np.ones((nx, ny))
density[nx//2:] = 2.0       # heavier fluid on top
density += 0.1*np.random.rand(nx, ny)  # small perturbations

# -----------------------------
# Helper functions
# -----------------------------
def build_b(b, density, dx, dy):
    b[1:-1,1:-1] = (density[1:-1,2:] - density[1:-1,0:-2])/(2*dx) + \
                    (density[2:,1:-1] - density[0:-2,1:-1])/(2*dy)
    return b

def pressure_poisson(p, b, dx, dy, nit=50):
    for _ in range(nit):
        p[1:-1,1:-1] = ((p[1:-1,2:] + p[1:-1,0:-2])*dy**2 +
                         (p[2:,1:-1] + p[0:-2,1:-1])*dx**2 -
                         b[1:-1,1:-1]*dx**2*dy**2)/(2*(dx**2 + dy**2))
        # Boundary conditions
        p[:,0] = p[:,1]
        p[:,-1] = p[:,-2]
        p[0,:] = p[1,:]
        p[-1,:] = p[-2,:]
    return p

def velocity_update(u, v, p, density, dx, dy, dt, nu):
    un = u.copy()
    vn = v.copy()

    u[1:-1,1:-1] = (un[1:-1,1:-1] -
                     un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,0:-2]) -
                     vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[0:-2,1:-1]) -
                     dt/(2*dx)*(p[1:-1,2:] - p[1:-1,0:-2]) +
                     nu*(dt/dx**2*(un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2]) +
                         dt/dy**2*(un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1])))

    v[1:-1,1:-1] = (vn[1:-1,1:-1] -
                     un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,0:-2]) -
                     vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[0:-2,1:-1]) -
                     dt/(2*dy)*(p[2:,1:-1] - p[0:-2,1:-1]) +
                     nu*(dt/dx**2*(vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,0:-2]) +
                         dt/dy**2*(vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[0:-2,1:-1])))

    # Boundary conditions
    u[0,:] = u[-1,:] = u[:,0] = u[:,-1] = 0
    v[0,:] = v[-1,:] = v[:,0] = v[:,-1] = 0

    return u, v

# -----------------------------
# 3D Visualization Setup
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
surf = ax.plot_surface(X, Y, density.T, cmap='plasma', edgecolor='k')

ax.set_zlim(0, 3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Density / Height')

# -----------------------------
# Animation Function
# -----------------------------
def update(frame):
    global u, v, p, b, density
    b = build_b(b, density, dx, dy)
    p = pressure_poisson(p, b, dx, dy)
    u, v = velocity_update(u, v, p, density, dx, dy, dt, nu)

    # Advect density
    density[1:-1,1:-1] -= dt*(u[1:-1,1:-1]*(density[1:-1,1:-1]-density[1:-1,0:-2])/dx +
                               v[1:-1,1:-1]*(density[1:-1,1:-1]-density[0:-2,1:-1])/dy)

    ax.clear()
    ax.set_zlim(0,3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density / Height')
    surf = ax.plot_surface(X, Y, density.T, cmap='plasma', edgecolor='k')
    return [surf]

ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)
plt.show()
