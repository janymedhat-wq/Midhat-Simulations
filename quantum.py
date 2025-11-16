
import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0545718e-34  # Reduced Planck constant (Joule second)
m_e = 9.10938356e-31  # Mass of electron (kg)
wavelength = 20e-9  # de Broglie wavelength (meters)
k0 = 2 * np.pi / wavelength  # Wave number

# Grid parameters
Lx, Ly = 1e-6, 1e-6  # Simulation domain size (meters)
Nx, Ny = 512, 512  # Number of grid points
dx, dy = Lx / Nx, Ly / Ny  # Grid spacing
x = np.linspace(-Lx / 2, Lx / 2, Nx)
y = np.linspace(-Ly / 2, Ly / 2, Ny)
X, Y = np.meshgrid(x, y)

# Time parameters
dt = 3e-14  # Time step (seconds)
Nt = 501  # Number of time steps

# Wave packet parameters
sigma_x = sigma_y = 60e-9  # Width of the Gaussian packet (meters)
x0, y0 = -2e-7, 0  # Initial position of the packet (meters)
kx0, ky0 = k0, 0  # Initial wave vector

# Barrier parameters
barrier_width = 1e-8  # Width of the square barrier (meters)
barrier_height = 1.05 * hbar**2 * k0**2 / (2 * m_e)  # Height of the barrier (Joules)

# Initial Gaussian wave packet
psi0 = np.exp(-(X - x0) ** 2 / (2 * sigma_x ** 2)) * np.exp(-(Y - y0) ** 2 / (2 * sigma_y ** 2))
psi0 = psi0.astype(np.complex128)
psi0 *= np.exp(1j * (kx0 * X + ky0 * Y))

# Normalize the initial wave packet
psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2))

# Potential energy (Square barrier in the center)
V = np.zeros_like(X)
V[np.abs(X) < barrier_width / 2] = barrier_height

# Fourier space components
kx = np.fft.fftfreq(Nx, dx) * 2 * np.pi
ky = np.fft.fftfreq(Ny, dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)
K2 = KX ** 2 + KY ** 2

# Split-step Fourier method
psi = psi0.copy()

transmission_prob = 0

for t in range(Nt):
    # (a) 1/2 Evolution for the potential energy in real space
    psi *= np.exp(-1j * V * dt / (2 * hbar))
    
    # (b) Forward transform
    psi_k = np.fft.fft2(psi)
    
    # (c) Full evolution for the kinetic energy in Fourier space
    psi_k *= np.exp(-1j * hbar * K2 * dt / (2 * m_e))
    
    # (d) Inverse Fourier transform
    psi = np.fft.ifft2(psi_k)
    
    # (e) 1/2 Evolution for the potential energy in real space
    psi *= np.exp(-1j * V * dt / (2 * hbar))
    
    # Calculate transmission probability after time step 300
    if t >= 0:
        Rho = np.abs(psi) ** 2
        transmission_prob = np.sum(Rho[X > barrier_width / 2])
    
    # Visualization of wave packet evolution
    if t % 50 == 0:
        plt.figure(figsize=(6, 5))
        Rho = np.abs(psi) ** 2
        plt.imshow(Rho/np.max(Rho) + V/np.max(V), extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2), cmap='hot')
        plt.colorbar()
        plt.title(f'Time step: {t}\nTransmission Probability: {transmission_prob:.3f}')
        plt.xlabel(r'$x$ (m)', fontsize=14)
        plt.ylabel(r'$y$ (m)', fontsize=14)
        plt.tick_params(which="major", axis="both", direction="in", top=True, right=True, length=5, width=1, labelsize=12)
        plt.pause(1)  # Pause for 1 second
        plt.close()

                