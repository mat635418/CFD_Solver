import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="Interactive CFD: Cavity Flow", layout="wide")
st.title("🌊 Interactive 2D Lid-Driven Cavity Flow")
st.markdown("""
This app simulates the **Navier-Stokes equations** for incompressible flow in a box. 
The top lid moves to the right, driving the fluid and creating a vortex.
""")

# --- 2. SIDEBAR PARAMETERS ---
st.sidebar.header("Simulation Parameters")

# Grid Size (nx, ny)
nx = st.sidebar.slider("Grid Size (Resolution)", min_value=20, max_value=60, value=41, step=1, 
                       help="Higher resolution is more accurate but slower.")
ny = nx  # Keep domain square for simplicity

# Time Steps (nt)
nt = st.sidebar.slider("Time Steps (Duration)", min_value=100, max_value=2000, value=500, step=100,
                       help="How long the simulation runs.")

# Physical Properties
nu = st.sidebar.slider("Viscosity (nu)", min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                       help="Thicker fluids (higher nu) flow more slowly.")
rho = st.sidebar.number_input("Fluid Density (rho)", value=1.0)

# Boundary Condition
lid_vel = st.sidebar.slider("Lid Velocity", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

# Advanced Solver Settings
nit = 50  # Iterations for pressure equation (kept constant for speed)
dt = 0.001 
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

# --- 3. THE SOLVER (Cached for Speed) ---
@st.cache_data
def solve_navier_stokes(nx, ny, nt, nit, rho, nu, dt, lid_vel):
    # Initialize arrays
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx)) 
    b = np.zeros((ny, nx))
    
    # Time Stepping Loop
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        # Source term for Pressure Poisson
        b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + 
                         (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                        2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                             (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                        ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
        
        # Pressure Poisson Solver
        for q in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                              (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                              (2 * (dx**2 + dy**2)) -
                              dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

            # Pressure Boundary Conditions
            p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
            p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
            p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
            p[-1, :] = 0        # p = 0 at y = 2
        
        # Velocity Updates
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * (dt / dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Velocity Boundary Conditions
        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = lid_vel  # Apply Lid Velocity
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0
        
    return u, v, p

# --- 4. RUN SIMULATION ---
with st.spinner(f"Simulating {nt} time steps..."):
    u, v, p = solve_navier_stokes(nx, ny, nt, nit, rho, nu, dt, lid_vel)

# --- 5. VISUALIZATION ---
st.subheader(f"Results (Re = {lid_vel * 2 / nu:.1f})")

# Coordinate mesh for plotting
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots(figsize=(10, 6))

# Pressure contour
contour = ax.contourf(X, Y, p, alpha=0.5, cmap='viridis', levels=20)
fig.colorbar(contour, ax=ax, label='Pressure')

# Velocity Streamlines
speed = np.sqrt(u**2 + v**2)
ax.streamplot(X, Y, u, v, color='k', density=1.5, linewidth=1, arrowsize=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'Velocity Streamlines & Pressure Field')

st.pyplot(fig)
