import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Virtual Wind Tunnel", layout="wide")
st.title("💨 Virtual Wind Tunnel: Lattice Boltzmann Demo")
st.markdown("""
This app simulates flow past an object using the **Lattice Boltzmann Method (LBM)**. 
Unlike the previous lid-driven cavity, this is an **open flow** simulation (Wind Tunnel).
""")

# --- 1. CONFIGURATION ---
st.sidebar.header("Wind Tunnel Settings")

# Resolution
ny = st.sidebar.slider("Tunnel Height (Resolution)", 40, 100, 60)
nx = st.sidebar.slider("Tunnel Length", 100, 400, 200)

# Physics
viscosity = st.sidebar.slider("Viscosity", 0.005, 0.1, 0.02, format="%.3f")
u_inlet = st.sidebar.slider("Inlet Wind Speed", 0.05, 0.2, 0.1)
steps = st.sidebar.slider("Simulation Steps", 500, 5000, 1500)

# Geometry Selection
shape = st.sidebar.selectbox("Select Obstacle Shape", ["Cylinder", "Square", "Wall", "Airfoil (NACA0012)"])

# --- 2. THE LBM SOLVER ---
# D2Q9 Lattice Constants
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
# Opposite directions for bounce-back
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

@st.cache_data
def get_obstacle_mask(nx, ny, shape):
    """Generates a boolean mask for the obstacle."""
    mask = np.full((ny, nx), False)
    cx, cy = nx // 4, ny // 2  # Center of obstacle
    
    if shape == "Cylinder":
        r = ny // 9
        y, x = np.ogrid[:ny, :nx]
        mask = (x - cx)**2 + (y - cy)**2 < r**2

    elif shape == "Square":
        r = ny // 9
        mask[cy-r:cy+r, cx-r:cx+r] = True
        
    elif shape == "Wall":
        mask[cy:cy+5, cx-10:cx+10] = True
        
    elif shape == "Airfoil (NACA0012)":
        # Simplified NACA0012 approximation
        chord = ny // 3
        for i in range(chord):
            x_pos = cx - chord//2 + i
            if 0 <= x_pos < nx:
                # Thickness distribution
                xu = i / chord
                yt = 0.6 * (0.2969 * np.sqrt(xu) - 0.1260 * xu - 0.3516 * xu**2 + 0.2843 * xu**3 - 0.1015 * xu**4)
                half_thick = int(yt * chord)
                mask[cy-half_thick : cy+half_thick, x_pos] = True

    return mask

@st.cache_data
def run_lbm(nx, ny, omega, u_in, steps, shape):
    # Initialize
    obstacle = get_obstacle_mask(nx, ny, shape)
    
    # Initial macroscopic variables
    vel = np.zeros((2, ny, nx))
    vel[0, :, :] = u_in * (1 + 1e-4 * np.sin(np.linspace(0, 4*np.pi, ny))[:, np.newaxis]) # Small perturbation
    rho = np.ones((ny, nx))
    
    # Initial distributions (equilibrium)
    f = np.zeros((9, ny, nx))
    for i in range(9):
        cu = 3 * (cx[i]*vel[0] + cy[i]*vel[1])
        f[i] = rho * w[i] * (1 + cu + 0.5*cu**2 - 1.5*(vel[0]**2 + vel[1]**2))
    
    # Main Loop
    for _ in range(steps):
        # 1. Macroscopic variables
        rho = np.sum(f, axis=0)
        vel[0] = np.sum(f * cx[:, np.newaxis, np.newaxis], axis=0) / rho
        vel[1] = np.sum(f * cy[:, np.newaxis, np.newaxis], axis=0) / rho
        
        # Force Inlet/Outlet (Zou/He simplified)
        vel[0, :, 0] = u_in
        vel[1, :, 0] = 0
        rho[:, 0] = 1 / (1 - u_in) * (np.sum(f[[0, 2, 4], :, 0], axis=0) + 2*np.sum(f[[3, 6, 7], :, 0], axis=0))
        
        # 2. Collision (BGK)
        feq = np.zeros_like(f)
        for i in range(9):
            cu = 3 * (cx[i]*vel[0] + cy[i]*vel[1])
            feq[i] = rho * w[i] * (1 + cu + 0.5*cu**2 - 1.5*(vel[0]**2 + vel[1]**2))
        
        f = f * (1 - omega) + feq * omega
        
        # 3. Streaming (Propagate)
        for i in range(9):
            f[i] = np.roll(f[i], cx[i], axis=1)
            f[i] = np.roll(f[i], cy[i], axis=0)
            
        # 4. Boundary Conditions (Bounce-back)
        # Find cells that streamed *into* obstacle
        bounced = f[:, obstacle]
        # Reflect them back to where they came from (swap directions)
        for i in range(9):
             f[opp[i], obstacle] = bounced[i]
             
    # Calculate final speed and vorticity for plotting
    speed = np.sqrt(vel[0]**2 + vel[1]**2)
    
    # Simple vorticity calc (curl)
    vorticity = (np.roll(vel[1], -1, axis=1) - np.roll(vel[1], 1, axis=1)) - \
                (np.roll(vel[0], -1, axis=0) - np.roll(vel[0], 1, axis=0))
    
    # Mask obstacle in output
    speed[obstacle] = np.nan
    vorticity[obstacle] = np.nan
    
    return speed, vorticity, obstacle

# --- 3. RUN SIMULATION ---
# Calculate relaxation parameter (omega) from viscosity
# viscosity = (1/3) * (1/omega - 0.5)
omega = 1.0 / (3.0 * viscosity + 0.5)

if omega >= 2.0:
    st.error("Viscosity too low! Simulation will be unstable. Increase Viscosity.")
else:
    with st.spinner(f"Simulating {shape} in Wind Tunnel..."):
        speed, vorticity, mask = run_lbm(nx, ny, omega, u_inlet, steps, shape)

    # --- 4. VISUALIZATION ---
    st.subheader(f"Flow Results")
    
    col1, col2 = st.columns(2)
    
    # Plot 1: Velocity Magnitude
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        im1 = ax1.imshow(speed, cmap='turbo', origin='lower')
        # Overlay obstacle
        ax1.imshow(mask, cmap='binary', alpha=0.5, origin='lower', interpolation='nearest')
        ax1.set_title("Velocity Magnitude")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.axis('off')
        st.pyplot(fig1)

    # Plot 2: Vorticity (Curl)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        # Use a divergent colormap for vorticity (red=clockwise, blue=counter-clockwise)
        im2 = ax2.imshow(vorticity, cmap='seismic', origin='lower', vmin=-0.1, vmax=0.1)
        ax2.imshow(mask, cmap='gray', alpha=0.5, origin='lower')
        ax2.set_title("Vorticity (Turbulence/Wake)")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.axis('off')
        st.pyplot(fig2)
