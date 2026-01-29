import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

st.set_page_config(page_title="Virtual Wind Tunnel", layout="wide")
st.title("💨 Virtual Wind Tunnel: Image-to-CFD")
st.markdown("""
Upload an image or choose a shape to test its aerodynamics. 
Use the **Rotation Controls** to adjust the angle of attack!
""")

# --- 1. CONFIGURATION ---
st.sidebar.header("Wind Tunnel Settings")

# Resolution
ny = st.sidebar.slider("Tunnel Height (Resolution)", 40, 150, 80)
nx = st.sidebar.slider("Tunnel Length", 100, 400, 300)

# Physics
viscosity = st.sidebar.slider("Viscosity", 0.005, 0.1, 0.02, format="%.3f")
u_inlet = st.sidebar.slider("Inlet Wind Speed", 0.05, 0.2, 0.1)
steps = st.sidebar.slider("Simulation Steps", 500, 5000, 1500)

# Geometry Selection
shape_options = ["Cylinder", "Square", "Airfoil (NACA0012)", "Custom Image Upload"]
shape = st.sidebar.selectbox("Select Obstacle Shape", shape_options)

# --- ROTATION STATE MANAGEMENT ---
# Initialize session state for rotation if not present
if 'rotation_angle' not in st.session_state:
    st.session_state.rotation_angle = 0

def update_slider():
    st.session_state.rotation_angle = st.session_state.num_input

def update_num():
    st.session_state.rotation_angle = st.session_state.slider_input

# --- CUSTOM IMAGE UPLOAD SECTION ---
uploaded_file = None
invert_colors = False
threshold = 128

if shape == "Custom Image Upload":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Object Configuration")
    
    # 1. Image Upload
    uploaded_file = st.sidebar.file_uploader("Upload Image (jpg, png)", type=['png', 'jpg', 'jpeg'])
    
    # 2. Rotation Controls (Synced)
    col_r1, col_r2 = st.sidebar.columns([2, 1])
    with col_r1:
        st.slider("Rotation (Slider)", -180, 180, 
                  key="slider_input", on_change=update_num, 
                  value=st.session_state.rotation_angle, label_visibility="collapsed")
    with col_r2:
        st.number_input("Angle (°)", -180, 180, 
                        key="num_input", on_change=update_slider, 
                        value=st.session_state.rotation_angle)
    
    # 3. Image Processing Settings
    invert_colors = st.sidebar.checkbox("Invert Colors", value=True, help="Check if object is white on black background.")
    threshold = st.sidebar.slider("Binary Threshold", 0, 255, 128)

# --- 2. THE LBM SOLVER ---
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

def process_image_to_mask(image_file, nx, ny, rotation, invert=False, thresh=128):
    """Converts uploaded image to rotated boolean mask."""
    # Open image
    img = Image.open(image_file).convert('L') # Convert to grayscale
    
    # Invert if necessary
    if invert:
        img = ImageOps.invert(img)
    
    # Rotate (bicubic is smoother for rotations) - Expand=True keeps the whole object
    img = img.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=0)

    # Resize logic: fit image height to 50% of tunnel height
    target_h = int(ny * 0.6)
    aspect_ratio = img.width / img.height
    target_w = int(target_h * aspect_ratio)
    
    # Resize
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    # Create canvas
    canvas = Image.new('L', (nx, ny), 0)
    
    # Paste in center
    paste_x = (nx - target_w) // 4
    paste_y = (ny - target_h) // 2
    canvas.paste(img, (paste_x, paste_y))
    
    # Convert to numpy array and threshold
    arr = np.array(canvas)
    mask = arr > thresh
    return mask

@st.cache_data
def get_obstacle_mask(nx, ny, shape, _uploaded_file, rotation, invert, thresh):
    mask = np.full((ny, nx), False)
    cx_pos, cy_pos = nx // 4, ny // 2
    
    if shape == "Custom Image Upload":
        if _uploaded_file is not None:
            mask = process_image_to_mask(_uploaded_file, nx, ny, rotation, invert, thresh)
        else:
            # Placeholder circle if no file
            r = ny // 10
            y, x = np.ogrid[:ny, :nx]
            mask = (x - cx_pos)**2 + (y - cy_pos)**2 < r**2
            
    elif shape == "Cylinder":
        r = ny // 9
        y, x = np.ogrid[:ny, :nx]
        mask = (x - cx_pos)**2 + (y - cy_pos)**2 < r**2

    elif shape == "Square":
        # Create a square image for rotation handling
        size = ny // 4
        img = Image.new('L', (size, size), 255)
        # Rotate logic reused for standard shapes if you wanted, but keeping simple here
        # (Standard shapes are static in this simple version, except Custom)
        r = ny // 9
        mask[cy_pos-r:cy_pos+r, cx_pos-r:cx_pos+r] = True
        
    elif shape == "Airfoil (NACA0012)":
        chord = ny // 3
        # Draw airfoil on a temporary PIL image to support rotation easily
        # 1. Create high-res canvas for drawing
        scale = 2 
        pil_h, pil_w = ny * scale, nx * scale
        img = Image.new('L', (pil_w, pil_h), 0)
        
        # Draw airfoil logic (simplified drawing on array then converting)
        # For simplicity, we keep the airfoil static or use the math directly
        # If we want to rotate the airfoil, we'd need to rotate coordinates.
        # Here we just keep it static 0 deg for the preset.
        for i in range(chord):
            x_pos = cx_pos - chord//2 + i
            if 0 <= x_pos < nx:
                xu = i / chord
                yt = 0.6 * (0.2969 * np.sqrt(xu) - 0.1260 * xu - 0.3516 * xu**2 + 0.2843 * xu**3 - 0.1015 * xu**4)
                half_thick = int(yt * chord)
                mask[cy_pos-half_thick : cy_pos+half_thick, x_pos] = True

    return mask

@st.cache_data
def run_lbm(nx, ny, omega, u_in, steps, shape, _uploaded_file, rotation, invert, thresh):
    obstacle = get_obstacle_mask(nx, ny, shape, _uploaded_file, rotation, invert, thresh)
    
    vel = np.zeros((2, ny, nx))
    vel[0, :, :] = u_in * (1 + 1e-4 * np.sin(np.linspace(0, 4*np.pi, ny))[:, np.newaxis])
    rho = np.ones((ny, nx))
    
    f = np.zeros((9, ny, nx))
    for i in range(9):
        cu = 3 * (cx[i]*vel[0] + cy[i]*vel[1])
        f[i] = rho * w[i] * (1 + cu + 0.5*cu**2 - 1.5*(vel[0]**2 + vel[1]**2))
    
    for _ in range(steps):
        rho = np.sum(f, axis=0)
        vel[0] = np.sum(f * cx[:, np.newaxis, np.newaxis], axis=0) / rho
        vel[1] = np.sum(f * cy[:, np.newaxis, np.newaxis], axis=0) / rho
        
        vel[0, :, 0] = u_in
        vel[1, :, 0] = 0
        rho[:, 0] = 1 / (1 - u_in) * (np.sum(f[[0, 2, 4], :, 0], axis=0) + 2*np.sum(f[[3, 6, 7], :, 0], axis=0))
        
        feq = np.zeros_like(f)
        for i in range(9):
            cu = 3 * (cx[i]*vel[0] + cy[i]*vel[1])
            feq[i] = rho * w[i] * (1 + cu + 0.5*cu**2 - 1.5*(vel[0]**2 + vel[1]**2))
        
        f = f * (1 - omega) + feq * omega
        
        for i in range(9):
            f[i] = np.roll(f[i], cx[i], axis=1)
            f[i] = np.roll(f[i], cy[i], axis=0)
            
        bounced = f[:, obstacle]
        for i in range(9):
             f[opp[i], obstacle] = bounced[i]
             
    speed = np.sqrt(vel[0]**2 + vel[1]**2)
    vorticity = (np.roll(vel[1], -1, axis=1) - np.roll(vel[1], 1, axis=1)) - \
                (np.roll(vel[0], -1, axis=0) - np.roll(vel[0], 1, axis=0))
    
    speed[obstacle] = np.nan
    vorticity[obstacle] = np.nan
    
    return speed, vorticity, obstacle

# --- 3. EXECUTION ---
omega = 1.0 / (3.0 * viscosity + 0.5)
rot_angle = st.session_state.rotation_angle

if omega >= 2.0:
    st.error("Viscosity too low! Simulation will be unstable.")
else:
    with st.spinner("Processing Mesh & Simulating Fluid..."):
        speed, vorticity, mask = run_lbm(nx, ny, omega, u_inlet, steps, shape, uploaded_file, rot_angle, invert_colors, threshold)

    # --- 4. VISUALIZATION ---
    st.subheader(f"Results")
    
    # Show mesh preview for Custom uploads
    if shape == "Custom Image Upload" and uploaded_file:
        with st.expander("See Generated Mesh Mask", expanded=True):
            st.image(mask.astype(float), caption=f"Simulation Mesh (Rotated {rot_angle}°)", clamp=True, width=400)

    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        im1 = ax1.imshow(speed, cmap='turbo', origin='lower')
        ax1.imshow(mask, cmap='binary', alpha=0.5, origin='lower', interpolation='nearest')
        ax1.set_title("Velocity Magnitude")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.axis('off')
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        im2 = ax2.imshow(vorticity, cmap='seismic', origin='lower', vmin=-0.1, vmax=0.1)
        ax2.imshow(mask, cmap='gray', alpha=0.5, origin='lower')
        ax2.set_title("Vorticity (Wake)")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.axis('off')
        st.pyplot(fig2)
