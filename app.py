import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from numba import jit
import time

# ==========================================
# 0. Page Config & CSS Styling
# ==========================================
st.set_page_config(page_title="Crowd Sim", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Global Text Color */
    body { color: #ffffff; background-color: #0e1117; }
    
    /* Button Styling */
    .stButton > button {
        background-color: #4A88C7; 
        color: white; 
        border-radius: 20px; 
        width: 100%;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #3b6ba0;
        color: white;
    }
    
    /* Disabled Button Styling */
    .stButton > button:disabled {
        background-color: #333333;
        color: #888888;
    }

    /* Input Fields Styling */
    .stNumberInput > div > div > input {
        color: white;
        background-color: #262730;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. State Management (Navigation)
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'inputs_disabled' not in st.session_state:
    st.session_state.inputs_disabled = False
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None

def go_to_guide():
    st.session_state.page = 'guide'

def go_to_sim():
    st.session_state.page = 'simulation'

def enable_editing():
    st.session_state.inputs_disabled = False
    st.session_state.simulation_data = None

# ==========================================
# 2. Simulation Logic (Full Backend)
# ==========================================

class Config:
    L_X = 100.0
    L_Y = 50.0
    DX = 0.5
    DY = 0.5
    T_MAX = 200.0 
    CFL = 0.4    
    MASS = 60.0
    PANIC_START_TIME = 150.0
    
    # Geometry: Market with Center Pillar
    OBSTACLES = np.array([
        [0.0, 10.0, 30.0, 50.0], [0.0, 10.0, 0.0, 20.0],
        [90.0, 100.0, 30.0, 50.0], [90.0, 100.0, 0.0, 20.0],
        [20.0, 80.0, 35.0, 45.0],  
        [20.0, 80.0, 20.0, 30.0],  # Center Pillar
        [20.0, 80.0, 5.0, 15.0], 
    ], dtype=np.float64)
    NEW_OBSTACLE = np.array([60, 65, 30, 33], dtype=np.float64)

@jit(nopython=True)
def get_f_rho(rho, coeff_A, coeff_B):
    # Fundamental Diagram: A * exp(B * rho^2)
    return coeff_A * np.exp(coeff_B * rho**2)

@jit(nopython=True)
def get_g_rho(rho, coeff_C):
    # Density Cost: C * rho^2
    return coeff_C * rho**2

@jit(nopython=True)
def clamp_idx(val, max_val):
    if val < 0: return 0
    if val > max_val: return max_val
    return val

# --- FULL EIKONAL SOLVER (4 Sweeps) ---
@jit(nopython=True)
def solve_eikonal_jit(cost_field, mask_target, dx, n_sweeps=4):
    ny, nx = cost_field.shape
    phi = np.full((ny, nx), 1e5)
    
    # Initialization
    for j in range(ny):
        for i in range(nx):
            if mask_target[j, i]:
                phi[j, i] = 0.0

    # Fast Sweeping Iterations
    for _ in range(n_sweeps):
        # 1. Top-Left -> Bottom-Right
        for j in range(ny):
            for i in range(nx):
                if mask_target[j, i]: continue
                phi_x = phi[j, i-1] if i > 0 else 1e5
                phi_y = phi[j-1, i] if j > 0 else 1e5
                f = cost_field[j, i]
                diff = abs(phi_x - phi_y)
                if diff >= f * dx:
                    val = min(phi_x, phi_y) + f * dx
                else:
                    discriminant = max(0.0, 2 * (f * dx)**2 - diff**2)
                    val = 0.5 * (phi_x + phi_y + np.sqrt(discriminant))
                phi[j, i] = min(phi[j, i], val)

        # 2. Top-Right -> Bottom-Left
        for j in range(ny):
            for i in range(nx-1, -1, -1):
                if mask_target[j, i]: continue
                phi_x = phi[j, i+1] if i < nx-1 else 1e5
                phi_y = phi[j-1, i] if j > 0 else 1e5
                f = cost_field[j, i]
                diff = abs(phi_x - phi_y)
                if diff >= f * dx:
                    val = min(phi_x, phi_y) + f * dx
                else:
                    discriminant = max(0.0, 2 * (f * dx)**2 - diff**2)
                    val = 0.5 * (phi_x + phi_y + np.sqrt(discriminant))
                phi[j, i] = min(phi[j, i], val)

        # 3. Bottom-Left -> Top-Right
        for j in range(ny-1, -1, -1):
            for i in range(nx):
                if mask_target[j, i]: continue
                phi_x = phi[j, i-1] if i > 0 else 1e5
                phi_y = phi[j+1, i] if j < ny-1 else 1e5
                f = cost_field[j, i]
                diff = abs(phi_x - phi_y)
                if diff >= f * dx:
                    val = min(phi_x, phi_y) + f * dx
                else:
                    discriminant = max(0.0, 2 * (f * dx)**2 - diff**2)
                    val = 0.5 * (phi_x + phi_y + np.sqrt(discriminant))
                phi[j, i] = min(phi[j, i], val)

        # 4. Bottom-Right -> Top-Left
        for j in range(ny-1, -1, -1):
            for i in range(nx-1, -1, -1):
                if mask_target[j, i]: continue
                phi_x = phi[j, i+1] if i < nx-1 else 1e5
                phi_y = phi[j+1, i] if j < ny-1 else 1e5
                f = cost_field[j, i]
                diff = abs(phi_x - phi_y)
                if diff >= f * dx:
                    val = min(phi_x, phi_y) + f * dx
                else:
                    discriminant = max(0.0, 2 * (f * dx)**2 - diff**2)
                    val = 0.5 * (phi_x + phi_y + np.sqrt(discriminant))
                phi[j, i] = min(phi[j, i], val)
                
    return phi

# --- FULL WENO3 RECONSTRUCTION ---
@jit(nopython=True)
def weno3_flux_reconstruction(Q, nx, ny, dx, dy, C0, A, B):
    dFdx = np.zeros_like(Q)
    dGdy = np.zeros_like(Q)
    
    rho = Q[0]
    u = np.zeros_like(rho)
    v = np.zeros_like(rho)
    
    # Velocity check
    for j in range(ny):
        for i in range(nx):
            if rho[j, i] > 1e-3:
                inv_rho = 1.0 / rho[j, i]
                u[j, i] = Q[1, j, i] * inv_rho
                v[j, i] = Q[2, j, i] * inv_rho
            else:
                u[j, i] = 0.0
                v[j, i] = 0.0

    # Flux F (x-dir)
    Fx = np.zeros_like(Q)
    Fx[0] = Q[1]
    Fx[1] = Q[1] * u + C0**2 * rho
    Fx[2] = Q[1] * v
    
    # Flux G (y-dir)
    Gy = np.zeros_like(Q)
    Gy[0] = Q[2]
    Gy[1] = Q[1] * v
    Gy[2] = Q[2] * v + C0**2 * rho
    
    alpha_x = np.max(np.abs(u)) + C0
    alpha_y = np.max(np.abs(v)) + C0
    
    eps = 1e-6
    g1, g2 = 1.0/3.0, 2.0/3.0
    
    # --- X-Fluxes ---
    for k in range(3):
        for j in range(ny):
            for i in range(-1, nx): 
                im1 = clamp_idx(i-1, nx-1)
                i0  = clamp_idx(i,   nx-1)
                ip1 = clamp_idx(i+1, nx-1)
                ip2 = clamp_idx(i+2, nx-1)
                
                # Positive Flux P
                P_m1 = 0.5 * (Fx[k,j,im1] + alpha_x * Q[k,j,im1])
                P_0  = 0.5 * (Fx[k,j,i0]  + alpha_x * Q[k,j,i0])
                P_p1 = 0.5 * (Fx[k,j,ip1] + alpha_x * Q[k,j,ip1])
                
                beta1 = (P_0 - P_m1)**2
                beta2 = (P_p1 - P_0)**2
                w1 = g1 / (eps + beta1)**2
                w2 = g2 / (eps + beta2)**2
                P_L = (w1 * (-0.5*P_m1 + 1.5*P_0) + w2 * (0.5*P_0 + 0.5*P_p1)) / (w1 + w2)
                
                # Negative Flux M
                M_0  = 0.5 * (Fx[k,j,i0]  - alpha_x * Q[k,j,i0])
                M_p1 = 0.5 * (Fx[k,j,ip1] - alpha_x * Q[k,j,ip1])
                M_p2 = 0.5 * (Fx[k,j,ip2] - alpha_x * Q[k,j,ip2])
                
                beta1n = (M_p1 - M_0)**2
                beta2n = (M_p2 - M_p1)**2
                w1n = g1 / (eps + beta2n)**2
                w2n = g2 / (eps + beta1n)**2
                M_R = (w1n * (-0.5*M_p2 + 1.5*M_p1) + w2n * (0.5*M_p1 + 0.5*M_0)) / (w1n + w2n)
                
                NumFlux = P_L + M_R
                
                if i >= 0:
                    dFdx[k, j, i] += NumFlux / dx
                if i < nx - 1:
                    dFdx[k, j, i+1] -= NumFlux / dx

    # --- Y-Fluxes ---
    for k in range(3):
        for i in range(nx):
            for j in range(-1, ny):
                jm1 = clamp_idx(j-1, ny-1)
                j0  = clamp_idx(j,   ny-1)
                jp1 = clamp_idx(j+1, ny-1)
                jp2 = clamp_idx(j+2, ny-1)
                
                P_m1 = 0.5 * (Gy[k,jm1,i] + alpha_y * Q[k,jm1,i])
                P_0  = 0.5 * (Gy[k,j0,i]  + alpha_y * Q[k,j0,i])
                P_p1 = 0.5 * (Gy[k,jp1,i] + alpha_y * Q[k,jp1,i])
                
                beta1 = (P_0 - P_m1)**2; beta2 = (P_p1 - P_0)**2
                w1 = g1/(eps+beta1)**2; w2 = g2/(eps+beta2)**2
                P_L = (w1*(-0.5*P_m1+1.5*P_0) + w2*(0.5*P_0+0.5*P_p1))/(w1+w2)
                
                M_0  = 0.5 * (Gy[k,j0,i]  - alpha_y * Q[k,j0,i])
                M_p1 = 0.5 * (Gy[k,jp1,i] - alpha_y * Q[k,jp1,i])
                M_p2 = 0.5 * (Gy[k,jp2,i] - alpha_y * Q[k,jp2,i])
                
                beta1n = (M_p1 - M_0)**2; beta2n = (M_p2 - M_p1)**2
                w1n = g1/(eps+beta2n)**2; w2n = g2/(eps+beta1n)**2
                M_R = (w1n*(-0.5*M_p2+1.5*M_p1) + w2n*(0.5*M_p1+0.5*M_0))/(w1n+w2n)
                
                NumFlux = P_L + M_R
                
                if j >= 0:
                    dGdy[k, j, i] += NumFlux / dy
                if j < ny - 1:
                    dGdy[k, j+1, i] -= NumFlux / dy
                
    return dFdx + dGdy

# --- FULL PHYSICS KERNEL ---
@jit(nopython=True)
def compute_rhs_jit(Q, P2, phi_e, mask_obs, dx, dy, C0, Tau, Mass, A, B):
    nx = Q.shape[2]
    ny = Q.shape[1]
    
    # 1. Flux Divergence
    flux_div = weno3_flux_reconstruction(Q, nx, ny, dx, dy, C0, A, B)
    RHS = -flux_div 
    
    rho = Q[0]
    u = np.zeros_like(rho)
    v = np.zeros_like(rho)
    
    # Explicit loop for velocity
    for j in range(ny):
        for i in range(nx):
            if rho[j, i] > 1e-3:
                inv_rho = 1.0 / rho[j, i]
                u[j, i] = Q[1, j, i] * inv_rho
                v[j, i] = Q[2, j, i] * inv_rho
    
    # Gradients
    p2_y = np.zeros_like(rho)
    p2_x = np.zeros_like(rho)
    p2_y[1:-1, :] = (P2[2:, :] - P2[:-2, :]) / (2*dy)
    p2_x[:, 1:-1] = (P2[:, 2:] - P2[:, :-2]) / (2*dx)
    
    phi_y = np.zeros_like(rho)
    phi_x = np.zeros_like(rho)
    phi_y[1:-1, :] = (phi_e[2:, :] - phi_e[:-2, :]) / (2*dy)
    phi_x[:, 1:-1] = (phi_e[:, 2:] - phi_e[:, :-2]) / (2*dx)
    
    grad_norm = np.sqrt(phi_x**2 + phi_y**2) + 1e-6
    nx_dir = -phi_x / grad_norm
    ny_dir = -phi_y / grad_norm
    
    # Equilibrium Speed using params A and B
    ve_mag = get_f_rho(rho, A, B)
    ue = ve_mag * nx_dir
    ve = ve_mag * ny_dir
    
    # Source Terms
    S_u = rho * (ue - u) / Tau - p2_x / Mass
    S_v = rho * (ve - v) / Tau - p2_y / Mass
    
    RHS[1] += S_u
    RHS[2] += S_v
    
    # Apply Obstacles
    for j in range(ny):
        for i in range(nx):
            if mask_obs[j, i]:
                RHS[0, j, i] = 0.0
                RHS[1, j, i] = 0.0
                RHS[2, j, i] = 0.0
                
    return RHS

@jit(nopython=True)
def compute_panic_source(rho, X, Y, t, panic_start, rho0, pushing_cap_coeff):
    ny, nx = rho.shape
    source = np.zeros((ny, nx))
    if t < panic_start: return source
    for j in range(ny):
        for i in range(nx):
            if rho[j, i] <= rho0: continue
            dist = np.sqrt((X[j, i] - 60.0)**2 + (Y[j, i] - 31.5)**2)
            term_dist = max(1.0 - dist/20.0, 0.0)
            k_val = pushing_cap_coeff * np.sqrt(rho[j, i] - rho0)
            source[j, i] = term_dist * k_val
    return source

# --- MAIN SIMULATION DRIVER ---
def run_simulation_backend(rho0, rhom, c0, mass, fd_A, fd_B, cost_C, push_K):
    # Run the actual heavy simulation using the parameters
    nx = int(Config.L_X / Config.DX)
    ny = int(Config.L_Y / Config.DY)
    x = np.linspace(0, Config.L_X, nx)
    y = np.linspace(0, Config.L_Y, ny)
    X, Y = np.meshgrid(x, y)
    
    Q = np.zeros((3, ny, nx))
    mask_obs = np.zeros((ny, nx), dtype=np.bool_)
    for obs in Config.OBSTACLES:
        mask_obs[(Y >= obs[2]) & (Y <= obs[3]) & (X >= obs[0]) & (X <= obs[1])] = True
    mask_dest = (X >= Config.L_X - 1.0)
    
    # Initial Population
    Q[0, :, :] = 2.0
    for j in range(ny):
        for i in range(nx):
            if mask_obs[j, i]: Q[0, j, i] = 0.0
    # Initial velocity
    Q[1, :, :] = Q[0] * get_f_rho(Q[0], fd_A, fd_B)
    
    t = 0.0
    history = []
    save_interval = 2.0
    last_save = -save_interval
    
    # Progress bar setup
    bar = st.progress(0)
    status = st.empty()
    
    while t < Config.T_MAX:
        # Panic Obstacle
        if t >= Config.PANIC_START_TIME:
            o = Config.NEW_OBSTACLE
            mask_obs[(Y >= o[2]) & (Y <= o[3]) & (X >= o[0]) & (X <= o[1])] = True

        # Inflow (High flux)
        rho_in = 5.5
        Q[0, :, 0:2] = rho_in
        Q[1, :, 0:2] = rho_in * get_f_rho(rho_in, fd_A, fd_B)
        Q[2, :, 0:2] = 0.0
        
        # Fields
        rho = Q[0]
        cost_nav = get_g_rho(rho, cost_C) + 1.0 / (get_f_rho(rho, fd_A, fd_B) + 1e-6)
        for j in range(ny):
            for i in range(nx):
                if mask_obs[j, i]: cost_nav[j, i] = 1000.0
        
        phi_e = solve_eikonal_jit(cost_nav, mask_dest, Config.DX)
        
        rhs_p2 = compute_panic_source(rho, X, Y, t, Config.PANIC_START_TIME, rho0, push_K)
        mask_p2_zero = (rho <= rho0)
        pot_p2 = solve_eikonal_jit(rhs_p2, mask_p2_zero, Config.DX)
        alpha = np.maximum((rho - rho0)/(rhom - rho0), 0.0)
        P2 = pot_p2 * alpha
        
        # Time Step
        mask_flow = rho > 1e-3
        max_v = c0
        if np.any(mask_flow):
            v_sq = (Q[1, mask_flow]/rho[mask_flow])**2 + (Q[2, mask_flow]/rho[mask_flow])**2
            max_v = np.sqrt(np.max(v_sq)) + c0
            
        dt = Config.CFL * Config.DX / (max_v + 1e-6)
        if dt > 0.1: dt = 0.1
        
        # RK3 Integration
        # Using passed parameters: C0=c0, Mass=mass, A=fd_A, B=fd_B
        # Tau is fixed in Config but could be passed if needed
        L1 = compute_rhs_jit(Q, P2, phi_e, mask_obs, Config.DX, Config.DY, c0, Config.TAU, mass, fd_A, fd_B)
        Q1 = Q + dt * L1
        L2 = compute_rhs_jit(Q1, P2, phi_e, mask_obs, Config.DX, Config.DY, c0, Config.TAU, mass, fd_A, fd_B)
        Q2 = 0.75 * Q + 0.25 * (Q1 + dt * L2)
        L3 = compute_rhs_jit(Q2, P2, phi_e, mask_obs, Config.DX, Config.DY, c0, Config.TAU, mass, fd_A, fd_B)
        Q_new = (1.0/3.0) * Q + (2.0/3.0) * (Q2 + dt * L3)
        
        for j in range(ny):
            for i in range(nx):
                if mask_obs[j, i]: Q_new[:, j, i] = 0.0
        
        Q = Q_new
        t += dt
        
        if t - last_save >= save_interval:
            history.append((t, Q[0].copy()))
            last_save = t
            progress = min(t / Config.T_MAX, 1.0)
            bar.progress(progress)
            status.text(f"Simulating: {t:.1f}s / {Config.T_MAX}s")
            
    return X, Y, history

# ==========================================
# 3. Page Content
# ==========================================

# --- PAGE 1: LANDING ---
if st.session_state.page == 'landing':
    st.markdown("<h1 style='text-align: center; margin-top: 100px;'>Crowd & Panic Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>A second-order macroscopic model for pedestrian flow analysis.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Next"):
            go_to_guide()
            st.rerun()

# --- PAGE 2: GUIDE ---
elif st.session_state.page == 'guide':
    st.markdown("<h1>User Guide</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### How it works
    1. **Configure**: Set physical parameters like crowd mass, density limits, and walking speeds.
    2. **Simulate**: The system solves partial differential equations to model flow.
    3. **Analyze**: Watch how crowd pressure builds up during panic events.
    """)
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Let's Get Started!"):
            go_to_sim()
            st.rerun()

# --- PAGE 3: SIMULATION ---
elif st.session_state.page == 'simulation':
    
    # Header
    st.markdown("<h3>Simulation Dashboard</h3>", unsafe_allow_html=True)
    
    # Main Layout: 1/3 Left (Inputs), 2/3 Right (Results)
    col_input, col_result = st.columns([1, 2], gap="large")
    
    with col_input:
        st.markdown("#### 1) Parameters")
        st.info("Hover over parameters for details.")
        
        # --- INPUT BLOCKS (Matches Figma) ---
        # We use st.session_state.inputs_disabled to lock them
        disabled = st.session_state.inputs_disabled
        
        crit_dens = st.number_input("Critical Density (ped/m²)", value=5.0, disabled=disabled)
        max_dens = st.number_input("Maximum Density (ped/m²)", value=7.0, disabled=disabled)
        sonic_spd = st.number_input("Sonic Speed (m/s)", value=1.2, disabled=disabled)
        avg_mass = st.number_input("Average Mass (kg)", value=60.0, disabled=disabled)
        
        st.markdown("**Fundamental Diagram**")
        c1, c2 = st.columns(2)
        fd_A = c1.number_input("Coeff A", value=0.5, disabled=disabled, help="Base speed (m/s)")
        fd_B = c2.number_input("Coeff B", value=-0.075, disabled=disabled, help="Density decay factor")
        st.caption(f"Speed = {fd_A} * exp({fd_B} * ρ²)")
        
        cost_C = st.number_input("Density Cost Function Coeff", value=0.01, disabled=disabled)
        push_K = st.number_input("Pushing Capacity Coeff", value=100.0, disabled=disabled)
        
        st.markdown("---")
        
        # --- ACTION BUTTONS ---
        if not st.session_state.inputs_disabled:
            # STATE 3: Ready to start
            if st.button("Start Simulation!"):
                st.session_state.inputs_disabled = True
                st.rerun() # Rerun to update UI to "Disabled" state immediately
        else:
            # STATE 4: Simulation Done/Running
            if st.session_state.simulation_data is None:
                # This block runs right after user clicks start (and rerun happens)
                with st.spinner("Simulating physics (this may take ~1 minute)..."):
                    # RUN BACKEND HERE
                    X, Y, hist = run_simulation_backend(crit_dens, max_dens, sonic_spd, avg_mass, fd_A, fd_B, cost_C, push_K)
                    st.session_state.simulation_data = (X, Y, hist)
                st.rerun() # Rerun to show results
            else:
                # Simulation is finished, data is present. Show "Edit" button.
                if st.button("Edit New Parameters"):
                    enable_editing()
                    st.rerun()

    with col_result:
        st.markdown("#### 2) Simulation Results")
        
        # Container for the result
        result_container = st.container()
        
        if st.session_state.simulation_data is None:
            # Placeholder State
            result_container.markdown(
                """
                <div style='height: 400px; border: 2px dashed #444; display: flex; align-items: center; justify-content: center; color: #888;'>
                    Awaiting parameter submission...
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            # Result State
            X, Y, history = st.session_state.simulation_data
            
            # Create Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Draw Static Obstacles
            for obs in Config.OBSTACLES:
                ax.add_patch(plt.Rectangle((obs[0], obs[2]), obs[1]-obs[0], obs[3]-obs[2], fc='black'))
            
            # Dynamic Obstacle Placeholder
            dyn_obs = plt.Rectangle((60, 30), 5, 3, fc='red', alpha=0)
            ax.add_patch(dyn_obs)
            
            # Initial Contours
            mesh = ax.contourf(X, Y, history[0][1], levels=np.linspace(0, 10, 100), cmap='jet', extend='both')
            fig.colorbar(mesh, ax=ax, label='Density')
            title = ax.set_title("Time: 0.0s")

            def update(frame_idx):
                t, rho = history[frame_idx]
                ax.clear()
                ax.contourf(X, Y, rho, levels=np.linspace(0, 10, 100), cmap='jet', extend='both')
                ax.set_title(f"Time: {t:.1f} s")
                for obs in Config.OBSTACLES:
                    ax.add_patch(plt.Rectangle((obs[0], obs[2]), obs[1]-obs[0], obs[3]-obs[2], fc='black'))
                if t >= Config.PANIC_START_TIME:
                    o = Config.NEW_OBSTACLE
                    ax.add_patch(plt.Rectangle((o[0], o[2]), o[1]-o[0], o[3]-o[2], fc='red'))

            anim = FuncAnimation(fig, update, frames=len(history), interval=100)
            
            # Display using Streamlit
            st.components.v1.html(anim.to_jshtml(), height=600, scrolling=True)
            result_container.success("Simulation Complete.")