import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Manifold Geometry (Sphere S^2) ---
def slerp(x0, x1, t):
    """Spherical Linear Interpolation (Geodesic path)"""
    # Clamp for numerical stability
    dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # Formula: sin((1-t)theta)/sin(theta) * x0 + sin(t*theta)/sin(theta) * x1
    scale0 = torch.sin((1 - t) * theta) / sin_theta
    scale1 = torch.sin(t * theta) / sin_theta
    
    # Fallback for small angles (linear approx)
    res = scale0 * x0 + scale1 * x1
    return torch.where(sin_theta < 1e-4, (1-t)*x0 + t*x1, res)

def geodesic_derivative(x0, x1, t):
    """Analytic Time-Derivative of Slerp (The Riemannian Target Field)"""
    dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # d/dt coefficients
    d_scale0 = -theta * torch.cos((1 - t) * theta) / sin_theta
    d_scale1 = theta * torch.cos(t * theta) / sin_theta
    
    res = d_scale0 * x0 + d_scale1 * x1
    return torch.where(sin_theta < 1e-4, x1 - x0, res)

# --- 2. Training the Toy Models ---
def train_toy_model(mode='euclidean'):
    # Simple MLP: Inputs (x,y,z, t) -> Output (vx, vy, vz)
    model = nn.Sequential(
        nn.Linear(4, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 3)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Training {mode.capitalize()} Flow...")
    for _ in range(1500):
        # Data: Source (Random Noise) -> Target (Cluster near North Pole)
        x0 = torch.randn(64, 3); x0 /= x0.norm(dim=1, keepdim=True)
        x1 = torch.randn(64, 3) * 0.5 + torch.tensor([0., 0., 2.]); x1 /= x1.norm(dim=1, keepdim=True)
        t = torch.rand(64, 1)

        if mode == 'euclidean':
            xt = (1 - t) * x0 + t * x1    # Straight line
            ut = x1 - x0                  # Constant velocity
        else:
            xt = slerp(x0, x1, t)         # Geodesic
            ut = geodesic_derivative(x0, x1, t) # Tangent vector

        vt = model(torch.cat([xt, t], dim=1))
        loss = ((vt - ut)**2).mean()
        
        opt.zero_grad(); loss.backward(); opt.step()
    return model

# --- 3. Generate Hero Image ---
def visualize_hero():
    m_e = train_toy_model('euclidean')
    m_r = train_toy_model('riemannian')
    
    # Test Trajectory: Start at X-axis
    start = torch.tensor([[1.0, 0.0, 0.0]])
    steps = 50
    dt = 1.0 / steps

    # Integration Loop
    path_e, path_r = [start], [start]
    curr_e, curr_r = start.clone(), start.clone()
    
    for i in range(steps):
        t = torch.tensor([[i * dt]])
        
        # Euclidean: Integrate directly
        v_e = m_e(torch.cat([curr_e, t], dim=1))
        curr_e = curr_e + v_e * dt
        path_e.append(curr_e.detach())
        
        # Riemannian: Integrate + Retract (Project back to Sphere)
        v_r = m_r(torch.cat([curr_r, t], dim=1))
        curr_r = curr_r + v_r * dt
        curr_r = curr_r / curr_r.norm() # <--- The "Slide" Magic
        path_r.append(curr_r.detach())

    # Plot
    pe, pr = torch.cat(path_e).numpy(), torch.cat(path_r).numpy()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sphere Wireframe
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v), color="gray", alpha=0.1)
    
    ax.plot(pe[:,0], pe[:,1], pe[:,2], 'r--', linewidth=2, label='Euclidean (Cuts Through)')
    ax.plot(pr[:,0], pr[:,1], pr[:,2], 'b-', linewidth=3, label='Riemannian (Slides)')
    ax.scatter([1], [0], [0], c='k', s=50, label='Start')
    ax.legend(); ax.set_axis_off(); plt.title("Visualizing Manifold Flow")
    plt.show()

if __name__ == "__main__":
    visualize_hero()
