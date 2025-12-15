import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class SphericalGeo:
    """
    Handles the geometry of the Hypersphere S^d.
    """
    @staticmethod
    def geodesic_path(x0, x1, t):
        """
        Spherical Linear Interpolation (SLERP).
        """
        dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
      
        safe_sin = torch.where(sin_theta < 1e-6, torch.ones_like(sin_theta), sin_theta)
        
        factor0 = torch.sin((1 - t) * theta) / safe_sin
        factor1 = torch.sin(t * theta) / safe_sin
        
        return factor0 * x0 + factor1 * x1

    @staticmethod
    def target_velocity(x0, x1, t):
        """
        Computes the time derivative of the geodesic path at time t.
        """
        dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        safe_sin = torch.where(sin_theta < 1e-6, torch.ones_like(sin_theta), sin_theta)

        # Derivative of the SLERP coefficients w.r.t time t
        d_factor0 = -theta * torch.cos((1 - t) * theta) / safe_sin
        d_factor1 = theta * torch.cos(t * theta) / safe_sin
        
        return d_factor0 * x0 + d_factor1 * x1

    @staticmethod
    def project_to_tangent(x, v):
        """
        Projects a vector v onto the tangent plane of point x.
        Ensures the velocity doesn't point 'off' the surface.
        """
        dot = torch.sum(x * v, dim=-1, keepdim=True)
        return v - dot * x

# ==========================================
# 2. The Neural Network
# ==========================================
class VectorFieldNetwork(nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        # Input: current_pos (3) + time (1) + target_cond (3) = 7 dims
        self.net = nn.Sequential(
            nn.Linear(dim * 2 + 1, 128),
            nn.Tanh(), 
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, dim)
        )

    def forward(self, x, t, target):
        # Concatenate inputs
        inp = torch.cat([x, t, target], dim=-1)
        return self.net(inp)

# ==========================================
# 3. Training & Sampling
# ==========================================
def train_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Initialize
    model = VectorFieldNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # --- Training Loop ---
    print("Training Riemannian Flow Matching...")
    for step in range(2000):
        optimizer.zero_grad()
        
        # Sample random points on a sphere
        batch_size = 128
        x0 = F.normalize(torch.randn(batch_size, 3), dim=-1).to(device) # Start
        x1 = F.normalize(torch.randn(batch_size, 3), dim=-1).to(device) # Target

        t = torch.rand(batch_size, 1).to(device)
        
        x_t = SphericalGeo.geodesic_path(x0, x1, t)
        u_t = SphericalGeo.target_velocity(x0, x1, t)
        
        v_pred = model(x_t, t, x1) # We condition on x1 (Target)
        
        # Project Prediction to Tangent Space
        v_pred_tangent = SphericalGeo.project_to_tangent(x_t, v_pred)
        
        # Loss
        loss = F.mse_loss(v_pred_tangent, u_t)
        
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item():.5f}")

    # --- Inference (Generation) ---
    print("\nGenerating Trajectories...")
    model.eval()
    
    # Demo: start at North Pole (0, 0, 1), End at Equator (1, 0, 0)
    start_point = torch.tensor([[0.0, 0.0, 1.0]]).to(device)
    target_point = torch.tensor([[1.0, 0.0, 0.0]]).to(device)
    
    # Euler Integration on Manifold
    steps = 50
    dt = 1.0 / steps
    current_x = start_point.clone()
    trajectory = [current_x.cpu().numpy()]
    
    with torch.no_grad():
        for i in range(steps):
            t_val = torch.ones(1, 1).to(device) * (i / steps)
            vel = model(current_x, t_val, target_point)
            vel = SphericalGeo.project_to_tangent(current_x, vel)
            
            # Update position (Euler step)
            next_x = current_x + vel * dt
            
            # Retraction (Project back to sphere surface)
            # This corrects the drift from the straight-line Euler step
            current_x = F.normalize(next_x, dim=-1)
            
            trajectory.append(current_x.cpu().numpy())
    
    traj_arr = np.concatenate(trajectory, axis=0)

    # --- 3D Visualization ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Sphere Wireframe
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.1)
    
    # Draw Trajectory
    ax.plot(traj_arr[:,0], traj_arr[:,1], traj_arr[:,2], 'b-', linewidth=3, label="Learned Flow")
    
    # Draw Start/End Markers
    ax.scatter(traj_arr[0,0], traj_arr[0,1], traj_arr[0,2], color='green', s=100, label='Start')
    ax.scatter(traj_arr[-1,0], traj_arr[-1,1], traj_arr[-1,2], color='red', s=100, label='Target')
    
    # Draw Ideal Geodesic for comparison (Dashed)
    ideal_traj = []
    ts = torch.linspace(0, 1, 50).view(-1, 1).to(device)
    # We expand start/target to match batch size of ts
    st_exp = start_point.repeat(50, 1)
    tg_exp = target_point.repeat(50, 1)
    ideal_pts = SphericalGeo.geodesic_path(st_exp, tg_exp, ts).cpu().numpy()
    ax.plot(ideal_pts[:,0], ideal_pts[:,1], ideal_pts[:,2], 'k--', alpha=0.5, label="Ideal Geodesic")

    ax.legend()
    ax.set_title("Riemannian Flow Matching Inference (Learned Geodesic)")
    plt.show()

if __name__ == "__main__":
    train_and_visualize()
