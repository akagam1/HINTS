import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# ============================================================
# Data generation (Poisson equation, Fourier series terms)
# ============================================================
def poisson_sample(N=31, terms=15, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    x = np.linspace(0, 1, N)
    y = np.linspace(-0.5, 0.5, N) 
    X, Y = np.meshgrid(x, y, indexing='ij')

    u = np.zeros((N, N), dtype=np.float64)
    f = np.zeros((N, N), dtype=np.float64)

    b = terms
    for i in range(1, terms + 1):
        a_n = rng.uniform(0, 1)
        basis = np.sin((i * np.pi * X)) * np.cos((i * np.pi * Y))
        u += a_n * basis / b
        f += (2 * ((i * np.pi)) ** 2 * a_n) * basis / b

    return u, f

# ============================================================
# NMSE Loss
# ============================================================
def nmse_loss(pred, target):
    return ((pred - target)**2).mean() / (target**2).mean()

# ============================================================
# Fast Jacobi solver for Poisson
# ============================================================
def jacobi_solver(f, u0=None, iterations=1):
    N = f.shape[0]
    h = 1.0 / (N - 1)
    h2 = h * h
    u = np.zeros_like(f) if u0 is None else u0.copy()
    for _ in range(iterations):
        u_new = u.copy()
        u_new[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] +
                                    u[1:-1, :-2] + u[1:-1, 2:] + h2 * f[1:-1, 1:-1])
        u = u_new
    return u

# ============================================================
# DeepONet 2D
# ============================================================
class BranchNet2D(nn.Module):
    def __init__(self,input_shape=(1,50,50), out_dim=80):
        super().__init__()
        self.out_dim = 80  # Fixed output size

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=3, stride=2, padding='valid'),  # -> (40, 15, 15)
            nn.ReLU(),
            nn.Conv2d(40, 60, kernel_size=3, stride=2, padding='valid'),  # -> (60, 7, 7)
            nn.ReLU(),
            nn.Conv2d(60, 100, kernel_size=3, stride=2, padding='valid'),  # -> (100, 3, 3)
            nn.ReLU(),
            nn.Conv2d(100, 180, kernel_size=3, stride=1, padding='valid'),  # -> (180, 1, 1)
            nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)  
            flat_size = self.conv(dummy).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, self.out_dim)
        )

    def forward(self, f):
        x = self.conv(f)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)

class TrunkNet2D(nn.Module):
    def __init__(self, out_dim=128):
        super(TrunkNet2D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 80),
            nn.Tanh(),
            nn.Linear(80, 80),
            nn.Tanh(),
            nn.Linear(80, 80),
            nn.Tanh()
        )

    def forward(self, coords):
        return self.net(coords)

class DeepONet2D(nn.Module):
    def __init__(self, input_shape=(1,50,50), branch_out_dim=128, trunk_out_dim=128):
        super(DeepONet2D, self).__init__()
        self.branch = BranchNet2D(input_shape=input_shape,out_dim=branch_out_dim)
        self.trunk = TrunkNet2D(out_dim=trunk_out_dim)

    def forward(self, f, coords):
        """
        f: (batch, 1, N, N)
        coords: (M, 2)
        """
        b = self.branch(f)         
        t = self.trunk(coords)   
        out = torch.matmul(t, b.T)
        return out.T               
# ============================================================
# HINTS Solver
# ============================================================
def hints_solver(model, f, u_true, nr=24, max_iters=200):
    N = f.shape[0]
    device = next(model.parameters()).device
    coords = np.array([[i / (N - 1), j / (N - 1)] for i in range(N) for j in range(N)])
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    u = np.zeros_like(f)
    errors = []
    f_t = torch.tensor(f, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    f_norm = torch.sqrt(torch.mean(f_t**2, dim=(1,2,3), keepdim=True))
    f_t_norm = f_t / (f_norm + 1e-10)

    for it in range(max_iters):
        h = 1.0 / (N - 1)
        h2 = h * h
        r = f - (
            (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
             np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) -
             4 * u) / h2
        )

        if it % nr == 0 and it > 0:
            with torch.no_grad():
                delta_u = model(f_t_norm, coords_t).cpu().numpy().reshape(N, N)
            alpha = 0.8
            u = (1-alpha)*delta_u + alpha*u
        else:
            u = jacobi_solver(f, u0=u, iterations=1)

        err = np.linalg.norm(u - u_true) / np.linalg.norm(u_true)
        errors.append(err)

    return u, errors

# ============================================================
# Main
# ============================================================
def main():
    # ------------------------------------------------------------
    # Generate training/validation data
    # ------------------------------------------------------------
    N = 50
    nval = 500
    U_val, F_val = [], []
    for i in range(nval):
        u, f = poisson_sample(N=N, terms=15)
        U_val.append(u)
        F_val.append(f)

    U_val, F_val = np.array(U_val), np.array(F_val)

    # ------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------
    def make_dataset(U, F):
        F_t = torch.tensor(F, dtype=torch.float32).unsqueeze(1)  # (B,1,N,N)
        F_norm = torch.sqrt(torch.mean(F_t**2, dim=(1,2,3), keepdim=True))
        F_t_norm = F_t / (F_norm + 1e-10)
        U_t = torch.tensor(U.reshape(U.shape[0], -1), dtype=torch.float32)  # flatten
        return TensorDataset(F_t_norm, U_t)

    val_loader = DataLoader(make_dataset(U_val, F_val), batch_size=10, shuffle=False)

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DeepONet2D(input_shape = (1,N,N), branch_out_dim=80, trunk_out_dim=80).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    coords = np.array([[i / (N - 1), j / (N - 1)] for i in range(N) for j in range(N)])
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    # ------------------------------------------------------------
    # Training loop with new randomized dataset per epoch
    # ------------------------------------------------------------
    epochs = 10000
    min_lr = 1e-5
    samples_per_epoch = 6400
    prog_bar = tqdm(range(epochs), desc="Training Progress", position=0)
    for ep in prog_bar:
        # === Generate new dataset for this epoch ===
        U_train, F_train = [], []
        for _ in range(samples_per_epoch):
            u, f = poisson_sample(N=N, terms=15)
            U_train.append(u)
            F_train.append(f)
        train_loader = DataLoader(
            make_dataset(np.array(U_train), np.array(F_train)),
            batch_size=16, shuffle=True
        )

        # === Maintain minimum LR ===
        for param_group in optimizer.param_groups:
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr

        # === Training ===
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", position=1, leave=False)
        for f_batch, u_batch in train_bar:
            f_batch, u_batch = f_batch.to(device), u_batch.to(device)
            optimizer.zero_grad()
            pred = model(f_batch, coords_t)
            loss = nmse_loss(pred, u_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (train_bar.n + 1)
            train_bar.set_postfix({"Train NMSE": f"{avg_loss:.4e}"})

        scheduler.step()

        # === Validation (use fixed validation set) ===
        model.eval()
        val_losses = []
        val_bar = tqdm(val_loader, desc=f"Validation {ep+1}/{epochs}", position=1, leave=False)
        with torch.no_grad():
            for f_batch, u_batch in val_bar:
                f_batch, u_batch = f_batch.to(device), u_batch.to(device)
                pred = model(f_batch, coords_t)
                val_losses.append(nmse_loss(pred, u_batch).item())
                val_bar.set_postfix({"Val NMSE": f"{np.mean(val_losses):.4e}"})

        prog_bar.set_postfix({
        "": f"Train={total_loss/len(train_loader):.6e} | "
            f"Val={np.mean(val_losses):.6e} | "
            f"LR={scheduler.get_last_lr()[0]:.1e}"
    })

        if (ep + 1) % 50 == 0:
            save_path = f"deeponet_epoch_{ep+1}.pth"
            torch.save({
                'epoch': ep + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': total_loss / len(train_loader),
                'val_loss': np.mean(val_losses)
            }, save_path)
            tqdm.write(f"✅ Saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()
