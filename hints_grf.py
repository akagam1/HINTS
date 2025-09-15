import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# ============================================================
# GRF sampling (spectral) and Poisson solver (FFT)
# ============================================================
def sample_grf(N=31, length_scale=0.25, sigma=1.0, rng=None):
    """
    Spectral sampling of a Gaussian Random Field on an N x N periodic grid.
    length_scale controls smoothness (larger -> smoother).
    """
    if rng is None:
        rng = np.random

    kx = np.fft.fftfreq(N)[:, None]   # shape (N,1)
    ky = np.fft.fftfreq(N)[None, :]   # shape (1,N)
    ksq = kx**2 + ky**2
    # squared-exponential kernel spectrum (Gaussian in k)
    spectrum = np.exp(-0.5 * (2*np.pi*length_scale)**2 * ksq)
    # complex white noise
    noise = rng.randn(N, N) + 1j * rng.randn(N, N)
    field = np.fft.ifft2(noise * np.sqrt(spectrum)).real
    # normalize to standard deviation sigma
    field = sigma * field / (np.std(field) + 1e-16)
    return field

def solve_poisson_fft(f):
    """
    Solve -Laplace(u) = f on periodic grid using spectral division:
    u_hat = f_hat / |k|^2  (zero mean enforced)
    Returns real u with zero mean.
    """
    N = f.shape[0]
    kx = np.fft.fftfreq(N)[:, None]
    ky = np.fft.fftfreq(N)[None, :]
    ksq = (2*np.pi*kx)**2 + (2*np.pi*ky)**2
    # avoid division by zero -- handle zero mode separately
    ksq[0,0] = 1.0
    f_hat = np.fft.fft2(f)
    u_hat = f_hat / ksq
    u_hat[0,0] = 0.0   # enforce zero mean solution (removes constant)
    u = np.fft.ifft2(u_hat).real
    return u

# ============================================================
# NMSE Loss
# ============================================================
def nmse_loss(pred, target):
    # pred, target: torch tensors
    return ((pred - target)**2).mean() / (target**2).mean()

# ============================================================
# Fast Jacobi solver for Poisson (used in HINTS inner loop)
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
# DeepONet 2D (Branch/Trunk) - same API as your original code
# ============================================================
class BranchNet2D(nn.Module):
    def __init__(self, out_dim=80):
        super().__init__()
        self.out_dim = out_dim  # Fixed output size (80 expected)
        # Convolutional feature extractor matching your hard-coded 31x31 design
        self.conv = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=3, stride=2, padding='valid'),  # -> (40, 15, 15)
            nn.ReLU(),
            nn.Conv2d(40, 60, kernel_size=3, stride=2, padding='valid'),  # -> (60, 7, 7)
            nn.ReLU(),
            nn.Conv2d(60, 100, kernel_size=3, stride=2, padding='valid'),  # -> (100, 3, 3)
            nn.ReLU()
        )

        # flatten dim = 100 * 3 * 3 = 900
        self.fc = nn.Sequential(
            nn.Linear(900, 80),
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
    def __init__(self, out_dim=80):
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
    def __init__(self, branch_out_dim=80, trunk_out_dim=80):
        super(DeepONet2D, self).__init__()
        self.branch = BranchNet2D(out_dim=branch_out_dim)
        self.trunk = TrunkNet2D(out_dim=trunk_out_dim)

    def forward(self, f, coords):
        """
        f: (batch, 1, N, N)
        coords: (M, 2)
        returns: (batch, M)
        """
        b = self.branch(f)         # (batch, branch_out_dim)
        t = self.trunk(coords)     # (M, trunk_out_dim)
        out = torch.matmul(t, b.T) # (M, batch)
        return out.T               # (batch, M)

# ============================================================
# HINTS solver that mixes Jacobi updates with model corrections
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
                # model predicts flattened u (M,) per batch; run single-sample batch
                delta_u = model(f_t_norm, coords_t).cpu().numpy().reshape(N, N)
            alpha = 0.8
            u = (1-alpha)*delta_u + alpha*u
        else:
            u = jacobi_solver(f, u0=u, iterations=1)

        err = np.linalg.norm(u - u_true) / np.linalg.norm(u_true)
        errors.append(err)

    return u, errors

# ============================================================
# Dataset creation helper using GRF + spectral Poisson solver
# ============================================================
def make_dataset_from_grf(U, F):
    """
    U: np array shape (B, N, N) (solutions)
    F: np array shape (B, N, N) (forces)
    returns TensorDataset with normalized F and flattened U
    """
    F_t = torch.tensor(F, dtype=torch.float32).unsqueeze(1)  # (B,1,N,N)
    F_norm = torch.sqrt(torch.mean(F_t**2, dim=(1,2,3), keepdim=True))
    F_t_norm = F_t / (F_norm + 1e-10)
    U_t = torch.tensor(U.reshape(U.shape[0], -1), dtype=torch.float32)  # flatten kernels
    return TensorDataset(F_t_norm, U_t)

# ============================================================
# Main training & evaluation pipeline
# ============================================================
def main():
    # ------------------------------------------------------------
    # Hyperparams and device
    # ------------------------------------------------------------
    N = 31
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validation dataset (fixed)
    nval = 500
    rng = np.random.RandomState(12345)
    F_val = []
    U_val = []
    for i in range(nval):
        f = sample_grf(N=N, length_scale=0.25, sigma=1.0, rng=rng)
        u = solve_poisson_fft(f)
        F_val.append(f)
        U_val.append(u)
    F_val = np.array(F_val)
    U_val = np.array(U_val)

    val_loader = DataLoader(make_dataset_from_grf(U_val, F_val), batch_size=10, shuffle=False)

    # ------------------------------------------------------------
    # Model, optimizer, scheduler
    # ------------------------------------------------------------
    model = DeepONet2D(branch_out_dim=80, trunk_out_dim=80).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    coords = np.array([[i / (N - 1), j / (N - 1)] for i in range(N) for j in range(N)])
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    # ------------------------------------------------------------
    # Training loop with new randomized GRF dataset each epoch
    # ------------------------------------------------------------
    epochs = 1000
    min_lr = 1e-5
    samples_per_epoch = 6400   # number of randomly generated samples used per epoch

    for ep in tqdm(range(epochs), desc="Training Progress", position=0):
        # Generate epoch dataset (GRFs sampled fresh)
        rng_epoch = np.random.RandomState(seed=ep + 999)  # optionally seed per epoch for reproducibility
        F_train = []
        U_train = []
        for _ in range(samples_per_epoch):
            f = sample_grf(N=N, length_scale=0.25, sigma=1.0, rng=rng_epoch)
            u = solve_poisson_fft(f)
            F_train.append(f)
            U_train.append(u)
        F_train = np.array(F_train)
        U_train = np.array(U_train)

        train_loader = DataLoader(make_dataset_from_grf(U_train, F_train), batch_size=16, shuffle=True)

        # Maintain minimum LR
        for param_group in optimizer.param_groups:
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr

        # Training pass
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", position=1, leave=False)
        seen = 0
        for f_batch, u_batch in train_bar:
            f_batch, u_batch = f_batch.to(device), u_batch.to(device)
            optimizer.zero_grad()
            pred = model(f_batch, coords_t)  # (batch, M)
            loss = nmse_loss(pred, u_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * f_batch.size(0)
            seen += f_batch.size(0)
            avg_loss = total_loss / seen
            train_bar.set_postfix({"Train NMSE": f"{avg_loss:.4e}"})

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for f_batch, u_batch in val_loader:
                f_batch, u_batch = f_batch.to(device), u_batch.to(device)
                pred = model(f_batch, coords_t)
                val_losses.append(nmse_loss(pred, u_batch).item())

        mean_val = np.mean(val_losses)
        mean_train = total_loss / len(train_loader.dataset)

        tqdm.write(
            f"[Epoch {ep+1}/{epochs}] "
            f"Train NMSE={mean_train:.6e} | "
            f"Val NMSE={mean_val:.6e} | "
            f"LR={scheduler.get_last_lr()[0]:.1e}"
        )

        if (ep + 1) % 50 == 0:
            save_path = f"deeponet_grf_epoch_{ep+1}.pth"
            torch.save({
                'epoch': ep + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': mean_train,
                'val_loss': mean_val
            }, save_path)
            tqdm.write(f"✅ Saved checkpoint: {save_path}")

    # ------------------------------------------------------------
    # Test: Generate a random GRF sample and show predictions
    # ------------------------------------------------------------
    rng_test = np.random.RandomState(2025)
    f_test = sample_grf(N=N, length_scale=0.25, sigma=1.0, rng=rng_test)
    u_true = solve_poisson_fft(f_test)

    f_t = torch.tensor(f_test, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    f_norm = torch.sqrt(torch.mean(f_t**2, dim=(1,2,3), keepdim=True))
    f_t_norm = f_t / (f_norm + 1e-10)
    f_t_norm = f_t_norm.to(device)

    model.eval()
    with torch.no_grad():
        u_pred = model(f_t_norm, coords_t).cpu().numpy().reshape(N, N)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(f_test, cmap="viridis"); axs[0].set_title("GRF f (test)")
    axs[1].imshow(u_true, cmap="viridis"); axs[1].set_title("Ground Truth u (spectral)")
    axs[2].imshow(u_pred, cmap="viridis"); axs[2].set_title("DeepONet Prediction")
    plt.show()

    # ------------------------------------------------------------
    # Run HINTS solver to combine model + Jacobi
    # ------------------------------------------------------------
    u_hints, errors = hints_solver(model, f_test, u_true, nr=25, max_iters=500)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(u_true, cmap="viridis"); axs[0].set_title("Ground Truth u")
    im = axs[1].imshow(u_hints, cmap="viridis"); axs[1].set_title("DeepONet + HINTS")
    plt.colorbar(im, ax=axs[1])
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(errors, label="Relative Error")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Relative Error")
    plt.title("HINTS Solver Error Curve")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
