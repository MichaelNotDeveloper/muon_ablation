import torch


class NuclearNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, z="zero", include_loss=False, tol=1e-8):
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = (S > tol).sum().item()

        ctx.save_for_backward(U[:, :r], Vh[:r].T)
        ctx.shape = W.shape
        ctx.z = z
        if include_loss is False:
            return W.new_zeros(())
        return S.sum()

    @staticmethod
    def backward(ctx, grad_out):
        U, V = ctx.saved_tensors
        m, n = ctx.shape
        G = U @ V.T

        if ctx.z == "random":
            A = torch.randn(m, n, device=U.device, dtype=U.dtype)

            Pu = torch.eye(m, device=U.device, dtype=U.dtype) - U @ U.T
            Pv = torch.eye(n, device=U.device, dtype=U.dtype) - V @ V.T

            Z = Pu @ A @ Pv
            Z = Z / torch.linalg.norm(Z, ord=2).clamp_min(1e-12)

            G = G + Z

        return grad_out * G, None, None, None


def reg(W, reg_type=None, grad_type="zero", include_loss=False):
    reg_type = normalize_reg_type(reg_type)
    if reg_type == "nuclear":
        return NuclearNorm.apply(W, grad_type, include_loss)
    return 0


def normalize_reg_type(reg_type):
    if reg_type is None:
        return None
    if isinstance(reg_type, str) and reg_type.lower() in ("", "no", "none", "null"):
        return None
    return reg_type


def regularization_grad(W, reg_type=None, grad_type="zero", tol=1e-8):
    reg_type = normalize_reg_type(reg_type)
    if reg_type is None:
        return None
    if reg_type != "nuclear":
        raise ValueError(f"Unknown regularization type: {reg_type}")

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = (S > tol).sum().item()
    U = U[:, :r]
    V = Vh[:r].T
    G = U @ V.T

    if grad_type == "random":
        m, n = W.shape
        A = torch.randn(m, n, device=W.device, dtype=W.dtype)
        Pu = torch.eye(m, device=W.device, dtype=W.dtype) - U @ U.T
        Pv = torch.eye(n, device=W.device, dtype=W.dtype) - V @ V.T
        Z = Pu @ A @ Pv
        Z = Z / torch.linalg.norm(Z, ord=2).clamp_min(1e-12)
        G = G + Z
    elif grad_type != "zero":
        raise ValueError(f"Unknown nuclear regularization grad_type: {grad_type}")

    print(G)
    return G
