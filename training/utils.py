from torch.optim.lr_scheduler import LambdaLR
from monai.utils import set_determinism

def set_global_seed(seed=17):

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    L.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed=seed)

def make_warmup_poly(optimizer, warmup_steps: int, total_steps: int, power: float = 1.2):
    """
    Linear warmup from 0 -> 1 over `warmup_steps`, then polynomial decay (1-t)^power to 0 by `total_steps`.
    Returns a LambdaLR scheduler. Use with interval='step'.
    """
    def lr_lambda(step: int):
        # linear warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # poly decay
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        t = min(max(t, 0.0), 1.0)
        return (1.0 - t) ** power
    return LambdaLR(optimizer, lr_lambda)
