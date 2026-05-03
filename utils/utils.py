from pennylane import numpy as np

# ======================================================
# Utility: parse subnet safely
# ======================================================
def parse_architecture_key(key, search_space_size):
    """Convert a stored architecture key into integer indices."""
    key_str = str(key).strip()
    if not key_str:
        return []
    if key_str.startswith('[') and key_str.endswith(']'):
        arr = np.fromstring(key_str.strip('[]'), sep=' ')
        if arr.size == 0:
            return []
        return np.clip(np.rint(arr), 0, search_space_size - 1).astype(int).tolist()
    if '-' in key_str:
        return [int(x) for x in key_str.split('-') if x]
    tokens = [tok for tok in key_str.replace(',', ' ').split() if tok]
    values = []
    for tok in tokens:
        try:
            val = int(np.clip(np.rint(float(tok)), 0, search_space_size - 1))
            values.append(val)
        except ValueError:
            continue
    return values if values else [int(key_str)]


def expert_evaluator(model, subnet, n_experts, cost_fn):
    """Choose the expert with the lowest energy for given subnet.

    Returns
    -------
    best_idx : int
    best_loss : float
        Energy already evaluated for best_idx — callers can use this
        directly instead of re-running the circuit.
    """
    best_idx, best_loss = 0, float('inf')
    for i in range(n_experts):
        model.params = model.get_params(subnet, i)
        loss = cost_fn(model.params)
        if loss < best_loss:
            best_loss, best_idx = loss, i
    return best_idx, best_loss