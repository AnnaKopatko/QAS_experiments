import argparse
import json
import os
import time
import pennylane as qml
from pennylane import numpy as np
from model import CircuitSearchModel, NAS_search_space
from evolution.evolution_sampler import EvolutionSampler


# ======================================================
# Argument Parsing
# ======================================================
def get_args():
    parser = argparse.ArgumentParser("Quantum Architecture Search (QAS)")
    parser.add_argument('--epochs', type=int, default=500, help='training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=100, help='warm-up epochs')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers per subnet')
    parser.add_argument('--n_experts', type=int, default=3, help='number of experts')
    parser.add_argument('--n_search', type=int, default=100, help='number of search iterations')
    parser.add_argument('--ea_pop_size', type=int, default=20, help='population size (evolution)')
    parser.add_argument('--ea_gens', type=int, default=10, help='number of generations (evolution)')
    parser.add_argument('--searcher', type=str, default='evolution', choices=['random', 'evolution'])
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # ---- Added noise/device args ----
    parser.add_argument('--noise', action='store_true', default=True, help='use noise model')
    parser.add_argument('--device', type=str, default='default', choices=['default', 'ibmq-sim', 'ibmq'],
                        help='which backend device to use')

    args = parser.parse_args()

    experiments_root = os.path.join("experiments")
    os.makedirs(experiments_root, exist_ok=True)
    args.save = os.path.join(experiments_root, f'search-{args.save}-{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(args.save, exist_ok=True)
    with open(os.path.join(args.save, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    return args


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
    """Choose the expert with the lowest energy for given subnet."""
    best_idx, best_loss = 0, float('inf')
    for i in range(n_experts):
        model.params = model.get_params(subnet, i)
        loss = cost_fn(model.params)
        if loss < best_loss:
            best_loss, best_idx = loss, i
    return best_idx


# ======================================================
# Main QAS logic
# ======================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    # ---- Optional noise import ----
    if args.noise or args.device in ['ibmq-sim', 'ibmq']:
        import qiskit
        import qiskit_aer.noise as noise     

    # ---- Define H₂ molecule ----
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, -0.6614],
                            [0.0, 0.0,  0.6614]])
    molecule = qml.qchem.Molecule(symbols, coordinates)
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(molecule)
    args.n_qubits = qubits

    print(f"\n=== Molecule: H₂ ===")
    print(f"Qubits: {qubits}")
    print("Hamiltonian:\n", hamiltonian)

    # ---- Device ----
    if args.device in ['ibmq-sim', 'ibmq']:
        from qiskit import IBMQ
        account_key = ''
        assert account_key != '', 'Please fill in your IBMQ account key.'
        IBMQ.save_account(account_key, overwrite=True)
        provider = IBMQ.enable_account(account_key)
        if args.device == 'ibmq':
            dev = qml.device('qiskit.ibmq', wires=qubits, backend='ibmq_ourense', provider=provider)
        else:
            backend = provider.get_backend('ibmq_ourense')
            noise_model = noise.NoiseModel().from_backend(backend)
            dev = qml.device('qiskit.aer', wires=qubits, noise_model=noise_model)
    else:
        if args.noise:
            prob_1 = 0.05  # 1-qubit gate
            prob_2 = 0.2   # 2-qubit gate
            error_1 = noise.depolarizing_error(prob_1, 1)
            error_2 = noise.depolarizing_error(prob_2, 2)
            noise_model = noise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
            print(noise_model)
            dev = qml.device('qiskit.aer', wires=qubits, noise_model=noise_model)
        else:
            dev = qml.device("default.qubit", wires=qubits)

    # ---- Model ----
    model = CircuitSearchModel(dev, args.n_qubits, args.n_layers, args.n_experts)

    @qml.qnode(dev)
    def circuit(params):
        model(params, wires=range(args.n_qubits))
        return qml.expval(hamiltonian)

    def cost(params):
        return circuit(params)

    opt = qml.AdamOptimizer(stepsize=0.2)
    exact_value = -1.136189454088  # FCI reference for H₂

    # ======================================================
    # Warm-up training
    # ======================================================
    print("\n=== Warm-up Training ===")
    for epoch in range(args.epochs):
        subnet = np.random.randint(0, len(NAS_search_space), args.n_layers).tolist()
        expert_idx = np.random.randint(args.n_experts) if epoch < args.warmup_epochs \
                     else expert_evaluator(model, subnet, args.n_experts, cost)
        model.params = model.get_params(subnet, expert_idx)
        model.params = opt.step(cost, model.params)
        model.set_params(model.params)
        if epoch % 50 == 0:
            energy = cost(model.params)
            print(f"Epoch {epoch}/{args.epochs}: Energy = {energy:.8f}")

    # ======================================================
    # Architecture search
    # ======================================================
    print("\n=== Start Architecture Search ===")
    result = {}
    if args.searcher == 'random':
        for i in range(args.n_search):
            subnet = np.random.randint(0, len(NAS_search_space), args.n_layers).tolist()
            expert_idx = expert_evaluator(model, subnet, args.n_experts, cost)
            model.params = model.get_params(subnet, expert_idx)
            energy = cost(model.params)
            result['-'.join(map(str, subnet))] = float(energy)
            print(f"{i+1}/{args.n_search}: subnet={subnet}, energy={energy:.6f}")
    else:
        sampler = EvolutionSampler(
            pop_size=args.ea_pop_size,
            n_gens=args.ea_gens,
            n_layers=args.n_layers,
            n_blocks=len(NAS_search_space)
        )
        def eval_func(subnet):
            expert_idx = expert_evaluator(model, subnet, args.n_experts, cost)
            model.params = model.get_params(subnet, expert_idx)
            energy = cost(model.params)
            return float(energy)
        sampler.sample(eval_func)
        result = sampler.subnet_eval_dict

    sorted_result = sorted(result.items(), key=lambda x: x[1])
    with open(os.path.join(args.save, 'nas_result_sorted.txt'), 'w') as f:
        for k, v in sorted_result:
            f.write(f"{k} {v}\n")

    if not sorted_result:
        print("No architectures found.")
        return

    # ======================================================
    # Evaluate all candidate architectures
    # ======================================================
    search_space_size = len(NAS_search_space)
    evaluated_entries = []
    best_energy_entry = None

    def safe_parse_subnet(raw_subnet):
        if isinstance(raw_subnet, str):
            try:
                return parse_architecture_key(raw_subnet, search_space_size)
            except Exception:
                pass
        elif isinstance(raw_subnet, (list, np.ndarray)):
            raw_subnet = np.asarray(raw_subnet, dtype=float)
            clipped = np.clip(np.rint(raw_subnet), 0, search_space_size - 1)
            return clipped.astype(int).tolist()
        raise ValueError(f"Cannot parse subnet: {raw_subnet}")

    for arch_key, stored_energy in sorted_result:
        try:
            subnet = safe_parse_subnet(arch_key)
        except ValueError as exc:
            print(f"Skipping architecture {arch_key}: {exc}")
            continue
        if len(subnet) != args.n_layers:
            print(f"Skipping architecture {arch_key}: expected {args.n_layers} layers, got {len(subnet)}")
            continue

        expert_idx = expert_evaluator(model, subnet, args.n_experts, cost)
        params = model.get_params(subnet, expert_idx)
        energy = float(cost(params))
        deviation = abs(energy - exact_value)
        subnet_str = '-'.join(map(str, subnet))
        entry = {
            "arch_key": arch_key,
            "subnet": subnet,
            "subnet_str": subnet_str,
            "energy": energy,
            "deviation": float(deviation),
            "expert_idx": int(expert_idx),
        }
        evaluated_entries.append(entry)

        if best_energy_entry is None or energy < best_energy_entry["energy"]:
            best_energy_entry = entry

    if not evaluated_entries:
        print("No architectures could be evaluated; aborting fine-tuning.")
        return

    detailed_path = os.path.join(args.save, 'nas_result_with_energy.txt')
    with open(detailed_path, 'w') as f:
        for item in evaluated_entries:
            f.write(
                "{} energy={:.12f} deviation={:.12f} expert={}\n".format(
                    item["subnet_str"],
                    item["energy"],
                    item["deviation"],
                    item["expert_idx"],
                )
            )

    print("\n=== Best Architecture (Lowest Energy) ===")
    print(f"Original key: {best_energy_entry['arch_key']}")
    print("Subnet:", best_energy_entry["subnet"])
    print("Expert:", best_energy_entry["expert_idx"])
    print(f"Energy: {best_energy_entry['energy']:.8f} Ha")
    print(f"Deviation: {best_energy_entry['deviation']:.8f} Ha")

    best_subnet = best_energy_entry["subnet"]
    best_expert = best_energy_entry["expert_idx"]

    # ---- Fine-tune ----
    fine_tune_epochs = 20
    model.params = model.get_params(best_subnet, best_expert)
    print("\n=== Fine-tuning Best Architecture ===")
    for epoch in range(fine_tune_epochs):
        model.params = opt.step(cost, model.params)
        model.set_params(model.params)
        energy = cost(model.params)
        deviation = abs(energy - exact_value)
        print(f"Epoch {epoch+1}/{fine_tune_epochs}: Energy = {energy:.8f}, ΔE = {deviation:.8f}")

    final_energy = cost(model.params)
    print(f"\nFinal Energy after fine-tuning: {final_energy:.8f} Ha")
    print(f"Deviation from FCI: {abs(final_energy - exact_value):.8f} Ha")

    np.save(os.path.join(args.save, 'best_finetuned_params.npy'), model.params)

    # ---- Visualization ----
    try:
        import matplotlib.pyplot as plt
        plt.plot(range(fine_tune_epochs),
                 [cost(model.params) for _ in range(fine_tune_epochs)], marker='o')
        plt.title("Fine-tuning Energy Convergence (H₂)")
        plt.xlabel("Epoch")
        plt.ylabel("Energy (Ha)")
        plt.grid(True)
        plt.savefig(os.path.join(args.save, 'fine_tune_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved fine-tuning plot to {args.save}/fine_tune_plot.png")
    except Exception as e:
        print(f"Skipping visualization: {e}")

    # ======================================================
    # Draw and save the final circuit diagram (as PNG)
    # ======================================================
    try:
        import matplotlib.pyplot as plt

        fig, ax = qml.draw_mpl(circuit)(model.params)
        fig.suptitle("Final Optimized Circuit (H₂)", fontsize=12)
        circuit_path = os.path.join(args.save, "final_circuit.png")
        fig.savefig(circuit_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Circuit diagram saved to {circuit_path}")
    except Exception as e:
        print(f"Could not draw circuit diagram: {e}")



if __name__ == "__main__":
    main()
