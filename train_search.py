import argparse
import json
import yaml
import os
import time

import pennylane as qml
from pennylane import numpy as np
from aim import Run, Image as AimImage


from models.circuit_model import CircuitModel
from models.search_space import SearchSpace
from models.circuit_search_model import CircuitSearchModel

# from evolution.evolution_sampler import EvolutionSampler
from custom_evolution.custom_evoltuion import EvolutionSampler
from utils.molecule_utils import load_molecule_and_hf
from utils.utils import parse_architecture_key, expert_evaluator
from utils.vis_and_logging import plot_energy_convergence, save_circuit_diagram, extract_and_save_specs


# ======================================================
# Helper functions for safe Aim logging
# ======================================================
def safe_log_metric(run, key, value, step=None, context=None):
    """Safely log metric with error handling"""
    try:
        run.track(float(value), name=key, step=step, context=context or {})
    except Exception as e:
        print(f"⚠️ Failed to log metric {key}={value}: {e}")


def safe_log_metrics(run, metrics, step=None, context=None):
    """Safely log multiple metrics"""
    try:
        for key, value in metrics.items():
            run.track(float(value), name=key, step=step, context=context or {})
    except Exception as e:
        print(f"⚠️ Failed to log metrics: {e}")


# ======================================================
# Argument Parsing
# ======================================================
def get_args():
    parser = argparse.ArgumentParser("Quantum Architecture Search (QAS)")
    parser.add_argument('--epochs', type=int, default=400, help='training epochs')
    parser.add_argument('--expr_tag', type=str, default="two_rots_per_layer_RyRy", help='the tag to add to logging')
    parser.add_argument('--warmup_epochs', type=int, default=200, help='warm-up epochs')
    parser.add_argument('--n_layers', type=int, default=8, help='number of layers per subnet')
    parser.add_argument('--n_experts', type=int, default=5, help='number of experts')
    parser.add_argument('--n_search', type=int, default=500, help='number of earch iterations')
    parser.add_argument('--ea_pop_size', type=int, default=25, help='population size (evolution)')
    parser.add_argument('--ea_gens', type=int, default=20, help='number of generations (evolution)')
    parser.add_argument('--mutation_prob', type = float, default = 0.25, help = 'mutation probability for the evoltuion algorithm')
    parser.add_argument('--use_controller', default=False, help='use architecture controller')
    parser.add_argument('--use_aging', default=False, action='store_true', help='use aging as in AmeubaNet')
    parser.add_argument('--searcher', type=str, default='evolution', choices=['random', 'evolution'])
    parser.add_argument('--finetune_epochs', type=int, default=150)
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--mol_name', type=str, default='LiH', choices=['H2', 'LiH', 'BeH2'],
                        help='Select molecule to simulate (H2 or LiH)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--log_experiment', action='store_true', default=True,
                        help='enable Aim experiment logging')
    parser.add_argument('--noise', action='store_true', default=True, help='use noise model')
    parser.add_argument('--device', type=str, default='default', choices=['default', 'ibmq-sim', 'ibmq'],
                        help='which backend device to use')
    parser.add_argument('--aim_repo', type=str, default='.aim', help='Aim repository path')
    parser.add_argument('--lr', type=float, default=0.2, help='optimizer learning rate (step size)')
    parser.add_argument('--qng_lam', type=float, default=0.001, help='QNG regularization parameter')
    parser.add_argument('--qng_approx', type=str, default='block-diag', 
                        choices=['block-diag', 'diag'], help='QNG metric tensor approximation')

    args = parser.parse_args()

    # Create experiment directory
    experiments_root = os.path.join("experiments")
    os.makedirs(experiments_root, exist_ok=True)
    args.save = os.path.join(experiments_root, f'search-{args.save}-{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(args.save, exist_ok=True)
    
    with open(os.path.join(args.save, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return args


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

    # ---- Load configuration ----
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    mol_name = args.mol_name
    mol_cfg = cfg["Molecules"][mol_name]
    

    # ======================================================
    # Initialize Aim Run
    # ======================================================
    run = None
    if args.log_experiment:
        noise_id = "noise" if args.noise else "no_noise"
        
        run_name = f"{mol_name}_{args.expr_tag}_{noise_id}_{args.searcher}_{time.strftime('%Y%m%d-%H%M%S')}"
        
        try:
            run = Run(
                repo=args.aim_repo,
                experiment=f"QAS_Experiment"
            )
            
            run['name'] = run_name
            run['searcher'] = args.searcher
            run['noise'] = args.noise
            run['mol_name'] = mol_name
            run['expr_tag'] = args.expr_tag
            run['minus_in_evolution'] = False
            run['hparams'] = {
                "epochs": args.epochs,
                "warmup_epochs": args.warmup_epochs,
                "n_layers": args.n_layers,
                "n_experts": args.n_experts,
                "n_search": args.n_search,
                "ea_pop_size": args.ea_pop_size,
                "ea_gens": args.ea_gens,
                "mutation_prob": args.mutation_prob,
                "use_controller": args.use_controller,
                "searcher": args.searcher,         
                "device": args.device,
                "seed": args.seed,
                "optimizer": "Adam",
                "lr": args.lr,
                "qng_lam": args.qng_lam,
                "qng_approx": args.qng_approx,
                "finetune_epochs": args.finetune_epochs,
                "aging": args.use_aging,
                
            }
            
            print(f"✅ Aim logging initialized: {run_name}")
        except Exception as e:
            print(f"⚠️ Failed to initialize Aim: {e}")
            run = None

    # ---- Load molecule and Hartree-Fock state ----
    hamiltonian, qubits, hf_state = load_molecule_and_hf(mol_cfg)
    noise_cfg = cfg["Noise"]
    
    search_space = SearchSpace(cfg, qubits, run)
    
    # ---- Device setup ----
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
            prob_1 = noise_cfg["prob_1q"]
            prob_2 = noise_cfg["prob_2q"]
            error_1 = noise.depolarizing_error(prob_1, 1)
            error_2 = noise.depolarizing_error(prob_2, 2)
            noise_model = noise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
            print(noise_model)
            dev = qml.device('qiskit.aer', wires=qubits, noise_model=noise_model)
        else:
            dev = qml.device("default.qubit", wires=qubits)

    args.n_qubits = qubits
    basis_state = hf_state
    print("HF basis state:", basis_state)
    
    # ---- Initialize Model ----
    model = CircuitSearchModel(dev, search_space, args.n_qubits, args.n_layers, args.n_experts, basis_state=basis_state)

    @qml.qnode(dev)
    def circuit(params):
        model(params, wires=range(args.n_qubits))
        return qml.expval(hamiltonian)

    def cost(params):
        return circuit(params)

    # Initialize QNG Optimizer
    # Initialize standard Adam optimizer
    opt = qml.AdamOptimizer(stepsize=args.lr)
    exact_value = mol_cfg["exact_energy"]


    # ======================================================
    # Warm-up training
    # ======================================================
    print("\n=== Warm-up Training ===")
    for epoch in range(args.epochs):
        subnet = np.random.randint(0, len(search_space), args.n_layers).tolist()

        if epoch < args.warmup_epochs:
            expert_idx = np.random.randint(args.n_experts)
        else:
            expert_idx = expert_evaluator(model, subnet, args.n_experts, cost)
        
        model.params = model.get_params(subnet, expert_idx)

        # ---- Adam update (standard) ----
        model.params = opt.step(cost, model.params)

        model.set_params(model.params)

        # ---- Logging ----
        energy = cost(model.params)
        deviation = abs(energy - exact_value)

        if run:
            safe_log_metric(run, "warmup_energy", energy, step=epoch, context={"stage": "warmup"})
            if epoch % 50 == 0:
                safe_log_metric(run, "warmup_deviation", deviation, step=epoch, context={"stage": "warmup"})
        
        print(f"Epoch {epoch}/{args.epochs}: Energy = {energy:.8f}, ΔE = {deviation:.8f}")


    # ======================================================
    # Architecture search
    # ======================================================
    print("\n=== Start Architecture Search ===")
    result = {}
    search_iter = 0

    if args.searcher == 'random':
        for i in range(args.n_search):
            subnet = np.random.randint(0, len(search_space), args.n_layers).tolist()
            expert_idx = expert_evaluator(model, subnet, args.n_experts, cost)
            model.params = model.get_params(subnet, expert_idx)
            energy = cost(model.params)
            subnet_key = '-'.join(map(str, subnet))
            result[subnet_key] = (float(energy), int(expert_idx))
            print(f"{i+1}/{args.n_search}: subnet={subnet}, expert={expert_idx}, energy={energy:.6f}")

            if run:
                safe_log_metrics(run, {
                    "search_energy": float(energy),
                    "search_deviation": float(abs(energy - exact_value)),
                    "search_expert_idx": expert_idx
                }, step=search_iter, context={"stage": "search", "searcher": "random"})
                
                try:
                    run.track(
                        f"Iter {i+1}: subnet={subnet}, expert={expert_idx}, energy={energy:.6f}",
                        name="search_progress",
                        step=search_iter,
                        context={"stage": "search"}
                    )
                except Exception as e:
                    print(f"⚠️ Failed to log text: {e}")
                
            search_iter += 1

    else:
        # ======================================================
        # Evolution Search
        # ======================================================
        sampler = EvolutionSampler(
            pop_size=args.ea_pop_size,
            n_gens=args.ea_gens,
            n_layers=args.n_layers,
            n_blocks=len(search_space),
            aim_run=run,
            search_space=search_space,
            mutation_prob=args.mutation_prob,
            use_controller=args.use_controller,
            aging=args.use_aging
        )

        # Fitness function for evolution
        def test_subnet_evolution(subnet):
            expert_idx = expert_evaluator(model, subnet, args.n_experts, cost)
            model.params = model.get_params(subnet, expert_idx)
            energy = cost(model.params)
            if run:
                safe_log_metrics(run, {
                        "search_evolution_energy": float(energy),
                    }, step=search_iter, context={"stage": "search", "searcher": "evolution"})
                    

            # Evolution uses a score (higher is better)
            score = np.abs(energy - exact_value)
            return score

        # Run evolutionary search
        sampler.sample(test_subnet_evolution)
        
        # Raw results: { '1-3-2': score, ... }
        raw_result = sampler.subnet_eval_dict

        # ======================================================
        # FIX: Convert raw evolution results into canonical format:
        #       subnet_key -> (energy, expert_idx)
        # ======================================================
        result = {}
        for subnet_key, score in raw_result.items():
            # Parse subnet from key (string like "1-3-2")
            subnet = parse_architecture_key(subnet_key, len(search_space))

            # Recompute expert and true energy
            expert_idx = expert_evaluator(model, subnet, args.n_experts, cost)
            model.params = model.get_params(subnet, expert_idx)
            energy = float(cost(model.params))

            # Store in correct tuple structure
            result[subnet_key] = (energy, int(expert_idx))

        print("✓ Evolution results converted to (energy, expert) tuples.")

    # ======================================================
    # Save and process results
    # ======================================================
    sorted_result = sorted(result.items(), key=lambda x: x[1][0])
    
    with open(os.path.join(args.save, 'nas_result_sorted.txt'), 'w') as f:
        for k, (energy, expert_idx) in sorted_result:
            f.write(f"{k} energy={energy:.12f} expert={expert_idx}\n")

    if run:
        try:
            with open(os.path.join(args.save, 'nas_result_sorted.txt'), 'r') as f:
                run.track(f.read(), name="nas_result_sorted", context={"type": "artifact"})
        except Exception as e:
            print(f"⚠️ Failed to log artifact: {e}")

    if not sorted_result:
        print("No architectures found.")
        if run:
            run.close()
        return

    # ======================================================
    # Process candidate architectures
    # ======================================================
    search_space_size = len(search_space)
    evaluated_entries = []

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

    for arch_key, (stored_energy, expert_idx) in sorted_result:
        try:
            subnet = safe_parse_subnet(arch_key)
        except ValueError as exc:
            print(f"Skipping architecture {arch_key}: {exc}")
            continue
        
        if len(subnet) != args.n_layers:
            print(f"Skipping architecture {arch_key}: expected {args.n_layers} layers, got {len(subnet)}")
            continue

        deviation = abs(stored_energy - exact_value)
        subnet_str = '-'.join(map(str, subnet))
        
        entry = {
            "arch_key": arch_key,
            "subnet": subnet,
            "subnet_str": subnet_str,
            "energy": stored_energy,
            "deviation": float(deviation),
            "expert_idx": expert_idx,
        }
        evaluated_entries.append(entry)

    if not evaluated_entries:
        print("No architectures could be evaluated; aborting fine-tuning.")
        if run:
            run.close()
        return

    # Save detailed results
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

    # Select best architecture (lowest energy)
    best_energy_entry = min(evaluated_entries, key=lambda x: x["energy"])

    print("\n=== Best Architecture (Lowest Energy) ===")
    print(f"Original key: {best_energy_entry['arch_key']}")
    print("Subnet:", best_energy_entry["subnet"])
    print("Expert:", best_energy_entry["expert_idx"])
    print(f"Energy from search: {best_energy_entry['energy']:.8f} Ha")
    print(f"Deviation from search: {best_energy_entry['deviation']:.8f} Ha")

    if run:
        safe_log_metrics(run, {
            "energy_from_search": best_energy_entry['energy'],
            "deviation_from_search": best_energy_entry['deviation']
        }, context={"stage": "best_arch"})
        

    # ======================================================
    # Fine-tune best architecture from scratch
    # ======================================================
    print("\n=== Training Best Architecture from Scratch ===")
    print("Creating new model instance for the best architecture...")

    best_subnet = best_energy_entry["subnet"]
    best_arch_str = '-'.join(map(str, best_subnet))

    fresh_model = CircuitModel(
        dev=dev,
        search_space=search_space,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        arch=best_arch_str,
        basis_state=basis_state
    )

    @qml.qnode(dev)
    def fixed_circuit(params):
        fresh_model(params, wires=range(args.n_qubits))
        return qml.expval(hamiltonian)

    def fixed_cost(params):
        return fixed_circuit(params)

    # Verify initial energy
    initial_random_energy = fixed_cost(fresh_model.params)
    print(f"Initial random energy: {initial_random_energy:.8f} Ha")
    print(f"Target (from search): {best_energy_entry['energy']:.8f} Ha")
    print(f"Exact FCI energy: {exact_value:.8f} Ha")

    if run:
        safe_log_metrics(run, {
            "initial_random_energy": initial_random_energy,
            "initial_random_deviation": abs(initial_random_energy - exact_value)
        }, context={"stage": "finetune_init"})


    # Train from scratch with Adam optimizer
    fine_tune_epochs = args.finetune_epochs
    fine_tune_opt = qml.AdamOptimizer(stepsize=args.lr)

    print(f"\nTraining for {fine_tune_epochs} epochs with QNGOptimizer...")
    fine_tune_energies = [initial_random_energy]

    best_finetune_energy = initial_random_energy
    best_finetune_params = fresh_model.params.copy()
    patience_counter = 0
    patience_limit = 50

    for epoch in range(fine_tune_epochs):
        # ---- Adam update (standard) ----
        fresh_model.params = fine_tune_opt.step(fixed_cost, fresh_model.params)

        energy = fixed_cost(fresh_model.params)
        fine_tune_energies.append(energy)
        deviation = abs(energy - exact_value)
        
        # Track best energy
        if energy < best_finetune_energy:
            best_finetune_energy = energy
            best_finetune_params = fresh_model.params.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Logging
        if run:
            safe_log_metrics(run, {
                "finetune_energy": energy,
                "finetune_deviation": deviation,
                "finetune_best_energy": best_finetune_energy
            }, step=epoch, context={"stage": "finetune"})
        
        if epoch % 20 == 0 or epoch < 5:
            print(
                f"Epoch {epoch+1}/{fine_tune_epochs}: "
                f"Energy = {energy:.8f}, "
                f"ΔE = {deviation:.8f}, "
                f"Best = {best_finetune_energy:.8f}"
            )



    # Use best parameters
    fresh_model.params = best_finetune_params
    final_energy = fixed_cost(fresh_model.params)

    # ======================================================
    # Results summary
    # ======================================================
    print(f"\n=== Fine-tuning Results ===")
    print(f"Initial random energy: {initial_random_energy:.8f} Ha")
    print(f"Best from search: {best_energy_entry['energy']:.8f} Ha")
    print(f"Best during training: {best_finetune_energy:.8f} Ha")
    print(f"Final energy: {final_energy:.8f} Ha")
    print(f"Improvement over random: {initial_random_energy - final_energy:.8f} Ha")
    print(f"Comparison to search: {final_energy - best_energy_entry['energy']:.8f} Ha")
    print(f"Final deviation from FCI: {abs(final_energy - exact_value):.8f} Ha")

    if final_energy <= best_energy_entry['energy']:
        print("✓ Training from scratch matched or improved search result!")
    else:
        print("✓ Training from scratch completed (slightly higher than search)")

    if run:
        safe_log_metrics(run, {
            "final_energy": final_energy,
            "best_finetune_energy": best_finetune_energy,
            "final_deviation": abs(final_energy - exact_value),
            "improvement_over_random": initial_random_energy - final_energy,
            "comparison_to_search": final_energy - best_energy_entry['energy'],
            "training_epochs_used": epoch + 1
        }, context={"stage": "final"})

    # Save trained parameters
    np.save(os.path.join(args.save, 'best_finetuned_params.npy'), best_finetune_params)
    print(f"\n✓ Saved best fine-tuned parameters to: {args.save}/best_finetuned_params.npy")

    model.params = best_finetune_params

    # ======================================================
    # Visualization & Aim logging
    # ======================================================
    try:
        energy_plot_path = plot_energy_convergence(fine_tune_energies, exact_value, args)
        circuit_path = save_circuit_diagram(
            circuit,
            model.params,
            os.path.join(args.save, "final_circuit.png"),
            title=f"Final Optimized Circuit ({args.mol_name})"
        )
        num_wires, num_gates, depth, specs_path = extract_and_save_specs(circuit, model.params, args.save)

        if run:
            try:
                if os.path.exists(energy_plot_path):
                    img = AimImage(energy_plot_path)
                    run.track(img, name="energy_convergence", step=0, context={"type": "visualization"})
                
                if os.path.exists(circuit_path):
                    img = AimImage(circuit_path)
                    run.track(img, name="circuit_diagram", step=0, context={"type": "visualization"})
                
                safe_log_metrics(run, {
                    "num_wires": num_wires,
                    "num_gates": num_gates,
                    "circuit_depth": depth
                }, context={"stage": "final", "type": "specs"})
                
                if os.path.exists(specs_path):
                    with open(specs_path, 'r') as f:
                        run.track(f.read(), name="circuit_specs", context={"type": "artifact"})
                        
            except Exception as e:
                print(f"⚠️ Failed to log visualizations to Aim: {e}")

    except Exception as e:
        print(f"⚠️ Visualization skipped: {e}")
    
    # Close Aim run
    if run:
        try:
            run.close()
            print("✅ Aim run closed successfully")
        except Exception as e:
            print(f"⚠️ Error closing Aim run: {e}")


if __name__ == "__main__":
    main()