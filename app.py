# app.py
# Streamlit app to demonstrate distributed (cut) execution of small quantum circuits
# using qiskit-addon-cutting with a simple, configurable UI.

import streamlit as st
import numpy as np

# Qiskit core & Aer primitives
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_aer import AerSimulator
from qiskit.transpiler import generate_preset_pass_manager

# Cutting utilities
from qiskit_addon_cutting import (
    partition_problem,
    generate_cutting_experiments,
    reconstruct_expectation_values,
)

# -----------------------------
# Helpers
# -----------------------------

def make_ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def make_qft(n: int) -> QuantumCircuit:
    # basic QFT (no swaps at the end to keep it small)
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            # controlled phase rotations
            qc.cp(np.pi / (2 ** (k - j)), k, j)
    return qc


def make_grover(n: int, marked: str = None, iterations: int = 1) -> QuantumCircuit:
    # tiny Grover on n qubits, mark a computational basis bitstring
    # default: all-zeros except the last bit = 1 (if not provided)
    if marked is None:
        marked = ("0" * (n - 1)) + "1"
    assert len(marked) == n and set(marked) <= {"0", "1"}

    qc = QuantumCircuit(n)
    # equal superposition
    for i in range(n):
        qc.h(i)

    def oracle(circ: QuantumCircuit):
        # phase-flip the |marked> state using multi-controlled Z via X basis trick
        for i, b in enumerate(marked):
            if b == "0":
                circ.x(i)
        circ.h(n - 1)
        circ.mcx(list(range(n - 1)), n - 1)
        circ.h(n - 1)
        for i, b in enumerate(marked):
            if b == "0":
                circ.x(i)

    def diffuser(circ: QuantumCircuit):
        for i in range(n):
            circ.h(i)
            circ.x(i)
        circ.h(n - 1)
        circ.mcx(list(range(n - 1)), n - 1)
        circ.h(n - 1)
        for i in range(n):
            circ.x(i)
            circ.h(i)

    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    return qc


def auto_observable(label: str, n: int) -> SparsePauliOp:
    # default observable: all-Zs of length n (e.g., ZZZZ)
    if label.strip() == "":
        label = "Z" * n
    # allow comma-separated list like "ZZZZ, IIZZ"
    terms = [t.strip().replace(" ", "") for t in label.split(",") if t.strip()]
    return SparsePauliOp(terms)


def make_partition_labels(n: int, cut_index: int) -> str:
    # A...A | B...B where cut_index is number of A's
    cut_index = max(1, min(n - 1, cut_index))
    return ("A" * cut_index) + ("B" * (n - cut_index))


def count_two_qubit_gates(circ: QuantumCircuit) -> int:
    return sum(1 for inst, *_ in circ.data if inst.num_qubits == 2)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Distributed Quantum Cutting Demo", layout="wide")
st.title("Distributed Quantum Circuit Cutting")

with st.sidebar:
    st.header("Configuration")
    algo = st.selectbox("Algorithm", ["GHZ", "Grover", "QFT"], index=0)
    n_qubits = st.slider("Number of qubits", min_value=3, max_value=8, value=4, step=1)

    if algo == "Grover":
        marked = st.text_input(
            "Marked bitstring (length n)", value=("0" * (n_qubits - 1) + "1")
        )
        iters = st.slider("Grover iterations", 1, 3, 1)
    else:
        marked, iters = None, 1

    cut_idx = st.slider(
        "Partition cut index (A|B)", min_value=1, max_value=n_qubits - 1, value=n_qubits // 2
    )
    part_labels = make_partition_labels(n_qubits, cut_idx)
    st.caption(f"Partition labels: **{part_labels}**")

    obs_text = st.text_input("Observable(s) (comma-separated Pauli strings)", value="" )
    shots = st.number_input("Shots per sub-experiment", min_value=1000, max_value=2_000_000, value=200_000, step=1000)
    opt_level = st.select_slider("Transpile optimization level", options=[0,1,2,3], value=1)
    noiseless = st.checkbox("Use noiseless Aer simulator", value=True)

    run_btn = st.button("Run benchmark", type="primary")

# Build circuit
if algo == "GHZ":
    qc = make_ghz(n_qubits)
elif algo == "QFT":
    qc = make_qft(n_qubits)
else:
    qc = make_grover(n_qubits, marked=marked, iterations=iters)

# Default observable
observable = auto_observable(obs_text, n_qubits)

# Baseline estimation (uncut)
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Circuit")
    st.text(qc.draw(output="text"))
    from qiskit.qasm3 import dumps
    st.code(dumps(qc))
with col2:
    st.subheader("Baseline (uncut)")
    try:
        est = EstimatorV2()
        exact = est.run([(qc, observable, [])]).result()[0].data.evs
    except Exception as e:
        exact = None
        st.error(f"Estimator error: {e}")

    depth = qc.depth()
    size = qc.size()
    twoq = count_two_qubit_gates(qc)

    st.write({"Exact <O>": float(exact) if exact is not None else None,
              "Depth": depth, "Size": size, "Two-qubit gates": twoq})

st.divider()

if run_btn:
    st.subheader("Distributed (cut)")

    try:
        # Partition/cut
        prob = partition_problem(
            circuit=qc,
            partition_labels=part_labels,
            observables=observable.paulis,
        )
        subcircuits = prob.subcircuits
        subobservables = prob.subobservables
        bases = prob.bases
        sampling_overhead = float(np.prod([b.overhead for b in bases]))

        # Generate sub-experiments
        subexperiments, coefficients = generate_cutting_experiments(
            circuits=subcircuits,
            observables=subobservables,
            num_samples=np.inf,
        )

        # Choose backend + transpile
        backend = AerSimulator() if noiseless else AerSimulator(method="automatic")
        pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
        isa_subexperiments = {lbl: pm.run(exps) for lbl, exps in subexperiments.items()}

        # Run with local SamplerV2 (Aer)
        sampler = SamplerV2()
        results = {}
        for lbl, exps in isa_subexperiments.items():
            # run list of circuits
            job = sampler.run(exps, shots=int(shots))
            results[lbl] = job.result()

        # Reconstruct full expectation value
        reconstructed_terms = reconstruct_expectation_values(
            results, coefficients, subobservables
        )
        # dot with original coefficients from SparsePauliOp
        recon = float(np.real(np.dot(reconstructed_terms, observable.coeffs)))

        # Show results
        st.success("Cutting run completed.")
        st.write({
            "Reconstructed <O>": recon,
            "Sampling overhead (theoretical)": sampling_overhead,
            "#Cuts (bases)": len(bases),
        })

        # Subcircuit stats
        rows = []
        for lbl, sub in subcircuits.items():
            rows.append({
                "Fragment": lbl,
                "Depth": sub.depth(),
                "Size": sub.size(),
                "Two-qubit": count_two_qubit_gates(sub),
                "#Sub-experiments": len(subexperiments[lbl]),
            })
        st.table(rows)

    except Exception as e:
        st.exception(e)

st.info(
    "Tip: If reconstructed values are far from the baseline, increase shots, reduce the number of cuts (move the A|B split), or switch to GHZ to sanity-check."
)

# st.caption(
#     "Packages you need: streamlit, qiskit[all]~=2.1, qiskit-aer~=0.17, qiskit-addon-cutting~=0.10.\n"
#     "Run with: `streamlit run app.py`."
# )
