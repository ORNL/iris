import numpy as np
from typing import Any, Dict, List


def _normalize_cut_int_states(cut: Any, bits_per_cut: int) -> Dict[str, int]:
    """
    Convert one QIR-EE cut with integer-encoded states into {bitstring: count}.

    cut is something like:
        [(0, N_entries),
         (state_index0, count0),
         (state_index1, count1),
         ...]
    or a NumPy array with the same structure.

    bits_per_cut is the number of qubits in this cut (bit-width).
    """
    # Handle None or empty
    if cut is None or len(cut) == 0:
        return {}

    # Treat both lists-of-rows and numpy arrays the same way
    first_row = cut[0]
    state0, n_entries = first_row[0], first_row[1]

    # Detect header (0, N_entries)
    is_header = (
        isinstance(state0, (int, np.integer))
        and int(state0) == 0
        and isinstance(n_entries, (int, np.integer))
    )
    start = 1 if is_header else 0

    out: Dict[str, int] = {}
    for row in cut[start:]:
        state_index_raw, count_raw = row[0], row[1]
        state_index = int(state_index_raw)

        # Decode integer index into a fixed-width binary string
        # Example: bits_per_cut = 4, state_index = 3 → "0011"
        bitstr = format(state_index, f"0{bits_per_cut}b")

        count = int(count_raw)
        out[bitstr] = out.get(bitstr, 0) + count

    return out


#def post_processing(params: List[Any]) -> Dict[str, int]:
def post_processing(spec, qpu0, qpu1, qpu2, qpu3, result):
    print("QPU1:", qpu0)
    print("QPU2:", qpu1)
    print("QPU1:", qpu2)
    print("QPU2:", qpu3)
    params = [qpu0, qpu1, qpu2, qpu3]
    """
    Reconstruct an approximate global counts distribution from
    M QIR-EE cuts with integer-encoded states.

    params: list of cuts.
        Each cut is a list or NumPy array like:
            [(0, N_entries),
             (state_index0, count0),
             (state_index1, count1),
             ...]

        state_index is an integer whose binary representation
        (with 'bits_per_cut' bits) encodes the qubit states for that cut.

    For your current setup:
        - 16 total qubits
        - 4 cuts
        - 4 qubits per cut → bits_per_cut = 4
    """
    if not params:
        return {}

    # 🔧 Set this according to how many qubits each cut covers
    bits_per_cut = 4  # you: 16 qubits / 4 cuts = 4 per cut

    # 1. Normalize each cut → {bitstring: count}
    cuts_dicts: List[Dict[str, int]] = [
        _normalize_cut_int_states(cut, bits_per_cut) for cut in params
    ]

    # 2. Convert each cut to probabilities
    cut_probs: List[Dict[str, float]] = []
    shot_counts: List[int] = []

    for d in cuts_dicts:
        total = sum(d.values())
        if total == 0:
            raise ValueError("One of the cuts has zero total shots; cannot normalize.")
        shot_counts.append(total)
        cut_probs.append({bits: cnt / total for bits, cnt in d.items()})

    # 3. Effective total shots (for reconstructed distribution)
    total_shots = min(shot_counts)

    # 4. Tensor-product over all cuts
    joint: Dict[str, float] = {"": 1.0}  # prefix → probability

    for probs in cut_probs:
        new_joint: Dict[str, float] = {}
        for prefix, p_pref in joint.items():
            for bits, p_bits in probs.items():
                full_bits = prefix + bits
                new_joint[full_bits] = new_joint.get(full_bits, 0.0) + p_pref * p_bits
        joint = new_joint

    # 5. Convert probabilities to counts
    full_counts: Dict[str, int] = {
        bits: int(round(p * total_shots))
        for bits, p in joint.items()
        if p > 0.0
    }
    print("Final: ")
    for k,v in full_counts.items():
        if v != 0:
            print(f"K:{k} => {v}")
    return full_counts

