from typing import List, Dict, Any
import numpy as np
import pdb

def _bits_to_str(bits: Any) -> str:
    """Convert various QIR-EE bit encodings to a '0101' string."""
    # already a proper string
    if isinstance(bits, str):
        return bits

    # numpy scalar or plain int
    if isinstance(bits, (int, np.integer)):
        # Fallback: just use the decimal representation
        # If you actually want binary, replace with: format(bits, 'b')
        return str(int(bits))

    # numpy array or other iterable of 0/1
    if isinstance(bits, (np.ndarray, list, tuple)):
        return "".join(str(int(b)) for b in bits)

    # generic fallback
    return str(bits)


def _normalize_cut_array(arr: np.ndarray) -> Dict[str, int]:
    """
    Convert a QIR-EE cut given as a NumPy array into
    a dict {bitstring: count}, dropping the header row.

    Expected format:
        arr[0] = [0, N]          # header
        arr[1:] = [<bits>, <count>]
    """
    if arr.size == 0:
        return {}

    # Assume header is first row if its first element is 0 and second is int-like
    start_idx = 0
    first0 = arr[0][0]
    first1 = arr[0][1]
    if isinstance(first0, (int, np.integer)) and int(first0) == 0 and isinstance(first1, (int, np.integer)):
        start_idx = 1

    d: Dict[str, int] = {}
    for row in arr[start_idx:]:
        bits_raw, count_raw = row[0], row[1]
        bitstr = _bits_to_str(bits_raw)
        count = int(count_raw)
        d[bitstr] = d.get(bitstr, 0) + count

    return d


def post_processing(spec, qpu0, qpu1, qpu2, qpu3, result):
#def post_processing(params: List[np.ndarray]) -> Dict[str, int]:
    print("QPU1:", qpu0)
    print("QPU2:", qpu1)
    print("QPU1:", qpu2)
    print("QPU2:", qpu3)
    params = [qpu0, qpu1, qpu2, qpu3]
    """
    Reconstruct an approximate global counts distribution from
    M QIR-EE cuts, each given as a NumPy array.

    params: list of NumPy arrays, one per cut.
        Each array arr has rows:
            [0, N]                        # header (QIR-EE metadata)
            [<bits0>, count0]
            [<bits1>, count1]
            ...
        <bits> may be a string '0101', a numpy array like [0,1,0,1], etc.

    The full bitstring is formed by concatenating the cut bitstrings
    in the same order as arrays appear in `params`.
    """

    if not params:
        return {}

    # 1. For each cut, get {bitstring: count}
    cuts_dicts = [
        _normalize_cut_array(arr) for arr in params
    ]

    # 2. Convert each cut to a probability distribution
    cut_probs: List[Dict[str, float]] = []
    shot_counts: List[int] = []

    for d in cuts_dicts:
        shots = sum(d.values())
        if shots == 0:
            raise ValueError("One of the cuts has zero total shots; cannot normalize.")
        shot_counts.append(shots)
        cut_probs.append({bits: count / shots for bits, count in d.items()})

    # 3. Effective total shot count: minimum across cuts
    total_shots = min(shot_counts)

    # 4. Tensor-product of all cuts' distributions
    joint_probs: Dict[str, float] = {"": 1.0}

    for probs in cut_probs:
        new_joint: Dict[str, float] = {}
        for prefix, p_pref in joint_probs.items():
            for bits, p_bits in probs.items():
                full_bits = prefix + bits
                new_joint[full_bits] = new_joint.get(full_bits, 0.0) + p_pref * p_bits
        joint_probs = new_joint

    # 5. Convert probabilities to integer counts
    full_counts: Dict[str, int] = {
        bits: int(round(p * total_shots))
        for bits, p in joint_probs.items()
        if p > 0.0
    }
    print("Final results:", full_counts)
    return full_counts

