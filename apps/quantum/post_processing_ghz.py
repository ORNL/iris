import numpy as np
from typing import Any, List, Dict


def _convert_qpu_result_to_dict(qpu_data: Any, bits_per_cut: int) -> Dict[str, int]:
    """
    Convert QPU result from array format to simple dictionary format.
    
    Input format:
        [(0, N_entries),
         (state_index0, count0),
         (state_index1, count1),
         ...]
    
    Output format:
        {"00": 49, "11": 51}
    """
    if qpu_data is None or len(qpu_data) == 0:
        return {}
    
    result = {}
    
    # Skip header row if present (first row with state=0)
    start_idx = 1 if (len(qpu_data) > 0 and int(qpu_data[0][0]) == 0) else 0
    
    for row in qpu_data[start_idx:]:
        state_index = int(row[0])
        count = int(row[1])
        
        # Convert state index to binary string
        bitstring = format(state_index, f"0{bits_per_cut}b")
        result[bitstring] = result.get(bitstring, 0) + count
    
    return result


def post_processing(spec, *circuit_results, result=None):
    """
    Collect measurement data from all QIR circuits across all fragments and convert to dictionary format.
    
    Each QPU result is converted from array format to dictionary format like:
        {"00": 49, "11": 51}
    
    Args:
        spec: IRIS spec (required parameter)
        *circuit_results: Variable number of circuit results, organized as 3 circuits per fragment
        result: Output buffer for results (passed as keyword argument)
    
    Returns:
        Dictionary with formatted results for all fragments
    """
    print("=" * 70)
    print("QPU RESULTS COLLECTION")
    print("=" * 70)
    
    bits_per_cut = 2  # Adjust based on your circuit
    
    # Organize results by fragment (3 circuits per fragment)
    num_circuits_per_fragment = 3
    num_fragments = len(circuit_results) // num_circuits_per_fragment
    
    all_results = {}
    
    for frag_idx in range(num_fragments):
        print(f"\nFragment {frag_idx}:")
        print("-" * 70)
        
        fragment_formatted = {}
        for circuit_idx in range(num_circuits_per_fragment):
            result_idx = frag_idx * num_circuits_per_fragment + circuit_idx
            qpu_data = circuit_results[result_idx]
            qpu_dict = _convert_qpu_result_to_dict(qpu_data, bits_per_cut)
            fragment_formatted[f"circuit_{circuit_idx}"] = qpu_dict
            print(f"  Circuit {circuit_idx}: {qpu_dict}")
        
        all_results[f"fragment_{frag_idx}"] = fragment_formatted
    
    print("\n" + "=" * 70)
    
    # Return the formatted results
    return all_results

