from collections import Counter
from typing import List, Tuple

import numpy as np
from braket.circuits import Circuit, Qubit, circuit
from braket.devices import Device
from braket.tasks import QuantumTask


def create_chsh_inequality_circuits(
    qubit0: Qubit = 0,
    qubit1: Qubit = 1,
    a: float = 0,
    a_: float = 2 * np.pi / 8,
    b: float = np.pi / 8,
    b_: float = 3 * np.pi / 8,
) -> List[Circuit]:
    """Create the four circuits for CHSH inequality. Default angles will give maximum violation of
    the inequality.

    Args:
        qubit0 (Qubit): First qubit.
        qubit1 (Qubit): Second qubit.
        a (Float): First basis rotation angle for first qubit
        a_ (Float): Second basis rotation angle for first qubit
        a (Float): First basis rotation angle for second qubit
        a_ (Float): Second basis rotation angle for second qubit

    Returns:
        List[QuantumTask]: List of quantum tasks.
    """
    circ_ab = bell_singlet_rotated_basis(qubit0, qubit1, a, b)
    circ_ab_ = bell_singlet_rotated_basis(qubit0, qubit1, a, b_).h(qubit1)
    circ_a_b = bell_singlet_rotated_basis(qubit0, qubit1, a_, b).h(qubit0)
    circ_a_b_ = bell_singlet_rotated_basis(qubit0, qubit1, a_, b_).h(qubit0).h(qubit1)
    return [circ_ab, circ_ab_, circ_a_b, circ_a_b_]


def run_chsh_inequality(
    circuits: List[Circuit],
    device: Device,
    shots: int = 1_000,
) -> List[QuantumTask]:

    """Submit four CHSH circuits to a device.

    Args:
        circuits (List[Circuit]): Four CHSH inequality circuits to run.
        device (Device): Quantum device or simulator.
        shots (int): Number of shots. Defaults to 1_000.

    Returns:
        List[QuantumTask]: List of quantum tasks.
    """
    tasks = [device.run(circ, shots=shots) for circ in circuits]
    return tasks


def get_chsh_results(
    tasks: List[QuantumTask], verbose: bool = True
) -> Tuple[List[Counter[float]], float, float, float]:
    """Return Bell task results after post-processing.

    Args:
        tasks (List[QuantumTask]): List of quantum tasks.
        verbose (bool): Controls printing of the inequality result. Defaults to True.

    Returns:
        Tuple[List[Counter[float]], float, float, float]: results, pAB, pAC, pBC
    """
    results = [task.result().result_types[0].value for task in tasks]
    prob_same = np.array([d[0] + d[3] for d in results])  # 00 and 11 states
    prob_different = np.array([d[1] + d[2] for d in results])  # 01 and 10 states
    E_ab, E_ab_, E_a_b, E_a_b_ = np.array(prob_same) - np.array(prob_different)
    chsh_value = E_ab - E_ab_ + E_a_b + E_a_b_

    if verbose:
        print(f"E(a,b) = {E_ab},E(a,b') = {E_ab_}, E(a',b) = {E_a_b}, E(a',b') = {E_a_b_}")
        print(f"\nCHSH inequality: {chsh_value} ≤ 2")

        if chsh_value > 2:
            print("CHSH inequality is violated!")
            print(
                "Notice that the quantity may not be exactly as predicted by Quantum theory. "
                "This is may be due to finite shots or the effects of noise on the QPU."
            )
        else:
            print("CHSH inequality is not violated.")
    return chsh_value, results, E_ab, E_ab_, E_a_b, E_a_b_


def bell_singlet_rotated_basis(
    qubit0: Qubit, qubit1: Qubit, rotation0: float, rotation1: float
) -> Circuit:
    """Prepare a Bell singlet state in a Ry-rotated meaurement basis.

    Args:
        qubit0 (Qubit): First qubit.
        qubit1 (Qubit): Second qubit.
        rotation0 (float): First qubit Rx rotation angle.
        rotation1 (float): Second qubit Rx rotation angle.

    Returns:
        Circuit: the Braket circuit that prepares the Bell circuit.
    """
    circ = Circuit().bell_singlet(qubit0, qubit1)
    if rotation0 != 0:
        circ.ry(qubit0, rotation0)
    circ.probability()
    return circ


@circuit.subroutine(register=True)
def bell_singlet(qubit0: Qubit, qubit1: Qubit) -> Circuit:
    """Prepare a Bell singlet state.

    Args:
        qubit0 (Qubit): First qubit.
        qubit1 (Qubit): Second qubit.

    Returns:
        Circuit: the Braket circuit that prepares the Bell single state.
    """
    return Circuit().h(qubit0).cnot(qubit0, qubit1)
