import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_wires = 4
graph = [(0, 1), (0, 3), (1, 2), (2, 3)]


# unitary operator U_B with parameter beta
def U_B(beta):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)


# unitary operator U_C with parameter gamma
def U_C(gamma):
    for edge in graph:
        qml.CNOT(wires=edge)
        qml.RZ(gamma, wires=edge[1])
        qml.CNOT(wires=edge)
        # Could also do
        # IsingZZ(gamma, wires=edge)

def bitstring_to_int(bit_string_sample):
    return int(2 ** np.arange(len(bit_string_sample)) @ bit_string_sample[::-1])

dev = qml.device("lightning.qubit", wires=n_wires)

@qml.set_shots(20)
@qml.qnode(dev)
def circuit(gammas, betas, return_samples=False):
    # apply Hadamards to get the n qubit |+> state
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for gamma, beta in zip(gammas, betas):
        U_C(gamma)
        U_B(beta)

    if return_samples:
        # sample bitstrings to obtain cuts
        return qml.sample()
    # during the optimization phase we are evaluating the objective using expval
    C = qml.sum(*(qml.Z(w1) @ qml.Z(w2) for w1, w2 in graph))
    return qml.expval(C)


def objective(params):
    """Minimize the negative of the objective function C by postprocessing the QNnode output."""
    return -0.5 * (len(graph) - circuit(*params))

def qaoa_maxcut(n_layers=1):
    print(f"\np={n_layers:d}")

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    params = init_params.copy()
    steps = 30
    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print(f"Objective after step {i+1:3d}: {-objective(params): .7f}")

    # sample 100 bitstrings by setting return_samples=True and the QNode shot count to 100
    bitstrings = qml.set_shots(circuit, shots=100)(*params, return_samples=True)
    # convert the samples bitstrings to integers
    sampled_ints = [bitstring_to_int(string) for string in bitstrings]

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(sampled_ints))
    most_freq_bit_string = np.argmax(counts)
    print(f"Optimized parameter vectors:\ngamma: {params[0]}\nbeta:  {params[1]}")
    print(f"Most frequently sampled bit string is: {most_freq_bit_string:04b}")

    return -objective(params), sampled_ints


def main():
    # perform QAOA on our graph with p=1,2 and keep the lists of sampled integers
    int_samples1 = qaoa_maxcut(n_layers=1)[1]
    int_samples2 = qaoa_maxcut(n_layers=2)[1]

    # plot the results
    xticks = range(0, 16)
    xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
    bins = np.arange(0, 17) - 0.5

    fig, _ = plt.subplots(1, 2, figsize=(8, 4))
    for i, samples in enumerate([int_samples1, int_samples2], start=1):
        plt.subplot(1, 2, i)
        plt.title(f"n_layers={i}")
        plt.xlabel("bitstrings")
        plt.ylabel("freq.")
        plt.xticks(xticks, xtick_labels, rotation="vertical")
        plt.hist(samples, bins=bins)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()