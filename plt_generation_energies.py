from matplotlib import pyplot as plt
import numpy as np

energies = []
directs = []
with open("compare_direct_path_output.txt", "r") as f:
    phase = -1
    for i, line in enumerate(f):
        if i % 4 == 0:
            line = line.strip()[1:-1].split(",")
            energies.append(list(map(float, line)))
        if i % 4 == 1:
            line = line.strip()[1:-1].split(",")
            directs.append(list(map(float, line)))
        if i > 41:
            break

fig, axes = plt.subplots(1, 5, figsize=(8,2.4))

maxlen = max([len(e) for e in energies])
maxlen = 1000
for i in range(5):
    ax = axes[i]
    ax.invert_xaxis()
    if i == 0:
        ax.set_ylabel("Relative Energy (kcal/mol)")
    ax.set_xlabel("Diffusion Step")

    c = "C{}".format(i)
    e = np.array(energies[i])
    ax.plot(list(reversed(range(e.shape[0]))),
            (e - e[-1]) * 23.06,
            color=c,
            label="Diffusion")

    e = np.array(directs[i])
    ax.plot(list(reversed(range(0, e.shape[0]*2, 2))),
            (e - e[-1]) * 23.06,
            linestyle="--",
            color=c,
            label="Direct")
plt.legend()
plt.tight_layout()
fig.subplots_adjust(wspace=0.35, hspace=0)
plt.savefig("writeup/figures/path_comparison.pdf")
#plt.show()
