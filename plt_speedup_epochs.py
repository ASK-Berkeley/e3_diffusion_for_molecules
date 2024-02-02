from matplotlib import pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial

from plt_denoising_results import load_results_pkl

def main(dataset_name):
    means = []
    medians = []
    sigmas = []
    q25s = []
    q75s = []

    means_smalle = []
    medians_smalle = []
    sigmas_smalle = []
    q25s_smalle = []
    q75s_smalle = []


    if dataset_name == "qm9":
        fn_base = "denoising_results/denoising_results_qm9{}_sT50.pkl" 
        epochs = [0, 250, 500, 750, 950, 1450, 2000, 2500, 3250, 5150]

    elif dataset_name == "geom":
        fn_base = "denoising_results/denoising_results_geom{}_sT50.pkl"
        epochs = list(range(0, 251, 10))

    mmff_results = load_results_pkl(fn_base.format("0"), dataset_name)

    for epoch in epochs:

        results = load_results_pkl(fn_base.format(epoch), "qm9")


        mmff_nsteps = mmff_results["mmff_nsteps"]
        diffused_nsteps = results["diffused_nsteps"]
        delta_es = results["diffused_relaxed_es"] - mmff_results["mmff_relaxed_es"]

        s = 100 * (1 - diffused_nsteps / mmff_nsteps)
        means.append(s.mean())
        sigmas.append(np.std(s))
        medians.append(np.percentile(s, 50))
        q25s.append(np.percentile(s, 25))
        q75s.append(np.percentile(s, 75))

        s = s[np.abs(delta_es) < 0.2]
        means_smalle.append(s.mean())
        sigmas_smalle.append(np.std(s))
        medians_smalle.append(np.percentile(s, 50))
        q25s_smalle.append(np.percentile(s, 25))
        q75s_smalle.append(np.percentile(s, 75))


    means = np.array(means)
    sigmas = np.array(sigmas)
    means_smalle = np.array(means_smalle)
    sigmas_smalle = np.array(sigmas_smalle)
    

    def plot(epochs, middle, bottom, top, ylabel):
        plt.figure(figsize=(4,3.5))
        plt.plot(epochs, middle, label="Median")
        plt.fill_between(epochs, bottom, top, alpha=0.2, label="25-75%ile")

        line = Polynomial(np.polyfit(epochs, middle, 1)[::-1])
        plt.plot(epochs, line(epochs), color="k", linestyle="--", alpha=0.6, label="Linear fit")

        plt.xlabel("Training Epochs")
        plt.ylabel(ylabel)
        #plt.ylim(-37, 37)
        plt.legend(loc="upper left")
        plt.grid(axis="y")
        plt.tight_layout()

    epochs = np.array(epochs, dtype=np.float32)
    #plot(epochs, means, means - sigmas, means + sigmas, "Mean Speedup (%)")
    #plot(epochs, means_smalle, means_smalle - sigmas_smalle, means_smalle + sigmas_smalle, "Mean Speedup (%, smalle)")
    plot(epochs, medians, q25s, q75s, "Median Speedup (%)")
    plt.savefig("writeup/figures/speedup_geom_500.pdf")
    plot(epochs, medians_smalle, q25s_smalle, q75s_smalle, "Median Speedup (%, smalle)")
    plt.savefig("writeup/figures/speedup_geom_500_lowE.pdf")

    #plt.show()
    #plt.savefig("writeup/figures/speedup_vs_epochs_lowE.pdf")

if __name__ == "__main__":
    main("geom")
