import showerdata
import numpy as np

ML_FILE      = "/n/home04/hhanif/AllShowers/results/20260301_171240_CNF-Transformer/samples01.h5"
CORSIKA_FILE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers_test.h5"
N_SHOWERS    = 500

print("Loading ML showers …")
showers_ml = showerdata.load(ML_FILE, stop=N_SHOWERS)

print("Loading CORSIKA showers …")
showers_corsika = showerdata.load(CORSIKA_FILE, stop=N_SHOWERS)

resolutions = []

for i in range(N_SHOWERS):
    energy_ml      = showers_ml[i].points.reshape(-1, 4)[:, 3].sum()
    energy_corsika = showers_corsika[i].points.reshape(-1, 4)[:, 3].sum()

    res = (energy_ml - energy_corsika) / energy_corsika
    resolutions.append(res)
    print(f"Shower {i:03d}: E_gen={energy_ml:.4f}, E_true={energy_corsika:.4f}, res={res*100:.2f}%")

mean_res = np.mean(resolutions)
print(f"\n<(E_gen - E_true) / E_true> = {mean_res*100:.2f}%")