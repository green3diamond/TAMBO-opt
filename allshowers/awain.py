import showerdata

cond_file = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers_test.h5"
showwwrrr= showerdata.observables.read_observables_from_file(cond_file)
print(showwwrrr)