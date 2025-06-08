
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file with HPO results.
# Replace the path with the actual location of your CSV file.
df = pd.read_csv("../evaluation/hpo_cv_results5.csv")

# Filter only the rows where the model type is GradientBoostingClassifier
gb_df = df[df["params_model_type"] == "GradientBoostingClassifier"]

# Create subplots: one for accuracy and one each for the three hyperparameters.
fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

# Plot Accuracy vs. Trial Number
axs[0].plot(gb_df["number"], gb_df["value"], marker="o", linestyle="-", color="blue")
axs[0].set_ylabel("Accuracy")
axs[0].set_title("Gradient Boosting Classifier: Accuracy & Hyperparameters over Trials")
axs[0].grid(True)

# Plot Number of Estimators vs. Trial Number
axs[1].plot(gb_df["number"], gb_df["params_gb_n_estimators"], marker="o", linestyle="-", color="red")
axs[1].set_ylabel("n_estimators")
axs[1].grid(True)

# Plot Learning Rate vs. Trial Number
axs[2].plot(gb_df["number"], gb_df["params_gb_learning_rate"], marker="o", linestyle="-", color="green")
axs[2].set_ylabel("learning_rate")
axs[2].grid(True)

# Plot Max Depth vs. Trial Number
axs[3].plot(gb_df["number"], gb_df["params_gb_max_depth"], marker="o", linestyle="-", color="purple")
axs[3].set_ylabel("max_depth")
axs[3].set_xlabel("Trial Number")
axs[3].grid(True)

plt.tight_layout()
plt.show()
plt.savefig("../results_hpo/accuracy_gradientboost2.png")

