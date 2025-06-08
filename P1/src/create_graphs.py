import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Read the CSV file into a DataFrame.
# The CSV is assumed to have no header row; we supply our own column names.
df = pd.read_csv("../evaluation/output_modified.csv", header=None,
                 names=["model_type", "encoder", "scaling", "accuracy", "std", "time"],
                 keep_default_na=False)

# Ensure that scaling values are sorted (for consistent ordering)
df["scaling"] = df["scaling"].astype(str)
scaling_order = sorted(df["scaling"].unique())  # e.g., ['None', 'StandardScaler']

# For each encoder, create a grouped bar chart.
unique_encoders = df["encoder"].unique()
for enc in unique_encoders:
    data_enc = df[df["encoder"] == enc]
    unique_models = data_enc["model_type"].unique()
    n_models = len(unique_models)
    n_scaling = len(scaling_order)
    
    # Create an x position for each model group.
    x = np.arange(n_models)
    # Determine bar width (spread them across the group)
    width = 0.8 / n_scaling
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # For each scaling option, plot a set of bars.
    for i, scale in enumerate(scaling_order):
        # For each model, get the accuracy and std for this scaling value.
        values = []
        errors = []
        for model in unique_models:
            row = data_enc[(data_enc["model_type"] == model) & (data_enc["scaling"] == scale)]
            if not row.empty:
                values.append(row["accuracy"].values[0])
                errors.append(row["std"].values[0])
            else:
                # If no data exists for this combination, append zero.
                values.append(0)
                errors.append(0)
        # Compute the position for these bars.
        # Shift each scaling group's bars within the model group.
        bar_positions = x - 0.4 + (i + 0.5) * width
        ax.bar(bar_positions, values, width=width, yerr=errors, capsize=5,
               label=f"scaling={scale}")
    
    ax.set_xticks(x)
    ax.set_xticklabels(unique_models, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy by Model Type for encoder: {enc}")
    ax.legend(title="Scaling")
    plt.tight_layout()
    plt.savefig(f"../results_modified/accuracy_{enc}.png")
    plt.show()

# Define the desired order for scaling and encoder.
scaling_order = ["None", "StandardScaler"]
encoder_order = ["OneHotEncoder", "OrdinalEncoder", "TargetEncoder"]

# There are 2 x 3 = 6 combinations.
num_combinations = len(scaling_order) * len(encoder_order)

# Build a list of combination labels in the desired order.
combination_labels = []
for s in scaling_order:
    for e in encoder_order:
        combination_labels.append(f"{s}+{e}")

# Get the unique model types (sorted for consistency).
unique_models = np.sort(df["model_type"].unique())
n_models = len(unique_models)

# Create a mapping from model type to x-axis positions.
x_positions = {model: i for i, model in enumerate(unique_models)}

# Set up the figure.
fig, ax = plt.subplots(figsize=(12, 6))

# Define group and bar widths.
group_width = 0.8     # total width allotted to one model group
bar_width = group_width / num_combinations

# Define a custom color palette for the 6 combinations.
# Feel free to change these hex codes to any colours you prefer.
custom_colors = ["#6e0233", "#8c0b7f", "#7517a6", "#4728bc", "#4362c8", "#6ca9c6"]

# Loop over each model and each combination, plotting a bar with error bars.
for i, model in enumerate(unique_models):
    # Filter rows for this model.
    model_df = df[df["model_type"] == model]
    for j, comb in enumerate(combination_labels):
        # Split the combination into scaling and encoder parts.
        s_val, e_val = comb.split("+")
        # Select the row corresponding to this combination.
        row = model_df[(model_df["scaling"] == s_val) & (model_df["encoder"] == e_val)]
        if not row.empty:
            acc = row["accuracy"].values[0]
            err = row["std"].values[0]
        else:
            acc = 0
            err = 0
        # Compute the x-coordinate for this bar.
        # The bars for each model are centered around x = x_positions[model].
        x = i - group_width/2 + j * bar_width + bar_width/2
        # Use the custom color for this combination.
        color = custom_colors[j]
        # Plot the bar.
        ax.bar(x, acc, width=bar_width, color=color, yerr=err, capsize=5)

# Set x-axis tick positions and labels.
ax.set_xticks(list(x_positions.values()))
ax.set_xticklabels(list(x_positions.keys()), rotation=45, ha="right")
ax.set_xlabel("Model Type")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy by Model Type, Scaling, and Encoder")

# Create a custom legend.
legend_patches = []
for j, comb in enumerate(combination_labels):
    color = custom_colors[j]
    patch = mpatches.Patch(color=color, label=comb)
    legend_patches.append(patch)
ax.legend(handles=legend_patches, title="Scaling+Encoder", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig("../results_modified/accuracy_barplot.png")
plt.show()

# Define the desired order for scaling and encoder.
scaling_order = ["None", "StandardScaler"]
encoder_order = ["OneHotEncoder", "OrdinalEncoder", "TargetEncoder"]

# There are 2 x 3 = 6 combinations.
num_combinations = len(scaling_order) * len(encoder_order)

# Build a list of combination labels in the desired order.
combination_labels = []
for s in scaling_order:
    for e in encoder_order:
        combination_labels.append(f"{s}+{e}")

# Get the unique model types (sorted for consistency).
unique_models = np.sort(df["model_type"].unique())
n_models = len(unique_models)

# Create a mapping from model type to x-axis positions.
x_positions = {model: i for i, model in enumerate(unique_models)}

# Set up the figure.
fig, ax = plt.subplots(figsize=(12, 6))

# Define group and bar widths.
group_width = 0.8     # total width allotted to one model group
bar_width = group_width / num_combinations

# Define a custom color palette for the 6 combinations.
# Feel free to change these hex codes to any colours you prefer.
custom_colors = ["#6e0233", "#8c0b7f", "#7517a6", "#4728bc", "#4362c8", "#6ca9c6"]

# Loop over each model and each combination, plotting a bar with error bars.
for i, model in enumerate(unique_models):
    # Filter rows for this model.
    model_df = df[df["model_type"] == model]
    for j, comb in enumerate(combination_labels):
        # Split the combination into scaling and encoder parts.
        s_val, e_val = comb.split("+")
        # Select the row corresponding to this combination.
        row = model_df[(model_df["scaling"] == s_val) & (model_df["encoder"] == e_val)]
        if not row.empty:
            acc = row["time"].values[0]
        else:
            acc = 0
        # Compute the x-coordinate for this bar.
        # The bars for each model are centered around x = x_positions[model].
        x = i - group_width/2 + j * bar_width + bar_width/2
        # Use the custom color for this combination.
        color = custom_colors[j]
        # Plot the bar.
        ax.bar(x, acc, width=bar_width, color=color, capsize=5)

# Set x-axis tick positions and labels.
ax.set_xticks(list(x_positions.values()))
ax.set_xticklabels(list(x_positions.keys()), rotation=45, ha="right")
ax.set_xlabel("Model Type")
ax.set_ylabel("Time")
ax.set_title("Time by Model Type, Scaling, and Encoder")

# Create a custom legend.
legend_patches = []
for j, comb in enumerate(combination_labels):
    color = custom_colors[j]
    patch = mpatches.Patch(color=color, label=comb)
    legend_patches.append(patch)
ax.legend(handles=legend_patches, title="Scaling+Encoder", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig("../results_modified/time_barplot.png")
plt.show()

# Create a separate bar plot for time taken.
plt.figure(figsize=(12, 6))
sns.barplot(x="model_type", y="time", hue="encoder", data=df, capsize=0.1, errwidth=1, ci=None)
plt.title("Time Taken by Model Type and Encoder")
plt.ylabel("Time (seconds)")
plt.xlabel("Model Type")
plt.legend(title="Encoder")
plt.tight_layout()
plt.savefig("../results_modified/time_plot.png")
plt.show()
