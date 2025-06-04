import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to plot and save the cosine similarity heatmap
def plot_and_save_heatmap(scores, output_filename="similarity_heatmap.png"):


    # Create a heatmap of the similarity matrix
    plt.figure(figsize=(10, 8))  # Set the size of the plot

    # Use Seaborn's heatmap to plot the similarity matrix
    sns.heatmap(scores, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=True, yticklabels=True)

    # Customize the plot with titles and labels
    plt.title("Cosine Similarity Heatmap")
    plt.xlabel("Sentence Index")
    plt.ylabel("Sentence Index")

    # Save the plot to a file
    plt.savefig(output_filename, format='png', dpi=300)  # Save with high resolution (300 DPI)

    # Display the plot
    plt.show()

# Example Usage:
# Assuming mean_pooled is already defined and contains your sentence embeddings
# Call the function to plot and save the heatmap
# plot_and_save_heatmap(mean_pooled, output_filename="similarity_heatmap.png")
