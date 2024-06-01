import pandas as pd
import matplotlib.pyplot as plt

# Read CSV data into DataFrame
df = pd.read_csv('results.csv')

# Create the plot
plt.figure(figsize=(10, 6))

# Group the data by thread configuration
for (threadsPerBlockX, threadsPerBlockY), group in df.groupby(['ThreadsPerBlockX', 'ThreadsPerBlockY']):
    plt.plot(group['ImageSize'], group['AverageTime(ms)'], marker='o', label=f'{threadsPerBlockX}x{threadsPerBlockY}')

# Add titles and labels
plt.title('Performance by Image Size and Thread Configuration')
plt.xlabel('Image Size')
plt.ylabel('Average Time (ms)')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.legend(title='Threads Per Block')
plt.grid(True, which="both", ls="--")

# Save and show the plot
plt.savefig('performance_plot_log_scale.png')
plt.show()
