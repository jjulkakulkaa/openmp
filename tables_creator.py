import pandas as pd
import matplotlib.pyplot as plt
import glob

# Find all files that contain "results" in their name
files = glob.glob("*results*.csv")

for filename in files:
    df = pd.read_csv(filename)


    # The first ParallelTime value is considered the sequential time
    sequential_time = df['ParallelTime'][0]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Scale the table
    table.scale(1, 1.5)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Save the table as an image
    plt.savefig(f'{filename.replace(".csv", "")}_table.png', bbox_inches='tight')
    plt.close()

print("Tables have been saved as images.")
