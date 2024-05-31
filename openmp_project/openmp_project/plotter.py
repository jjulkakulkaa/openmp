import pandas as pd
import matplotlib.pyplot as plt

files = []
for file in files:

    df = pd.read_csv('results.csv')

    # Wykres czasu wykonania
    plt.figure(figsize=(10, 5))
    plt.plot(df['Threads'], df['ParallelTime'], marker='o', label='Parallel Time')
    plt.axhline(y=df['ParallelTime'][0], color='r', linestyle='--', label='Sequential Time')
    plt.xlabel('Number of Threads')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time vs Number of Threads')
    plt.legend()
    plt.grid(True)
    plt.savefig('execution_time.png')
    plt.show()

    # Wykres przyspieszenia według prawa Amdahla
    plt.figure(figsize=(10, 5))
    plt.plot(df['Threads'], df['SpeedupAmdahl'], marker='o', label='Amdahl Speedup')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Number of Threads (Amdahl\'s Law)')
    plt.legend()
    plt.grid(True)
    plt.savefig('speedup_amdahl.png')
    plt.show()

    # Wykres przyspieszenia według prawa Gustafsona
    plt.figure(figsize=(10, 5))
    plt.plot(df['Threads'], df['SpeedupGustafson'], marker='o', label='Gustafson Speedup')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Number of Threads (Gustafson\'s Law)')
    plt.legend()
    plt.grid(True)
    plt.savefig('speedup_gustafson.png')
    plt.show()
