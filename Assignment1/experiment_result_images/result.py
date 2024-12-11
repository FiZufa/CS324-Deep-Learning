import matplotlib.pyplot as plt

scenario1_acc = [99.0, 99.0, 99.0, 98.0, 97.5, 98.5, 98.5, 97.5, 99.0, 98.5, 87.0]
scenario2_acc = [84.4, 83.5, 85.5, 85.0, 85.0, 84.5, 85.5, 84.5, 85.5, 85.0, 77.0]
scenario3_acc = [97.5, 98.0, 97.5, 98.0, 98.5, 98.0, 98.5, 98.5, 97.5, 97.5, 85.0]

batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]

# Plot the accuracies over batch size for each scenario
plt.plot(batch_size, scenario1_acc, label='Scenario 1', color='blue', marker='o')
plt.plot(batch_size, scenario2_acc, label='Scenario 2', color='green', marker='s')
plt.plot(batch_size, scenario3_acc, label='Scenario 3', color='red', marker='^')

plt.xlabel('Batch Size')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Over Batch Size for mini-batch mode')
plt.legend()
plt.xscale('log')  # Using a logarithmic scale for the x-axis
plt.grid(True)
plt.savefig('result_mlp.png')