import matplotlib.pyplot as plt
import numpy as np

x = np.arange(5,16)
y = [1.00,1.00,1.00,1.00,0.10,1.00,0.10,0.10,1.0,1.0,0.10]

plt.title("LSTM accuracy")
plt.xlabel("Sequence length")
plt.ylabel("Accuracy")
plt.plot(x,y)
plt.grid()

plt.savefig("LSTM_accuracies.png")
