import torch
import matplotlib.pyplot as plt

# Example tensors
t_u = torch.linspace(30, 100, 10)  # Example temperature inputs
t_c = 0.5 * t_u - 5  # Example actual Celsius values
t_p = 0.52 * t_u - 4.8  # Example model predictions

# Create the plot
fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy(), label="Predictions")
plt.plot(t_u.numpy(), t_c.numpy(), 'o', label="Actual Data")
plt.legend()
plt.show()