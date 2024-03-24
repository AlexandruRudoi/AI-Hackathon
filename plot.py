import matplotlib.pyplot as plt

# Sample data
deep_fake_value = 0.7  # Example deep fake value
sentimental_value = 0.4  # Example sentimental value

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(deep_fake_value, sentimental_value, color='blue', marker='o')
plt.title('Sentimental Value vs. Deep Fake Value')
plt.xlabel('Deep Fake Value')
plt.ylabel('Sentimental Value')
plt.grid(True)
plt.show()
