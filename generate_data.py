import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 2000

# Normal operating conditions
temperature = np.random.normal(70, 5, n_samples)
vibration = np.random.normal(0.5, 0.1, n_samples)
pressure = np.random.normal(100, 10, n_samples)
rpm = np.random.normal(1500, 50, n_samples)
humidity = np.random.normal(60, 5, n_samples)

# Create failure conditions for 10% of data
failure_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
temperature[failure_idx] += np.random.uniform(20, 40, len(failure_idx))
vibration[failure_idx] += np.random.uniform(0.5, 1.5, len(failure_idx))
pressure[failure_idx] += np.random.uniform(30, 60, len(failure_idx))
rpm[failure_idx] += np.random.uniform(200, 500, len(failure_idx))

# Create warning conditions for 15% of data
warning_idx = np.random.choice(
    [i for i in range(n_samples) if i not in failure_idx],
    size=int(n_samples * 0.15),
    replace=False
)
temperature[warning_idx] += np.random.uniform(10, 20, len(warning_idx))
vibration[warning_idx] += np.random.uniform(0.2, 0.5, len(warning_idx))
pressure[warning_idx] += np.random.uniform(15, 30, len(warning_idx))

# Create labels
labels = np.zeros(n_samples, dtype=int)
labels[failure_idx] = 2      # 2 = failure
labels[warning_idx] = 1      # 1 = warning
                              # 0 = healthy

df = pd.DataFrame({
    "temperature": temperature.round(2),
    "vibration": vibration.round(3),
    "pressure": pressure.round(2),
    "rpm": rpm.round(1),
    "humidity": humidity.round(2),
    "status": labels
})

df.to_csv("sensor_data.csv", index=False)
print(f"Dataset created: {len(df)} samples")
print(df["status"].value_counts().rename({0: "healthy", 1: "warning", 2: "failure"}))