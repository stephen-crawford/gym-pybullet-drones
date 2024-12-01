import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import re

# Define paths
datapath60m = '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/free_flight_data_hover_downwash/downwash_crossing_20240908_1613_sep_0.6_m.mat'
datapath57m = '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/free_flight_data_hover_downwash/downwash_crossing_20240908_1657_sep_0.57_m.mat'
datapath55m = '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/free_flight_data_hover_downwash/downwash_crossing_20240908_1633_sep_0.55_m.mat'
datapath53m = '/home/stephencrawford/MATLAB/Projects/DownwashAnalysis/free_flight_data_hover_downwash/downwash_crossing_20240908_1642_sep_0.53_m.mat'

# Load data
data = scipy.io.loadmat(datapath55m)

# Extract separation distance from the file name
sepDist_match = re.search(r'sep_(\d+\.\d+)_m', datapath55m)
if sepDist_match:
    sepDist = float(sepDist_match.group(1))  # Convert to numeric value
else:
    raise ValueError("Separation distance not found in the file path")

# Labels for the states
labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'thrust']

# Drone 1 (Lower) rows
drone1_rows = [0, 1, 3, 5, 7, 9, 11, 13]  # Adjusted for Python (0-indexed)
drone1_data = np.array(data['states_array'])[drone1_rows, :].T
time1 = drone1_data[:, 0]
print("Drone 1 data shape: " + str(drone1_data.shape))

# Drone 2 (Above) rows
drone2_rows = [0, 2, 4, 6, 8, 10, 12, 14]  # Adjusted for Python (0-indexed)
drone2_data = np.array(data['states_array'])[drone2_rows, :].T
time2 = drone2_data[:, 0]
print("Drone 2 data shape: " + str(drone2_data.shape))

# Plot the data for Drone 1 and Drone 2 (if needed)
# This step is commented out but can be added if necessary for visualization
plt.figure()
for i in range(1, drone1_data.shape[1]  - 1):
    print("Trying to plot drone 1 data for i: " + str(i))
    plt.plot(time1, drone1_data[:, i], '-o', label=f'Drone 1 - {labels[i]}')
for i in range(1, drone2_data.shape[1] - 1):
    plt.plot(time2, drone2_data[:, i], '--x', label=f'Drone 2 - {labels[i]}')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.legend()
plt.title(f'Drone 1 and Drone 2 Free Flight Data vs. Time at {sepDist} m separation')
plt.show()

# Time threshold and target x-position for Drone 1
time_threshold = 13
target_x_position = 1.908

# Find the first index where Drone 1's x-position is close to 1.908 meters
x_position_condition = np.abs(drone1_data[:, 1] - target_x_position) < 0.01
start_idx = np.where(x_position_condition)[0][0]  # First index where Drone 1 hits 1.908 meters

if start_idx is None:
    raise ValueError("Drone 1 never crosses the target x-position of 1.908 meters")

# Slice the data for Drone 1 starting from the identified index
time1_limited = time1[start_idx:]
drone1_data_limited = drone1_data[start_idx:, :]

# Limit Drone 1's data to the first 13 seconds after crossing the threshold
idx1 = time1_limited < time_threshold
time1_limited = time1_limited[idx1]
drone1_data_limited = drone1_data_limited[idx1, :]

# For Drone 2, align it to the same time range
time2_limited = time2[(time2 >= time1_limited[0]) & (time2 < time_threshold)]
drone2_data_limited = drone2_data[(time2 >= time1_limited[0]) & (time2 < time_threshold), :]

# Calculate velocities and accelerations for Drone 1 and Drone 2
def calculate_velocity_acceleration(data, time):
    velocity = np.diff(data, axis=0) / np.diff(time)[:, None]  # Compute velocity
    acceleration = np.diff(velocity, axis=0) / np.diff(time[1:])[:, None]  # Compute acceleration
    return velocity, acceleration

# Drone 1 velocities and accelerations
velocity1_x, acceleration1_x = calculate_velocity_acceleration(drone1_data_limited[:, 1], time1_limited)
velocity1_y, acceleration1_y = calculate_velocity_acceleration(drone1_data_limited[:, 2], time1_limited)
velocity1_z, acceleration1_z = calculate_velocity_acceleration(drone1_data_limited[:, 3], time1_limited)

# Drone 2 velocities and accelerations
velocity2_x, acceleration2_x = calculate_velocity_acceleration(drone2_data_limited[:, 1], time2_limited)
velocity2_y, acceleration2_y = calculate_velocity_acceleration(drone2_data_limited[:, 2], time2_limited)
velocity2_z, acceleration2_z = calculate_velocity_acceleration(drone2_data_limited[:, 3], time2_limited)

# Calculate average and standard deviation of accelerations for Drone 1
avg_accel1_x, std_accel1_x = np.mean(acceleration1_x), np.std(acceleration1_x)
avg_accel1_y, std_accel1_y = np.mean(acceleration1_y), np.std(acceleration1_y)
avg_accel1_z, std_accel1_z = np.mean(acceleration1_z), np.std(acceleration1_z)

# Calculate average and standard deviation of accelerations for Drone 2
avg_accel2_x, std_accel2_x = np.mean(acceleration2_x), np.std(acceleration2_x)
avg_accel2_y, std_accel2_y = np.mean(acceleration2_y), np.std(acceleration2_y)
avg_accel2_z, std_accel2_z = np.mean(acceleration2_z), np.std(acceleration2_z)

# Calculate velocity errors between Drone 1 and Drone 2
vel_error_x = velocity1_x - velocity2_x
vel_error_y = velocity1_y - velocity2_y
vel_error_z = velocity1_z - velocity2_z

# Optionally, print results
print(f"Average acceleration Drone 1 (x): {avg_accel1_x}, Standard deviation: {std_accel1_x}")
print(f"Average acceleration Drone 1 (y): {avg_accel1_y}, Standard deviation: {std_accel1_y}")
print(f"Average acceleration Drone 1 (z): {avg_accel1_z}, Standard deviation: {std_accel1_z}")
print(f"Average acceleration Drone 2 (x): {avg_accel2_x}, Standard deviation: {std_accel2_x}")
print(f"Average acceleration Drone 2 (y): {avg_accel2_y}, Standard deviation: {std_accel2_y}")
print(f"Average acceleration Drone 2 (z): {avg_accel2_z}, Standard deviation: {std_accel2_z}")