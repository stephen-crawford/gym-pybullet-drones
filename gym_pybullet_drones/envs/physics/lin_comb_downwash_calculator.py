import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from gym_pybullet_drones.envs.physics.constants import *


def calculate_downwash_force(drone1_x, drone1_y, drone1_z, drone2_x, drone2_y, drone2_z):

    interpolated_force_data = pd.read_csv('/home/stephencrawford/gym-pybullet-drones/gym_pybullet_drones/envs/physics/interpolated_force_data.csv')

    delta_xy = np.sqrt((drone1_x - drone2_x)**2 + (drone1_y - drone2_y)**2)
    delta_z = np.sqrt((drone1_z - drone2_z)**2)
    delta_xy = delta_xy / crayflie_arm_length_meters
    delta_z = delta_z / crayflie_arm_length_meters

    X = X = interpolated_force_data.iloc[:, :2]
    y = interpolated_force_data['force']  # Target column

    model = LinearRegression()
    model.fit(X, y)
    input_data = pd.DataFrame([[delta_xy, delta_z]], columns=interpolated_force_data.columns[:2])
    predicted_value = model.predict(input_data)[0]
    print(predicted_value)
    return predicted_value


def main():
    calculate_downwash_force(1, 2, 3, 1, 2, 4)

if __name__ == '__main__':
    main()
