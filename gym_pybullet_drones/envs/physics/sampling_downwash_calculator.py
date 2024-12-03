import pandas as pd
import numpy as np

from gym_pybullet_drones.envs.physics.constants import *


def calculate_downwash_force(drone1_x, drone1_y, drone1_z, drone2_x, drone2_y, drone2_z):

    interpolated_force_data = pd.read_csv('/home/stephencrawford/gym-pybullet-drones/gym_pybullet_drones/envs/physics/interpolated_force_data.csv')

    delta_xy = np.sqrt((drone1_x - drone2_x)**2 + (drone1_y - drone2_y)**2)
    delta_z = np.sqrt((drone1_z - drone2_z)**2)
    delta_xy = delta_xy / crayflie_arm_length_meters
    delta_z = delta_z / crayflie_arm_length_meters

    closest_idx = np.argmin(
        (interpolated_force_data['delta_x/l'] - delta_xy) ** 2 + (interpolated_force_data['delta_z/l'] - delta_z) ** 2
    )

    interpolated_force_data['row_separation'] = interpolated_force_data.index - closest_idx

    interpolated_force_data['probabilities'] = np.exp(-0.5 * (interpolated_force_data['row_separation'] / decay_scale) ** 2)

    interpolated_force_data['probabilities'] /= interpolated_force_data['probabilities'].sum()

    sampled_index = np.random.choice(interpolated_force_data.index, size=1, p=interpolated_force_data['probabilities'])
    sampled_force = interpolated_force_data.loc[sampled_index]['force']

    # start_idx = max(0, closest_idx - 10)
    # end_idx = min(len(interpolated_force_data), closest_idx + 10)
    # print(interpolated_force_data.iloc[start_idx:end_idx])

    return sampled_force


def main():
    calculate_downwash_force(1, 2, 3, 1, 2, 4)

if __name__ == '__main__':
    main()
