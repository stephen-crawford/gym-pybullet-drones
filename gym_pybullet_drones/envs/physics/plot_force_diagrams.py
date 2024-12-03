import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def read_data(file_name):
    try:
        print("Trying to read " + "../../assets/" + file_name)
        return np.loadtxt("../../assets/" + file_name, skiprows=4)
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return None

def combined_zones():

    # Function to read data
    # Define the arm length and induced velocity
    l = 32.5  # arm length in mm
    vi = np.sqrt((0.03 * 9.81) / (2 * 1.2 * np.pi * (22.5 / 1000) ** 2))

    # Font settings
    font_size = 35
    label_interpreter = 'latex'

    # --------------- Plotting for Zone D direct overlap ---------------
    data = read_data('ZoneD_directOverlap0001.dat')

    if data is None:
        raise ValueError("Data could not be read. Check the file format and delimiter.")

    # Assuming the data has eight columns
    x_ZoneD_dirOlap = data[:, 0]
    y_ZoneD_dirOlap = data[:, 1]
    V_ZoneD_dirOlap = data[:, 4]

    rows = 273
    cols = 744

    # Ensure the total number of elements matches rows * cols
    if len(x_ZoneD_dirOlap) != rows * cols:
        raise ValueError('Data size mismatch. Cannot reshape the data into a matrix.')

    # Reshape data
    x_piv_ZoneD_dirOlap = (x_ZoneD_dirOlap - 6.4442).reshape(rows, cols)
    y_piv_ZoneD_dirOlap = (y_ZoneD_dirOlap - 36.2846).reshape(rows, cols)
    V_piv_ZoneD_dirOlap = V_ZoneD_dirOlap.reshape(rows, cols)

    # Plotting the contourf
    plt.figure()
    plt.contourf(x_piv_ZoneD_dirOlap / l, y_piv_ZoneD_dirOlap / l, V_piv_ZoneD_dirOlap / vi, 600)

    # Adding patches
    x_patch = np.array([-5, 5, 5, -5])
    y_patch = np.array([2.3, 2.3, 3.5, 3.5])
    plt.fill(x_patch, y_patch, color=[0.25, 0.25, 0.25], edgecolor='none')

    x1_patch = np.array([-5, 5, 5, -5])
    y1_patch = np.array([-0.55, -0.55, 0.54, 0.54])
    plt.fill(x1_patch, y1_patch, color=[0.25, 0.25, 0.25], edgecolor='none')

    # Formatting the plot
    plt.axis('equal')
    plt.xticks()
    plt.yticks(np.arange(-12.5, 5.1, 2.5))
    plt.xlim([-4.4, 4.4])
    plt.ylim([-12.5, 5])
    plt.clim([0, 1])
    plt.Colormap('turbo')
    plt.colorbar(label=r'$\overline{V}/V_{ind}$')

    # Setting labels
    plt.xlabel(r'$\Delta \:x/l$', labelpad=20)
    plt.ylabel(r'$\Delta \:z/l$', labelpad=20)

    plt.show()

    # --------------- Repeat for other data files ---------------

    # Reading and processing for max F std. deviation plot
    data = read_data('ZoneD_directOverlap0002.dat')

    if data is None:
        raise ValueError("Data could not be read. Check the file format and delimiter.")

    xstd_ZoneD_dirOlap = data[:, 0]
    ystd_ZoneD_dirOlap = data[:, 1]
    Vstd_ZoneD_dirOlap = data[:, 4]

    # Reshape data
    xstd_piv_ZoneD_dirOlap = (xstd_ZoneD_dirOlap - 10.4742).reshape(rows, cols)
    ystd_piv_ZoneD_dirOlap = (ystd_ZoneD_dirOlap - 36.2846).reshape(rows, cols)
    Vstd_piv_ZoneD_dirOlap = Vstd_ZoneD_dirOlap.reshape(rows, cols)

    # Plotting std deviation
    plt.figure()
    plt.contourf(xstd_piv_ZoneD_dirOlap / l, ystd_piv_ZoneD_dirOlap / l, Vstd_piv_ZoneD_dirOlap / vi, 600)

    # Adding patches
    x_patch = np.array([-5, 5, 5, -5])
    y_patch = np.array([2.3, 2.3, 3.5, 3.5])
    plt.fill(x_patch, y_patch, color=[0.1, 0.1, 0.1], edgecolor='none')

    x1_patch = np.array([-5, 5, 5, -5])
    y1_patch = np.array([-0.52, -0.52, 0.54, 0.54])
    plt.fill(x1_patch, y1_patch, color=[0.1, 0.1, 0.1], edgecolor='none')

    # Formatting the plot
    plt.axis('equal')
    plt.xticks()
    plt.yticks(np.arange(-12.5, 5.1, 2.5))
    plt.xlim([-4.4, 4.4])
    plt.ylim([-12.5, 5])
    plt.clim([0, 0.35])
    plt.Colormap('bone')
    plt.colorbar(label=r'$\overline{V_{std}}/V_{ind}$')

    # Show the plot
    plt.show()

def velocity_analysis():
    # Constants and parameters
    l = 32.5  # scaling factor
    vi = np.sqrt((0.03 * 9.81) / (2 * 1.2 * np.pi * (22.5 / 1000) ** 2))  # velocity scale
    dz_l = np.array([-1, -2.5, -5, -7.5, -10])  # y-axis slices for velocity

    # Define the custom color map for blue and green shades
    shades_of_blue = [
        [0, 0.2, 0.6],  # Dark blue
        [0, 0.4, 0.8],  # Medium-dark blue
        [0, 0.6, 1.0],  # Medium blue
        [0.4, 0.8, 1.0],  # Light-medium blue
        [0.8, 0.9, 1.0]  # Light blue
    ]

    shades_of_green = [
        [0, 0.3, 0],  # Dark green
        [0, 0.5, 0],  # Medium-dark green
        [0, 0.7, 0],  # Medium green
        [0.4, 0.8, 0.4],  # Light-medium green
        [0.7, 0.9, 0.7]  # Light green
    ]


    # Function to extract velocity profiles from the data
    def extract_velocity_profiles(x_piv, y_piv, V_piv, l, vi, dz_l):
        x_slice = x_piv[:, 0] / l  # Normalize x values based on l
        extracted_values = np.column_stack([x_slice])

        # Extract velocity profiles for each dz_l value
        for dz in dz_l:
            # Find the closest y value in y_piv to dz*l
            y_idx = np.argmin(np.abs(y_piv - dz * l))

            # Extract corresponding velocity values and normalize
            v_slice = V_piv[:, y_idx] / vi

            # Append the velocity slice to the extracted_values array
            extracted_values = np.column_stack([extracted_values, v_slice])

        return extracted_values


    # Zone data - substitute with actual data arrays
    zones = ['ZoneD_dirOlap', 'ZoneD_maxM', 'ZoneC_dirOlap', 'ZoneC_maxM']
    zone_names = ['Contour Plot 1', 'Contour Plot 2', 'Contour Plot 3', 'Contour Plot 4']

    # Example data arrays for zone x, y, and V - replace with actual data
    x_piv_data = {
        'ZoneD_dirOlap': np.random.rand(100, 1),
        'ZoneD_maxM': np.random.rand(100, 1),
        'ZoneC_dirOlap': np.random.rand(100, 1),
        'ZoneC_maxM': np.random.rand(100, 1),
    }

    y_piv_data = {
        'ZoneD_dirOlap': np.random.rand(100),
        'ZoneD_maxM': np.random.rand(100),
        'ZoneC_dirOlap': np.random.rand(100),
        'ZoneC_maxM': np.random.rand(100),
    }

    V_piv_data = {
        'ZoneD_dirOlap': np.random.rand(100, 100),
        'ZoneD_maxM': np.random.rand(100, 100),
        'ZoneC_dirOlap': np.random.rand(100, 100),
        'ZoneC_maxM': np.random.rand(100, 100),
    }

    # Loop over zones and plot the velocity profiles
    for zone_idx, zone_name in enumerate(zones):
        x_piv = x_piv_data[zone_name]
        y_piv = y_piv_data[zone_name]
        V_piv = V_piv_data[zone_name]

        # Extract velocity profiles
        extracted_values = extract_velocity_profiles(x_piv, y_piv, V_piv, l, vi, dz_l)

        # Plot the extracted velocity profiles
        plt.figure()
        for i in range(1, extracted_values.shape[1]):
            plt.plot(extracted_values[:, 0], extracted_values[:, i], linewidth=2.5, color=shades_of_blue[i - 1])

        plt.xlabel('$x/l$')
        plt.ylabel(r'$\bar{V}/V_{ind}$', labelpad=5)
        plt.legend([f'$z/l$ = {dz}' for dz in dz_l], loc='best')
        plt.xlim([-4.25, 4.25])
        plt.ylim([0, 1])
        plt.grid(True)
        plt.title(f'Profiles of velocity magnitudes in {zone_names[zone_idx]}')
        plt.gca().tick_params(axis='both', which='major', labelsize=20)
        plt.show()

    # Extract and plot velocity std devs (fluctuations)
    for zone_idx, zone_name in enumerate(zones):
        xstd_piv = x_piv_data[zone_name]  # Replace with actual std data if available
        ystd_piv = y_piv_data[zone_name]  # Replace with actual std data if available
        Vstd_piv = V_piv_data[zone_name]  # Replace with actual std data if available

        # Extract std dev velocity profiles
        extractedstd_values = extract_velocity_profiles(xstd_piv, ystd_piv, Vstd_piv, l, vi, dz_l)

        # Plot the extracted std dev velocity profiles
        plt.figure()
        for i in range(1, extractedstd_values.shape[1]):
            plt.plot(extractedstd_values[:, 0], extractedstd_values[:, i], linewidth=2.5, color=shades_of_green[i - 1])

        plt.xlabel('$x/l$')
        plt.ylabel("$V'/V_{ind}$", labelpad=10)
        plt.legend([f'$z/l$ = {dz}' for dz in dz_l], loc='best')
        plt.xlim([-4.25, 4.25])
        plt.ylim([0, 0.31])
        plt.grid(True)
        plt.title(f'Profiles of velocity std. devs in {zone_names[zone_idx]}')
        plt.gca().tick_params(axis='both', which='major', labelsize=20)
        plt.show()

def main():
    combined_zones()
    velocity_analysis()

if __name__ == '__main__':
    main()