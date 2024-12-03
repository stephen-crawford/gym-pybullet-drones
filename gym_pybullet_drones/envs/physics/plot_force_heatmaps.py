import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib import cm
import seaborn as sns

# Define constants
sampling_time = 30  # [s] Sampling time
frot = 320  # [Hz] Frequency of blade rotation
fsampling = 20000  # [Hz] Sampling frequency
rot_rpm = (frot / (2 * np.pi)) * 60  # [rpm] Blade rotation frequency in rpm

# Load the MATLAB workspace data
mat_file = '../../assets/workspace.mat'
data = sio.loadmat(mat_file)

# Assign necessary variables from the workspace data
# (assuming the MATLAB variables are in the .mat file as dictionaries)
zl13 = data['zl13']
zl17 = data['zl17']
zl21 = data['zl21']
zl25 = data['zl25']
zl29 = data['zl29']
zl35 = data['zl35']
zl40 = data['zl40']
zl50 = data['zl50']
zl60 = data['zl60']
zl70 = data['zl70']
zl80 = data['zl80']
zl90 = data['zl90']
zl100 = data['zl100']
zl110 = data['zl110']

# Arm length and weight
l_arm = 3.25  # arm length in cm
lW = 0.295  # weight of lower crazyflie in N

# Create a grid for the x and y axes
xl_new_fine = np.array(data['xvals']).flatten()  # Assuming xvals are in the data as a 1D array
yl_new_fine = np.array([13, 17, 21, 25, 29, 35, 40, 50, 60, 70, 80, 90, 100, 110])

X, Y = np.meshgrid(xl_new_fine / l_arm, yl_new_fine / l_arm)

# Function to plot contour
def plot_contour(X, Y, Z, title, xlabel, ylabel, colorbar_label, cmap='Blues', levels=200):
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    plt.colorbar(contour, label=colorbar_label)
    plt.gca().invert_yaxis()  # Invert the y-axis for proper alignment
    plt.show()

# Fz lower Crazyflie (mean force)
lF_new_fine = np.array([zl13[:, 4], zl17[:, 4], zl21[:, 4], zl25[:, 4], zl29[:, 4], zl35[:, 4], zl40[:, 4],
                        zl50[:, 4], zl60[:, 4], zl70[:, 4], zl80[:, 4], zl90[:, 4], zl100[:, 4], zl110[:, 4]]) / lW
plot_contour(X, Y, lF_new_fine, r'$\bar{F_{z}}$ lower Crazyflie', r'Horizontal separation, ${\Delta x}/l$',
             r'Vertical separation, ${\Delta z}/l$', r'Normalized $F_{z}$')

# Fz lower unsteadiness (standard deviation)
lFstd_new_fine = np.array([zl13[:, 5], zl17[:, 5], zl21[:, 5], zl25[:, 5], zl29[:, 5], zl35[:, 5], zl40[:, 5],
                           zl50[:, 5], zl60[:, 5], zl70[:, 5], zl80[:, 5], zl90[:, 5], zl100[:, 5], zl110[:, 5]]) / lW
plot_contour(X, Y, lFstd_new_fine, r'$F_{z}^\prime$ lower Crazyflie', r'Horizontal separation, ${\Delta x}/l$',
             r'Vertical separation, ${\Delta z}/l$', r'Normalized $F_{z}^\prime$ st. dev.', cmap='Purples', levels=51)

# M lower Crazyflie (mean moment)
lM_new_fine = np.array([zl13[:, 6], zl17[:, 6], zl21[:, 6], zl25[:, 6], zl29[:, 6], zl35[:, 6], zl40[:, 6],
                        zl50[:, 6], zl60[:, 6], zl70[:, 6], zl80[:, 6], zl90[:, 6], zl100[:, 6], zl110[:, 6]]) / (lW * l_arm)
plot_contour(X, Y, lM_new_fine, r'$\bar{M_{z}}$ lower Crazyflie', r'Horizontal separation, ${\Delta x}/l$',
             r'Vertical separation, ${\Delta z}/l$', r'Normalized $M_{z}$')

# M lower unsteadiness (standard deviation)
lMstd_new_fine = np.array([zl13[:, 7], zl17[:, 7], zl21[:, 7], zl25[:, 7], zl29[:, 7], zl35[:, 7], zl40[:, 7],
                           zl50[:, 7], zl60[:, 7], zl70[:, 7], zl80[:, 7], zl90[:, 7], zl100[:, 7], zl110[:, 7]]) / (lW * l_arm)
plot_contour(X, Y, lMstd_new_fine, r'$M_{z}^\prime$ lower Crazyflie', r'Horizontal separation, ${\Delta x}/l$',
             r'Vertical separation, ${\Delta z}/l$', r'Normalized $M_{z}^\prime$ st. dev.', cmap='Purples', levels=51)

# Fz upper Crazyflie (mean force)
uW = 0.3354  # New weight for upper Crazyflie
uF_new_fine = np.array([data['zu13'][:, 4], data['zu17'][:, 4], data['zu21'][:, 4], data['zu25'][:, 4],
                        data['zu29'][:, 4], data['zu35'][:, 4], data['zu40'][:, 4], data['zu50'][:, 4],
                        data['zu60'][:, 4], data['zu70'][:, 4], data['zu80'][:, 4], data['zu90'][:, 4],
                        data['zu100'][:, 4], data['zu110'][:, 4]]) / uW
plot_contour(X, Y, uF_new_fine, r'$\bar{F_{z}}$ upper Crazyflie', r'Horizontal separation, ${\Delta x}/l$',
             r'Vertical separation, ${\Delta z}/l$', r'Normalized $F_{z}$', cmap='YlOrRd', levels=201)

# Fz upper unsteadiness (standard deviation)
uFstd_new_fine = np.array([data['zu13'][:, 5], data['zu17'][:, 5], data['zu21'][:, 5], data['zu25'][:, 5],
                           data['zu29'][:, 5], data['zu35'][:, 5], data['zu40'][:, 5], data['zu50'][:, 5],
                           data['zu60'][:, 5], data['zu70'][:, 5], data['zu80'][:, 5], data['zu90'][:, 5],
                           data['zu100'][:, 5], data['zu110'][:, 5]]) / uW
plot_contour(X, Y, uFstd_new_fine, r'$F_{z}^\prime$ upper Crazyflie', r'Horizontal separation, ${\Delta x}/l$',
             r'Vertical separation, ${\Delta z}/l$', r'Normalized $F_{z}^\prime$ st. dev.', cmap='YlOrRd', levels=51)

# M upper Crazyflie (mean moment)
uM_new_fine = np.array([data['zu13'][:, 6], data['zu17'][:, 6], data['zu21'][:, 6], data['zu25'][:, 6],
                        data['zu29'][:, 6], data['zu35'][:, 6], data['zu40'][:, 6], data['zu50'][:, 6],
                        data['zu60'][:, 6], data['zu70'][:, 6], data['zu80'][:, 6], data['zu90'][:, 6],
                        data['zu100'][:, 6], data['zu110'][:, 6]]) / (uW * l_arm)
plot_contour(X, Y, uM_new_fine, r'$\bar{M_{z}}$ upper Crazyflie', r'Horizontal separation, ${\Delta x}/l$',
             r'Vertical separation, ${\Delta z}/l$', r'Normalized $M_{z}$', cmap='YlOrRd', levels=201)

# M upper unsteadiness (standard deviation)
uMstd_new_fine = np.array([data['zu13'][:, 7], data['zu17'][:, 7], data['zu21'][:, 7], data['zu25'][:, 7],
                           data['zu29'][:, 7], data['zu35'][:, 7], data['zu40'][:, 7], data['zu50'][:, 7],
                           data['zu60'][:, 7], data['zu70'][:, 7], data['zu80'][:, 7], data['zu90'][:, 7],
                           data['zu100'][:, 7], data['zu110'][:, 7]]) / (uW * l_arm)
plot_contour(X, Y, uMstd_new_fine, r'$M_{z}^\prime$ upper Crazyflie', r'Horizontal separation, ${\Delta x}/l$',
             r'Vertical separation, ${\Delta z}/l$', r'Normalized $M_{z}^\prime$ st. dev.', cmap='YlOrRd', levels=51)
