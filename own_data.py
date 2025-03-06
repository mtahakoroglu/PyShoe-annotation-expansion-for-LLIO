import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ins_tools.util import *
import ins_tools.visualize as visualize
from ins_tools.INS import INS
import os
import logging
import glob
import scipy.io as sio  # Import scipy.io for loading .mat files
from scipy.interpolate import interp1d # required when adopting compensated imu data at 250Hz

# Directory of sensor data files
sensor_data_dir = 'data/own'
sensor_data_files = glob.glob(os.path.join(sensor_data_dir, 'SensorConnectData_*.csv'))
GCP_files = glob.glob(os.path.join(sensor_data_dir, 'SensorConnectData_*.mat'))

# Set up logging
output_dir = "results/figs/own"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'output.log')
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

extracted_training_data_dir = "data/" # training data (imu, zv) for LSTM retraining & (displacement, heading change, stride indexes, timestamps) for LLIO training

# Detectors and labels
det_list = ['lstm'] # 'ared', 'shoe'
# Define thresholds for each detector
thresh_list = [0] # 0.55, 8.5e7 
W_list = [0] # 5, 5, 0
legend = ['PyShoe (LSTM)'] # 'ARED', 'SHOE'

# this function is used in stride detection
def count_zero_to_one_transitions(arr):
    arr = np.asarray(arr) # Ensure the array is a NumPy array
    transitions = np.where((arr[:-1] == 0) & (arr[1:] == 1))[0] # Find the locations where transitions from 0 to 1 occur
    return len(transitions), transitions + 1  # Add 1 to get the index of the '1'

# Function to count one-to-zero transitions to determine stride indexes
def count_one_to_zero_transitions(zv):
    strides = np.where(np.diff(zv) < 0)[0] + 1
    return len(strides), strides

# elimination of incorrect stride detections in raw zv_opt
def heuristic_zv_filter_and_stride_detector(zv, k):
    if zv.dtype == 'bool':
        zv = zv.astype(int)
    zv[:50] = 1 # make sure all labels are zero at the beginning as the foot is stationary
    # detect strides (falling edge of zv binary signal) and respective indexes
    n, strideIndexFall = count_one_to_zero_transitions(zv)
    strideIndexFall = strideIndexFall - 1 # make all stride indexes the last samples of the respective ZUPT phase
    strideIndexFall = np.append(strideIndexFall, len(zv)-1) # last sample is the last stride index
    # detect rising edge indexes of zv labels
    n2, strideIndexRise = count_zero_to_one_transitions(zv)
    for i in range(len(strideIndexRise)):
        if (strideIndexRise[i] - strideIndexFall[i] < k):
            zv[strideIndexFall[i]:strideIndexRise[i]] = 1 # make all samples in between one
    # after the correction is completed, do the stride index detection process again
    n, strideIndexFall = count_one_to_zero_transitions(zv)
    strideIndexFall = strideIndexFall - 1 # make all stride indexes the last samples of the respective ZUPT phase
    strideIndexFall = np.append(strideIndexFall, len(zv)-1) # last sample is the last stride index
    return zv, n, strideIndexFall

# Function to calculate displacement and heading change between stride points
def calculate_displacement_and_heading(traj, strideIndex):
    displacements, heading_changes = [], []
    for j in range(1, len(strideIndex)):
        delta_position = traj[strideIndex[j], :2] - traj[strideIndex[j-1], :2]
        displacement = np.linalg.norm(delta_position)
        heading_change = np.arctan2(delta_position[1], delta_position[0])
        displacements.append(displacement)
        heading_changes.append(heading_change)
    return np.array(displacements), np.array(heading_changes)

# Function to reconstruct trajectory from displacements and heading changes
def reconstruct_trajectory(displacements, heading_changes, initial_position):
    trajectory = [initial_position]
    current_heading = 0.0

    for i in range(len(displacements)):
        delta_position = np.array([
            displacements[i] * np.cos(heading_changes[i]),
            displacements[i] * np.sin(heading_changes[i])
        ])
        new_position = trajectory[-1] + delta_position
        trajectory.append(new_position)
        current_heading += heading_changes[i]

    trajectory = np.array(trajectory)
    # trajectory[:, 0] = -trajectory[:, 0] # change made by mtahakoroglu to match with GT alignment
    return trajectory

def rotate_trajectory(trajectory, theta):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return trajectory @ rotation_matrix.T

def downsample(data, original_freq, target_freq):
    # Calculate the time points for the original and target frequencies
    original_time_points = np.arange(0, len(data)) / original_freq
    target_time_points = np.arange(0, len(data) * target_freq / original_freq) / target_freq
    # Create an interpolation function
    interpolation_function = interp1d(original_time_points, data, axis=0, kind='linear')
    # Use the interpolation function to get the downsampled data
    downsampled_data = interpolation_function(target_time_points)
    return downsampled_data

# Flag to save LLIO training data - DO NOT CHANGE THIS VALUE AS IT IS AUTOMATICALLY UPDATED BELOW
extract_LLIO_training_data = False # used to save csv files for LLIO SHS training (displacement, heading change) and (stride indexes, timestamps, GCP stride coordinates)
traveled_distances = [] # to keep track of the traveled distances in each experiment for LLIO training data generation
traverse_times = [] # to keep track of experiment times and eventually total experiment time for LLIO training data generation
number_of_stride_wise_verified_experiments = 0 # detected stride points must be equal to the actual number

g = 9.8029 # gravity constant
# processing zero velocity labels to turn a sampling frequency driven system into gait driven system (stride and heading system)
expCount = 0 # experiment index
dpi = 400 # for figure resolution 
# Process each sensor data file
for file in sensor_data_files:
    logging.info(f"===================================================================================================================")
    logging.info(f"Processing file {file}")
    expCount = expCount+1 # next experiment

    sensor_data = pd.read_csv(file)
    
    # Remove the '.csv' extension from the filename
    base_filename = os.path.splitext(os.path.basename(file))[0]
    GCP_data = sio.loadmat(sensor_data_dir + '/' + base_filename + '.mat')
    # extract the number after _ in base_filename
    expNumberData = int(base_filename.split('_')[1])

    # Extract the relevant columns from csv data files (SensorConnect by LORD Microstrain)
    timestamps = sensor_data['Time'].values
    if expNumberData <= 40: # experiments before 41 captured compensated imu data at 200Hz that is compatible with PyShoe LSTM detector
        imu_data = sensor_data[['inertial-6253.76535:scaledAccelX', 'inertial-6253.76535:scaledAccelY', 'inertial-6253.76535:scaledAccelZ',
            'inertial-6253.76535:scaledGyroX', 'inertial-6253.76535:scaledGyroY', 'inertial-6253.76535:scaledGyroZ']].copy()
        # Multiply accelerations by gravity
        imu_data[['inertial-6253.76535:scaledAccelX', 'inertial-6253.76535:scaledAccelY', 'inertial-6253.76535:scaledAccelZ']] *= g
        euler_angles = sensor_data[['inertial-6253.76535:roll', 'inertial-6253.76535:pitch', 'inertial-6253.76535:yaw']].copy()
    else: # experiments after 40 captured compensated imu data at 250Hz and then downsampled to 200Hz
        imu_data = sensor_data[['inertial-6253.76535:estCompensatedAccelX', 'inertial-6253.76535:estCompensatedAccelY', 'inertial-6253.76535:estCompensatedAccelZ',
            'inertial-6253.76535:estAngularRateX', 'inertial-6253.76535:estAngularRateY', 'inertial-6253.76535:estAngularRateZ']].copy()
        euler_angles = sensor_data[['inertial-6253.76535:estRoll', 'inertial-6253.76535:estPitch', 'inertial-6253.76535:estYaw']].copy()
        # all timestamps, imu_data and euler_angles are captured at 250Hz.
        # downsample all data to 200Hz to match with the rest of the data
        timestamps = downsample(timestamps, 250, 200)
        imu_data = downsample(imu_data, 250, 200)
        euler_angles = downsample(euler_angles, 250, 200)
    
    # Extract Ground Control Points (GCP) info from mat files
    # logging.info(f"Keys in GCP_data: {list(GCP_data.keys())}")
    expNumber = GCP_data['expID'].item()
    if expNumber != expNumberData:
        logging.info(f"Experiment number mismatch between csv and mat files for file {base_filename}.")
        break
    if GCP_data['GCP_exist_and_correct'].item() == True:
        logging.info(f"GCP are available & correct for file {base_filename}.")
        GCP = GCP_data['GCP_meters']
        GCP[:,1] = -GCP[:,1] # change made by mtahakoroglu to match with GT alignment
    else:
        logging.info(f"GCP are either not available or not correct for file {base_filename}.")
    GCP_stride_numbers = np.squeeze(GCP_data['GCP_stride_numbers'])
    numberOfStrides =  GCP_data['numberOfStrides'].item() # total number of strides is equal to the last GCP stride number, i.e., GCP_stride_numbers[-1]
    
    if expNumber > 30: # update this statement later to include only the experiments that are manually annotated for LLIO training
        extract_LLIO_training_data = True

    # Initialize INS object with correct parameters - adopted the exact parameters used in PyShoe research (makes sense as our sensor is in the same family)
    if expNumber <= 40:
        ins = INS(imu_data.values, sigma_a=0.00098, sigma_w=8.7266463e-5, T=1.0/200)
    else:
        ins = INS(imu_data, sigma_a=0.00098, sigma_w=8.7266463e-5, T=1.0/200)

    traj_list, zv_list = [], []
    for i, detector in enumerate(det_list):
        logging.info(f"Processing {detector.upper()} detector for file {base_filename}.")
        zv = ins.Localizer.compute_zv_lrt(W=W_list[i], G=thresh_list[i], detector=detector)
        x, acc_n = ins.baseline(zv=zv)
        traj_list.append(x)
        zv_list.append(zv)

    strideIndex = None # stride indexes will be used to chop imu data for LLIO training & are provided by PyShoe (LSTM) detector
    for i, zv in enumerate(zv_list):
        logging.info(f"Plotting zero velocity detection for {det_list[i].upper()} detector for file {base_filename}.")
        # Apply a heuristic filter to zero velocity labels (via LSTM) to eliminate undesired jumps & achieve correct stride detection
        if det_list[i] == 'lstm':
            k = 75 # temporal window size for checking if detected strides are too close
            if expNumber in [33]:
                k = 100
            zv_lstm_filtered, n, strideIndex = heuristic_zv_filter_and_stride_detector(zv, k)
            logging.info(f"There are {n}/{numberOfStrides} strides detected in experiment #{expNumber}.")
            # print(f"Stride indexes: {strideIndex}")
            # print(f"Time indexes: {timestamps[strideIndex]}")

            plt.figure()
            plt.plot(timestamps[:len(zv_lstm_filtered)], zv_lstm_filtered)
            plt.scatter(timestamps[strideIndex], zv_lstm_filtered[strideIndex], c='r', marker='x')
            plt.title(f'LSTM filtered ({n}/{numberOfStrides}) - {base_filename}')
            plt.xlabel('Time [s]'); plt.ylabel('Zero Velocity')
            plt.savefig(os.path.join(output_dir, f'{base_filename}_ZV_{det_list[i].upper()}_filtered.png'), dpi=dpi, bbox_inches='tight')
            plt.close()

    # Reconstruct the trajectory from displacements and heading changes to build a Stride & Heading INS (SHS)
    displacements, heading_changes = calculate_displacement_and_heading(traj_list[-1][:, :2], strideIndex)
    initial_position = traj_list[-1][strideIndex[0], :2] # Starting point
    reconstructed_traj = reconstruct_trajectory(displacements, heading_changes, initial_position)

    # Align the trajectory wrt the selected stride - obsolete (this is an arbitrary rotation to align the trajectory along with x axis)
    # Align the trajectory wrt the selected GCP - this is a rotation from navigation coordinate frame to world-fix coordinate frame
    strideAlign = 3; GCP_align = strideAlign
    if expNumber == 40: # Exp#40 is conducted at Science Road
        strideAlign = 10 # this stride number is selected according to the trajectory plot
        GCP_align = strideAlign
    elif expNumber == 42: # Exp#42 is conducted at Science Road
        strideAlign = 10 # this stride number is selected according to the trajectory plot
        GCP_align = strideAlign
    elif expNumber == 43:
        strideAlign = 5 # this stride number is selected according to the trajectory plot
        GCP_align = strideAlign  
    # elif expNumber == 45:
    #     strideAlign = 8 # this stride number is selected according to the trajectory plot
    #     GCP_align = strideAlign
    
    _, thetaPyShoe = calculate_displacement_and_heading(traj_list[-1][:, :2], strideIndex[np.array([0,strideAlign])])
    _, thetaGCP = calculate_displacement_and_heading(GCP, np.array([0,GCP_align]))
    theta = thetaPyShoe - thetaGCP # later we apply theta to GCP to align world coordinate frame with nav coordinate frame
    print(f"theta = {np.degrees(theta)} degrees for experiment #{expNumber}.")
    # Apply the rotation to GCP points instead of INS trajectory - this way we do not change IMU data
    # !!! NOTICE THAT GCP POINTS CHANGE AFTER THIS POINT !!!
    # traj_list[-1][:,:2] = np.squeeze(rotate_trajectory(traj_list[-1][:,:2], -theta))
    # reconstructed_traj = np.squeeze(rotate_trajectory(reconstructed_traj, -theta))
    GCP_wcf = GCP.copy(); # for trajectory in WCF
    GCP = np.squeeze(rotate_trajectory(GCP, theta))
    # save plots in world coordinate frame as well - following data are not saved to file
    traj_wcf = np.squeeze(rotate_trajectory(traj_list[-1][:, :2], -theta)) # rotate PyShoe INS to make it compatible with GCP
    reconstructed_traj_wcf = np.squeeze(rotate_trajectory(reconstructed_traj, -theta)) # rotate PyShoe SHS to make it compatible with GCP
    GCP_wcf[:,1] = -GCP_wcf[:,1] # change made by mtahakoroglu to match with GT alignment
    traj_wcf[:,1] = -traj_wcf[:,1] # change made by mtahakoroglu to match with GT alignment
    reconstructed_traj_wcf[:,1] = -reconstructed_traj_wcf[:,1] # change made by mtahakoroglu to match with GT alignment
    # acc_w = np.squeeze(rotate_trajectory(acc_n[:, :2], -theta)) # rotate acc_n to make it acc_w
    
    # reverse data in x direction to match with GCP and better illustration in the paper
    # traj_list[-1][:,:2][:,0] = -traj_list[-1][:,:2][:,0]
    # reconstructed_traj[:,0] = -reconstructed_traj[:,0]

    # PERFORMANCE EVALUTATION via METRICS
    if GCP_data['GCP_exist_and_correct'].item() and n == numberOfStrides:
        k, number_of_GCP = 0, GCP.shape[0]
        logging.info(f"File {base_filename}, i.e., experiment {expNumber} will be used in performance evaluation.")
        # Calculate the RMSE between the GCP and GCP stride in SHS trajectories
        rmse_GCP = np.sqrt(np.sum((reconstructed_traj[GCP_stride_numbers] - GCP)**2, axis=1))
        logging.info(f"There are {rmse_GCP.shape[0]} GCP for file {base_filename}, i.e., experiment {expNumber}.")
        for i in range(rmse_GCP.shape[0]):
            k += 1 # GCP index
            logging.info(f"RMSE for GCP {k} stepped on at stride {GCP_stride_numbers[i]} is {rmse_GCP[i]:.4f}")
        logging.info(f"Average RMSE for all GCP strides in experiment {expNumber} is {np.mean(rmse_GCP):.4f}")
    else:
        logging.info(f"File {base_filename}, i.e., experiment {expNumber} will not be used in performance evaluation.")
    
    ############################ PLOTS ###############################
    # NAVIGATION COORDINATE SYSTEM
    plt.figure()
    if GCP_data['GCP_exist_and_correct'].item():
        # plt.scatter(GCP[:,0], GCP[:,1], color='r', s=30, marker='s', edgecolors='k', label="GCP")
        plt.plot(GCP[:,0], GCP[:,1], 'ks-', linewidth=1.4, markersize=5, markerfacecolor='r', markeredgecolor='k', markeredgewidth=1.2, label="GCP")
    if n == numberOfStrides and not extract_LLIO_training_data: # experiments after 30 are conducted for expanding/enlarging LLIO training dataset
        plt.scatter(reconstructed_traj[GCP_stride_numbers,0], reconstructed_traj[GCP_stride_numbers,1], color='r', s=45, 
                    marker='o', facecolor='none', linewidths=1.5, label="GCP stride")
    plt.plot(traj_list[-1][:,:2][:,0], traj_list[-1][:,:2][:,1], linewidth = 1.5, color='b', label=legend[-1])
    plt.legend(fontsize=15); plt.xlabel('x [m]', fontsize=22); plt.ylabel('y [m]', fontsize=22)
    plt.title(f'{base_filename}_NCF', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.axis('equal')
    if expNumber == 28:
        plt.ylim(-10,20)
    plt.grid(True, which='both', linestyle='--', linewidth=1.5)
    plt.savefig(os.path.join(output_dir, f'{base_filename}_NCF.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    plt.figure()
    if GCP_data['GCP_exist_and_correct'].item():
        # plt.scatter(GCP[:,0], GCP[:,1], color='r', s=30, marker='s', edgecolors='k', label="GCP")
        plt.plot(GCP[:,0], GCP[:,1], 'ks-', linewidth=1.4, markersize=5, markerfacecolor='r', markeredgewidth=1.2, markeredgecolor='k', label="GCP")
    plt.plot(reconstructed_traj[:,0], reconstructed_traj[:,1], 'b.-', linewidth = 1.4, markersize=5, markeredgewidth=1.2, label="PyShoe (LSTM) SHS")
    # plt.plot(reconstructed_traj[-3:,0], reconstructed_traj[-3:,1], 'bx-', linewidth = 1.4, markersize=5, markeredgewidth=1.2, label="PyShoe (LSTM) SHS last three")
    if n == numberOfStrides and not extract_LLIO_training_data: # experiments after 30 are conducted for expanding/enlarging LLIO training dataset
        plt.scatter(reconstructed_traj[GCP_stride_numbers,0], reconstructed_traj[GCP_stride_numbers,1], color='r', s=45, 
                marker='o', facecolor='none', linewidths=1.5, label="GCP stride")
    plt.legend(fontsize=15); plt.xlabel('x [m]', fontsize=22); plt.ylabel('y [m]', fontsize=22)
    plt.title(f'Exp#{expNumber}: {n}/{numberOfStrides} strides detected', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.axis('equal')
    if expNumber == 28:
        plt.ylim(-10,20)
    plt.grid(True, which='both', linestyle='--', linewidth=1.5)
    plt.savefig(os.path.join(output_dir, f'{base_filename}_SHS_NCF.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # WORLD COORDINATE SYSTEM
    plt.figure()
    if GCP_data['GCP_exist_and_correct'].item():
        # plt.scatter(GCP_wcf[:,0], GCP_wcf[:,1], color='r', s=30, marker='s', edgecolors='k', label="GCP")
        plt.plot(GCP_wcf[:,0], GCP_wcf[:,1], 'ks-', linewidth=1.4, markersize=5, markerfacecolor='r', markeredgecolor='k', markeredgewidth=1.2, label="GCP")
    if n == numberOfStrides and not extract_LLIO_training_data: # experiments after 30 are conducted for expanding/enlarging LLIO training dataset
        plt.scatter(reconstructed_traj_wcf[GCP_stride_numbers,0], reconstructed_traj_wcf[GCP_stride_numbers,1], color='r', s=45, 
                    marker='o', facecolor='none', linewidths=1.5, label="GCP stride")
    plt.plot(traj_wcf[:,0], traj_wcf[:,1], linewidth = 1.5, color='b', label=legend[-1])
    plt.legend(fontsize=15); plt.xlabel('x [m]', fontsize=22); plt.ylabel('y [m]', fontsize=22)
    plt.title(f'{base_filename}_WCF', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.axis('equal')
    if expNumber == 28:
        plt.ylim(-10,20)
    plt.grid(True, which='both', linestyle='--', linewidth=1.5)
    plt.savefig(os.path.join(output_dir, f'{base_filename}_WCF.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    plt.figure()
    if GCP_data['GCP_exist_and_correct'].item():
        # plt.scatter(GCP_wcf[:,0], GCP_wcf[:,1], color='r', s=30, marker='s', edgecolors='k', label="GCP")
        plt.plot(GCP_wcf[:,0], GCP_wcf[:,1], 'ks-', linewidth=1.4, markersize=5, markerfacecolor='r', markeredgecolor='k', markeredgewidth=1.2, label="GCP")
    plt.plot(reconstructed_traj_wcf[:,0], reconstructed_traj_wcf[:,1], 'b.-', linewidth = 1.4, markersize=5, markeredgewidth=1.2, label="PyShoe (LSTM) SHS")
    # plt.plot(reconstructed_traj[-3:,0], reconstructed_traj[-3:,1], 'bx-', linewidth = 1.4, markersize=5, markeredgewidth=1.2, label="PyShoe (LSTM) SHS last three")
    if n == numberOfStrides and not extract_LLIO_training_data: # experiments after 30 are conducted for expanding/enlarging LLIO training dataset
        plt.scatter(reconstructed_traj_wcf[GCP_stride_numbers,0], reconstructed_traj_wcf[GCP_stride_numbers,1], color='r', s=45, 
                marker='o', facecolor='none', linewidths=1.5, label="GCP stride")
    plt.legend(fontsize=15); plt.xlabel('x [m]', fontsize=22); plt.ylabel('y [m]', fontsize=22)
    plt.title(f'Exp#{expNumber}: {n}/{numberOfStrides} strides detected', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.axis('equal')
    if expNumber == 28:
        plt.ylim(-10,20)
    plt.grid(True, which='both', linestyle='--', linewidth=1.5)
    plt.savefig(os.path.join(output_dir, f'{base_filename}_SHS_WCF.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

    # Plot stride indexes on IMU data, i.e., the magnitudes of acceleration and angular velocity
    plt.figure()
    if expNumber <= 40:
        plt.plot(timestamps, np.linalg.norm(imu_data.iloc[:, :3].values, axis=1), label=r'$\Vert\mathbf{a}\Vert$')
        plt.plot(timestamps, np.linalg.norm(imu_data.iloc[:, 3:].values, axis=1), label=r'$\Vert\mathbf{\omega}\Vert$')
        plt.scatter(timestamps[strideIndex], np.linalg.norm(imu_data.iloc[strideIndex, :3].values, axis=1), 
                    c='r', marker='x', label='Stride', zorder=3)
        plt.scatter(timestamps[strideIndex], np.linalg.norm(imu_data.iloc[strideIndex, 3:].values, axis=1), 
                    c='r', marker='x', zorder=3)
    else:
        plt.plot(timestamps, np.linalg.norm(imu_data[:, :3], axis=1), label=r'$\Vert\mathbf{a}\Vert$')
        plt.plot(timestamps, np.linalg.norm(imu_data[:, 3:], axis=1), label=r'$\Vert\mathbf{\omega}\Vert$')
        plt.scatter(timestamps[strideIndex], np.linalg.norm(imu_data[strideIndex, :3], axis=1),
                    c='r', marker='x', label='Stride', zorder=3)
        plt.scatter(timestamps[strideIndex], np.linalg.norm(imu_data[strideIndex, 3:], axis=1),
                    c='r', marker='x', zorder=3)
    plt.title(f'{base_filename} - Stride Detection on IMU Data')
    plt.xlabel('Time [s]'); plt.ylabel(r'Magnitude'); plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=1.5)
    plt.savefig(os.path.join(output_dir, f'{base_filename}_stride_detection.png'), dpi=600, bbox_inches='tight')
    plt.close()

    # Plot Euler angles vs. time
    plt.figure()
    plt.subplot(3, 1, 1)
    if expNumber <= 40:
        plt.plot(timestamps, euler_angles.values[:, 0], c='b', label="Roll (IMU)", linewidth=0.9)
    else:
        plt.plot(timestamps, euler_angles[:, 0], c='b', label="Roll (IMU)", linewidth=0.9)
    plt.plot(timestamps, x[:, 6], c='r', label="Roll (PyShoe)", linewidth=0.8)
    plt.title(f'{base_filename} - Euler Angles')
    plt.ylabel('Angle [rad]')
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    plt.legend()
    plt.subplot(3, 1, 2)
    if expNumber <= 40:
        plt.plot(timestamps, euler_angles.values[:, 1], c='b', label="Pitch (IMU)", linewidth=0.9)
    else:
        plt.plot(timestamps, euler_angles[:, 1], c='b', label="Pitch (IMU)", linewidth=0.9)
    plt.plot(timestamps, x[:, 7], c='r', label="Pitch (PyShoe)", linewidth=0.8)
    # plt.title(f'{base_filename} - Pitch Angle')
    plt.ylabel('Angle [rad]')
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    plt.legend()
    plt.subplot(3, 1, 3)
    if expNumber <= 40:
        plt.plot(timestamps, euler_angles.values[:, -1], c='b', label="Yaw (IMU)", linewidth=0.9)
    else:
        plt.plot(timestamps, euler_angles[:, -1], c='b', label="Yaw (IMU)", linewidth=0.9)
    plt.plot(timestamps, x[:, -1], c='r', label="Yaw (PyShoe)", linewidth=0.8)
    # plt.title(f'{base_filename} - Yaw Angle')
    plt.xlabel('Time [s]'); plt.ylabel('Angle [rad]')
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_filename}_euler_angles.png'), dpi=600, bbox_inches='tight')
    plt.close()

    #################### STRIDE INDEX ANNOTATION (IN CASE OF PYSHOE (LSTM) MISSES ZV INTERVALS) ##################################
    if expNumber == 32 and strideIndex[-2] == 14203:
        strideIndex[-2] = 14131-1
        print(f"strideIndex[-2] is manually corrected for experiment #{expNumber} after MATLAB inspection.")
    elif expNumber == 33 and strideIndex[12] == 2979:
        strideIndex[12] = 2921-1
        print(f"strideIndex[12] is manually corrected for experiment #{expNumber} after MATLAB inspection.")
    elif expNumber == 34: # Stride #7 and #15 are missed; at MATLAB side they are manually annotated and inserted into the list
        strideIndex = np.insert(strideIndex, 7, 1705-1) # Stride #7 index is inserted
        strideIndex = np.insert(strideIndex, 15, 3278-1) # Stride #15 index is inserted
    elif expNumber == 35: # Stride #{2, 5, 17, 18, 19} are missed; at MATLAB side they are manually annotated and inserted into the list
        missedStride, missedStrideIndex = [2, 5, 17, 18, 19], [951-1, 1453-1, 3591-1, 3740-1, 3892-1]
        for i in range(len(missedStride)):
            strideIndex = np.insert(strideIndex, missedStride[i], missedStrideIndex[i]) # Stride #i index is inserted
    elif expNumber == 36 and strideIndex[17] == 4470:
        strideIndex[17] = 4416-1
        print(f"strideIndex[17] is manually corrected for experiment #{expNumber} after MATLAB inspection.")
    elif expNumber == 37 and strideIndex[33] == 8776 and strideIndex[34] == 8998: # stride indexes are the same as MATLAB side
        strideIndex[33], strideIndex[34] = 8722-1, 8942-1
        if strideIndex[41] == 10548:
            strideIndex[41] = 10500-1
        if strideIndex[43] == 10979:
            strideIndex[43] = 10933-1
        if strideIndex[60] == 14765:
            strideIndex[60] = 14710-1
    elif expNumber == 38: # At MATLAB side missed strides are manually annotated and inserted into the list
        missedStride = [5, 6, 7, 17, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 41]
        missedStrideIndex = [1348-1, 1535-1, 1732-1, 3636-1, 3983-1, 4155-1, 4329-1, 4500-1, 4672-1, 4843-1, 
                             5013-1, 5180-1, 5517-1, 5687-1, 5854-1, 6017-1, 7947-1]
        for i in range(len(missedStride)):
            strideIndex = np.insert(strideIndex, missedStride[i], missedStrideIndex[i]) # Stride #i index is inserted
        # At MATLAB side some strides are manually corrected as follows
        if strideIndex[45] == 8831-1:
            strideIndex[45] = 8763-1
        if strideIndex[59] == 11758-1:
            strideIndex[59] = 11686-1
    elif expNumber == 40: # At MATLAB side missed strides are manually annotated and inserted into the list
        missedStride, missedStrideIndex = [3], [981-1]
        for i in range(len(missedStride)):
            strideIndex = np.insert(strideIndex, missedStride[i], missedStrideIndex[i]) # Stride #i index is inserted
    elif expNumber == 42: # At MATLAB side missed strides are manually annotated and inserted into the list
        missedStride, missedStrideIndex = [1], [545-1]
        for i in range(len(missedStride)):
            strideIndex = np.insert(strideIndex, missedStride[i], missedStrideIndex[i])
    # starting at exp#43, now any incorrect stride index is deleted first and then annotated from scratch instead of correction
    elif expNumber == 43: # there are 9 missed strides indexes and an incorrect stride index | incorrect stride index is deleted
        strideIndex = np.delete(strideIndex, 34) # remove incorrectly annotated (by PyShoe (LSTM)) stride index
        print(f"strideIndex[34] is removed for experiment #{expNumber} after MATLAB inspection.")
        missedStride, missedStrideIndex = [19, 20, 21, 23, 37, 41, 42, 43, 44], [3963, 4148, 4332, 4692, 7594, 8326, 8508, 8688, 8868]
        for i in range(len(missedStride)):
            strideIndex = np.insert(strideIndex, missedStride[i], missedStrideIndex[i]) # Stride #i index is inserted
    elif expNumber == 44: # 52/65: 2 stride indexes are removed and 15 missed stride indexes are added
        strideIndex = np.delete(strideIndex, 7) # remove incorrectly annotated (by PyShoe (LSTM)) stride index
        print(f"strideIndex[7] is removed for experiment #{expNumber} after MATLAB inspection.")
        strideIndex = np.delete(strideIndex, -3) # remove incorrectly annotated (by PyShoe (LSTM)) stride index
        print(f"strideIndex[-3] is removed for experiment #{expNumber} after MATLAB inspection.")
        missedStride = [8, 10, 11, 15, 25, 26, 27, 42, 43, 47, 54, 55, 57, 58, 63]
        missedStrideIndex = [1998, 2350, 2532, 3245, 5171, 5357, 5547, 8472, 8657, 9402, 10701, 10886, 11264, 11439, 12535]
        for i in range(len(missedStride)):
            strideIndex = np.insert(strideIndex, missedStride[i], missedStrideIndex[i]) # Stride #i index is inserted
    elif expNumber == 45: # 74/74 but 11th, 16th & 27th stride indexes are incorrectly detected by PyShoe LSTM
        strideIndex = np.delete(strideIndex, 11) # remove incorrectly annotated (by PyShoe (LSTM)) stride index
        print(f"strideIndex[11] is removed for experiment #{expNumber} after MATLAB inspection.")
        strideIndex = np.delete(strideIndex, 16-1) # remove incorrectly annotated (by PyShoe (LSTM)) stride index
        print(f"strideIndex[16] is removed for experiment #{expNumber} after MATLAB inspection.")
        strideIndex = np.delete(strideIndex, 27-2) # remove incorrectly annotated (by PyShoe (LSTM)) stride index
        print(f"strideIndex[27] is removed for experiment #{expNumber} after MATLAB inspection.")
        missedStride = [11, 16, 27]
        missedStrideIndex = [2992, 4166, 6800]
        for i in range(len(missedStride)):
            strideIndex = np.insert(strideIndex, missedStride[i], missedStrideIndex[i]) # Stride #i index is inserted
        
    ############################### CORRECTED PLOTS ########################################
    if expNumber in [32, 33, 34, 35, 36, 37, 38, 40, 42, 43, 44, 45]: # these experiments either needed stride index correction or introduction
        # Plot annotated stride indexes on IMU data, i.e., the magnitudes of acceleration and angular velocity
        plt.figure()
        if expNumber <= 40:
            plt.plot(timestamps, np.linalg.norm(imu_data.iloc[:, :3].values, axis=1), label=r'$\Vert\mathbf{a}\Vert$')
            plt.plot(timestamps, np.linalg.norm(imu_data.iloc[:, 3:].values, axis=1), label=r'$\Vert\mathbf{\omega}\Vert$')
            plt.scatter(timestamps[strideIndex], np.linalg.norm(imu_data.iloc[strideIndex, :3].values, axis=1), 
                        c='r', marker='x', label='Stride', zorder=3)
            plt.scatter(timestamps[strideIndex], np.linalg.norm(imu_data.iloc[strideIndex, 3:].values, axis=1), 
                        c='r', marker='x', zorder=3)
        else:
            plt.plot(timestamps, np.linalg.norm(imu_data[:, :3], axis=1), label=r'$\Vert\mathbf{a}\Vert$')
            plt.plot(timestamps, np.linalg.norm(imu_data[:, 3:], axis=1), label=r'$\Vert\mathbf{\omega}\Vert$')
            plt.scatter(timestamps[strideIndex], np.linalg.norm(imu_data[strideIndex, :3], axis=1), 
                        c='r', marker='x', label='Stride', zorder=3)
            plt.scatter(timestamps[strideIndex], np.linalg.norm(imu_data[strideIndex, 3:], axis=1), 
                        c='r', marker='x', zorder=3)
        plt.title(f'{base_filename} - Stride Detection on IMU Data')
        plt.xlabel('Time [s]'); plt.ylabel(r'Magnitude'); plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=1.5)
        plt.savefig(os.path.join(output_dir, f'{base_filename}_stride_detection_annotation.png'), dpi=600, bbox_inches='tight')
        plt.close()
    ################################################ SAVE TRAINING DATA for LLIO TRAINING ###############################################
    if extract_LLIO_training_data:
        # Stride coordinates (GCP) is the target in LLIO training yet we can save polar coordinates for the sake of completeness
        # combined_data = np.column_stack((displacements, heading_changes)) # Combine displacement and heading change data into one array
        print(f"strideIndex.shape = {strideIndex.shape} len(strideIndex) = {len(strideIndex)}")
        print(f"strideIndex = {strideIndex}")
        print(f"timestamps(strideIndex) = {timestamps[strideIndex]}")
        print(f"GCP.shape = {GCP.shape}")
        print(f"imu_data shape: {imu_data.shape}")
        
        if len(strideIndex)-1 == numberOfStrides:
            number_of_stride_wise_verified_experiments += 1
            logging.info(f"There are {len(strideIndex)-1}/{numberOfStrides} strides detected in experiment #{expNumber}.")
            combined_data = np.column_stack((strideIndex, timestamps[strideIndex], GCP[:,0], GCP[:,1]))
            combined_csv_filename = os.path.join(extracted_training_data_dir, f'LLIO_training_data/{base_filename}_strideIndex_timestamp_gcpX_gcpY.csv')
            np.savetxt(combined_csv_filename, combined_data, delimiter=',', header='strideIndex,timestamp,gcpX,gcpY', comments='')

        logging.info(f"Experiment #{expNumber} is annotated stride-wise (GCP) & going to be used in LLIO training/testing.")
        # compute stride distances and sum them up to get the traveled distance made in the current walk
        traveled_distance = np.sum(np.linalg.norm(np.diff(GCP, axis=0), axis=1))
        logging.info(f"Traveled distance is {traveled_distance:.3f} meters in experiment #{expNumber}.")
        traverse_time = timestamps[-1] - timestamps[0]
        logging.info(f"Travel time is {traverse_time:.3f} seconds in experiment #{expNumber}.")
        traveled_distances.append(traveled_distance) # sum all traveled distances cumulatively to get the total distance made in the experiments for LLIO training
        traverse_times.append(traverse_time) # sum all traversal times cumulatively to obtain the total experiment time for LLIO training
        
        # imu_data = imu_data.values
        # accX = imu_data[:,0]; accY = imu_data[:,1]; accZ = imu_data[:,2]
        # omegaX = imu_data[:,3]; omegaY = imu_data[:,4]; omegaZ = imu_data[:,5]

        # save stride indexes, timestamps, GCP stride coordinates and IMU data to mat file
        if expNumber <=40:
            sio.savemat(os.path.join(extracted_training_data_dir, f'LLIO_training_data/{base_filename}_LLIO.mat'), 
                        {'strideIndex': strideIndex, 'timestamps': timestamps, 'GCP': GCP, 'imu_data': imu_data.values, 
                        'pyshoeTrajectory': traj_list[-1][:,:2], 'euler_angles': x[:,6:], 'acc_n': acc_n,
                        'euler_angles_imu': euler_angles.values, 'thetaPyShoe': thetaPyShoe, 'thetaGCP': thetaGCP, 
                        'theta': theta, 'traveled_distance': traveled_distance, 'traverse_time': traverse_time, 
                        'GCP_wcf': GCP_wcf, 'traj_wcf': traj_wcf, 'reconstructed_traj_wcf': reconstructed_traj_wcf})
        else:
            sio.savemat(os.path.join(extracted_training_data_dir, f'LLIO_training_data/{base_filename}_LLIO.mat'), 
                        {'strideIndex': strideIndex, 'timestamps': timestamps, 'GCP': GCP, 'imu_data': imu_data, 
                        'pyshoeTrajectory': traj_list[-1][:,:2], 'euler_angles': x[:,6:], 'acc_n': acc_n,
                        'euler_angles_imu': euler_angles, 'thetaPyShoe': thetaPyShoe, 'thetaGCP': thetaGCP, 
                        'theta': theta, 'traveled_distance': traveled_distance, 'traverse_time': traverse_time, 
                        'GCP_wcf': GCP_wcf, 'traj_wcf': traj_wcf, 'reconstructed_traj_wcf': reconstructed_traj_wcf})
    else:
        # still save the stride indexes and the associated timestamps for further analysis in MATLAB side
        if expNumber <= 40:
            sio.savemat(os.path.join(extracted_training_data_dir, f'LLIO_nontraining_data/{base_filename}_LLIO_nontraining_data.mat'), 
                        {'strideIndex': strideIndex, 'timestamps': timestamps, 'imu_data': imu_data.values})
        else:
            sio.savemat(os.path.join(extracted_training_data_dir, f'LLIO_nontraining_data/{base_filename}_LLIO_nontraining_data.mat'), 
                        {'strideIndex': strideIndex, 'timestamps': timestamps, 'imu_data': imu_data})

total_distance, total_traverse_time = sum(traveled_distances), sum(traverse_times)
logging.info(f"===================================================================================================================")
logging.info(f"Total traveled distance in {number_of_stride_wise_verified_experiments} hallway experiments (to be used for LLIO training/test) is {total_distance:.3f} meters.")
logging.info(f"Total traverse time in {number_of_stride_wise_verified_experiments} hallway experiments (to be used for LLIO training/test) is {total_traverse_time:.3f}s = {total_traverse_time/60:.3f}mins.")
logging.info(f"===================================================================================================================")
logging.info(f"There are {expCount} experiments processed.")
logging.info("Processing complete for all files.")