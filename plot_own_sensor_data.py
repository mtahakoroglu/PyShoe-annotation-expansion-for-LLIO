# version 1 uses heuristic filter to detect strides
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
from scipy.signal import medfilt  # Import median filter

# Directory of sensor data files
sensor_data_dir = 'data/own'
sensor_data_files = glob.glob(os.path.join(sensor_data_dir, 'SensorConnectData_*.csv'))

# Set up logging
output_dir = "results/figs/own"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'output.log')
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Detectors and labels
det_list = ['ared', 'shoe', 'lstm']
# Define thresholds for each detector
thresh_list = [0.55, 8.5e7, 0] 
W_list = [5, 5, 0] # 5, 5, 5, 0
legend = ['ARED', 'SHOE', 'LSTM']

# gravity contant
g = 9.8029
# Plotting the results and zero velocity detections
# processing zero velocity labels to turn a sampling frequency driven system into gait driven system (stride and heading system)
numberOfStrides = [75, 24, 28, 28, 65] # we counted respective number of strides during the experiments
# different size kernels for correct detection of number of strides taken by the pedestrian
# kernel sizes are determined automatically in a brute-force search yet the correct values are noted here for time-saving purposes
# k = [21, 55]
j = 0 # experiment index
# Process each sensor data file
for file in sensor_data_files:
    logging.info(f"Processing file: {file}")
    sensor_data = pd.read_csv(file)

    # Extract the relevant columns and multiply accelerations by gravity
    timestamps = sensor_data['Time'].values
    imu_data = sensor_data[
        [
            'inertial-6253.76535:scaledAccelX',
            'inertial-6253.76535:scaledAccelY',
            'inertial-6253.76535:scaledAccelZ',
            'inertial-6253.76535:scaledGyroX',
            'inertial-6253.76535:scaledGyroY',
            'inertial-6253.76535:scaledGyroZ'
        ]
    ].copy()

    imu_data[['inertial-6253.76535:scaledAccelX',
              'inertial-6253.76535:scaledAccelY',
              'inertial-6253.76535:scaledAccelZ']] *= g

    # Initialize INS object with correct parameters
    ins = INS(imu_data.values, sigma_a=0.00098, sigma_w=8.7266463e-5, T=1.0/200)

    traj_list = []
    zv_list = []

    for i, detector in enumerate(det_list):
        logging.info(f"Processing {detector} detector for file: {file}")
        zv = ins.Localizer.compute_zv_lrt(W=W_list[i], G=thresh_list[i], detector=detector)
        x = ins.baseline(zv=zv)
        traj_list.append(x)
        zv_list.append(zv)

    # Remove the '.csv' extension from the filename
    base_filename = os.path.splitext(os.path.basename(file))[0]

    for i, zv in enumerate(zv_list):
        logging.info(f"Plotting zero velocity detection for {det_list[i]} detector for file: {file}")
        plt.figure()
        plt.plot(timestamps[:len(zv)], zv)
        plt.title(f'{legend[i]} - {base_filename}')
        plt.xlabel('Time')
        plt.ylabel('Zero Velocity')
        plt.savefig(os.path.join(output_dir, f'zv_{det_list[i]}_{base_filename}.png'), dpi=800, bbox_inches='tight')
        # Apply a heuristic filter to adaptively detected zero velocity labels (via LSTM) to eliminate undesired jumps & achieve correct stride detection
        if det_list[i] == 'lstm':
            logging.info(f"Plotting zero velocity detection for heuristically filtered {det_list[i]} detector for file: {file}")
            k = 45 # temporal window size for checking if detected strides are too close or not
            print(f"zv size: {zv.shape}")
            print(f"zv content: {zv}")
            zv_lstm_filtered, n, strideIndex = heuristic_zv_filter_and_stride_detector(zv, k)
            if n == numberOfStrides[j]:
                print("Number of strides and the indices are correctly detected.")
                print(f"There are {n}/{numberOfStrides[j]} strides detected in the experiment.")
                print(f"The stride indices are {strideIndex}")
            plt.figure()
            plt.plot(timestamps[:len(zv_lstm_filtered)], zv_lstm_filtered)
            plt.scatter(timestamps[strideIndex], zv_lstm_filtered[strideIndex], c='r', marker='x')
            plt.title(f'{legend[i]} filtered ({n}/{numberOfStrides[j]}) - {base_filename}')
            plt.xlabel('Time [s]')
            plt.ylabel('Zero Velocity')
            plt.savefig(os.path.join(output_dir, f'zv_{det_list[i]}_heuristically_filtered_{base_filename}.png'), dpi=800, bbox_inches='tight')

    plt.figure()
    visualize.plot_topdown(traj_list, title=f'{base_filename}', legend=legend)
    plt.scatter(-traj_list[2][strideIndex, 0], traj_list[2][strideIndex, 1], c='r', marker='x')
    plt.savefig(os.path.join(output_dir, f'{base_filename}.png'), dpi=600, bbox_inches='tight')

    plt.figure()
    plt.plot(-traj_list[2][strideIndex, 0], traj_list[2][strideIndex, 1], c='r', marker='x', label=f"adaptive ZUPT (LSTM) {base_filename}")
    plt.legend(fontsize=15)
    plt.ylabel('y [m]', fontsize=22)
    plt.xlabel('x [m]', fontsize=22)
    plt.title(f'LSTM stride detections w/ filtering ({n}/{numberOfStrides[j]}) - {base_filename}', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.grid(True, which='both', linestyle='--', linewidth=1.5)
    plt.savefig(os.path.join(output_dir, f'{base_filename}_LSTM_ZUPT_filtered_strides.png'), dpi=800, bbox_inches='tight')

    j = j+1
    print(f"There are {j} experiments conducted.")
    
    # plt.figure()
    # for traj in traj_list:
    #     plt.plot(timestamps[:len(traj)], traj[:, 2])  # Ensure timestamps and trajectory lengths match
    # plt.title(f'Vertical Trajectories - {base_filename}')
    # plt.xlabel('Time')
    # plt.ylabel('Z Position')
    # plt.legend(legend)
    # plt.savefig(os.path.join(output_dir, f'vertical_{base_filename}.png'))

logging.info("Processing complete for all files.")
