import numpy as np
import matplotlib.pyplot as plt
from ins_tools.util import *
import ins_tools.visualize as visualize
from ins_tools.INS import INS
import os
import logging
import glob
import scipy.io as sio
from scipy.signal import medfilt
from scipy.linalg import orthogonal_procrustes

vicon_data_dir = 'data/vicon/processed/' # Directory containing VICON data files
vicon_data_files = glob.glob(os.path.join(vicon_data_dir, '*.mat'))

output_dir = "results/figs/vicon/" # Set up logging and output directory
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'output.log')
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

extracted_training_data_dir = "data/" # training data (imu, zv) for LSTM retraining & (displacement, heading change, stride indexes, timestamps) for LLIO training

# Following optimal ZV detectors are extracted from the mat files yet some "not optimal" detectors needed to be corrected manually
# 16th experiment: Despite showing MBGTD is the optimal detector in the mat file, VICON & ARED performs a lot better. Optimal detector is selected as ARED.
# 51st experiment: Optimal detector is changed from MBGTD to VICON
detector = ['shoe', 'ared', 'shoe', 'shoe', 'shoe', 'ared', 'shoe', 'shoe',
            'vicon', 'shoe', 'shoe', 'vicon', 'vicon', 'shoe', 'vicon', 'ared',
            'shoe', 'shoe', 'ared', 'vicon', 'shoe', 'shoe', 'vicon', 'shoe',
            'vicon', 'shoe', 'shoe', 'shoe', 'vicon', 'vicon', 'vicon', 'shoe',
            'shoe', 'vicon', 'vicon', 'shoe', 'shoe', 'shoe', 'shoe', 'ared',
            'shoe', 'shoe', 'ared', 'shoe', 'shoe', 'shoe', 'ared', 'shoe',
            'shoe', 'ared', 'vicon', 'shoe', 'vicon', 'shoe', 'shoe', 'vicon']

# thresholds of experiments (#16 and #51) are changed  - there are more inconsistencies in PyShoe dataset
thresh = [2750000, 0.1, 6250000, 15000000, 5500000, 0.08, 3000000, 3250000,
          0.02, 97500000, 20000000, 0.0825, 0.1, 30000000, 0.0625, 0.1250,
          92500000, 9000000, 0.015, 0.05, 3250000, 4500000, 0.1, 100000000,
          0.0725, 100000000, 15000000, 250000000, 0.0875, 0.0825, 0.0925, 70000000,
          525000000, 0.4, 0.375, 150000000, 175000000, 70000000, 27500000, 1.1,
          12500000, 65000000, 0.725, 67500000, 300000000, 650000000, 1, 4250000,
          725000, 0.0175, 0.0225, 42500000, 0.0675, 9750000, 3500000, 0.175]

# Function to calculate displacement and heading change between stride points
def calculate_displacement_and_heading(traj, strideIndex):
    displacements, heading_changes = [], []
    for j in range(1, len(strideIndex)):
        delta_position = traj[strideIndex[j], :2] - traj[strideIndex[j - 1], :2]
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

# this function is used in stride detection
def count_zero_to_one_transitions(arr):
    # Ensure the array is a NumPy array
    arr = np.asarray(arr)
    
    # Find the locations where transitions from 0 to 1 occur
    transitions = np.where((arr[:-1] == 0) & (arr[1:] == 1))[0]
    
    # Return the count and the indexes
    return len(transitions), transitions + 1  # Add 1 to get the index of the '1'

# Function to count one-to-zero transitions to determine stride indexes
def count_one_to_zero_transitions(zv):
    strides = np.where(np.diff(zv) < 0)[0] + 1
    return len(strides), strides

# Elimination of incorrect stride detections in the raw zv_opt
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

# Function to align trajectories using Procrustes analysis with scaling
def align_trajectories(traj_est, traj_gt):
    traj_est_2d = traj_est[:, :2]
    traj_gt_2d = traj_gt[:, :2]

    # Trim both trajectories to the same length
    min_length = min(len(traj_est_2d), len(traj_gt_2d))
    traj_est_trimmed = traj_est_2d[:min_length]
    traj_gt_trimmed = traj_gt_2d[:min_length]

    # Center the trajectories
    traj_est_mean = np.mean(traj_est_trimmed, axis=0)
    traj_gt_mean = np.mean(traj_gt_trimmed, axis=0)
    traj_est_centered = traj_est_trimmed - traj_est_mean
    traj_gt_centered = traj_gt_trimmed - traj_gt_mean

    # Compute scaling factor
    norm_est = np.linalg.norm(traj_est_centered)
    norm_gt = np.linalg.norm(traj_gt_centered)
    scale = norm_gt / norm_est

    traj_est_scaled = traj_est_centered * scale # Apply scaling
    R, _ = orthogonal_procrustes(traj_est_scaled, traj_gt_centered) # Compute the optimal rotation matrix
    traj_est_rotated = np.dot(traj_est_scaled, R) # Apply rotation
    traj_est_aligned = traj_est_rotated + traj_gt_mean # Translate back

    return traj_est_aligned, traj_gt_trimmed, scale

def rotate_trajectory(trajectory, theta):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return trajectory @ rotation_matrix.T

i = 0  # experiment index
count_training_exp = 0
# following two lines are used to run selected experiment results
# training_data_tag = [0]*56; training_data_tag[8] = 1 # we left off at exp#8
# training_data_tag are the experiments to be used in extracting displacement and heading change data for LLIO training
training_data_tag = [1, 1, 1, -1, 1, -1, 1, 0, 1, 1, -1, 1, 0, 1, 1, 1, 1, -1, 1, 1, # Exp 8 is excluded due to an extreme jump in a ZV region
                    1, 1, 1, 1, 1, 1, 0, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, # Exp 27 is excluded after further examination 
                    1, 1, -1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0] # Exp {49,53,54,56} are excluded after further examination
annotated_experiment_index = [4, 6, 11, 18, 27, 30, 32, 36, 38, 43, 49]
nGT = [22, 21, 21, 18, 26, 24, 18, 20, 29, 35, 29, 22, 30, 34, 24, 36, 20, 15, 10, 33, 
       22, 19, 13, 16, 17, 21, 20, 28, 18, 12, 13, 26, 34, 25, 24, 24, 43, 42, 15, 12, 
       13, 14, 24, 27, 25, 26, 0, 28, 13, 41, 33, 26, 16, 16, 11, 9] # number of actual strides
training_data_tag = [abs(x) for x in training_data_tag]
extract_bilstm_training_data = False # used to save csv files for zv and stride detection training
extract_LLIO_training_data = True # used to save csv files for LLIO SHS training - (displacement, heading change) and (stride indexes, timestamps)
# if sum(training_data_tag) == 56: # if total of 56 experiments are plotted (5 of them is not training data)
#     extract_bilstm_training_data = False # then do not write imu and zv data to file for BiLSTM training
traveled_distances = [] # to keep track of the traveled distances in each experiment for LLIO training data generation
traverse_times = [] # to keep track of experiment times and eventually total experiment time for LLIO training data generation
number_of_stride_wise_verified_experiments = 0 # detected stride points must be equal to the actual number

# Process each VICON room training data file
for file in vicon_data_files:
    if training_data_tag[i]:
        logging.info(f"===================================================================================================================")
        logging.info(f"Processing file {file}")
        data = sio.loadmat(file)

        # Remove the '.mat' suffix from the filename
        base_filename = os.path.splitext(os.path.basename(file))[0]

        # Extract the relevant columns
        imu_data = np.column_stack((data['imu'][:, :3], data['imu'][:, 3:6]))  # Accel and Gyro data
        timestamps = data['ts'][0]
        gt = data['gt']  # Ground truth from VICON dataset

        # Initialize INS object with correct parameters
        ins = INS(imu_data, sigma_a=0.00098, sigma_w=8.7266463e-5, T=1.0 / 200)

        logging.info(f"Processing {detector[i].upper()} detector for file {file}")
        ins.Localizer.set_gt(gt)  # Set the ground truth data required by 'vicon' detector
        ins.Localizer.set_ts(timestamps)  # Set the sampling time required by 'vicon' detector
        zv = ins.Localizer.compute_zv_lrt(W=5 if detector[i] != 'mbgtd' else 2, G=thresh[i], detector=detector[i])
        zv_lstm = ins.Localizer.compute_zv_lrt(W=0, G=0, detector='lstm')
        # zv_bilstm = ins.Localizer.compute_zv_lrt(W=0, G=0, detector='bilstm')
        # print(f"zv_bilstm = {zv_bilstm} \t len(zv_bilstm) = {len(zv_bilstm)}")
        x = ins.baseline(zv=zv)
        x_lstm = ins.baseline(zv=zv_lstm)
        # Align trajectories using Procrustes analysis with scaling
        # !!!DEACTIVATED!!! for LLIO training data extraction
        # aligned_x_lstm, aligned_gt, scale_lstm = align_trajectories(x_lstm, gt)

        # Apply filter to zero velocity detection results for stride detection corrections
        logging.info(f'Applying heuristic filter to optimal ZUPT detector {detector[i].upper()} generated ZV values for correct stride detection.')
        k = 75 # temporal window size for checking if detected strides are too close or not
        if i+1 == 54: # remove false positive by changing filter size for experiment 54
            k = 95
        elif i+1 == 9:
            k = 70
        # elif i+1 == 13: # not considered as part of training data due to sharp 180 degree changes in position
        #     k = 85
        zv_filtered, n, strideIndex = heuristic_zv_filter_and_stride_detector(zv, k)
        zv_lstm_filtered, n_lstm_filtered, strideIndexLSTMfiltered = heuristic_zv_filter_and_stride_detector(zv_lstm, k)
        gt = gt - gt[strideIndex[0],:] # subtract the initial point to make the GT start from 0 - important when evaluating predicted trajectories
        # zv_bilstm_filtered, n_bilstm_filtered, strideIndexBiLSTMfiltered = heuristic_zv_filter_and_stride_detector(zv_bilstm, k)
        # zv_filtered = medfilt(zv_filtered, 15)
        # n, strideIndex = count_one_to_zero_transitions(zv_filtered)
        # strideIndex = strideIndex - 1 # make all stride indexes the last samples of the respective ZUPT phase
        # strideIndex[0] = 0 # first sample is the first stride index
        # strideIndex = np.append(strideIndex, len(timestamps)-1) # last sample is the last stride index
        logging.info(f"Detected {n}/{nGT[i]} strides with (filtered) optimal detector {detector[i].upper()} in experiment {i+1}.")
        print(f"Detected {n_lstm_filtered}/{nGT[i]} strides with (filtered) LSTM ZV detector in experiment {i+1}.")
        # print(f"BiLSTM filtered ZV detector found {n_bilstm_filtered}/{nGT[i]} strides in the experiment {i+1}.")

        # Align the trajectory wrt the selected stride (GT data)
        # this is a rotation from the navigation coordinate frame to the world-fix coordinate frame
        strideAlign = 1; GT_align = strideAlign
        _, thetaPyShoe = calculate_displacement_and_heading(x_lstm[:, :2], strideIndex[np.array([0, strideAlign])])
        _, thetaGT = calculate_displacement_and_heading(gt[:, :2], strideIndex[np.array([0, GT_align])])
        theta = thetaPyShoe - thetaGT
        print(f"theta = {np.degrees(theta)} degrees for experiment #{i+1}.")

        # Apply the rotation to the GT data instead of PyShoe trajectory - this way we do not change IMU data
        # !!! NOTICE THAT GT DATA CHANGE AFTER THIS POINT !!!
        gt = gt[:, :2] # only interested in x-y positioning info
        gt = np.squeeze(rotate_trajectory(gt, theta))

        # Calculate displacement and heading changes between stride points for ground truth and PyShoe trajectory, respectively
        displacements_GT, heading_changes_GT = calculate_displacement_and_heading(gt[:, :2], strideIndex)
        displacements_PyShoe, heading_changes_PyShoe = calculate_displacement_and_heading(x_lstm[:, :2], strideIndex)
        # Reconstruct the trajectory from displacements and heading changes
        initial_position = gt[strideIndex[0], :2] # Get the starting point from the GT
        reconstructed_traj_GT = reconstruct_trajectory(displacements_GT, heading_changes_GT, initial_position)
        # reconstructed_traj_PyShoe = reconstruct_trajectory(displacements_PyShoe, heading_changes_PyShoe, initial_position)

        # reverse data in x direction to match with GCP and better illustration in the paper
        # x_lstm[:,0] = -x_lstm[:,0]
        # gt[:,0] = -gt[:,0]
        # reconstructed_traj[:,0] = -reconstructed_traj[:,0]

        # Plotting the reconstructed trajectory and the ground truth
        plt.figure()
        visualize.plot_topdown([reconstructed_traj_GT, gt[:, :2]], title=f"Exp#{i+1} ({base_filename}) - {detector[i].upper()}", 
                               legend=[f'GT (stride-wise) - {n}/{nGT[i]}', 'GT (sample-wise)'])  
        plt.scatter(reconstructed_traj_GT[:, 0], reconstructed_traj_GT[:, 1], c='b', marker='o')
        plt.savefig(os.path.join(output_dir, f'trajectory_exp_{i+1}.png'), dpi=600, bbox_inches='tight')
        plt.close()

        # Plot LSTM trajectory results
        plt.figure()
        visualize.plot_topdown([x_lstm, gt[:, :2]], title=f"Exp#{i+1} ({base_filename}) - PyShoe (LSTM)", legend=['PyShoe (LSTM)', 'GT'])
        plt.scatter(x_lstm[strideIndexLSTMfiltered, 0], x_lstm[strideIndexLSTMfiltered, 1], c='b', marker='o')
        plt.savefig(os.path.join(output_dir, f'trajectory_exp_{i+1}_lstm_ins.png'), dpi=600, bbox_inches='tight')
        plt.close()

        # Plotting ZV raw and filtered signals an detected stride indexes
        plt.figure()
        plt.plot(timestamps[:len(zv)], zv, label='Raw ZV signal')
        plt.plot(timestamps[:len(zv_filtered)], zv_filtered, label='Filtered ZV signal')
        plt.scatter(timestamps[strideIndex], zv_filtered[strideIndex], c='r', marker='x', label='Stride index')
        plt.title(f'Exp#{i+1} ({base_filename}) {n}/{nGT[i]} strides detected ({detector[i].upper()})')
        plt.xlabel('Time [s]'); plt.ylabel('ZV label')
        plt.grid(True, which='both', linestyle='--', linewidth=1.5)
        plt.legend()
        plt.yticks([0,1])
        plt.savefig(os.path.join(output_dir, f'zv_labels_exp_{i+1}.png'), dpi=600, bbox_inches='tight')
        plt.close()

        # Plotting the zero velocity detection for LSTM filtered data
        plt.figure()
        plt.plot(timestamps[:len(zv_lstm)], zv_lstm, label='Raw ZV signal')
        plt.plot(timestamps[:len(zv_lstm_filtered)], zv_lstm_filtered, label='Filtered ZV signal')
        plt.scatter(timestamps[strideIndexLSTMfiltered], zv_lstm_filtered[strideIndexLSTMfiltered], c='r', marker='x')
        plt.title(f'Exp#{i+1} ({base_filename}) {n_lstm_filtered}/{nGT[i]} strides detected ({"lstm".upper()})')
        plt.xlabel('Time [s]'); plt.ylabel('ZV label')
        plt.grid(True, which='both', linestyle='--', linewidth=1.5)
        plt.legend()
        plt.yticks([0,1])
        plt.savefig(os.path.join(output_dir, f'zv_labels_exp_{i+1}_lstm.png'), dpi=600, bbox_inches='tight')
        plt.close()

        # Plot stride indexes on IMU data, i.e., the magnitudes of acceleration and angular velocity
        plt.figure()
        plt.plot(timestamps, np.linalg.norm(imu_data[:, :3], axis=1), label=r'$\Vert\mathbf{a}\Vert$')
        plt.plot(timestamps, np.linalg.norm(imu_data[:, 3:], axis=1), label=r'$\Vert\mathbf{\omega}\Vert$')
        plt.scatter(timestamps[strideIndex], np.linalg.norm(imu_data[strideIndex, :3], axis=1), 
                    c='r', marker='x', label='Stride index', zorder=3)
        plt.scatter(timestamps[strideIndex], np.linalg.norm(imu_data[strideIndex, 3:], axis=1), 
                    c='r', marker='x', zorder=3)
        plt.title(f'Exp#{i+1} ({base_filename}) - Stride Detection on IMU Data')
        plt.xlabel('Time [s]'); plt.ylabel(r'Magnitude'); plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=1.5)
        plt.savefig(os.path.join(output_dir, f'stride_detection_exp_{i+1}.png'), dpi=600, bbox_inches='tight')
        plt.close()

        # while some experiments are excluded due to being non bipedal locomotion motion (i.e., crawling experiments)
        # some other bipedal locomotion experimental data requires correction for some ZV labels and stride detections 
        # correction indexes are extracted manually (see detect_missed_strides.m for details)
        if i+1 == 4: # Experiment needs ZV correction in 10th stride (8th from the end)
            zv_filtered[2800:2814] = 1 # correction indexes for the missed stride
            zv[2800:2814] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
        elif i+1 == 6: # Experiment needs ZV correction in 9th stride (start counting the strides to south direction)
            zv_filtered[2544:2627] = 1 # correction indexes for the missed stride
            zv[2544:2627] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
        elif i+1 == 11: # Experiment needs ZV correction in 7th stride
            zv_filtered[2137:2162] = 1 # correction indexes for the missed stride
            zv[2137:2162] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
        elif i+1 == 18: # Experiment needs ZV correction in 7th stride
            zv_filtered[1882:1940] = 1 # correction indexes for the missed stride
            zv[1882:1940] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
        elif i+1 == 27: # Experiment needs ZV correction in {9, 16, 17, 18}th strides (first 3 by VICON and the last one by MBGTD)
            zv_filtered[1816:1830] = 1 
            zv_filtered[2989:3002] = 1
            zv_filtered[3154:3168] = 1
            zv_filtered[3329-3:3329+3] = 1
            zv[1816:1830] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
            zv[2989:3002] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
            zv[3154:3168] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
            zv[3329-3:3329+3] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
        elif i+1 == 30: # Experiment needs ZV correction in {2, 10}th strides (both detected by SHOE (supplementary) detector)
            zv_filtered[620:630] = 1
            zv_filtered[1785:1790] = 1
            zv[620:630] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
            zv[1785:1790] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
        elif i+1 == 32: # 32nd experiment: missed strides {9, 11, 20}. First two are recovered by VICON but the last one needed to be introduced by manual annotation.
            zv_filtered[1851-3:1851+3] = 1
            zv_filtered[2138:2146] = 1
            zv_filtered[3997:4004] = 1 # this is manual annotation
            zv[1851-3:1851+3] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
            zv[2138:2146] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
            zv[3997:4004] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
        elif i+1 == 36: # 36th experiment: 7th stride is missed
            zv_filtered[1864:1890] = 1
            zv[1864:1890] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
        elif i+1 == 38: # 38th experiment: missed strides {3,27,33}. All three strides are recovered by VICON.
            zv_filtered[874-3:874+3] = 1 # stride 3
            zv_filtered[4520-3:4520+3] = 1 # stride 27
            zv_filtered[5410:5421] = 1 # stride 33
            zv[874-3:874+3] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
            zv[4520-3:4520+3] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
            zv[5410:5421] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
        elif i+1 == 43: # 43rd experiment: missed strides {3, 14, 16}. All three strides are recovered by VICON.
            zv_filtered[905:944] = 1 # stride 3
            zv_filtered[2613:2662] = 1 # stride 14
            zv_filtered[2925:2974] = 1 # stride 16
            zv[905:944] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
            zv[2613:2662] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
            zv[2925:2974] = 1 # zv is the target data in LSTM retraining for robust ZUPT aided INS
        # EXPERIMENT 49 IS LATER EXCLUDED AFTER EXAMINING TOTAL TRAVERSE TIME INFORMATION
        elif i+1 == 49: # 49th experiment: Detected last 4 strides are left outside of the experiment as they do not cause any x-y motion.
            zv_filtered = zv_filtered[:13070] # data cropped to exclude the last 4 strides
            zv = zv[:13070] # zv is the target data in LSTM retraining for robust ZUPT aided INS
            reconstructed_traj_GT = reconstructed_traj_GT[:13070]
            gt = gt[:13070]
            timestamps = timestamps[:13070]
            imu_data = imu_data[0:13070,:]

        # PRODUCE CORRECTED ZV and TRAJECTORY PLOTS
        if i+1 in annotated_experiment_index:
            # Apply filter to zero velocity detection
            logging.info(f"Applying stride detection to the combined zero velocity detection results for experiment {i+1}.")
            zv_filtered, n, strideIndex = heuristic_zv_filter_and_stride_detector(zv_filtered, 1)
            logging.info(f"Detected {n}/{nGT[i]} strides detected with the combined ZV detector in the experiment {i+1}.")

            # Calculate displacement and heading changes between ground truth values of stride points
            displacements_GT, heading_changes_GT = calculate_displacement_and_heading(gt[:, :2], strideIndex)

            # Reconstruct the trajectory from displacements and heading changes
            initial_position = gt[strideIndex[0], :2]  # Starting point from the GT trajectory
            reconstructed_traj_GT = reconstruct_trajectory(displacements_GT, heading_changes_GT, initial_position)

            # Plotting the reconstructed trajectory and the ground truth without stride indices
            plt.figure()
            visualize.plot_topdown([reconstructed_traj_GT, gt[:, :2]], title=f"Exp#{i+1} ({base_filename}) - Combined",
                                legend=[f'GT (stride-wise) - {n}/{nGT[i]}', 'GT (sample-wise)']) 
            plt.scatter(reconstructed_traj_GT[:, 0], reconstructed_traj_GT[:, 1], c='b', marker='o')
            plt.savefig(os.path.join(output_dir, f'trajectory_exp_{i+1}_corrected.png'), dpi=600, bbox_inches='tight')
            plt.close()

            # Plotting the zero velocity detection for the combined ZV detector without stride indices
            plt.figure()
            plt.plot(timestamps, zv, label='Raw ZV signal')
            plt.plot(timestamps, zv_filtered, label='Filtered ZV signal')
            plt.scatter(timestamps[strideIndex], zv_filtered[strideIndex], c='r', marker='x')
            plt.title(f'Exp#{i+1} ({base_filename}) {n}/{nGT[i]} strides detected (combined)')
            plt.xlabel('Time [s]'); plt.ylabel('ZV label')
            plt.grid(True, which='both', linestyle='--', linewidth=1.5)
            plt.legend(); plt.yticks([0,1])
            plt.savefig(os.path.join(output_dir, f'zv_labels_exp_{i+1}_corrected.png'), dpi=600, bbox_inches='tight')
            plt.close()

            # Plot stride indexes on IMU data, i.e., the magnitudes of acceleration and angular velocity
            plt.figure()
            plt.plot(timestamps, np.linalg.norm(imu_data[:, :3], axis=1), label=r'$\Vert\mathbf{a}\Vert$')
            plt.plot(timestamps, np.linalg.norm(imu_data[:, 3:], axis=1), label=r'$\Vert\mathbf{\omega}\Vert$')
            plt.scatter(timestamps[strideIndex], np.linalg.norm(imu_data[strideIndex, :3], axis=1), 
                        c='r', marker='x', label='Stride index', zorder=3)
            plt.scatter(timestamps[strideIndex], np.linalg.norm(imu_data[strideIndex, 3:], axis=1), 
                        c='r', marker='x', zorder=3)
            plt.title(f'Exp#{i+1} ({base_filename}) - Stride Annotation on IMU Data')
            plt.xlabel('Time [s]'); plt.ylabel('Magnitude'); plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=1.5)
            plt.savefig(os.path.join(output_dir, f'stride_detection_exp_{i+1}_corrected.png'), dpi=600, bbox_inches='tight')
            plt.close()
        
        #################### SAVE TRAINING DATA RIGHT AT THIS SPOT for LSTM RETRAINING #################
        if extract_bilstm_training_data:
            # Combine IMU data and ZV data into one array
            combined_data = np.column_stack((timestamps, imu_data, zv))

            # Save the combined IMU and ZV data to a CSV file
            combined_csv_filename = os.path.join(extracted_training_data_dir, f'LSTM_ZV_detector_training_data/{base_filename}_imu_zv.csv')

            np.savetxt(combined_csv_filename, combined_data, delimiter=',',
                    header='t,ax,ay,az,wx,wy,wz,zv', comments='')
        #################### SAVE TRAINING DATA for LLIO TRAINING #################
        if extract_LLIO_training_data:
            number_of_stride_wise_verified_experiments += 1
            # Stride coordinates (GCP) is the target in Gradient Boosting (LLIO) training yet we can save polar coordinates for the sake of completeness
            # combined_data = np.column_stack((displacements, heading_changes)) # Combine displacement and heading change data into one array
            GCP = gt[strideIndex, :2] # extract the stride coordinates (GCP) from the ground truth data
            print(f"strideIndex.shape = {strideIndex.shape}")
            print(f"GCP.shape = {GCP.shape}")
            print(f"imu_data shape: {imu_data.shape}")
            # accX = imu_data[:,0]; accY = imu_data[:,1]; accZ = imu_data[:,2]
            # omegaX = imu_data[:,3]; omegaY = imu_data[:,4]; omegaZ = imu_data[:,5]

            # Save the combined data to a CSV file
            combined_data = np.column_stack((strideIndex, timestamps[strideIndex], GCP[:,0], GCP[:,1]))
            combined_csv_filename = os.path.join(extracted_training_data_dir, f'LLIO_training_data/{base_filename}_strideIndex_timestamp_gcpX_gcpY.csv')
            np.savetxt(combined_csv_filename, combined_data, delimiter=',', header='strideIndex,timestamp,gcpX,gcpY', comments='')

            # Save stride indexes, timestamps, GCP stride coordinates and IMU data to mat file
            sio.savemat(os.path.join(extracted_training_data_dir, f'LLIO_training_data/{base_filename}_LLIO.mat'),
                        {'strideIndex': strideIndex, 'timestamps': timestamps[strideIndex], 'GCP': GCP, 'imu_data': imu_data, 
                         'timestamps_all': timestamps, 'euler_angles': x_lstm[:,6:]})
            
        logging.info(f"Experiment #{i+1} is annotated stride-wise & going to be used in LLIO training/testing.")
        # compute stride distances and sum them up to get the traveled distance made in the current walk
        traveled_distance = np.sum(np.linalg.norm(np.diff(GCP, axis=0), axis=1))
        logging.info(f"Traveled distance is {traveled_distance:.3f} meters in experiment #{i+1}.")
        traverse_time = timestamps[-1] - timestamps[0]
        logging.info(f"Travel time is {traverse_time:.3f} seconds in experiment #{i+1}.")
        traveled_distances.append(traveled_distance) # sum all traveled distances cumulatively to get the total distance made in the experiments for LLIO training
        traverse_times.append(traverse_time) # sum all traversal times cumulatively to obtain the total experiment time for LLIO training
        
        count_training_exp += 1
    else:
        logging.info(f"===================================================================================================================")
        logging.info(f"Processing file {file}")
        print(f"Experiment {i+1} data is not considered as bipedal locomotion data for LLIO training.".upper())
        # 13th experiment shows a lot of 180° turns, which causes multiple ZV phase and stride detections during the turns.
        # Labeled as 0, i.e., non bi-pedal locomotion data, temporarily. It will be included in future for further research. 
        # 20th experiment: The pedestrian stops in every 5 or 6 strides for a while but it is a valid bipedal locomotion data (confirmed by GCetin's ML code)
        # 47th experiment is a crawling experiment so it is not a bipedal locomotion data.
        # 48th experiment shows a lot of 180° turns, which causes multiple ZV phase and stride detections during the turns.
        # Labeled as 0, i.e., non bi-pedal locomotion data, temporarily. It will be included in future for further research.
        # 50th experiment: shows a lot of 180° turns, which causes multiple ZV phase and stride detections during the turns.
        # Labeled as 0, i.e., non bi-pedal locomotion data, temporarily. It will be included in future for further research.
        # 55 needs cropping at the beginning or the end - left out of the training dataset yet will be considered as training data in future
         
    i += 1  # Move to the next experiment

total_distance, total_traverse_time = sum(traveled_distances), sum(traverse_times)
logging.info(f"===================================================================================================================")
logging.info(f"Total traveled distance in {number_of_stride_wise_verified_experiments} VICON room experiments (to be used for LLIO training/test) is {total_distance:.3f} meters.")
logging.info(f"Total experiment time in {number_of_stride_wise_verified_experiments} VICON room experiments (to be used for LLIO training/test) is {total_traverse_time:.3f}s = {total_traverse_time/60:.3f}mins.")
logging.info(f"===================================================================================================================")
logging.info("Processing complete for all files.")
