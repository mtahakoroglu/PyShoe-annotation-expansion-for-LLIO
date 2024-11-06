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

vicon_data_dir = 'data/vicon/processed/' # Directory containing Vicon data files
vicon_data_files = glob.glob(os.path.join(vicon_data_dir, '*.mat'))

output_dir = "results/figs/vicon_corrected/" # Set up logging and output directory
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'output.log')
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

extracted_training_data_dir = "data/" # training data (imu, zv) for LSTM retraining & (displacement, heading change, stride indexes, timestamps) for LLIO training

# 16th experiment: Despite showing MBGTD is the optimal detector in the mat file, VICON & ARED performs a lot better. Optimal detector is selected as ARED.
# 51st experiment: Optimal detector is changed from MBGTD to VICON
detector = ['shoe', 'ared', 'shoe', 'shoe', 'shoe', 'ared', 'shoe', 'shoe',
            'vicon', 'shoe', 'shoe', 'vicon', 'vicon', 'shoe', 'vicon', 'ared',
            'shoe', 'shoe', 'ared', 'vicon', 'shoe', 'shoe', 'vicon', 'shoe',
            'vicon', 'shoe', 'shoe', 'shoe', 'vicon', 'vicon', 'vicon', 'shoe',
            'shoe', 'vicon', 'vicon', 'shoe', 'shoe', 'shoe', 'shoe', 'ared',
            'shoe', 'shoe', 'ared', 'shoe', 'shoe', 'shoe', 'ared', 'shoe',
            'shoe', 'ared', 'vicon', 'shoe', 'vicon', 'shoe', 'shoe', 'vicon']
# corresponding thresholds are changed (#16 and #51)
thresh = [2750000, 0.1, 6250000, 15000000, 5500000, 0.08, 3000000, 3250000,
          0.02, 97500000, 20000000, 0.0825, 0.1, 30000000, 0.0625, 0.1250,
          92500000, 9000000, 0.015, 0.05, 3250000, 4500000, 0.1, 100000000,
          0.0725, 100000000, 15000000, 250000000, 0.0875, 0.0825, 0.0925, 70000000,
          525000000, 0.4, 0.375, 150000000, 175000000, 70000000, 27500000, 1.1,
          12500000, 65000000, 0.725, 67500000, 300000000, 650000000, 1, 4250000,
          725000, 0.0175, 0.0225, 42500000, 0.0675, 9750000, 3500000, 0.175]

# Function to calculate displacement and heading change between stride points
def calculate_displacement_and_heading(gt, strideIndex):
    displacements = []
    heading_changes = []
    for j in range(1, len(strideIndex)):
        delta_position = gt[strideIndex[j], :2] - gt[strideIndex[j - 1], :2]
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


i = 0  # experiment index
count_training_exp = 0
# following two lines are used to run selected experiment results
# training_data_tag = [1]*55
# training_data_tag.append(1)
# training_data_tag are the experiments to be used in extracting displacement and heading change data for LLIO training
training_data_tag = [1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 0, 1, 1, 1, 1, -1, 1, 1, 
                    1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 
                    1, 1, -1, 1, 1, 1, 0, 0, -1, 0, 1, 1, 1, 1, 0, 1]
corrected_data_index = [4, 6, 11, 18, 27, 30, 32, 36, 38, 43, 49] # corrected experiment indexes
nGT = [22, 21, 21, 18, 26, 24, 18, 20, 28, 35, 29, 22, 30, 34, 24, 36, 20, 15, 10, 33, 
       22, 19, 13, 16, 17, 21, 20, 28, 18, 12, 13, 26, 34, 25, 24, 24, 43, 42, 15, 12, 
       13, 14, 24, 27, 25, 26, 0, 28, 13, 41, 33, 26, 16, 16, 11, 9] # number of actual strides
training_data_tag = [abs(x) for x in training_data_tag]
extract_bilstm_training_data = False # used to save csv files for zv and stride detection training
extract_LLIO_training_data = False # used to save csv files for LLIO SHS training - (displacement, heading change) and (stride indexes, timestamps)
# if sum(training_data_tag) == 56: # if total of 56 experiments are plotted (5 of them is not training data)
#     extract_bilstm_training_data = False # then do not write imu and zv data to file for BiLSTM training

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
        gt = data['gt']  # Ground truth from Vicon dataset

        # Initialize INS object with correct parameters
        ins = INS(imu_data, sigma_a=0.00098, sigma_w=8.7266463e-5, T=1.0 / 200)

        logging.info(f"Processing {detector[i]} detector for file {file}")
        ins.Localizer.set_gt(gt)  # Set the ground truth data required by 'vicon' detector
        ins.Localizer.set_ts(timestamps)  # Set the sampling time required by 'vicon' detector
        zv = ins.Localizer.compute_zv_lrt(W=5 if detector[i] != 'mbgtd' else 2, G=thresh[i], detector=detector[i])
        zv_lstm = ins.Localizer.compute_zv_lrt(W=0, G=0, detector='lstm')
        # zv_bilstm = ins.Localizer.compute_zv_lrt(W=0, G=0, detector='bilstm')
        # print(f"zv_bilstm = {zv_bilstm} \t len(zv_bilstm) = {len(zv_bilstm)}")
        x = ins.baseline(zv=zv)
        x_lstm = ins.baseline(zv=zv_lstm)
        # Align trajectories using Procrustes analysis with scaling
        aligned_x_lstm, aligned_gt, scale_lstm = align_trajectories(x_lstm, gt)

        # Apply filter to zero velocity detection results for stride detection corrections
        logging.info(f'Applying heuristic filter to optimal ZUPT detector {detector[i].upper()} for correct stride detection.')
        k = 75 # temporal window size for checking if detected strides are too close or not
        if i+1 == 54: # remove false positive by changing filter size for experiment 54
            k = 95
        # elif i+1 == 13: # not considered as part of training data due to sharp 180 degree changes in position
        #     k = 85
        zv_filtered, n, strideIndex = heuristic_zv_filter_and_stride_detector(zv, k)
        zv_lstm_filtered, n_lstm_filtered, strideIndexLSTMfiltered = heuristic_zv_filter_and_stride_detector(zv_lstm, k)
        # zv_bilstm_filtered, n_bilstm_filtered, strideIndexBiLSTMfiltered = heuristic_zv_filter_and_stride_detector(zv_bilstm, k)
        # zv_filtered = medfilt(zv_filtered, 15)
        # n, strideIndex = count_one_to_zero_transitions(zv_filtered)
        # strideIndex = strideIndex - 1 # make all stride indexes the last samples of the respective ZUPT phase
        # strideIndex[0] = 0 # first sample is the first stride index
        # strideIndex = np.append(strideIndex, len(timestamps)-1) # last sample is the last stride index
        logging.info(f"Detected {n}/{nGT[i]} strides with (filtered) optimal detector {detector[i].upper()} in experiment {i+1}.")
        print(f"Detected {n_lstm_filtered}/{nGT[i]} strides with (filtered) LSTM ZV detector in experiment {i+1}.")
        # print(f"BiLSTM filtered ZV detector found {n_bilstm_filtered}/{nGT[i]} strides in the experiment {i+1}.")
        # Calculate displacement and heading changes between stride points based on ground truth
        displacements, heading_changes = calculate_displacement_and_heading(gt[:, :2], strideIndex)
        # Reconstruct the trajectory from displacements and heading changes
        initial_position = gt[strideIndex[0], :2]  # Starting point from the GT trajectory
        reconstructed_traj = reconstruct_trajectory(displacements, heading_changes, initial_position)

        # reverse data in x direction to match with GCP and better illustration in the paper
        aligned_x_lstm[:,0] = -aligned_x_lstm[:,0]
        aligned_gt[:,0] = -aligned_gt[:,0]
        reconstructed_traj[:,0] = -reconstructed_traj[:,0]

        # Plotting the reconstructed trajectory and the ground truth without stride indices
        plt.figure()
        visualize.plot_topdown([reconstructed_traj, gt[:, :2]], title=f"Exp#{i+1} ({base_filename}) - {detector[i].upper()}", 
                               legend=['GT (stride & heading)', 'GT (sample-wise)'])
        # to visualize selected stides from the experiment of interest, change the parameters below
        # if i+1==49:
        #     # plt.plot(gt[-5:,0], gt[-5:,1], c='r')
        #     # plt.scatter(-reconstructed_traj[-3:, 0], reconstructed_traj[-5:, 1], c='b', marker='x')
        #     plt.scatter(-reconstructed_traj[:13, 0], reconstructed_traj[:13, 1], c='b', marker='x')
        #     plt.scatter(-reconstructed_traj[14, 0], reconstructed_traj[14, 1], c='g', marker='s')
        #     # hms = 34 # "how many strides" to show from the beginning (including the initial stride)
        #     plt.plot(-reconstructed_traj[0:14, 0], reconstructed_traj[0:14, 1], c='r')
        #     # plt.scatter(-reconstructed_traj[0, 0], reconstructed_traj[0, 1], c='b', marker='s')
        #     # plt.scatter(-reconstructed_traj[0:hms-1, 0], reconstructed_traj[0:hms-1, 1], c='b', marker='x')
        #     # plt.scatter(-reconstructed_traj[hms-1, 0], reconstructed_traj[hms-1, 1], c='g', marker='o')
        # else:    
        plt.scatter(reconstructed_traj[:, 0], reconstructed_traj[:, 1], c='b', marker='o')
        plt.savefig(os.path.join(output_dir, f'trajectory_exp_{i+1}.png'), dpi=600, bbox_inches='tight')

        # Plot LSTM trajectory results
        plt.figure()
        visualize.plot_topdown([aligned_x_lstm, aligned_gt[:, :2]], title=f"Exp#{i+1} ({base_filename}) - LSTM", 
                               legend=['LSTM INS', 'GT'])
        plt.scatter(aligned_x_lstm[strideIndexLSTMfiltered, 0], aligned_x_lstm[strideIndexLSTMfiltered, 1], c='b', marker='o')
        plt.savefig(os.path.join(output_dir, f'trajectory_exp_{i+1}_lstm_ins.png'), dpi=600, bbox_inches='tight')

        # # plotting vertical trajectories
        # plt.figure()
        # plt.plot(timestamps[:len(gt)], gt[:, 2], label='GT (sample-wise)')  # Plot GT Z positions
        # plt.plot(timestamps[:len(reconstructed_traj)], reconstructed_traj[:, 1],
        #         label='Stride & Heading')  # Plot reconstructed Z positions (use Y axis for visualization)
        # plt.title(f'Vertical Trajectories - {base_filename} - ZUPT detector={detector[i]} for exp#{i+1}')
        # plt.grid(True, which='both', linestyle='--', linewidth=1.5)
        # plt.xlabel('Time [s]')
        # plt.ylabel('Z Position')
        # plt.legend()
        # plt.savefig(os.path.join(output_dir, f'vertical_{base_filename}.png'), dpi=600, bbox_inches='tight')

        # Plotting the zero velocity detection for filtered data without stride indices
        plt.figure()
        plt.plot(timestamps[:len(zv)], zv, label='Raw')
        plt.plot(timestamps[:len(zv_filtered)], zv_filtered, label='Filtered')
        plt.scatter(timestamps[strideIndex], zv_filtered[strideIndex], c='r', marker='x')
        plt.title(f'Exp#{i+1} ({base_filename}) {n}/{nGT[i]} strides detected ({detector[i].upper()})')
        plt.xlabel('Time [s]')
        plt.ylabel('Zero Velocity')
        plt.grid(True, which='both', linestyle='--', linewidth=1.5)
        plt.legend()
        plt.yticks([0,1])
        plt.savefig(os.path.join(output_dir, f'zv_labels_exp_{i+1}.png'), dpi=600, bbox_inches='tight')

        # Plotting the zero velocity detection for LSTM filtered data
        plt.figure()
        plt.plot(timestamps[:len(zv_lstm)], zv_lstm, label='Raw')
        plt.plot(timestamps[:len(zv_lstm_filtered)], zv_lstm_filtered, label='Filtered')
        plt.scatter(timestamps[strideIndexLSTMfiltered], zv_lstm_filtered[strideIndexLSTMfiltered], c='r', marker='x')
        plt.title(f'Exp#{i+1} ({base_filename}) {n_lstm_filtered}/{nGT[i]} strides detected ({"lstm".upper()})')
        plt.xlabel('Time [s]')
        plt.ylabel('Zero Velocity')
        plt.grid(True, which='both', linestyle='--', linewidth=1.5)
        plt.legend()
        plt.yticks([0,1])
        plt.savefig(os.path.join(output_dir, f'zv_labels_exp_{i+1}_lstm.png'), dpi=600, bbox_inches='tight')

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
        elif i+1 == 49: # 49th experiment: Detected last 4 strides are left outside of the experiment as they do not cause any x-y motion.
            zv_filtered = zv_filtered[:13070] # data cropped to exclude the last 4 strides
            zv = zv[:13070] # zv is the target data in LSTM retraining for robust ZUPT aided INS
            reconstructed_traj = reconstructed_traj[:13070]
            gt = gt[:13070]
            timestamps = timestamps[:13070]
            imu_data = imu_data[0:13070,:]

        # PRODUCE CORRECTED ZV and TRAJECTORY PLOTS
        if i+1 in corrected_data_index:
            # Apply filter to zero velocity detection
            logging.info(f"Applying stride detection to the combined zero velocity detection results for experiment {i+1}.")
            zv_filtered, n, strideIndex = heuristic_zv_filter_and_stride_detector(zv_filtered, 1)
            logging.info(f"Detected {n}/{nGT[i]} strides detected with the combined ZV detector in the experiment {i+1}.")

            # Calculate displacement and heading changes between stride points ground truth data
            displacements, heading_changes = calculate_displacement_and_heading(gt[:, :2], strideIndex)

            # Reconstruct the trajectory from displacements and heading changes
            initial_position = gt[strideIndex[0], :2]  # Starting point from the GT trajectory
            reconstructed_traj = reconstruct_trajectory(displacements, heading_changes, initial_position)

            # Plotting the reconstructed trajectory and the ground truth without stride indices
            plt.figure()
            visualize.plot_topdown([reconstructed_traj, gt[:, :2]], title=f"Exp#{i+1} ({base_filename}) - Combined",
                                legend=['Stride & Heading', 'GT (sample-wise)']) 
            plt.scatter(reconstructed_traj[:, 0], reconstructed_traj[:, 1], c='b', marker='o')
            plt.savefig(os.path.join(output_dir, f'trajectory_exp_{i+1}_corrected.png'), dpi=600, bbox_inches='tight')

            # Plotting the zero velocity detection for the combined ZV detector without stride indices
            plt.figure()
            plt.plot(timestamps, zv, label='Raw')
            plt.plot(timestamps, zv_filtered, label='Filtered')
            plt.scatter(timestamps[strideIndex], zv_filtered[strideIndex], c='r', marker='x')
            plt.title(f'Exp#{i+1} ({base_filename}) {n}/{nGT[i]} strides detected (combined)')
            plt.xlabel('Time [s]')
            plt.ylabel('Zero Velocity')
            plt.grid(True, which='both', linestyle='--', linewidth=1.5)
            plt.legend()
            plt.yticks([0,1])
            plt.savefig(os.path.join(output_dir, f'zv_labels_exp_{i+1}_corrected.png'), dpi=600, bbox_inches='tight')
        
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
            # Stride indexes and timestamps will be used to calculate (dx,dy) in Gradient Boosting (LLIO) training yet we saved other for completeness
            combined_data = np.column_stack((displacements, heading_changes)) # Combine displacement and heading change data into one array
            combined_data2 = np.column_stack((strideIndex, timestamps[strideIndex])) # Combine stride indexes and timestamps into one array

            # Save the combined displacement and heading change data to a CSV file
            combined_csv_filename = os.path.join(extracted_training_data_dir, f'LLIO_training_data/{base_filename}_displacement_heading_change.csv')
            # Save the combined stride indexes and timestamps data to a CSV file
            combined_csv_filename2 = os.path.join(extracted_training_data_dir, f'LLIO_training_data/{base_filename}_strideIndex_timestamp.csv')

            # print(f"strideIndex.shape = {strideIndex.shape}")
            np.savetxt(combined_csv_filename, combined_data, delimiter=',', header='displacement,heading_change', comments='')
            np.savetxt(combined_csv_filename2, combined_data2, delimiter=',', header='strideIndex,timestamp', comments='')
        
        count_training_exp += 1

    else:
        logging.info(f"===================================================================================================================")
        logging.info(f"Processing file {file}")
        print(f"Experiment {i+1} data is not considered as bipedal locomotion data for the retraining process.".upper())
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

logging.info(f"===================================================================================================================")
print(f"Out of {i} experiments, {count_training_exp} of them will be used in retraining LSTM robust ZV detector.")
logging.info("Processing complete for all files.")
