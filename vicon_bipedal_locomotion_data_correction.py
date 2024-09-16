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

# Directory containing your Vicon data files
vicon_data_dir = 'data/vicon/processed/'
vicon_data_files = glob.glob(os.path.join(vicon_data_dir, '*.mat'))

# Set up logging
output_dir = "results/figs/vicon_bipedal_locomotion/"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'output.log')
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Detector thresholds. Detector 16 is changed from MBGTD to ARED
detector = ['shoe', 'ared', 'shoe', 'shoe', 'shoe', 'ared', 'shoe', 'shoe',
            'vicon', 'shoe', 'shoe', 'vicon', 'vicon', 'shoe', 'vicon', 'ared',
            'shoe', 'shoe', 'ared', 'vicon', 'shoe', 'shoe', 'vicon', 'shoe',
            'vicon', 'shoe', 'shoe', 'shoe', 'vicon', 'vicon', 'vicon', 'shoe',
            'shoe', 'vicon', 'vicon', 'shoe', 'shoe', 'shoe', 'shoe', 'ared',
            'shoe', 'shoe', 'ared', 'shoe', 'shoe', 'shoe', 'ared', 'shoe',
            'shoe', 'ared', 'mbgtd', 'shoe', 'vicon', 'shoe', 'shoe', 'vicon']
thresh = [2750000, 0.1, 6250000, 15000000, 5500000, 0.08, 3000000, 3250000,
          0.02, 97500000, 20000000, 0.0825, 0.1, 30000000, 0.0625, 0.225,
          92500000, 9000000, 0.015, 0.05, 3250000, 4500000, 0.1, 100000000,
          0.0725, 100000000, 15000000, 250000000, 0.0875, 0.0825, 0.0925, 70000000,
          525000000, 0.4, 0.375, 150000000, 175000000, 70000000, 27500000, 1.1,
          12500000, 65000000, 0.725, 67500000, 300000000, 650000000, 1, 4250000,
          725000, 0.0175, 0.125, 42500000, 0.0675, 9750000, 3500000, 0.175]

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

    return np.array(trajectory)

i = 0  # experiment index
training_data_tag = [0]*35
training_data_tag.append(1)

corrected_data_index = [4, 6, 11, 18, 27, 30, 32, 36] # corrected experiment indexes
nGT = [22, 21, 21, 18, 26, 24, 18, 20, 28, 35,
       29, 22, 30, 34, 24, 36, 20, 15, 10, 33, 
       22, 19, 13, 16, 17, 21, 20, 28, 18, 12,
       13, 26, 34, 25, 24, 24] # number of actual strides
training_data_tag = [abs(x) for x in training_data_tag]
# Process each VICON room training data file
for file in vicon_data_files:
    if training_data_tag[i]:
        logging.info(f"Processing file: {file}")
        data = sio.loadmat(file)

        # Extract the relevant columns
        imu_data = np.column_stack((data['imu'][:, :3], data['imu'][:, 3:6]))  # Accel and Gyro data
        timestamps = data['ts'][0]
        gt = data['gt']  # Ground truth from Vicon dataset

        # Initialize INS object with correct parameters
        ins = INS(imu_data, sigma_a=0.00098, sigma_w=8.7266463e-5, T=1.0 / 200)

        logging.info(f"Processing {detector[i]} detector for file: {file}")
        ins.Localizer.set_gt(gt)  # Set the ground truth data required by 'vicon' detector
        ins.Localizer.set_ts(timestamps)  # Set the sampling time required by 'vicon' detector
        zv = ins.Localizer.compute_zv_lrt(W=5 if detector[i] != 'mbgtd' else 2, G=thresh[i], detector=detector[i])
        x = ins.baseline(zv=zv)

        # Apply filter to zero velocity detection results for stride detection corrections
        logging.info(f'Applying heuristic filter to optimal ZUPT detector "{detector[i]}" for correct stride detection.')
        k = 75 # temporal window size for checking if detected strides are too close or not
        zv_filtered, n, strideIndex = heuristic_zv_filter_and_stride_detector(zv, k)
        # zv_filtered = medfilt(zv_filtered, 15)
        # n, strideIndex = count_one_to_zero_transitions(zv_filtered)
        # strideIndex = strideIndex - 1 # make all stride indexes the last samples of the respective ZUPT phase
        # strideIndex[0] = 0 # first sample is the first stride index
        # strideIndex = np.append(strideIndex, len(timestamps)-1) # last sample is the last stride index
        logging.info(f"Detected {n}/{nGT[i]} strides in the experiment {i+1}.")

        # Calculate displacement and heading changes between stride points based on ground truth
        displacements, heading_changes = calculate_displacement_and_heading(gt[:, :2], strideIndex)
        # Reconstruct the trajectory from displacements and heading changes
        initial_position = gt[strideIndex[0], :2]  # Starting point from the GT trajectory
        reconstructed_traj = reconstruct_trajectory(displacements, heading_changes, initial_position)

        # Remove the '.mat' suffix from the filename
        base_filename = os.path.splitext(os.path.basename(file))[0]
        # Plotting the reconstructed trajectory and the ground truth without stride indices
        plt.figure()
        visualize.plot_topdown([reconstructed_traj, gt[:, :2]], title=f"Exp#{i+1} ({base_filename}) - {detector[i].upper()}", 
                               legend=['Stride & Heading', 'GT (sample-wise)'])
        # if i+1==36:
        #     # plt.plot(gt[-5:,0], gt[-5:,1], c='r')
        #     plt.scatter(-reconstructed_traj[-5:, 0], reconstructed_traj[-5:, 1], c='b', marker='x')
        #     plt.scatter(-reconstructed_traj[0:3, 0], reconstructed_traj[0:3, 1], c='r', marker='x')
        #     # hms = 34 # "how many strides" to show from the beginning (including the initial stride)
        #     plt.plot(-reconstructed_traj[-5:, 0], reconstructed_traj[-5:, 1], c='b')
        #     # plt.scatter(-reconstructed_traj[0, 0], reconstructed_traj[0, 1], c='b', marker='s')
        #     # plt.scatter(-reconstructed_traj[0:hms-1, 0], reconstructed_traj[0:hms-1, 1], c='b', marker='x')
        #     # plt.scatter(-reconstructed_traj[hms-1, 0], reconstructed_traj[hms-1, 1], c='g', marker='o')
        # else:    
        plt.scatter(-reconstructed_traj[:, 0], reconstructed_traj[:, 1], c='b', marker='o')
        plt.savefig(os.path.join(output_dir, f'trajectory_exp_{i+1}.png'), dpi=600, bbox_inches='tight')

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

        # while some experiments are excluded due to being non bipedal locomotion motion (i.e., crawling experiments)
        # some other bipedal locomotion experimental data requires correction for some ZV labels and stride detections 
        # correction indexes are extracted manually (see detect_missed_strides.m for details)
        if i+1 == 4: # Experiment needs ZV correction in 10th stride
            zv_filtered[2800:2814] = 1 # correction indexes for the missed stride
        elif i+1 == 6: # Experiment needs ZV correction in 9th stride
            zv_filtered[2544:2627] = 1 # correction indexes for the missed stride
        elif i+1 == 11: # Experiment needs ZV correction in 7th stride
            zv_filtered[2137:2162] = 1 # correction indexes for the missed stride
        elif i+1 == 18: # Experiment needs ZV correction in 7th stride
            zv_filtered[1882:1940] = 1 # correction indexes for the missed stride
        elif i+1 == 27: # Experiment needs ZV correction in {9, 16, 17, 18}th strides (first 3 by VICON and the last one by MBGTD)
            zv_filtered[1816:1830] = 1 
            zv_filtered[2989:3002] = 1
            zv_filtered[3154:3168] = 1
            zv_filtered[3329-3:3329+3] = 1
        elif i+1 == 30: # Experiment needs ZV correction in {2, 10}th strides (both detected by SHOE (supplementary) detector)
            zv_filtered[620:630] = 1
            zv_filtered[1785:1790] = 1
        elif i+1 == 32: # 32nd experiment: missed strides {9, 11, 20}. First two strides are recovered by VICON but the last one needed to be introduced by manual annotation.
            zv_filtered[1851-3:1851+3] = 1
            zv_filtered[2138:2146] = 1
            zv_filtered[3997:4004] = 1 # this is manual annotation
        elif i+1 == 36:
            zv_filtered[1864:1890] = 1
        
        if i+1 in corrected_data_index:
            # Apply filter to zero velocity detection
            logging.info(f"Applying stride detection to the combined zero velocity detection results for experiment {i+1}.")
            zv_filtered, n, strideIndex = heuristic_zv_filter_and_stride_detector(zv_filtered, 1)
            logging.info(f"Detected {n}/{nGT[i]} strides detected with the combined ZV detector in the experiment {i+1}.")

            # Calculate displacement and heading changes between stride points based on ground truth
            displacements, heading_changes = calculate_displacement_and_heading(gt[:, :2], strideIndex)

            # Reconstruct the trajectory from displacements and heading changes
            initial_position = gt[strideIndex[0], :2]  # Starting point from the GT trajectory
            reconstructed_traj = reconstruct_trajectory(displacements, heading_changes, initial_position)

            # Plotting the reconstructed trajectory and the ground truth without stride indices
            plt.figure()
            visualize.plot_topdown([reconstructed_traj, gt[:, :2]], title=f"Exp#{i+1} ({base_filename}) - Combined",
                                legend=['Stride & Heading', 'GT (sample-wise)']) 
            plt.scatter(-reconstructed_traj[:, 0], reconstructed_traj[:, 1], c='b', marker='o')
            plt.savefig(os.path.join(output_dir, f'trajectory_exp_{i+1}_corrected.png'), dpi=600, bbox_inches='tight')

            # Plotting the zero velocity detection for the combined ZV detector without stride indices
            plt.figure()
            plt.plot(timestamps[:len(zv)], zv, label='Raw')
            plt.plot(timestamps[:len(zv_filtered)], zv_filtered, label='Filtered')
            plt.scatter(timestamps[strideIndex], zv_filtered[strideIndex], c='r', marker='x')
            plt.title(f'Exp#{i+1} ({base_filename}) {n}/{nGT[i]} strides detected (combined)')
            plt.xlabel('Time [s]')
            plt.ylabel('Zero Velocity')
            plt.grid(True, which='both', linestyle='--', linewidth=1.5)
            plt.legend()
            plt.yticks([0,1])
            plt.savefig(os.path.join(output_dir, f'zv_labels_exp_{i+1}_corrected.png'), dpi=600, bbox_inches='tight')

    i += 1  # Move to the next experiment

logging.info("Processing complete for all files.")
