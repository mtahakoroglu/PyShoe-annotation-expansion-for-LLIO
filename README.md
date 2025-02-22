<h2>PyShoe Dataset Annotation & Expansion/Enlargement for Loose Learned Inertial Odometry (LLIO)</h2>

<p align="justify">Similar to <a href="https://github.com/mtahakoroglu/gradient-boosting-based-LLIO">LLIO</a> repository, the following installations are required to run the scripts and reproduce the results presented here and in the paper. Given scripts for creating the <b>pyshoe</b> virtual environment have worked successfully on various Windows systems, although we cannot guarantee compatibility with all system configurations.</p>

<p align="justify"><b>Note:</b> The Anaconda virtual environment is named <b>pyshoe</b> because initially the research was completely based on <a href="https://scholar.google.ca/citations?user=I_x8PVcAAAAJ&hl=en">Brandon Wagstaff's PyShoe study</a> and <a href="https://ieee-dataport.org/open-access/university-toronto-foot-mounted-inertial-navigation-dataset">public dataset</a>. Despite the transition to using exclusively self-collected data and building own inertial odometry public dataset, the original name was retained.</p>

<h3>Creating <b>pyshoe</b> Virtual Environment in Anaconda</h3>
<p align="justify">After installing Anaconda, launch <b>Anaconda PowerShell</b> and then type</p>

```
conda create --name pyshoe python=3.7
```

<p align="justify">to create <b>pyshoe</b> virtual environment (venv). Subsequently, type</p>

```
conda activate pyshoe
```

<p align="justify">to activate <b>pyshoe</b> venv.</p>

<h3>Installing Required Packages</h3>

<p align="justify">Type and enter the following commands in Anaconda PS terminal to install the required packages and libraries to run PyShoe codes and reproduce the results in the page and the paper. We thank <a href="https://scholar.google.com.tr/citations?user=F2NkKNAAAAAJ&hl=tr">Dr. Ramazan Özgür Doğan</a> for the assistance in setting up the environment.</p>

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```


<p align="justify">Continue and complete the installation by entering the following commands in Anaconda PS in <b>pyshoe</b> virtual environment.<p>

```
conda install scikit-learn==0.19.1
```

```
conda install matplotlib
```

```
pip install pandas==1.1.5
```

<h3>Installing <b>liegroups</b> package</h3>

<p align="justify">After cloning this repository to your local computer, you must install <b><a href="https://github.com/utiasSTARS/liegroups" target="_blank">liegroups</a></b> package if you would like to reproduce the results shown here in this repo or the paper.</p>

<h4>Video Tutorial</h4>

<p align="justify">If one fails or has troublesome experience in setting up the virtual environment <b>pyshoe</b>, the video in <a href="https://www.youtube.com/watch?v=CnA4yF7448c">pyshoe venv setting</a> will soon be available.</p>

<h3>VICON Room Experiments - Annotation & Corrections for LLIO</h3>

<p align="justify">We aim to form a <b>gait-driven system</b> (in other words a <b>stride & heading system</b> - <b>SHS</b>) from VICON room experiments of PyShoe dataset where VICON room experiments are sample-wise annotated pedestrian trajectories. Stride indexes are also required in training a stride-wise INS.</p>

<p align="justify">Here, some troublesome experiments are shown to understand Zero Velocity (ZV) interval and stride detection problems. The optimal ZV detectors are selected (e.g., SHOE for experiment 4, ARED for experiment 6) with the corresponding optimal threshold values (optimal values are supplied by Wagstaff <i>et. al.</i> in the structure of the mat files for VICON room experiments of PyShoe dataset) for all VICON room experiments.</p>

<p align="justify">The process depicted below is the ZV interval & stride index annotation (or correction) of some troublesome trajectories produced in VICON room experiments (PyShoe dataset). Eventually, ZV interval and stride index detection errors in VICON data are fixed. Extracted dataset is going to be used in training <a href="https://github.com/mtahakoroglu/LLIO">Loose Learned Inertial Odometry (LLIO)</a>, a gait & data driven INS. Some researchers call a data-driven (or modern) INS as learned inertial odometry.</p>

<p align="justify">Please use <b>detect_missed_strides.m</b> located at <b>data/vicon/processed</b> if you like to reproduce the figures related to ZV interval annotation & correction.</p>

<h4>VICON Room Experiments - Experiment 4 (2017-11-22-11-25-20) Annotation</h4>

<p align="justify">When carefully tracked starting from the initial stride, one can see that the 10<sup>th</sup> stride is not detected in the trajectory plot shown below. To detect the missed ZV interval(s), supplementary ZUPT detectors such as VICON, ARED, MBGTD or AMVD can be utilized. In general, VICON detector was able to generate ZV labels correctly; therefore, in many cases, only VICON ZUPT detector is used as the supplementary detector.</p>

<!--
<img src="results/figs/vicon_obsolete/exp4.jpg" alt="Optimal detector results for experiment 4 (2017-11-22-11-25-20) VICON dataset" width=%100 height=auto>
-->

<img src="data/vicon/processed/experiment4_ZUPT_detectors_strides.png" alt="ZV labels for experiment 4 (2017-11-22-11-25-20) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the filtered supplementary ZUPT detector (i.e., VICON) enabled successful detection of the ZV interval as shown in the combined ZUPT detector plot above (located at the bottom). The corrected ground-truth data (as a sample-wise and a stride & heading system trajectory) and ZV signals can be seen below. Note that the annotation is only going to be used in extracting x-y axes displacement (or displacement and heading change) values for LLIO training dataset generation; therefore, corrected ZV labels are not used in any trajectory generation.</p>

<!--
<img src="results/figs/vicon_obsolete/exp4_corrected.jpg" alt="Experiment trajectory after corrections - VICON dataset" width=%100 height=auto>
-->

<img src="results/figs/vicon_obsolete/gif/exp4.gif" alt="ZV correction results" width=%100 height=auto>

<!--
<img src="results/figs/vicon_obsolete/stride_detection_exp_4.png" alt="Stride detection results on imu data - VICON dataset">
-->

<h4>Experiment 6 (2017-11-22-11-26-46) - VICON training dataset</h4>

<p align="justify">We see that the 9th stride is not detected in the plots below.</p>

<img src="results/figs/vicon_obsolete/exp6.jpg" alt="optimal detector results for experiment 6 (2017-11-22-11-26-46) VICON dataset" width=%100 height=auto>

<p align="justify">Just like we did to compensate for the errors ZUPT phase and stride detection in experiment 4, here VICON ZUPT detector is selected again as the supplementary detector to correctly detect the missed stride.</p>

<img src="data/vicon/processed/experiment6_ZUPT_detectors_strides.png" alt="ZV labels for experiment 6 (2017-11-22-11-26-46) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detector (i.e., filtered VICON) enabled successfull detection of the missed stride as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below for the experiment 6.</p>

<img src="results/figs/vicon_obsolete/exp6_corrected.jpg" alt="corrected results for experiment 6 (2017-11-22-11-26-46) VICON dataset - trajectory" width=%100 height=auto>

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file inserted below.</p>

<img src="results/figs/vicon_obsolete/gif/exp6.gif" alt="experiment 6 results after ZV correction" width=%100 height=auto>

<img src="results/figs/vicon_obsolete/stride_detection_exp_6.png" alt="stride detection results on imu data for experiment 6 of VICON dataset">

<h4>Experiment 11 (2017-11-22-11-35-59) - VICON training dataset</h4>

<p align="justify">We see that the 7th stride is not detected in the plots below.</p>

<img src="results/figs/vicon_obsolete/exp11.jpg" alt="optimal detector results for experiment 11 (2017-11-22-11-35-59) VICON dataset" width=%100 height=auto>

<p align="justify">Just like we did to compensate for the errors in ZUPT phase and stride detection in experiments 4 and 6, here VICON ZUPT detector is selected again as the supplementary detector to correctly detect the missed stride.</p>

<img src="data/vicon/processed/experiment11_ZUPT_detectors_strides.png" alt="ZV labels for experiment 11 (2017-11-22-11-35-59) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detector (i.e., filtered VICON) enabled successfull detection of the missed stride as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below.</p>

<img src="results/figs/vicon_obsolete/exp11_corrected.jpg" alt="corrected results for experiment 11 (2017-11-22-11-35-59) VICON dataset" width=%100 height=auto>

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file inserted below.</p>

<img src="results/figs/vicon_obsolete/gif/exp11.gif" alt="experiment 11 results after ZV correction" width=%100 height=auto>

<img src="results/figs/vicon_obsolete/stride_detection_exp_11.png" alt="stride detection results on imu data for experiment 11 of VICON dataset">

<h4>Experiment 18 (2017-11-22-11-48-35) - VICON training dataset</h4>

<p align="justify">We see that the 7<sup>th</sup> stride is not detected in the plots below. Notice that this is the example experiment that is shown in the paper.</p>

<img src="results/figs/vicon_obsolete/exp18.jpg" alt="optimal detector results for experiment 18 (2017-11-22-11-48-35) VICON dataset" width=%100 height=auto>

<p align="justify">Just like we did to compensate for the errors in ZV interval and stride index detection in experiments 4, 6, and 11, here VICON ZUPT detector is selected again as the supplementary detector to correctly include the missed stride to the stride & heading system trajectory.</p>

<img src="data/vicon/processed/experiment18_ZUPT_detectors_strides.png" alt="ZV labels for experiment 18 (2017-11-22-11-48-35) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detector (i.e., filtered VICON) enabled successfull detection of the missed stride as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below.</p>

<img src="results/figs/vicon_obsolete/exp18_corrected.jpg" alt="corrected results for experiment 18 (2017-11-22-11-48-35) VICON dataset" width=%100 height=auto>

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file inserted below.</p>

<img src="results/figs/vicon_obsolete/gif/exp18.gif" alt="experiment 18 results after ZV correction" width=%100 height=auto>

<img src="results/figs/vicon_obsolete/stride_detection_exp_18.png" alt="stride detection results on imu data for experiment 18 of VICON dataset">

<!-- EXPERIMENT 27 is excluded after further examination of the aligned trajectory in the navigation coordinate frame
<h4>Experiment 27 (2017-11-27-11-12-44) - VICON training dataset</h4>

<p align="justify">We see that the strides {9, 16, 17, 18} are not detected in the plots below.</p>

<img src="results/figs/vicon/exp27.jpg" alt="optimal detector results for experiment 27 (2017-11-27-11-12-44) VICON dataset" width=%100 height=auto>

<p align="justify">Unlike for the compensation for the errors in ZUPT phase and stride detection in experiments 4, 6, 11 and 18, here VICON ZUPT detector was not sufficient to detect all missed strides. An additional ZV detector (MBGTD) is exploited to detect the last missed stride.</p>

<img src="data/vicon/processed/experiment27_ZUPT_detectors_strides.png" alt="ZV labels for experiment 27 (2017-11-27-11-12-44) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detectors (i.e., filtered VICON and filtered MBGTD) enabled successfull detection of the missed strideS as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below for the experiment 27.</p>

<img src="results/figs/vicon/exp27_corrected.jpg" alt="corrected results for experiment 27 (2017-11-27-11-12-44) VICON dataset" width=%100 height=auto>

<p align="justify">To see the correction by the supplementary ZUPT detectors, check the gif file inserted below.</p>

<img src="results/figs/vicon/gif/exp27.gif" alt="experiment 27 results after ZV correction" width=%100 height=auto>

<img src="results/figs/vicon/stride_detection_exp_27.png" alt="stride detection results on imu data for experiment 27 of VICON dataset">
-->

<h4>Experiment 30 (2017-11-27-11-14-03) - VICON training dataset</h4>

<p align="justify">We see that the strides {2, 10} are not detected in the plots below.</p>

<img src="results/figs/vicon_obsolete/exp30.jpg" alt="optimal detector results for experiment 30 (2017-11-27-11-14-03) VICON dataset" width=%100 height=auto>

<p align="justify">Unlike experiments {4, 6, 11, 18, 27}, here SHOE ZUPT detector is selected as the supplementary detector to correctly detect the missed strides.</p>

<img src="data/vicon/processed/experiment30_ZUPT_detectors_strides.png" alt="ZV labels for experiment 30 (2017-11-27-11-14-03) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector VICON with the supplementary ZUPT detector (i.e., filtered SHOE) enabled successfull detection of the missed stride as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below for the experiment 30.</p>

<img src="results/figs/vicon_obsolete/exp30_corrected.jpg" alt="corrected results for experiment 30 (2017-11-27-11-14-03) VICON dataset" width=%100 height=auto>

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file inserted below.</p>

<img src="results/figs/vicon_obsolete/gif/exp30.gif" alt="experiment 30 results after ZV correction" width=%100 height=auto>

<img src="results/figs/vicon_obsolete/stride_detection_exp_30.png" alt="stride detection results on imu data for experiment 30 of VICON dataset">

<h4>Experiment 32 (2017-11-27-11-17-28) - VICON training dataset</h4>

<p align="justify">We see that the strides {9, 11, 20} are not detected in the plots below.</p>

<img src="results/figs/vicon_obsolete/exp32.jpg" alt="optimal detector results for experiment 32 (2017-11-27-11-17-28) VICON dataset" width=%100 height=auto>

<p align="justify">Unlike experiments {4, 6, 11, 18, 27, 30}, here supplementary detectors were not able to detect all missed strides. While first two was recovered by VICON ZV detector, the last stride needed to be introduced via manual annotation as can be seen below.</p>

<img src="data/vicon/processed/experiment32_ZUPT_detectors_strides.png" alt="ZV labels for experiment 32 (2017-11-27-11-17-28) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detector (i.e., filtered VICON) and the MANUAL ANNOTATION enabled successfull detection of all missed strides as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below for the experiment 32.</p>

<img src="results/figs/vicon_obsolete/exp32_corrected.jpg" alt="corrected results for experiment 32 (2017-11-27-11-17-28) VICON dataset" width=%100 height=auto>

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file inserted below.</p>

<img src="results/figs/vicon_obsolete/gif/exp32.gif" alt="experiment 32 results after ZV correction" width=%100 height=auto>

<img src="results/figs/vicon_obsolete/stride_detection_exp_32.png" alt="stride detection results on imu data for experiment 32 of VICON dataset">

<h4>Experiment 36 (2017-11-27-11-23-18) - VICON training dataset</h4>

<p align="justify">We see that the 7<sup>th</sup> stride is not detected in the plots below.</p>

<img src="results/figs/vicon_obsolete/exp36.jpg" alt="optimal detector results for experiment 36 (2017-11-27-11-23-18) VICON dataset" width=%100 height=auto>

<p align="justify">Just like the 4<sup>th</sup> experiment, here the supplementary detector is selected as VICON, which was able to recover the missed stride.</p>

<img src="data/vicon/processed/experiment36_ZUPT_detectors_strides.png" alt="ZV labels for experiment 36 (2017-11-27-11-23-18) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detector (i.e., filtered VICON) enabled successfull detection of the missed stride as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below for the experiment 36.</p>

<img src="results/figs/vicon_obsolete/exp36_corrected.jpg" alt="corrected results for experiment 36 (2017-11-27-11-23-18) VICON dataset" width=%100 height=auto>

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file inserted below.</p>

<img src="results/figs/vicon_obsolete/gif/exp36.gif" alt="experiment 36 results after ZV correction" width=%100 height=auto>

<img src="results/figs/vicon_obsolete/stride_detection_exp_36.png" alt="stride detection results on imu data for experiment 36 of VICON dataset">

<h4>Experiment 38 (2017-11-27-11-25-12) - VICON training dataset</h4>

<p align="justify">We see that the strides {3, 27, 33} are not detected in the plots below.</p>

<img src="results/figs/vicon_obsolete/exp38.jpg" alt="optimal detector results for experiment 38 (2017-11-27-11-25-12) VICON dataset" width=%100 height=auto>

<p align="justify">The supplementary detector is selected as VICON, which was able to recover the missed strides all.</p>

<img src="data/vicon/processed/experiment38_ZUPT_detectors_strides.png" alt="ZV labels for experiment 38 (2017-11-27-11-25-12) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detector (i.e., filtered VICON) enabled successfull detection of the missed strides as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below for the experiment 38.</p>

<img src="results/figs/vicon_obsolete/exp38_corrected.jpg" alt="corrected results for experiment 38 (2017-11-27-11-25-12) VICON dataset" width=%100 height=auto>

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file inserted below.</p>

<img src="results/figs/vicon_obsolete/gif/exp38.gif" alt="experiment 38 results after ZV correction" width=%100 height=auto>

<img src="results/figs/vicon_obsolete/stride_detection_exp_38.png" alt="stride detection results on imu data for experiment 38 of VICON dataset">

<h4>Experiment 43 (2017-12-15-18-01-18) - VICON training dataset</h4>

<p align="justify">We see that the strides {3, 14, 16} are not detected in the plots below.</p>

<img src="results/figs/vicon_obsolete/exp43.jpg" alt="optimal detector results for experiment 43 (2017-12-15-18-01-18) VICON dataset" width=%100 height=auto>

<p align="justify">The supplementary detector is selected as VICON, which was able to recover the missed strides all.</p>

<img src="data/vicon/processed/experiment43_ZUPT_detectors_strides.png" alt="ZV labels for experiment 43 (2017-12-15-18-01-18) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detector (i.e., filtered VICON) enabled successfull detection of the missed strides as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below for the experiment 43.</p>

<img src="results/figs/vicon_obsolete/exp43_corrected.jpg" alt="corrected results for experiment 43 (2017-12-15-18-01-18) VICON dataset" width=%100 height=auto>

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file inserted below.</p>

<img src="results/figs/vicon_obsolete/gif/exp43.gif" alt="experiment 43 results after ZV correction" width=%100 height=auto>

<img src="results/figs/vicon_obsolete/stride_detection_exp_43.png" alt="stride detection results on imu data for experiment 43 of VICON dataset">

<p align="justify">Thus far, due to some undetected steps in VICON room experiments data (recall that Wagstaff <i>et. al.</i> conducted crawling motion experiments in PyShoe) and self-collected data, we examined 56 experiments coarsely in the training dataset <i><b>(i)</b></i> to correct for undetected steps (they are classified as 0 in ZV signal plot despite them actually being 1, i.e., it is false-negative) and <i><b>(ii)</b></i> to exclude motions like crawling, which are not of type bipedal locomotion. As can be seen above, experiments {4, 6, 11, 18, 30, 32, 36, 38, 43} are corrected with manual ZV interval and stride index annotations by utilizing supplementary detectors. In <b>plot_vicon_data.py</b> file, right before processing experimental data in a loop, annotated experiments are tagged as -1 as follows:</p>

```
training_data_tag = [1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 0, 1, 1, 1, 1, -1, 1, 1, 
                    1, 1, 1, 1, 1, 1, 0, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1,  
                    1, 1, -1, 1, 1, 1, 0, 0, -1, 0, 1, 1, 1, 1, 0, 1]
```

<p align="justify">where</p>

```
len(training_data_tag) = 56
```

<p align="justify">In the signal processing main loop (where the loop visits each experiment one by one), experiments are accepted as training data for LLIO according to the absolute values of given tags. In other words, experiments labeled as 1's directly satisfy conditions while the ones labeled as -1s needed corrections on ZV intervals and stride indexes to be included in LLIO training dataset. On the other hand, 0s stand for eliminated experiments due to being nonbipedal locomotion data or unrecoverable errors in ZV intervals and/or stride indexes (we actually emailed Brandon Wagstaff about motion types of experiments yet no documentation was made regarding the motion types (e.g., walking, running, crawling) in the experiments at the time of PyShoe dataset generation). In the decision of labeling an experiment as 0, trajectory (spatial) plots along with IMU data (time series) plots with stride indexes are coarsely examined by the second author of the paper by using MATLAB and Python environments. Eventually, the elimination of VICON room data yielded shrink in the traveled distance by the pedestrian, which negatively affcets LLIO training to produce a data-driven INS. Therefore, additional training data was required in LLIO training. The next section describes dataset expansion/enlargement process.</p>

<h3>Training Dataset Expansion/Enlargement for LLIO</h3>

<p align="justify">Here robust (pre-trained LSTM based ZV detector) pedestrian INS is applied on our own-collected data where our sensor is <a href="https://www.microstrain.com/sites/default/files/applications/files/3dm-gx5-25_datasheet_8400-0093_rev_n.pdf">3DM-GX5-25</a> and sensor data capture software is <a href="https://www.microstrain.com/software/sensorconnect">SensorConnect</a>.</p>

| 3DM-GX5-25 Sensor | SensorConnect software screen |
| :---: | :---: |
| <img src="https://www.zse.de/db_pics/shop_content/500x500/3dm-gx5-25.png" alt="3DM-GX5-25" width=auto height=200> | <img src="https://www.microstrain.com/sites/default/files/bitmap.png" alt="SensorConnect" width=auto height=200> |


<p align="justify">Experiments conducted here are manually annotated by using a ruler (see the paper for more details). As the VICON room walks/experiments of PyShoe dataset is very different than hallway traversals, proposed <a href="https://github.com/mtahakoroglu/LLIO">LLIO</a> system required a bigger dataset that accounts for straight walk gait characteristics (at various walking paces). Additionally, faster motion experiments are conducted to make the dataset diverse.</p>

<p align="justify">Similar to the notation used for VICON room experiments, the start point is called as stride #0, i.e., initial stride. If <b>plot_vicon_data.py</b> is checked, one can see that ZV labels are filtered for accurate stride index detection. However the filtered ZV values are not used in the trajectory generation. In other words, the pedestrian trajectories are obtained with the raw (not filtered) LSTM based PyShoe generated ZV labels while the strides that are visualized on the trajectories with x correspond to the last index of the ZV intervals of the filtered ZV signals (i.e., stride index).</p>

<p align="justify">As all strides produced a foot-print on the ground thar are later measured to form ground-truth data for the experiments, each stride location is called a Ground Control Point (GCP).</p>

<p align="justify"><b>Note:</b> One can run <b>plot_own_sensor_data.py</b> to obtain the same results shown below. To learn more about experiment info and results, one can view <b>output.txt</b> log file (located at <b>results/figs/own</b>) recorded while the code was running.</p>

<h4>Experiment 31</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_31_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_31_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_31_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_31_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 32</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_32_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_32_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_32_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_32_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 33</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_33_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_33_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_33_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_33_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<!--
<img src="results/figs/own/SensorConnectData_33_stride_annotation.png" alt="Stride indexes plotted on top of IMU data" width=%100 height=auto>
-->

<h4>Experiment 34</h4>

<p align="justify">Here, the motion speed varies during the experiment as can be seen in the <a href="https://www.youtube.com/shorts/kLczRGQx4Ds">video</a>. PyShoe (LSTM based ZUPT aided ESKF) is able to detect only 22 of 24 strides as can be seen below.</p>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_34_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_34_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_34_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_34_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<!--
<img src="results/figs/own/SensorConnectData_34_stride_annotation.png" alt="Stride indexes plotted on top of IMU data" width=%100 height=auto>
-->

<h4>Experiment 35</h4>

<p align="justify">Here, the motion speed varies during the experiment as can be seen in the <a href="https://www.youtube.com/shorts/R2UBuftXXrE">video</a>. PyShoe (LSTM based ZUPT aided ESKF) is able to detect only 23 of 28 strides as can be seen below.</p>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_35_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_35_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_35_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_35_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<!--
<img src="results/figs/own/SensorConnectData_35_stride_annotation.png" alt="Stride indexes plotted on top of IMU data" width=%100 height=auto>
-->

<h4>Experiment 36</h4>

<p align="justify">Here, the motion speed is slow and constant during the experiment as can be seen in the <a href="https://youtu.be/uT7wL-taX08">video</a>. PyShoe (LSTM based ZUPT aided ESKF) is able to detect all 58 strides successfully. Stride#17 annotation is slightly corrected after examination of detected stride indexes on IMU data, i.e., the magnitudes of acceleration and angular velocity vectors.</p>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_36_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_36_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_36_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_36_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<!--
| Final Stride Indexes on IMU data |
|  :---:  |
| <img src="results/figs/own/SensorConnectData_36_stride_annotation.png" alt="Stride indexes plotted on top of IMU data" width=%100 height=auto> |
-->

<h4>Experiment 37</h4>

<p align="justify">Here, the pedestrian speed is normal (walking style) during the experiment. The video of this experiment was recorded but due to a computer SSD hard disk failure, unfortunately it could not be retrieved. Stride #{33, 34, 41, 43, 60} annotations are slightly corrected after coarse examination of detected stride indexes on IMU data, i.e., the magnitudes of acceleration and angular velocity vectors.</p>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_37_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_37_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_37_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_37_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 38</h4>

<p align="justify">Here, the pedestrian motion becomes extreme in some moments. Therefore, PyShoe LSTM was able to detect 43/60 ZV intervals (and strides) in the trajectory. Missing 17 strides are manually annotated after careful examination of IMU data. Also two stride indexes (i.e., stride 45 and 59) are manually corrected to form the training data for LLIO.</p>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_38_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_38_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_38_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_38_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 39</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_39_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_39_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_39_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_39_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 40</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_40_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_40_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_40_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_40_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 41 (Compensated IMU data)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_41_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_41_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_41_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_41_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 42 (Compensated IMU data)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_42_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_42_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_42_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_42_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 43 (Compensated IMU data)</h4>

<p align="justify">Experiment 43 video, captured by Gökhan Çetin, can be viewed <a href="https://www.youtube.com/shorts/-nBwdj_sYYA">here</a>.</p>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_43_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_43_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_43_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_43_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 44 (Compensated IMU data)</h4>

<p align="justify">Experiment 44 video is <a href="https://www.youtube.com/watch?v=xBWOx4vfLdg">here</a>.</p>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_44_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_44_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_44_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_44_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 45 (Compensated IMU data)</h4>

<p align="justify">Experiment 45 video, captured by Yasin Emir, can be viewed <a href="https://www.youtube.com/shorts/SWT8pragH8U">here</a>.</p>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_45_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_45_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_45_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_45_SHS_WCF.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<!---
| Final Stride Indexes on IMU data |
|  :---:  |
| <img src="results/figs/own/SensorConnectData_37_stride_annotation.png" alt="Stride indexes superimposed IMU data" width=%100 height=auto> |
--->

<!---
<h3>Experimental Test Results (Own Sensor Data)</h3>

<p align="justify">Note that some experiments do not contain GCP or contain incorrectly documented GCP and thereby are not considered in performance evaluation. Also, in some experiments correct number of strides are not detected (with LSTM based robust ZV detector). These experiments are not used in evalation as well. However, in future, if a better ZV interval detector can be developed and detect all strides without missing any, then mentioned experiments could be involved in evaluation process.</p>

<p align="justify"><b>Note:</b> One can run <b>plot_own_sensor_data.py</b> to obtain the same results shown below. To learn more about experiment info and results, one can view <b>output.txt</b> log file (located at <b>results/figs/own</b>) recorded while the code was running.</p>

<h4>Experiment 6 (GCP not available - not used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_06_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_06.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_06_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_06_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 11 (GCP not available - not used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_11_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_11.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_11_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_11_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 13 (GCP not available - not used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_13_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_13.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_13_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_13_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 14 (GCP not available - not used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_06_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_14.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_14_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_14_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 15 (GCP available & all strides detected correctly - can be used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_15_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_15.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_15_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_15_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 16 (GCP available but not correct - not used for performance evaluation)</h4>

<p align="justify">Despite correct detection of all strides, due to errors in GCP documentation, this experiment is not considered in performance evaluation.</p>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_16_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_16.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_16_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_16_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 17 (can be used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_17_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_17.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_17_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_17_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 18 (can be used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_18_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_18.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_18_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_18_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 19 (can be used for performance evaluation)</h4>

| Zero Velocity |  Trajectory  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_19_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_19_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 20 (can be used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (SHS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_16_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_16.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_20_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_20_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 21 (42/43 strides detected - not used for performance evaluation) - Failure (23<sup>rd</sup> ZV interval (and the corresponding stride) is missed)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_21_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_21.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_21_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_21_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 22 (can be used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_22_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_22.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_22_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_22_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 23 (109/110 strides detected - not used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_23_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_23.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_23_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_23_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 24 (can be used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_24_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_24.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_24_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_24_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 25 (can be used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_25_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_25.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_25_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_25_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 26 (can be used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_26_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_26.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_26_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_26_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 27 (can be used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_27_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_27.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_27_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_27_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 28 - a longer pedestrian traverse (can be used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_28_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_28.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_28_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_28_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 29 - same path with exp 28 but at a higher pace (can be used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_29_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_29.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_29_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_29_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

<h4>Experiment 30 - same path with exp 28-29 but at the highest pace, i.e., almost running (54/85 strides detected - not used for performance evaluation)</h4>

| Stride Indexes |  Trajectory (INS)  |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_30_stride_detection.png" alt="Stride indexes plotted on top of IMU data" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_30.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |

| Zero Velocity |  Trajectory (SHS) |
|  :---:  |  :---:  |
| <img src="results/figs/own/SensorConnectData_30_ZV_LSTM_filtered.png" alt="ZV labels produced with robust ZUPT (LSTM filtered) detector" width=400 height=auto> | <img src="results/figs/own/SensorConnectData_30_SHS.png" alt="trajectory obtained with robust ZUPT detector (LSTM) aided (Error-State Kalman Filter based) foot-mounted INS" width=400 height=auto> |
-->

<h2>REFERENCES</h2>
<p align="justify" id="ref1"><a href="#gobacktoref1">[1]</a> X. Liu, N. Li and Y. Zhang, <a href="https://ieeexplore.ieee.org/document/9956821" target="_blank">"A Novel Adaptive Zero Velocity Detection Algorithm Based on Improved General Likelihood Ratio Test Detector,"</a> in <i>IEEE Sensors Journal</i>, vol. 22, no. 24, pp. 24479-24492, 2022.</p>
<p align="justify" id="ref2"><a href="#gobacktoref2">[2]</a> B. Wagstaff, V. Peretroukhin and J. Kelly, <a href="https://ieeexplore.ieee.org/document/8115947" target="_blank">"Improving foot-mounted inertial navigation through real-time motion classification,"</a> in <i>2017 International Conference on Indoor Positioning and Indoor Navigation (IPIN)</i>, Sapporo, Japan, 2017, pp. 1-8.</p>
<p align="justify" id="ref3"><a href="#gobacktoref3">[3]</a> E. Sangenis, C. -S. Jao and A. M. Shkel, <a href="https://ieeexplore.ieee.org/abstract/document/9967027" target="_blank">"SVM-based Motion Classification Using Foot-mounted IMU for ZUPT-aided INS,"</a> in <i>2022 IEEE Sensors</i>, Dallas, TX, USA, 2022, pp. 1-4.</p>
<p align="justify" id="ref4"><a href="#gobacktoref4">[4]</a> Y. Wang and A. M. Shkel, <a href="https://ieeexplore.ieee.org/document/8861375" target="_blank">"Adaptive Threshold for Zero-Velocity Detector in ZUPT-Aided Pedestrian Inertial Navigation,"</a> in <i>IEEE Sensors Letters</i>, vol. 3, no. 11, pp. 1-4, 2019.</p>
<p align="justify" id="ref5"><a href="#gobacktoref5">[5]</a> Y. Wang and A. M. Shkel, <a href="https://ieeexplore.ieee.org/document/9133730" target="_blank">"A Review on ZUPT-Aided Pedestrian Inertial Navigation,"</a> in <i>2020 27th Saint Petersburg International Conference on Integrated Navigation Systems (ICINS)</i>, St. Petersburg, Russia, 2020.</p>
<p align="justify" id="ref6"><a href="#gobacktoref6">[6]</a> J. Wahlström, I. Skog, F. Gustafsson, A. Markham and N. Trigoni, <a href="https://ieeexplore.ieee.org/abstract/document/8715398" target="_blank">"Zero-Velocity Detection—A Bayesian Approach to Adaptive Thresholding,"</a> in <i>IEEE Sensors Letters</i>, vol. 3, no. 6, pp. 1-4, June 2019.</p>
<p align="justify" id="ref7"><a href="#gobacktoref7">[7]</a> Z. Meng, Z. Deng, P. Zhang and Z. Li, <a href="https://ieeexplore.ieee.org/document/10472598" target="_blank">"Adaptive Mid-Stance Phase Observer-Aided Pedestrian Inertial Navigation System for Varying Gait Speeds,"</a> in <i>IEEE Internet of Things Journal</i>, vol. 11, no. 12, pp. 21904-21915, 15 June, 2024.</p>
<p align="justify" #id="ref8"><a href="#gobacktoref8">[8]</a> C. . -S. Jao, K. Stewart, J. Conradt, E. Neftci and A. M. Shkel, <a href="https://ieeexplore.ieee.org/document/9244906" target="_blank">"Zero Velocity Detector for Foot-mounted Inertial Navigation System Assisted by a Dynamic Vision Sensor,"</a> in <i>2020 DGON Inertial Sensors and Systems (ISS)</i>, Braunschweig, Germany, 2020, pp. 1-18.</p>
<p align="justify" #id="ref9"><a href="#gobacktoref9">[9]</a> C. -S. Jao, Y. Wang and A. M. Shkel, <a href="https://ieeexplore.ieee.org/document/9278755" target="_blank">"A Zero Velocity Detector for Foot-mounted Inertial Navigation Systems Aided by Downward-facing Range Sensor,"</a> in <i>2020 IEEE SENSORS</i>, Rotterdam, Netherlands, 2020, pp. 1-4.</p>
<p align="justify" #id="ref10"><a href="#gobacktoref10">[10]</a> <a href="https://starslab.ca/foot-mounted-inertial-navigation-dataset/">University of Toronto STARS Lab. Foot-Mounted Inertial Navigation Dataset</a></p>
<p align="justify" #id="ref11"><a href="#gobacktoref11">[11]</a> Brandon Wagstaff, Valentin Peretroukhin, Jonathan Kelly, July 20, 2021, <a href="https://ieee-dataport.org/open-access/university-toronto-foot-mounted-inertial-navigation-dataset" target="_blank">"University of Toronto Foot-Mounted Inertial Navigation Dataset"</a>, IEEE Dataport, doi: <a href="https://dx.doi.org/10.21227/v1z6-9z84">https://dx.doi.org/10.21227/v1z6-9z84</a>.</p>
<p align="justify" #id="ref12"><a href="#gobacktoref10">[12]</a> J. Wahlström and I. Skog, <a href="https://ieeexplore.ieee.org/document/9174869" target="_blank">"Fifteen Years of Progress at Zero Velocity: A Review,"</a> in <i>IEEE Sensors Journal</i>, vol. 21, no. 2, pp. 1139-1151, 15 Jan., 2021.</p>
<p align="justify"#id="ref13"><a href="#gobacktoref13">[13]</a> Guimarães, V.; Sousa, I.; Correia, M.V. <a href="https://www.mdpi.com/1424-8220/21/11/3940" target="_blank">Orientation-Invariant Spatio-Temporal Gait Analysis Using Foot-Worn Inertial Sensors.</a> <i>Sensors</i> 2021, 21, 3940.</p>
<p align="justify"#id="ref14"><a href="#gobacktoref14">[14]</a> Guimarães, V.; Sousa, I.; Correia, M.V. <a href="https://www.mdpi.com/1424-8220/21/22/7517" target="_blank">A Deep Learning Approach for Foot Trajectory Estimation in Gait Analysis Using Inertial Sensors.</a> <i>Sensors</i> 2021, 21, 7517.</p>
<p align="justify"#id="ref15"><a href="#gobacktoref15">[15]</a> J. Li et al., <a href="https://ieeexplore.ieee.org/abstract/document/10229495" taget="_blank">"Learning-Based Stance Phase Detection and Multisensor Data Fusion for ZUPT-Aided Pedestrian Dead Reckoning System,"</a> in <i>IEEE Internet of Things Journal</i>, vol. 11, no. 4, pp. 5899-5911, 15 Feb., 2024.</p>
</ul>