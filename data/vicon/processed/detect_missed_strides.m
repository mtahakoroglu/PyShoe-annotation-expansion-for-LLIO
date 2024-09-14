clear all; close all; clc;
k = 75; % heuristic filter window size
%% EXPERIMENT 4
load('2017-11-22-11-25-20'); % misses 10th stride
expIndex = 4; nGT = 18; % actual number of strides
figure(expIndex); clf; set(gcf, 'position', [565 165 842 545]);
subplot(4,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, 1);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.77    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Optimal ZUPT Detector (SHOE) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);

subplot(4,1,2); % OPTIMAL DETECTOR (FILTERED DATA)
[zv_shoe_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt_filtered, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv_shoe_opt_filtered(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.54    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (SHOE filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);

subplot(4,1,3); % SUPPLEMENTARY DETECTOR (FILTERED DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.31    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (VICON filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);

subplot(4,1,4); % COMBINED DETECTOR
tolerance = 1e-4;  % Define a small tolerance
indexStart = find(abs(ts-14.005) < tolerance);
indexEnd = find(abs(ts-14.0701) < tolerance);
T = 0;
zv = zv_shoe_opt_filtered; zv(indexStart-T:indexEnd+T) = 1;
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
fprintf(sprintf('There are %i strides detected by combined ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.08    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Combined ZUPT Detector - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);
h = xlabel('Time [s]', 'FontSize', 14); set(h, 'Position', [13.2921   -0.1649   -1.0000]);
print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpng','-r800');
%% EXPERIMENT 6
load('2017-11-22-11-26-46'); % misses 9th stride
expIndex = 6; nGT = 24; % actual number of strides
figure(expIndex); clf; set(gcf, 'position', [565 165 842 545]);
subplot(4,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_ared_opt, 1);
fprintf(sprintf('There are %i strides detected by (filtered) ARED ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_ared_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.77    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Optimal ZUPT Detector (ARED) - %i/%i strides detected in experiment %i', n, nGT, expIndex);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);

subplot(4,1,2); % OPTIMAL DETECTOR (FILTERED DATA)
[zv_ared_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_ared_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) ARED ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_ared_opt_filtered, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv_ared_opt_filtered(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.54    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (ARED filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);

subplot(4,1,3); % SUPPLEMENTARY DETECTOR (FILTERED DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.31    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (VICON filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);

subplot(4,1,4); % COMBINED DETECTOR
tolerance = 1e-4;  % Define a small tolerance
indexStart = find(abs(ts-12.725) < tolerance);
indexEnd = find(abs(ts-13.135) < tolerance);
T = 0;
zv = zv_ared_opt_filtered; zv(indexStart-T:indexEnd+T) = 1;
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
fprintf(sprintf('There are %i strides detected by combined ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.08    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Combined ZUPT Detector - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);
h = xlabel('Time [s]', 'FontSize', 14); set(h, 'Position', [13.2921   -0.1649   -1.0000]);
print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpng','-r800');
%% EXPERIMENT 11
load('2017-11-22-11-35-59'); % misses 7th stride
figure(11);
subplot(2,1,1);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf('There are %i strides detected by (filtered) SHOE zupt detector in experiment 11.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [shoe filtered]');
legend('ZV labels', 'strides');
subplot(2,1,2);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf('There are %i strides detected by (filtered) VICON zupt detector in experiment 11.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [vicon filtered]');
legend('ZV labels', 'strides');
%% EXPERIMENT 18
load('2017-11-22-11-48-35'); % misses 7th stride
figure(18);
subplot(2,1,1);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf('There are %i strides detected by (filtered) SHOE zupt detector in experiment 18.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [shoe filtered]');
legend('ZV labels', 'strides');
subplot(2,1,2);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf('There are %i strides detected by (filtered) VICON zupt detector in experiment 18.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [vicon filtered]');
legend('ZV labels', 'strides');
%% EXPERIMENT 27
load('2017-11-27-11-12-44.mat'); % misses strides {9, 16, 17, 18}
% first three strides can be retrieved by VICON zupt detector while the
% last one can be retrieved by MBGTD
figure(27);
subplot(4,1,1);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf('There are %i strides detected by (filtered) SHOE zupt detector in experiment 27.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [shoe filtered]');
legend('ZV labels', 'strides');
subplot(4,1,2);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf('There are %i strides detected by (filtered) VICON zupt detector in experiment 27.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [vicon filtered]');
legend('ZV labels', 'strides');
subplot(4,1,3);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_ared_opt, k);
fprintf('There are %i strides detected by (filtered) ARED zupt detector in experiment 27.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [ared filtered]');
legend('ZV labels', 'strides');
subplot(4,1,4);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_mbgtd_opt, k);
fprintf('There are %i strides detected by (filtered) MBGTD zupt detector in experiment 27.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [MBGTD filtered]');
legend('ZV labels', 'strides');