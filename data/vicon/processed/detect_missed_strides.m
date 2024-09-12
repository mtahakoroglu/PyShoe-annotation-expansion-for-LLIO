clear all; close all; clc;
k = 75; % heuristic filter window size
%% EXPERIMENT 4
load('2017-11-22-11-25-20'); % misses 10th stride
nGT = 18; % There are 18 strides in reality.
figure(1); clf; set(gcf, 'position', [565 238 842 368]);
subplot(3,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, 1);
fprintf('There are %i strides detected by (filtered) SHOE zupt detector in experiment 4.\n', n);
plot(ts, zv_shoe_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.72    0.948    0.21]);
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Optimal ZUPT Detector (SHOE) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);
subplot(3,1,2); % OPTIMAL DETECTOR (FILTERED DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf('There are %i strides detected by (filtered) SHOE zupt detector in experiment 4.\n', n);
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.415    0.948    0.21]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (SHOE filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);
subplot(3,1,3); % SUPPLEMENTARY DETECTOR (FILTERED DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf('There are %i strides detected by (filtered) VICON zupt detector in experiment 4.\n', n);
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.1    0.948    0.21]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
h = xlabel('Time [s]', 'FontSize', 14); set(h, 'Position', [13.2921   -0.1649   -1.0000]);
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (VICON filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);
print('-f1','experiment4_ZUPT_detectors_strides','-dpng','-r800');
%% EXPERIMENT 6
load('2017-11-22-11-26-46'); % misses 9th stride
figure(2);
subplot(2,1,1);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_ared_opt, k);
fprintf('There are %i strides detected by (filtered) ARED zupt detector in experiment 6.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [ared filtered]');
legend('ZV labels', 'strides');
subplot(2,1,2);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf('There are %i strides detected by (filtered) VICON zupt detector in experiment 6.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [vicon filtered]');
legend('ZV labels', 'strides');
%% EXPERIMENT 11
load('2017-11-22-11-35-59'); % misses 7th stride
figure(3);
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
figure(4);
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
fprintf('There are %i strides detected by (filtered) VICON zupt detector in experiment 11.\n', n);
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
figure(5);
subplot(4,1,1);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf('There are %i strides detected by (filtered) SHOE zupt detector in experiment 18.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [shoe filtered]');
legend('ZV labels', 'strides');
subplot(4,1,2);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf('There are %i strides detected by (filtered) VICON zupt detector in experiment 11.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [vicon filtered]');
legend('ZV labels', 'strides');
subplot(4,1,3);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_ared_opt, k);
fprintf('There are %i strides detected by (filtered) ARED zupt detector in experiment 11.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [ared filtered]');
legend('ZV labels', 'strides');
subplot(4,1,4);
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_mbgtd_opt, k);
fprintf('There are %i strides detected by (filtered) MBGTD zupt detector in experiment 11.\n', n);
plot(ts, zv, 'LineWidth', 1.2);
hold on;
plot(ts(strideIndex), zv(strideIndex), 'rx', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]'); ylabel('ZV labels [MBGTD filtered]');
legend('ZV labels', 'strides');