clear all; clc;
dinfo = dir('*.mat');
filename = {dinfo.name};
detector = {}; threshold = zeros(1,length(filename));
for i=1:length(filename)
    data = load(filename{i});
    detector{i} = data.best_detector;
    thresholdString = sprintf('G_%s_opt', detector{i});
    threshold(1,i) = data.(thresholdString);
end
% detector count
shoe_count = 0; ared_count = 0; vicon_count = 0; mbgtd_count = 0;
for i=1:length(filename)
    if detector{i} == "shoe"
        shoe_count = shoe_count + 1;
    elseif detector{i} == "ared"
        ared_count = ared_count + 1;
    elseif detector{i} == "vicon"
        vicon_count = vicon_count + 1;
    elseif detector{i} == "mbgtd"
        mbgtd_count = mbgtd_count + 1;
    else
        disp('There is something wrong!');
    end
end

if shoe_count + ared_count + vicon_count + mbgtd_count == length(filename)
    fprintf('There are %i experients conducted in PyShoe dataset.\n', length(filename));
end