function [zv, n, strideIndexFall] = heuristic_zv_filter_and_stride_detector(zv, k)
    % Ensure zv is an integer array if it's logical (boolean)
    if islogical(zv)
        zv = double(zv);
    end

    % Ensure the first 50 samples are considered stationary (set to 1)
    zv(1:50) = 1;
    
    % Detect falling edge (1 to 0 transition) and respective indexes
    [n, strideIndexFall] = count_one_to_zero_transitions(zv);
    strideIndexFall = strideIndexFall - 1; % Make stride indexes the last samples of the respective ZUPT phase
    strideIndexFall = [strideIndexFall, length(zv)]; % Last sample is the last stride index
    
    % Detect rising edge (0 to 1 transition) and respective indexes
    [n2, strideIndexRise] = count_zero_to_one_transitions(zv);
    
    % Correction for small stride phases
    for i = 1:length(strideIndexRise)
        if strideIndexRise(i) - strideIndexFall(i) < k
            zv(strideIndexFall(i):strideIndexRise(i)) = 1; % Make all samples in between one
        end
    end
    
    % Perform the stride index detection again after correction
    [n, strideIndexFall] = count_one_to_zero_transitions(zv);
    strideIndexFall = strideIndexFall - 1; % Make stride indexes the last samples of the respective ZUPT phase
    strideIndexFall = [strideIndexFall, length(zv)]; % Last sample is the last stride index
end