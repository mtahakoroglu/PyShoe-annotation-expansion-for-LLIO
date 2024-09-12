function [n, strides] = count_one_to_zero_transitions(zv)
    % Find where the difference between consecutive elements is negative (1 to 0 transition)
    strides = find(diff(zv) < 0) + 1;
    
    % Return the number of transitions and the corresponding indices
    n = length(strides);
end
