function [n, transitions] = count_zero_to_one_transitions(arr)
    % Find locations where transitions from 0 to 1 occur
    transitions = find((arr(1:end-1) == 0) & (arr(2:end) == 1));
    
    % Return the count and the adjusted indexes
    transitions = transitions + 1; % Add 1 to get the index of the '1'
    n = length(transitions);
end
