% Load the data
load('/Users/thib/Desktop/Master_thesis/DataForThesis/fMRI/Subj12_UT/ResponseData_Subj_1110_Run_100_060625_14h13_TEMP.mat');

% Extract trigger codes and times
triggerCodes = info.triggerCodes;
triggerTimes = info.triggerTimes;

% Initialize arrays
trial_type_array = categorical();
onset_array = [];
duration_array = [];

i = 1;
n = length(triggerCodes);

while i <= n - 1  % Ensure i+1 does not exceed array bounds
    code = triggerCodes(i);
    
    if code == 64
        % Skip instruction codes (they come in pairs)
        i = i + 2;
    else
        % Process block codes (also come in pairs)
        block_code = code;
        block_start_time = triggerTimes(i);
        block_end_time = triggerTimes(i + 1);
        duration = block_end_time - block_start_time;
        
        % Map Block Number to Condition
        switch block_code
            case 1
                condition = 'Allo Faces';
            case 3
                condition = 'Color Faces';
            case 7
                condition = 'Ego Faces';
            case 11
                condition = 'Allo Flags';
            case 13
                condition = 'Ego Flags';
            case 15
                condition = 'Color Flags';
            otherwise
                condition = 'Unknown';
        end
        
        % Append to arrays
        trial_type_array(end + 1) = condition;
        onset_array(end + 1) = block_start_time;
        duration_array(end + 1) = duration;
        
        % Move to the next pair
        i = i + 2;
    end
end

% Create the table
T = table(trial_type_array', onset_array', duration_array', ...
    'VariableNames', {'trial_type', 'onset', 'duration'});

% Specify the filename
filename = fullfile('/Users/thib/Desktop/Master_thesis/DataForThesis/fMRI/Subj12_UT', 'Subj12_Localizer-Blocks.tsv');

% Write the table to a TSV file
writetable(T, filename, 'FileType', 'text', 'Delimiter', '\t');
