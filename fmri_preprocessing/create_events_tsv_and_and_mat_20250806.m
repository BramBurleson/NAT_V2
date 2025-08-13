clear, clc

subject_ids = {"sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-08", "sub-09"};

thisFileDir = fileparts(mfilename('fullpath'));
cd(thisFileDir)
ROOT_DATASET = fileparts(fileparts(thisFileDir));


behav_folder = [ROOT_DATASET, '/data', '/behav'];
derivatives_folder = [ROOT_DATASET, '/data/fmri/derivatives/']



for i = 1:numel(subject_ids)
    subject_id = subject_ids{i}


    subject_behav_folder = fullfile(behav_folder, subject_id, 'ff')
    events_folder = fullfile(derivatives_folder, subject_id, 'events');
    files = dir(fullfile(subject_behav_folder, 'ResponseData*'));
    file = files(1)
    load(fullfile(file.folder, file.name))

    
    % Extract trigger codes and times
    triggerCodes = info.triggerCodes;
    triggerTimes = info.triggerTimes;

    % Initialize arrays
    trial_type = categorical();
    onset = [];
    duration = [];

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
            block_duration = block_end_time - block_start_time;
            
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
            trial_type(end + 1) = condition;
            onset(end + 1) = block_start_time;
            duration(end + 1) = block_duration;
            
            % Move to the next pair
            i = i + 2;
        end
    end

    %%save mat file
    names = trial_type;
    onsets = onset;
    durations = duration;
    filename = fullfile(events_folder, subject_id + "_ff_events.mat")
    save(filename, 'names', 'onsets', 'durations', '-mat')
 
    %% save events_tsv
    T = table(trial_type', onset', duration', ...
        'VariableNames', {'trial_type','onset','duration'});   % <? keep names

    filename = fullfile(events_folder, subject_id + "_ff_events.tsv");
    writetable(T, filename, 'FileType', 'text', 'Delimiter', '\t', ...
        'WriteVariableNames', true);  

    clear info

end

