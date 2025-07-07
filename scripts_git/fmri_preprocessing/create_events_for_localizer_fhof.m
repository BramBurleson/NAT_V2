% Load the data
% Define subject IDs and corresponding localizer IDs
subject_ids = {'sub04', 'sub02', 'sub05'}';
localizer_ids = {'Subj0', 'Subj4', 'Subj3'}';

% Create a table
id_table = table(subject_ids, localizer_ids);



behav_folder_path = 'K:\BramBurleson\01_Data\NAT_V1_fMRI_pilots\localizer\behav_output';
events_folder_path = 'K:\BramBurleson\01_Data\NAT_V1_fMRI_pilots\localizer\fhof_events';
cd(behav_folder_path)
folders = dir('Subj*')


for i = 1:numel(folders)
    localizer_id = folders(i).name;
    disp(localizer_id)

    if strcmp(localizer_id, 'Subj1')
        disp('Subj1 Does not contain trigger codes')
    else


        subject_id = id_table.subject_ids{strcmp(id_table.localizer_ids, localizer_id)};



        cd(fullfile(behav_folder_path, folders(i).name, 'run100'))
        files = dir('ResponseData*');
        file    = files(end).name; %get last should be most recent.
    
        load(file);
        
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
        filename = fullfile(events_folder_path, ['localizer_events_', subject_id,'.tsv']);
        disp(filename)
        % Write the table to a TSV file
        writetable(T, filename, 'FileType', 'text', 'Delimiter', '\t');
    end
end
