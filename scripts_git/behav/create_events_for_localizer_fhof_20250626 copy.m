% Load the data
% Define subject IDs and corresponding localizer IDs
% subject_ids = {'sub04', 'sub02', 'sub05'}';
% localizer_ids = {'Subj0', 'Subj4', 'Subj3'}';

% K:\BramBurleson\000_datasets_and_scripts\20250625_data_from_Prosper\FHO_Data4_Bram\BehavioralLogs
localizer_ids = {"sbj01", "sbj02", "sbj03", "sbj04", "sbj05", "sbj06", "sbj07", "sbj08", "sbj09", "sbj10", "sbj11", "sbj12"};
subject_ids = {"sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-08", "sub-09", "sub-10", "sub-11", "sub-12"};
% Create a table
id_table = table(subject_ids, localizer_ids);

behav_folder_path = 'K:\BramBurleson\000_datasets_and_scripts\FF_20250625_data_from_Prosper\FHO_Data4_Bram\BehavioralLogs';
events_folder_path = 'K:\BramBurleson\000_datasets_and_scripts\20250625_data_from_Prosper\FHO_Data4_Bram\events_bram_20250626';
cd(behav_folder_path)
subject_localizer_folders = dir('sbj*')


for i = 1:numel(subject_localizer_folders)
    localizer_id = subject_localizer_folders(i).name;
    disp(localizer_id)

    subject_id = id_table.subject_ids{strcmp(string(id_table.localizer_ids), localizer_id)};
    disp(subject_id)

    cd(fullfile(behav_folder_path, subject_localizer_folders(i).name))
       
    run_localizer_folders = dir('Run*');
    for r = 1:numel(run_localizer_folders)
        run_id = run_localizer_folders(r).name;
        disp(run_id);
        cd(fullfile(behav_folder_path, subject_localizer_folders(i).name, run_localizer_folders(r).name));
        files = dir('RespData*FHOegoallo*');
        file = files(1).name;
        disp(file);
        load(file);
        
        block_start = reshape(info.block_start', 1, []);
        block_names = reshape(repmat(info.block_names', [1,2]), 1, []);
        block_duration = reshape(repmat(info.block_duration', [1,2]), 1, []);

        disp(block_start)
        disp(block_names)
        disp(block_duration)

        % Create the table
        T = table(block_names', block_start', block_duration', 'VariableNames', {' block_names', 'block_start', 'block_duration'});
        
        % Specify the filename
        filename = fullfile(events_folder_path, "FHOF_events_" + subject_id + "_" + run_id + ".tsv");
        disp(filename)
        % Write the table to a TSV file
        writetable(T, filename, 'FileType', 'text', 'Delimiter', '\t');
    end
end

