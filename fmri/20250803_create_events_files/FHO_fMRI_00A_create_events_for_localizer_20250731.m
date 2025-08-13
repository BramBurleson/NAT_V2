% Load the data
% Define subject IDs and corresponding localizer IDs
% subject_ids = {'sub04', 'sub02', 'sub05'}';
% localizer_ids = {'Subj0', 'Subj4', 'Subj3'}';
clear, clc
% K:\BramBurleson\000_datasets_and_scripts\20250625_data_from_Prosper\FHO_Data4_Bram\BehavioralLogs
localizer_ids = {"sbj01", "sbj02", "sbj03", "sbj04", "sbj05", "sbj06", "sbj07", "sbj08", "sbj09", "sbj10", "sbj11", "sbj12"};
subject_ids = {"sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-08", "sub-09", "sub-10", "sub-11", "sub-12"};

thisFileDir = fileparts(mfilename('fullpath'));
cd(thisFileDir)
ROOT_DATASET = fileparts(fileparts(fileparts(thisFileDir)));

% Create a table
id_table = table(subject_ids, localizer_ids);

behav_folder_path = [ROOT_DATASET, '/data', '/behav'];

cd(behav_folder_path)
subject_localizer_folders = dir('sbj*');


for i = 1:numel(subject_localizer_folders)
    localizer_id = subject_localizer_folders(i).name;
    disp(localizer_id)

    subject_id = id_table.subject_ids{strcmp(string(id_table.localizer_ids), localizer_id)};
    disp(subject_id)

    cd(fullfile(behav_folder_path, subject_localizer_folders(i).name))
    events_folder_path = fullfile(ROOT_DATASET, '/data/fmri/derivatives/', subject_id, 'events');

    run_localizer_folders = dir('Run*');
    for r = 1:numel(run_localizer_folders)
        run_id = ['run-' run_localizer_folders(r).name(end)];
        disp(run_id);
        cd(fullfile(behav_folder_path, subject_localizer_folders(i).name, run_localizer_folders(r).name));
        files = dir('RespData*FHOegoallo*');
        file = files(1).name;
        disp(file);
        load(file);

        block_start = reshape(info.block_start', 1, []);
        block_names = reshape(repmat(info.block_names', [1,2]), 1, []);
        block_duration = reshape(repmat(info.block_duration', [1,2]), 1, []);
        % 
        % disp(block_start)
        % disp(block_names)
        % disp(block_duration)

    	%Create.tsv for nilearn
        T = table(block_names', block_start', block_duration', 'VariableNames', {'trial_type', 'onset', 'duration'});
        filename = fullfile(events_folder_path, "FHOF_events_" + subject_id + "_" + run_id + ".tsv");
        disp(["saving nilearn .tsv as " filename])
        writetable(T, filename, 'FileType', 'text', 'Delimiter', '\t');

        %Save variables to .mat for SPM
        unique_names = unique(block_names);
        for c = 1:numel(unique_names)
            cond = unique_names{c};
            idx = strcmp(block_names, cond);
        
            names{c}     = cond;
            onsets{c}    = block_start(idx);
            durations{c} = cell2mat(block_duration(idx));
        end


        filename = fullfile(events_folder_path, ["FHOF_events_" + subject_id + "_" + run_id + ".mat"]);   
        save(filename, 'names', 'onsets', 'durations', '-mat');
        disp(["saving spm .mat as " filename])

        
    end
end

