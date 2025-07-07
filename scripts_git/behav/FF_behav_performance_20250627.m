%Get current file path and locate subject folders
this_file = mfilename("fullpath");
ROOT_DATASET = fileparts(fileparts(fileparts(this_file)));
datafolder = ROOT_DATASET + "\data\behav";
subjects = dir(fullfile(datafolder, '*sub*'));
% 
% %Define condition mapping:
keys = [1, 3, 7, 11, 13, 15]
values = {'allo_faces', 'color_faces', 'ego_faces',
        'allo_flags', 'color_flags', 'ego_flags'};;
condition = containers.Map(keys, values);

%what is the minimum data that I need?
%conditions, trials,  which_key, trial_expected_response, reaction times.
fields = {'resptype', 'trial_expected_resp', 'RT'};  % List of field names (could be from fieldnames)

%extract trial and block level data save to csv and bring to python so we're
%out of this f-ing quagmire.
%you can do like rep10 and then add it to dict for conditions and blocks.
n_trial = 6*6*10;
% iterate through subject folders and load FF data

        %what is the minimum data that I need?
        %conditions, trials,  which_key, trial_expected_response, reaction times

for s = 1:length(subjects)
    trial = 1:n_trial;
    trial_condition       = strings(n_trial,1);
    trial_block           = nan(n_trial, 1);
    trial_performance     = nan(n_trial, 1);

   subject_ff = dir(fullfile(subjects(s).folder + "\" + subjects(s).name + "\ff", "corr*ResponseData*"));
    if ~isempty(subject_ff)
        disp(subject_ff.name)
        data = load(subject_ff.folder + "\" + subject_ff.name)     

        trial_resp_type= horzcat(data.info.resptype{:})';
        empties = cellfun('isempty', trial_resp_type);
        trial_resp_type(empties) = {NaN};         % replace empty cells with NaN
        trial_resp_type= cell2mat(trial_resp_type);         % convert to numeric array
 
        trial_expected_resp = cell2mat(vertcat(data.info.expected_resp{:}))

        diff_result = trial_resp_type- trial_expected_resp;
        disp("number of errors or misses = " + string(sum(diff_result~=0)))

        trial_RT = nan(size(trial_resp_type));

        trial_RT(~isnan(trial_resp_type)) = cell2mat([data.info.RT{:}]');

    

        goodTriggers = data.info.triggerCodes(data.info.triggerCodes~=64)
        for c = 1:2:length(goodTriggers)
            triggerCode = goodTriggers(c);
            disp(triggerCode)
            idx = (c + 1) / 2; 
            trials_in_block = (idx-1)*10+1:idx*10;
            disp(trials_in_block)
            trial_condition(trials_in_block) = condition(triggerCode);
            trial_block(trials_in_block) = idx;
            trial_performance(trials_in_block) = data.info.performance(idx);
        end
           subj_data = table(trial_block, trial_condition, trial_performance, trial_resp_type, trial_expected_resp, trial_RT);
           filename = fullfile(subject_ff.folder, [subjects(s).name '_corr_ff.csv']);
           writetable(subj_data, filename);
    end
end

