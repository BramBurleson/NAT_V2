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
%conditions, trials,  which_key, expected_response, reaction times.
fields = {'resptype', 'expected_resp', 'RT'};  % List of field names (could be from fieldnames)

%extract trial and block level data save to csv and bring to python so we're
%out of this f-ing quagmire.
%you can do like rep10 and then add it to dict for conditions and blocks.
n_trial = 6*6*10;
% iterate through subject folders and load FF data

        %what is the minimum data that I need?
        %conditions, trials,  which_key, expected_response, reaction times

for s = 1:length(subjects)
    trial = 1:n_trial;
    trial_condition       = strings(n_trial,1);
    trial_block           = nan(n_trial, 1);
    trial_performance     = nan(n_trial, 1);

   subject_ff = dir(fullfile(subjects(s).folder + "\" + subjects(s).name + "\ff", "*ResponseData*"));
    if ~isempty(subject_ff)
        disp(subject_ff.name)
        data = load(subject_ff.folder + "\" + subject_ff.name)
        
        trial_resp_type = [data.info.resptype{:}]';
        trial_expected_resp = vertcat(data.info.expected_resp{:})

        trial_expected_resp2 = reshape(data.info.expected_resp, [], 1)

        resp = data.info.resptype{:};
        empties = cellfun('isempty', resp);
        resp(empties) = {NaN};         % replace empty cells with NaN
        resp = cell2mat(resp);         % convert to numeric array

        % expected = data.info.expected_resp{1};
        resp = resp(:);
        expected = trial_expected_resp2(:);
        diff_result = resp - expected;

        trial_RT = [data.info.RT{:}]';

        goodTriggers = data.info.triggerCodes(data.info.triggerCodes~=64)
        for c = 1:2:length(goodTriggers)
            triggerCode = goodTriggers(c);
            idx = (c + 1) / 2; 
            trials_in_block = (idx-1)*10+1:idx*10;
            trial_condition(trials_in_block) = condition(triggerCode);
            trial_block(trials_in_block) = idx;
            trial_performance(trials_in_block) = data.info.performance(idx);
        end
           subj_data = table(trial_block, trial_condition, trial_performance, trial_resp_type, trial_expected_resp, trial_RT);
           filename = fullfile(subject_ff.folder, [subjects(s).name '_ff.csv']);
           writetable(subj_data, filename);
    end
end

