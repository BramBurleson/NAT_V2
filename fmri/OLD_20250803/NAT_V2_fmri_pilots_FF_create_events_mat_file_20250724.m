% Setup
%-----------------------------------------------------------------------
clear, clc
thisFileDir       = mfilename('fullpath');
ROOT_DATASET      = fileparts(fileparts(fileparts(thisFileDir)));
derivatives_folder = fullfile(ROOT_DATASET,'data','fmri','derivatives');



subjects = ["sub-01","sub-02","sub-03","sub-04","sub-05","sub-07","sub-08","sub-09"];


for subject = subjects
    disp(subject)

    evt_dir  = fullfile(derivatives_folder,subject,"events");
    evt_list = dir(fullfile(evt_dir,"*.tsv"));  % catches all BIDS event files
    disp(length(evt_list));

    % Cross?platform folder paths
    datafolder   = fullfile(ROOT_DATASET, 'data', 'fmri', 'derivatives', subject, 'func');
    events_folder = fullfile(ROOT_DATASET, 'data', 'fmri', 'derivatives', subject, 'events');
    firstlevel_folder = fullfile(ROOT_DATASET,'data', 'fmri',subject, 'spm_first_level_analysis');

    for f = 1:numel(evt_list)
%         disp(f)
        
        % --------- read the events.tsv ---------
        t = readtable(fullfile(evt_list(f).folder,evt_list(f).name), ...
                      'FileType','text','Delimiter','\t');
%         disp(t)

        % --------- build names/onsets/durations ---------
        [trial_types,~,ic] = unique(t.trial_type,'stable');

        names     = cellstr(trial_types(:)');              % 1×K
        onsets    = accumarray(ic, t.onset,    [], @(x){x(:)'});  % 1×K
        durations = accumarray(ic, t.duration, [], @(x){x(:)'});  % 1×K

        % --------- save exactly what SPM wants ---------
        [~,base,~] = fileparts(evt_list(f).name);          % strip .tsv
%         disp(base)
        filename = fullfile(evt_dir,[base '.mat']);
        disp(['saving ' filename]);
        save(filename, 'names','onsets','durations');
   
    end
end
