%-----------------------------------------------------------------------
clear, clc
thisFileDir = fileparts(mfilename('fullpath'));
ROOT_DATASET = fileparts(fileparts(thisFileDir));

subjects = ["sub-01","sub-02","sub-03","sub-04","sub-05","sub-07","sub-08","sub-09"];


for subject = subjects

    matlabbatch = {};

    % Cross?platform folder paths
    datafolder   = fullfile(ROOT_DATASET, 'data', 'fmri', 'derivatives', subject, 'func');
    events_folder = fullfile(ROOT_DATASET, 'data', 'fmri', 'derivatives', subject, 'events');
    firstlevel_folder = fullfile(ROOT_DATASET,'data', 'fmri', 'derivatives', subject, 'spm_first_level_analysis_ff');
    % Create subject-specific output folder
    if ~exist(firstlevel_folder, 'dir')
        mkdir(firstlevel_folder);
    end

    nii_files = dir(fullfile(datafolder, '*ff*bold_smooth_6*.nii'));
    
    [~, sortIdx] = sort({nii_files.name});  % Ensure files are sorted by run number
    nii_files = nii_files(sortIdx);
    
    matlabbatch{1}.spm.stats.fmri_spec.dir = cellstr(firstlevel_folder);
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 1.5;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 44; %based on raw BOLD data
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 22;

        
    for i = 1:numel(nii_files)
        nii_file = nii_files(i);
        disp(nii_file.name)
        %get scans
        V = spm_vol(fullfile(nii_file.folder, nii_file.name));
        n_vols = numel(V);
        scans = cell(n_vols, 1);
        for ii = 1:n_vols
            scans{ii} = [fullfile(nii_file.folder, nii_file.name) ',' num2str(ii)];
        end
%         run_id = regexp(nii_files(i).name, 'run-\d+', 'match', 'once');
        events_file = dir(fullfile(events_folder, sprintf('*%s*_events.mat', 'ff')));
        disp(events_file(1).name)
        selected_confounds_file = dir(fullfile(nii_file.folder, sprintf('*%s*confounds_timeseries_SELECTED.txt', 'ff')));
        disp(selected_confounds_file(1).name)
                
        events_path = fullfile(events_file(1).folder, events_file(1).name);
        confounds_path = fullfile(selected_confounds_file(1).folder, selected_confounds_file(1).name);
        matlabbatch{1}.spm.stats.fmri_spec.sess(i).multi = cellstr(events_path);
        matlabbatch{1}.spm.stats.fmri_spec.sess(i).scans = scans;
        matlabbatch{1}.spm.stats.fmri_spec.sess(i).cond = struct([]);
        matlabbatch{1}.spm.stats.fmri_spec.sess(i).regress = struct('name', {}, 'val', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(i).multi_reg = cellstr(confounds_path);
        matlabbatch{1}.spm.stats.fmri_spec.sess(i).hpf = 100;
    %     potential difference in nilearn it is 100 
        
        matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
        matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
        matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
        matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
        matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
        matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
    end
    % save('matlabbatch.mat', 'matlabbatch')
    disp("here")
    
    spm('defaults', 'FMRI');
    spm_jobman('interactive', matlabbatch);   % opens the batch tree
    spm_jobman('run', matlabbatch);
end
