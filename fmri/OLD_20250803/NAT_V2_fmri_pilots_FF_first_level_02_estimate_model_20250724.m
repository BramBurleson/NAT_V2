%-----------------------------------------------------------------------
clear, clc
thisFileDir = fileparts(mfilename('fullpath'));
ROOT_DATASET = fileparts(fileparts(thisFileDir));

subjects = ["sub-01","sub-02","sub-03","sub-04","sub-05","sub-07","sub-08","sub-09"];


for subject = subjects

    matlabbatch = {};
    firstlevel_folder = fullfile(ROOT_DATASET,'data', 'fmri', 'derivatives', subject, 'spm_first_level_analysis_ff');

    spmMatPath    = fullfile(firstlevel_folder, 'SPM.mat');

    % 2) MODEL ESTIMATION  (uses SPM.mat produced by step1)
    % ---------------------------------------------------------------------
    matlabbatch{1}.spm.stats.fmri_est.spmmat         = {char(spmMatPath)};
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = true;   % (default)
    matlabbatch{1}.spm.stats.fmri_est.write_residuals   = false; % don't write ResMS.nii
    
    % RUN 
    spm('defaults','FMRI');
    spm_jobman('initcfg');              % good practice in scripts
    spm_jobman('run',matlabbatch);
end