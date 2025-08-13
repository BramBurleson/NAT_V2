%-----------------------------------------------------------------------
clear, clc
thisFileDir = fileparts(mfilename('fullpath'));
ROOT_DATASET = fileparts(fileparts(thisFileDir));

subjects = ["sub-01","sub-02","sub-03","sub-04","sub-05","sub-07","sub-08","sub-09"];


for subject = subjects

    firstlevel_folder = fullfile(ROOT_DATASET,'data', 'fmri', 'derivatives', subject, 'spm_first_level_analysis_ff');
    spmMatPath    = fullfile(firstlevel_folder, 'SPM.mat');

    % ---------------------------------------------------------------------
    % 3) CONTRAST MANAGER
    % ---------------------------------------------------------------------
    matlabbatch             = {};
    matlabbatch{1}.spm.stats.con.spmmat = {char(spmMatPath)};
    % --- T?contrast: OTA_Allo?Ego
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name        = 'Faces_Ego-Color';
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights     = [0 -1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep     = 'repl';  % same weights per run
    
    matlabbatch{1}.spm.stats.con.delete = 1;  % keep any existing contrasts
    
    % ---------------------------------------------------------------------
    % RUN EVERYTHING
    % ---------------------------------------------------------------------
    spm('defaults','FMRI');
    spm_jobman('initcfg');              % good practice in scripts
    spm_jobman('run',matlabbatch);
end