%-----------------------------------------------------------------------
clear, clc
thisFileDir = fileparts(mfilename('fullpath'));
ROOT_DATASET = fileparts(fileparts(thisFileDir));
derivatives_folder = fullfile(ROOT_DATASET, 'data', 'fmri', 'derivatives');

secondlevel_folder = fullfile(ROOT_DATASET,'data', 'fmri', 'derivatives', 'spm_second_level_analysis_ff');
spmMatPath    = fullfile(secondlevel_folder, 'SPM.mat');

% 2) MODEL ESTIMATION  (uses SPM.mat produced by step1)
% ---------------------------------------------------------------------
matlabbatch{1}.spm.stats.fmri_est.spmmat         = {char(spmMatPath)};
matlabbatch{1}.spm.stats.fmri_est.method.Classical = true;   % (default)
matlabbatch{1}.spm.stats.fmri_est.write_residuals   = false; % don't write ResMS.nii

% 3) CONTRAST MANAGER
% ---------------------------------------------------------------------
matlabbatch{2}.spm.stats.con.spmmat = {char(spmMatPath)};
% --- T?contrast: OTA_Allo?Ego
matlabbatch{2}.spm.stats.con.consess{1}.tcon.name        = 'Faces_Allo-Ego';
matlabbatch{2}.spm.stats.con.consess{1}.tcon.weights     = 1;
matlabbatch{2}.spm.stats.con.consess{1}.tcon.sessrep     = 'repl';  % same weights per run

matlabbatch{2}.spm.stats.con.delete = 1;  % keep any existing contrasts

spm('defaults', 'FMRI');
fprintf('Running second-level estimation and contrast for: %s\n', spmMatPath);
spm_jobman('run', matlabbatch);
