%-----------------------------------------------------------------------
clear, clc
thisFileDir = fileparts(mfilename('fullpath'));
ROOT_DATASET = fileparts(fileparts(thisFileDir));
derivatives_folder = fullfile(ROOT_DATASET, 'data', 'fmri', 'derivatives');

secondlevel_folder = fullfile(ROOT_DATASET,'data', 'fmri', 'derivatives', 'spm_second_level_analysis_ff');
% Create subject-specific output folder
if ~exist(secondlevel_folder, 'dir')
    mkdir(secondlevel_folder);
end


subjects = ["sub-01","sub-02","sub-03","sub-04","sub-05","sub-07","sub-08","sub-09"];
all_subject_beta_maps = {};


for s = 1:numel(subjects)
    subject = subjects(s)
    firstlevel_folder = fullfile(derivatives_folder, subject, 'spm_first_level_analysis_ff');    
    subject_beta_map = dir(fullfile(firstlevel_folder, 'con*.nii'));

    all_subject_beta_maps{s} = fullfile(subject_beta_map.folder, subject_beta_map.name)
end

matlabbatch{1}.spm.stats.factorial_design.dir = {secondlevel_folder};
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = all_subject_beta_maps';
% matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
% matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
% matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
% matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
% matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
% matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
% matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;


spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch);
