%-----------------------------------------------------------------------
clear, clc
thisFileDir = fileparts(mfilename('fullpath'));
ROOT_DATASET = fileparts(fileparts(fileparts(thisFileDir)));
derivatives_folder = fullfile(ROOT_DATASET, 'data', 'fmri', 'derivatives');

secondlevel_folder = fullfile(ROOT_DATASET,'data', 'fmri', 'derivatives', 'spm_second_level_analysis_ff');

contrasts = {
    "Faces_Allo_Ego", "Flags_Allo_Ego", "Faces_Allo_Color", "Flags_Allo_Color"
};

for c = 1:numel(contrasts)
    matlabbatch = {};
    contrast = contrasts{c}
    contrast_folder = fullfile(secondlevel_folder, contrast)
    spmMatPath    = fullfile(contrast_folder, 'SPM.mat');

    % 3) CONTRAST MANAGER
    % ---------------------------------------------------------------------
    matlabbatch{1}.spm.stats.con.spmmat = {char(spmMatPath)};
    % --- T?contrast: OTA_Allo?Ego
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name        = 'Faces_Allo-Ego';
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights     = 1;
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep     = 'repl';  % same weights per run

    matlabbatch{1}.spm.stats.con.delete = 1;  % keep any existing contrasts

    spm('defaults', 'FMRI');
    fprintf('Running second-level estimation and contrast for: %s\n', spmMatPath);
    spm_jobman('run', matlabbatch);
end
