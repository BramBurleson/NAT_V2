%-----------------------------------------------------------------------
clear, clc
thisFileDir = fileparts(mfilename('fullpath'));
ROOT_DATASET = fileparts(fileparts(fileparts(thisFileDir)));

subjects = ["sub-01","sub-02","sub-03","sub-04","sub-05","sub-07","sub-08","sub-09"];

%for id - ing contrasts : idx = cellfun(@(x) contains(x, 'Allo Faces'), SPM.xX.name);

contrasts = {
    'Faces_Allo_Ego', [1   0   -1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0];
    'Flags_Allo_Ego', [0   0   0   1   -1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0];
    'Faces_Allo_Color', [1   -1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0];
    'Flags_Allo_Color', [0   0   0   1   0   -1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0];
};


for subject = subjects
    disp(subject)
    subjectfolder = fullfile(ROOT_DATASET, 'data', 'fmri', 'derivatives', subject);
    datafolder   = fullfile(subjectfolder, 'func');
    firstlevel_folder   = fullfile(subjectfolder, 'spm_first_level_analysis_ff');
    spmMatPath    = fullfile(firstlevel_folder, 'SPM.mat');

    for i = 1:size(contrasts,1)
        contrast_name = contrasts{i, 1};
        contrast_vector = contrasts{i, 2};

        con_file_name     = ['con_' contrast_name '.nii'];
        spmt_file_name    = ['SPMT_' contrast_name '.nii'];
        con_file_wildcard = 'con_0*.nii';
        spmt_file_wildcard = 'SPMT_0*.nii';
        spmt2_file_wildcard = 'spmT_0*.nii';
        % 
        old_contrasts = {con_file_name, spmt_file_name, con_file_wildcard, spmt_file_wildcard, spmt2_file_wildcard};
        
        for j = 1:numel(old_contrasts)
            files = dir(fullfile(firstlevel_folder, old_contrasts{j}));
            for k = 1:numel(files)
                fullpath = fullfile(files(k).folder, files(k).name);
                if exist(fullpath, 'file')
                    fprintf('Deleting %s\n', fullpath);
                    delete(fullpath);
                end
            end
        end
    
    
        
        %create batch and assign contrast name and contrast vector
        matlabbatch = {};
        matlabbatch{1}.spm.stats.con.spmmat = cellstr(spmMatPath);

        matlabbatch{1}.spm.stats.con.consess{1}.tcon.name        = contrast_name;
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights     = contrast_vector;
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep     = 'repl';  % same weights per run
        matlabbatch{1}.spm.stats.con.delete = false;  % keep any existing contrasts
    
        % run SPM
        spm('defaults','FMRI');
        spm_jobman('initcfg');              % good practice in scripts
        spm_jobman('run',matlabbatch);

        % Rename output files
        con_file   = dir(fullfile(firstlevel_folder, 'con_0*.nii'));
        con_file = fullfile(con_file.folder, con_file.name);
        spmt_file  = dir(fullfile(firstlevel_folder, 'spmT_0*.nii'));
        spmt_file  = fullfile(spmt_file.folder, spmt_file.name);
        
                
        movefile(con_file,  fullfile(firstlevel_folder, con_file_name));
        movefile(spmt_file, fullfile(firstlevel_folder, spmt_file_name));
    end
end