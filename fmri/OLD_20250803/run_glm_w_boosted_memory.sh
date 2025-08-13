#!/bin/bash
## Bash script to run single subject nilearn glm boosted.
#source ~/.bash_profile
# Load my environment variables
# source ~/.bash_profile

DOES NOT WORK AND LIKELY WILL NOT


nthreads=12
mem=20 #gb
container=docker #docker or singularity

#Begin:
#Convert virtual memory from gb to mb
mem=`echo "${mem//[!0-9]/}"` #remove gb at end
mem_mb=`echo $(((mem*1000)-5000))` #reduce some memory for buffer space during pre-processing
# subject names



#for subID  in $sbNames 
for subID  in "${sbNames[@]}"
do
  # Display current subject 
  echo "Running fMRIPrep preprocessing for " ${subID}
  
  # subject number  
  STR=$subID
  subj=${STR:4:5}
  #echo $subj
  
  # Run fmriprep preprocessing for the current subject( without fmap correction & no --fs-no-reconall) 
 fmriprep-docker $bids_root_dir $bids_root_dir/derivatives participant --participant-label $subj --md-only-boilerplate --fs-license-file $FS_LICENSE --fs-no-reconall --output-spaces MNI152NLin2009cAsym:res-2 anat:res-2 --nthreads $nthreads --stop-on-first-crash --mem_mb $mem_mb --work-dir /home/bramb --ignore fieldmaps 
done

