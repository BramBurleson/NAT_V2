#git bash:
#cd to folder with video then:
#read -p "please enter file directory: " file_directory
#read -p "please enter input file name: " input_name
#read -p "please enter output file name: " output_name
#read -p "please enter cut_start: " cut_start
#read -p "please enter cut_end: " cut_end

#cd file_directory

#ffmpeg -i input_name -c:v libx264 -preset slow -crf 23 -vf scale=1280:-2 -r 30 -an output_name"

#ffmpeg -i input.mp4 -ss cut_start -t cut_end -c:v copy -c:a copy output_name_cut.mp4


#!/usr/bin/env bash
# Minimal FFmpeg helper — keeps just the essentials

read -rp "Directory containing the video: " dir
read -rp "Input file name (e.g. input.mp4): " in
read -rp "Output file name (e.g. output.mp4): " out
read -rp "Clip start time (HH:MM:SS[.ms]): " ss
read -rp "Clip end time (HH:MM:SS[.ms]): " to

cd "${dir}" || exit 1

# Re‑encode whole video
ffmpeg -i "${in}" -c:v libx264 -preset slow -crf 23 -vf scale=1280:-2 -r 30 -an "${out}"

# Quick copy‑mode clip
ffmpeg -ss "${ss}" -to "${to}" -i "${in}" -c copy "${out%.*}_cut.${out##*.}"