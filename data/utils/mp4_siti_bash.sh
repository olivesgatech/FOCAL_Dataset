#!/bin/bash
# Go to the directory you want to start from
# Execute this script from there
#/Users/yash-yee/Desktop/FOCAL/blurred/deepen3ddata-s6nwj6hi-defaultproject-24Ir8HAnKidAuLlxPZf5zOQh/processed/images
search_dir=$(pwd)

# suffix to be trimmed
suffix="/"
# file extension
extension=".mp4"
jsonExtension=".json"

for dirName in $(ls -d */);
do
  cd $search_dir/$dirName
  # echo ">>$dirName";
  name=${dirName%"$suffix"}
  videoName="${name}${extension}"
  jsonFileName="${name}${jsonExtension}"
  # echo ">>$trimmedName"
  # echo "Pwd:"
  # pwd
 
  # go into first folder found
  for g in $(ls -d */ | head -1);
  do
    # echo ">> >>$g"
    cd $g
    # echo ">> >>Pwd:"
    # pwd
    for h in $(ls -d */ | head -1);
    do
      # echo ">> >> >> $h"
      cd $h
      # echo "Pwd: "
      # pwd
      # call script
      # rm *$extension

      FILE=00000.pcd.jpg

	  if [ -f "$FILE" ]; then
	    ffmpeg -r 5 -framerate 10 -i %5d.pcd.jpg -vcodec mpeg4 $videoName
	  else 
	    ffmpeg -r 5 -framerate 10 -i cam1_%5d.jpg -vcodec mpeg4 $videoName
	  fi

      # rm *$jsonExtension
      # ffmpeg -r 5 -framerate 10 -i %5d.pcd.jpg -vcodec mpeg4 $videoName
      sleep 2 
      siti -f $videoName > $jsonFileName
      cd ..
    done
    cd ..
  done
 
  echo "$(tput setaf 4)Processed $name$(tput sgr 0)";
  cd ..
done
echo "DONE"