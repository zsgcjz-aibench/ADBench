#!/bin/sh
#this bash script use fsl to process brain MRI into MNI template

export FSLOUTPUTTYPE='NIFTI'
true_path="/home/lxs/yes"
arr=$1
for file_path in ${arr[*]}; do
    file_dir=${file_path%/*}
    file_name=${file_path##*/}
	fir=${file_name%.*}
	if [ ! -d "$2/tmp/${fir}" ]; then
		mkdir $2/tmp/${fir}
	fi
	cd ${file_dir}
    cp ${file_name} $2/tmp/${fir}/
    cd $2/tmp/${fir}/
	echo ${file_name}
	${true_path}/bin/fslreorient2std ${file_name} T1.nii  
	line=`${true_path}/bin/robustfov -i T1.nii | grep -v Final | head -n 1`

    x1=`echo ${line} | awk '{print $1}'`
    x2=`echo ${line} | awk '{print $2}'`
    y1=`echo ${line} | awk '{print $3}'`
    y2=`echo ${line} | awk '{print $4}'`
    z1=`echo ${line} | awk '{print $5}'`
    z2=`echo ${line} | awk '{print $6}'`

    x1=`printf "%.0f", $x1`
    x2=`printf "%.0f", $x2`
    y1=`printf "%.0f", $y1`
    y2=`printf "%.0f", $y2`
    z1=`printf "%.0f", $z1`
    z2=`printf "%.0f", $z2`

    #step3 is to cut the brain to get area of interest (roi), sometimes it cuts part of the brain
    ${true_path}/bin/fslmaths T1.nii -roi $x1 $x2 $y1 $y2 $z1 $z2 0 1 T1_roi.nii
    #cp T1.nii T1_roi.nii

    #step4: remove skull -g 0.1 -f 0.45
    ${true_path}/bin/bet T1_roi.nii T1_brain.nii -R

    #step5: registration from cut to MNI
    ${true_path}/bin/flirt -in T1_brain.nii -ref $true_path/data/standard/MNI152_T1_1mm_brain -omat orig_to_MNI.mat
    #${true_path}/bin/flirt -in T1_roi.nii -ref $true_path/data/standard/MNI152_T1_1mm -omat orig_to_MNI.mat

    #step6: apply matrix onto original image
    ${true_path}/bin/flirt -in T1.nii -ref $true_path/data/standard/MNI152_T1_1mm_brain -applyxfm -init orig_to_MNI.mat -out T1_MNI.nii

    #step7: skull remove -f 0.3 -g -0.0
    ${true_path}/bin/bet T1_MNI.nii T1_MNI_brain.nii -R -f $3 -g $4

    # step8: register the skull removed scan to MNI_brain_only template again to fine tune the alignment
    ${true_path}/bin/flirt -in T1_MNI_brain.nii -ref $true_path/data/standard/MNI152_T1_1mm_brain -out T1_MNI_brain.nii

    #step9: rename and move final file
    mv T1_MNI_brain.nii $2/scans/${file_name}
	# clear tmp folder
	rm -rf $2/tmp/${fir}	
done
