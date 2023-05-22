input_dir="submission/PFNANO_V9_17Feb23_PostNanoTools/"
output_dir="merged_files/PFNANO_V9_17Feb23_PostNanoTools/"
mkdir -p ${output_dir}
cp ${input_dir}*coffea ${output_dir}
bumerge ${output_dir} -o ${output_dir} -j 4

