
bias: ./make-bias-matrix.cu ./kernel_col_asm.ptx
	./ptx-inline-converter -i kernel_col_asm.ptx -o converted_kernel_col_asm.ptx
	./ptx-inline-converter -i kernel_row_asm.ptx -o converted_kernel_row_asm.ptx
	nvcc make-bias-matrix.cu -std=c++11 -lcublas
