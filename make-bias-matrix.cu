#include <iostream>
#include <chrono>
#include <boost/program_options.hpp>

int N = 1000;
int batch_size = 32;
int output_size = 14*14;
const int blocks = 2048;

__global__ void kernel_row(float* bias_matrix,float* bias,int output_size,int batch_size){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( output_size <= tid )return;
	float bias_value = bias[tid];
	for(int i = batch_size-1;i >= 0;i--){
		bias_matrix[i * output_size + tid] = bias_value;
	}
}
__global__ void kernel_col(float* bias_matrix,float* bias,int output_size,int batch_size){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( batch_size <= tid )return;
	for(int i = output_size - 1;i >= 0 ;i--){
		bias_matrix[tid * output_size + i] = bias[i];
	}
}
__global__ void kernel_col_asm(float* bias_matrix,float* bias,int output_size,int batch_size){
	asm volatile("{\n\t"
#include "converted_kernel_col_asm.ptx"
		"}"
		::"l"(bias_matrix),"l"(bias),"r"(output_size),"r"(batch_size)
			);
}

__global__ void kernel_row_asm(float* bias_matrix,float* bias,int output_size,int batch_size){
	asm volatile("{\n\t"
#include "converted_kernel_row_asm.ptx"
		"}"
		::"l"(bias_matrix),"l"(bias),"r"(output_size),"r"(batch_size)
			);
}

int main(int argc,char** argv){
	boost::program_options::options_description opt("option");
	opt.add_options()
		("help,h","show help")
		("output,o",boost::program_options::value<int>()->default_value(output_size),	"matrix output size")
		("batch,b",boost::program_options::value<int>()->default_value(batch_size),	"matrix batch size")
		("calc,c",boost::program_options::value<int>()->default_value(N),	"calculation count");
	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::parse_command_line(argc,argv,opt),vm);
	boost::program_options::notify(vm);
	if(vm.count("help")!=0){
		std::cout<<opt<<std::endl;
		return 0;
	}
	N = vm["calc"].as<int>();
	batch_size = vm["batch"].as<int>();
	output_size = vm["output"].as<int>();

	float *d_bias,*d_bias_matrix;
	float *h_bias,*h_bias_matrix;
	cudaMalloc( (void**)&d_bias, sizeof(float) * output_size);
	cudaMalloc( (void**)&d_bias_matrix, sizeof(float) * output_size * batch_size);
	cudaMallocHost( (void**)&h_bias, sizeof(float) * output_size);
	cudaMallocHost( (void**)&h_bias_matrix, sizeof(float) * output_size * batch_size);
	for(int i = 0;i < output_size;i++) h_bias[i] = (i+1)/100.0f;

	/*cudaMemset( d_bias_matrix, 0, sizeof(float) * output_size * batch_size);
	cudaMemcpy( d_bias, h_bias, sizeof(float) * output_size , cudaMemcpyHostToDevice);
	{
		auto start = std::chrono::system_clock::now();
		for(int i = 0;i < N;i++){
			for(int j = 0;j < batch_size;j++) cudaMemcpy( d_bias_matrix + j * output_size, d_bias, sizeof(float) * output_size, cudaMemcpyDeviceToDevice );
		}
		cudaThreadSynchronize();
		auto stop = std::chrono::system_clock::now();
		float calc_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/(float)N;
		std::cout<<"cudaMemcpy copy : "<<calc_time<<" [us]"<<std::endl;
	}*/

	cudaMemset( d_bias_matrix, 0, sizeof(float) * output_size * batch_size);
	cudaMemcpy( d_bias, h_bias, sizeof(float) * output_size , cudaMemcpyHostToDevice);

	{
		//auto start = std::chrono::system_clock::now();
		for(int i = 0;i < N;i++){
			kernel_col<<<blocks,(batch_size+blocks-1)/blocks>>>(d_bias_matrix,d_bias,output_size,batch_size);
		}
		cudaThreadSynchronize();
		//auto stop = std::chrono::system_clock::now();
		//float calc_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/(float)N;
		//std::cout<<"kernel_col copy : "<<calc_time<<" [us]"<<std::endl;
	}
	cudaMemset( d_bias_matrix, 0, sizeof(float) * output_size * batch_size);
	cudaMemcpy( d_bias, h_bias, sizeof(float) * output_size , cudaMemcpyHostToDevice);
	{
		//auto start = std::chrono::system_clock::now();
		for(int i = 0;i < N;i++){
			kernel_row<<<blocks,(output_size+blocks)/blocks>>>(d_bias_matrix,d_bias,output_size,batch_size);
		}
		cudaThreadSynchronize();
		//auto stop = std::chrono::system_clock::now();
		//float calc_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/(float)N;
		//std::cout<<"kernel_row copy : "<<calc_time<<" [us]"<<std::endl;
	}
	cudaMemset( d_bias_matrix, 0, sizeof(float) * output_size * batch_size);
	cudaMemcpy( d_bias, h_bias, sizeof(float) * output_size , cudaMemcpyHostToDevice);
	{
		//auto start = std::chrono::system_clock::now();
		for(int i = 0;i < N;i++){
			kernel_col_asm<<<128,(batch_size+127)/128>>>(d_bias_matrix,d_bias,output_size,batch_size);
		}
		cudaThreadSynchronize();
		//auto stop = std::chrono::system_clock::now();
		//float calc_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/(float)N;
		//std::cout<<"kernel_asm_col copy : "<<calc_time<<" [us]"<<std::endl;
	}
	cudaMemset( d_bias_matrix, 0, sizeof(float) * output_size * batch_size);
	cudaMemcpy( d_bias, h_bias, sizeof(float) * output_size , cudaMemcpyHostToDevice);
	{
		//auto start = std::chrono::system_clock::now();
		for(int i = 0;i < N;i++){
			kernel_row_asm<<<128,(output_size+127)/128>>>(d_bias_matrix,d_bias,output_size,batch_size);
		}
		cudaThreadSynchronize();
		//auto stop = std::chrono::system_clock::now();
		//float calc_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/(float)N;
		//std::cout<<"kernel_asm_row copy : "<<calc_time<<" [us]"<<std::endl;
	}

	cudaMemcpy( h_bias_matrix, d_bias_matrix, sizeof(float) * output_size * batch_size, cudaMemcpyDeviceToHost);
	/*for(int j = 0;j < output_size;j++){
		for(int i = 0;i < batch_size;i++){
			printf("%.3f ",h_bias_matrix[i * output_size + j]);
		}
		printf("\n");
	}*/

	cudaFree( d_bias );
	cudaFreeHost( h_bias );
	cudaFree( d_bias_matrix );
	cudaFreeHost( h_bias_matrix );
}
