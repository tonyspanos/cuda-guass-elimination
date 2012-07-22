/* Part of program for solving linear equations using CUDA C */
/* Uses Gaussian elimination */ 
/* Files requirede : Kernel.cu (this file) , main.cpp, fread.cpp, common.h, Devi ceFunc.cu, data.txt */

//Kernel function that executes on the device 

#include "Common.h" 
#include<cuda.h> 

__device__ __global__ void Kernel(float *a_d , float *b_d ,int size) 
{ 
	int idx = threadIdx.x ;
	int idy = threadIdx.y ; 
	
	//int width = size ; 
	//int height = size ; 
	
	//Allocating memory in the share memory of the device 
	
	__shared__ float temp[16][16]; 
	
	//Copying the data to the shared memory 
	
	temp[idy][idx] = a_d[(idy * (size+1)) + idx] ;
	
	for(int i =1 ; i<size ;i++) 
	{ 
		if((idy + i) < size) // NO Thread divergence here 
		{ 
			float var1 =(-1)*( temp[i-1][i-1]/temp[i+idy][i-1]);
			temp[i+idy][idx] = temp[i-1][idx] +((var1) * (temp[i+idy ][idx]));
		} 
		
		__syncthreads(); //Synchronizing all threads before Next iterat ion 
	} 
	
	b_d[idy*(size+1) + idx] = temp[idy][idx]; 
}

