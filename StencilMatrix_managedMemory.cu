#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <ostream>

using namespace std;

#define DSIZE 512 
#define RADIUS 3
#define BLOCK_SIZE 32

__global__ void stencil_2d(int *in, int *out) {
	
	int row = threadIdx.y + blockIdx.y * blockDim.y; //row - can switch row and column?
    int column = threadIdx.x + blockIdx.x * blockDim.x; //column

	// Read input elements into shared memory
	int result = in[column + DSIZE * row]; 
    out[column+DSIZE*row] = result;

	if (row >= RADIUS && column >= RADIUS && (DSIZE-row) > RADIUS && (DSIZE-column) > RADIUS ) {
		// Apply the stencil
        for (int offset = -RADIUS; offset <= RADIUS; offset++){
            __syncthreads(); //makes sure we have access to everything in temp accessed across multiple threads
            
            //avoid double-counting 
            if(offset!=0){
                result += in[(row + offset)*DSIZE + column];
                result += in[row*DSIZE + column + offset];
            }
        }
        // Store the result
        out[column+DSIZE*row] = result;
	}
}

// Square matrix multiplication on CPU : C = A * B
void matrix_mul_cpu(const int *A, const int *B, int *C, int size) {
  // i iterates over rows of matrix A
  for (int i = 0; i<size; i++){
    // j iterates over columns of matrix B
    for (int j = 0; j<size; j++){
        int temp = 0;
        // k indexes which item in the ith row of A and jth column of B we are multiplying
        for (int k = 0; k<size; k++){
            //i is analagous to idx, j to idy, size to n
            temp += A[i * size + k] * B [k * size + j];
        }
    C[i*size + j]= temp;
    }
  }
}

// Square matrix multiplication on GPU : C = A * B
__global__ void matrix_mul_gpu(const int *A, const int *B, int *C, int size) {

    // create thread x index
    // create thread y index
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idy = blockIdx.x * blockDim.x + threadIdx.x;;
    // Make sure we are not out of range
    if ((idx < size) && (idy < size)) {
        int temp = 0;
        for (int i = 0; i < size; i++){
            //Add dot product of row and column
            temp += A [idx * size +i] * B [i * size +idy];
        }
        C[idx*size+idy] = temp;                    
    }

}

// Error Checking for stencil
int error_stencil (int* stencilled, int* original){
    for (int i = 0; i < DSIZE; ++i) {
        for (int j = 0; j < DSIZE; ++j) {

            if (i < RADIUS || DSIZE-i<= RADIUS) {
                if (stencilled[i*DSIZE+j] != original[i*DSIZE+j]) {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, stencilled[i*DSIZE+j], original[i*DSIZE+j]);
                    return -1;
                }
            }
            else if (j < RADIUS || DSIZE-j<= RADIUS) {
                if (stencilled[i*DSIZE+j] != original[i*DSIZE+j]) {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, stencilled[i*DSIZE+j], original[i*DSIZE+j]);
                    return -1;
                }
            }		 
            else { 
                int expectedResult = original[i*DSIZE+j];
                for (int k=1; k<=RADIUS; k++){
                    expectedResult +=original[(i+k)*DSIZE+j];
                    expectedResult +=original[(i-k)*DSIZE+j];
                    expectedResult +=original[i*DSIZE+(j+k)];
                    expectedResult +=original[i*DSIZE+(j-k)];
                }
                if (stencilled[i*DSIZE+j] != expectedResult) {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, stencilled[i*DSIZE+j], expectedResult);
                    return -1;
                }
            }
        }
    }
    printf("no stencil error \n");
    return 0;
}

// Error checking for matrix multiplication - modify from cc to work for gpu
int error_matrix_mul_gpu(const int *A, const int *B, int *C, int size){
     for (int i=0; i<size; i++){
        for(int j=0; j<size; j++){
            int expectedResult=0;
            for (int k=0; k<size; k++){
                expectedResult += A[i*size+k]*B[k*size+j];
            }
            if (C[i*size+j]!=expectedResult){
                printf("Multiplication wrong at [%d,%d], was: %d, should be: %d\n", i,j, C[i*size+j], expectedResult);
                    return -1;
            }
        }
     }
     printf("no multiplication error \n");
     return 0;
}

// Cuda error checking macro (from matrix multiplication assignment)
#define cudaCheckErrors(msg)                                   \
   do {                                                        \
       cudaError_t __err = cudaGetLastError();                 \
       if (__err != cudaSuccess) {                             \
           fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                   msg, cudaGetErrorString(__err),             \
                   __FILE__, __LINE__);                        \
           fprintf(stderr, "*** FAILED - ABORTING\n");         \
           exit(1);                                            \
       }                                                       \
   } while (0)

//Elliott helped me with print Matrix function
int printMatrix(int *A, int limit = 8) {
    std::cout<<"-              -\n";
    for (int i = 0; i < limit; i++) {
        std::cout<<"| ";
        for (int j = 0; j < limit; j++) {
            std::cout<<A[i*DSIZE+j]<<" ";
        }
        std::cout<<" |\n";
    }
    std::cout<<"-              -\n\n";
    return 0;
}

int main(void) {
    //with managed memory, we only need one version of each matrix (not host AND device)
    int *A, *B, *C, *A_stencilled, *B_stencilled; //host copies
    
    // These are used for timing
    clock_t t0, t1, t2;
    double t1sum=0.0;
    double t2sum=0.0;

    // start timing
    t0 = clock();

    // Allocate device memory 
    int size = (DSIZE)*(DSIZE) * sizeof(int);
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&A_stencilled, size);
    cudaMallocManaged(&B_stencilled, size);
    cudaMallocManaged(&C, size);
    cudaCheckErrors("After Managed Memory Allocation");

    //initialize values
    for (int i = 0; i < DSIZE*DSIZE; i++){
        A[i] = (rand() % 10);
        B[i] = (rand() % 10);
        A_stencilled[i]=0;
        B_stencilled[i]=0;
        C[i] = 0;
    }
    cudaCheckErrors("After Filling Initial Values");

    printf("Matrix A: \n");
    printMatrix(A);
    printf("Matrix B: \n");
    printMatrix(B);

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

	// Launch stencil_2d() kernel on GPU
	int gridSize = DSIZE/BLOCK_SIZE; //from Asignment 2 mult_matrix.cu
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	
	stencil_2d<<<grid,block>>>(A , A_stencilled); //QUESTION: confused how the plus works?
    cudaDeviceSynchronize();
    cudaCheckErrors("After applying stencil to matrix A");
	stencil_2d<<<grid,block>>>(B , B_stencilled);
    cudaDeviceSynchronize();
    cudaCheckErrors("After applying stencil to matrix B");

    //Launch matrix_mul kernel on GPU
    matrix_mul_gpu<<<grid,block>>>(A_stencilled, B_stencilled, C, DSIZE);
    cudaDeviceSynchronize();
    cudaCheckErrors("After applying matrix multiplication");

    error_stencil(A_stencilled, A);
    error_stencil(B_stencilled, B);
    error_matrix_mul_gpu(A_stencilled, B_stencilled, C, DSIZE);

    printf("Matrix A stencilled: \n");
    printMatrix(A_stencilled);
    printf("Matrix B stencilled: \n");
    printMatrix(B_stencilled);
    printf("Matrix A stencilled * Matrix B stencilled: \n");
    printMatrix(C);

    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t2sum);
    
	// Free memory 
    cudaFree(A);
    cudaFree(B);
    cudaFree(A_stencilled);
    cudaFree(B_stencilled);
    cudaFree(C);
}


