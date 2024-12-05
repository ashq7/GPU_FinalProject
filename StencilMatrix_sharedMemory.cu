#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <ostream>

using namespace std;

//#define N 64 -what was N? I think N+2*Radius = DSIZE: DSIZE should be N
#define DSIZE 512 //NEED TO CHANGE BACK TO 512
#define RADIUS 3
#define BLOCK_SIZE 32


__global__ void stencil_2d(int *in, int *out) {

	__shared__ int temp[BLOCK_SIZE][BLOCK_SIZE]; //QUESTION: why block size, not DSIZE? Ans: can't access elements not in that block

	int row = threadIdx.y + blockIdx.y * blockDim.y; //row - can switch row and column?
    int column = threadIdx.x + blockIdx.x * blockDim.x; //column

    int lindex_x = threadIdx.x;
	int lindex_y = threadIdx.y;

	// Read input elements into shared memory
    temp[lindex_x][lindex_y] = in[gindex_y + size * gindex_x];

    if (lindex_x < RADIUS) {
		temp[lindex_x-RADIUS][lindex_y]=in[gindex_y + size * (gindex_x - RADIUS)]; //QUESTION: what does this find if we are on the very edge? zero? what if it's negative size /left side
		temp[lindex_x + BLOCK_SIZE][lindex_y] = in[gindex_y + size * (gindex_x + BLOCK_SIZE)]; //think about geometry! right side
	}

	if (lindex_y < RADIUS ) {
		temp[lindex_x][lindex_y-RADIUS]=in[(gindex_y - RADIUS)+ size * gindex_x];
		temp[lindex_x][lindex_y + BLOCK_SIZE] = in[gindex_y + BLOCK_SIZE + size * gindex_x];
	}

	if (row >= RADIUS && column >= RADIUS && (DSIZE-row) > RADIUS && (DSIZE-column) > RADIUS ) {
		// Apply the stencil
        //int result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; offset++){
            //__syncthreads(); //makes sure we have access to everything in temp accessed across multiple threads
            
            //avoid double-counting 
            if(offset!=0){
                result += in[(row + offset)*DSIZE + column];
                result += in[row*DSIZE + column + offset];
            }
        }
        // Store the result
        out[column+DSIZE*row] = result;
	}
    //printf("at index [%d,%d], result is: %d\n", row,column, result);
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


void fill_ints(int *x, int n) {
   // Store the result
   // https://en.cppreference.com/w/cpp/algorithm/fill_n
   fill_n(x, n, 1);
   //takes in matrix, starts at pointer and fills subsequent n with value (1 here)
}

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

    int *h_A, *h_A_stencilled, *h_B, *h_B_stencilled, *h_C; //host copies
    int *d_A, *d_A_stencilled, *d_B, *d_B_stencilled, *d_C; //device copies

    // These are used for timing
    clock_t t0, t1, t2;
    double t1sum=0.0;
    double t2sum=0.0;

    // start timing
    t0 = clock();

    //Alloc space for host copies 
    int size = (DSIZE)*(DSIZE) * sizeof(int);
    h_A = (int*)malloc(size);
    h_A_stencilled = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_B_stencilled = (int*)malloc(size);
    h_C = (int*)malloc(size);

    //initialize host values
    for (int i = 0; i < DSIZE*DSIZE; i++){
        h_A[i] = (rand() % 10);
        h_B[i] = (rand() % 10);
        h_A_stencilled[i]=0;
        h_B_stencilled[i]=0;
        h_C[i] = 0;
    }
    printf("Matrix A: \n");
    printMatrix(h_A);
    printf("Matrix B: \n");
    printMatrix(h_B);

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);
    
    // Allocate device memory 
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_A_stencilled, size);
    cudaMalloc((void **)&d_B_stencilled, size);
    cudaMalloc((void **)&d_C, size);
    cudaCheckErrors("After Memory Allocation");

    // Copy from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_stencilled, h_A_stencilled, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_stencilled, h_B_stencilled, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    cudaCheckErrors("After copy from host to device");

	// Launch stencil_2d() kernel on GPU
	int gridSize = DSIZE/BLOCK_SIZE; //from Asignment 2 mult_matrix.cu
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	
	stencil_2d<<<grid,block>>>(d_A , d_A_stencilled); //QUESTION: confused how the plus works?
    cudaCheckErrors("After applying stencil to matrix A");
	stencil_2d<<<grid,block>>>(d_B , d_B_stencilled);
    cudaCheckErrors("After applying stencil to matrix B");

    cudaMemcpy(h_A_stencilled, d_A_stencilled, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B_stencilled, d_B_stencilled, size, cudaMemcpyDeviceToHost);

    //Launch matrix_mul kernel on GPU
    matrix_mul_gpu<<<grid,block>>>(d_A_stencilled, d_B_stencilled, d_C, DSIZE);
    cudaCheckErrors("After applying matrix multiplication");

	// Copy result back to host
	cudaMemcpy(h_A_stencilled, d_A_stencilled, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B_stencilled, d_B_stencilled, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    error_stencil(h_A_stencilled, h_A);
    error_stencil(h_B_stencilled, h_B);
    error_matrix_mul_gpu(h_A_stencilled, h_B_stencilled, h_C, DSIZE);

    printf("Matrix A stencilled: \n");
    printMatrix(h_A_stencilled);
    printf("Matrix B stencilled: \n");
    printMatrix(h_B_stencilled);
    printf("Matrix A stencilled * Matrix B stencilled: \n");
    printMatrix(h_C);

    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t2sum);

	// Free memory 
    free(h_A);
    free(h_B);
    free(h_A_stencilled);
    free(h_B_stencilled);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_A_stencilled);
    cudaFree(d_B_stencilled);
    cudaFree(d_C);

}


