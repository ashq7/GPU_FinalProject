#include <stdio.h>
#include <algorithm>

using namespace std;

//#define N 64 -what was N? I think N+2*Radius = DSIZE
#define DSIZE 512
#define RADIUS 3
#define BLOCK_SIZE 32


__global__ void stencil_2d(int *in, int *out) {

	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];
	int gindex_x = threadIdx.x + blockIdx.x * blockDim.x; 
	int lindex_x = threadIdx.x + RADIUS;
	int gindex_y = threadIdx.y + blockIdx.y * blockDim.y;
	int lindex_y = threadIdx.y + RADIUS;

	// Read input elements into shared memory
	//int size = N + 2 * RADIUS; //becomes DSIZE
	temp[lindex_x][lindex_y] = in[gindex_y + DSIZE * gindex_x]; 

	if (threadIdx.x < RADIUS) {
		temp[lindex_x-RADIUS][lindex_y]=in[gindex_y + DSIZE * (gindex_x - RADIUS)];
		temp[lindex_x + BLOCK_SIZE][lindex_y] = in[gindex_y + DSIZE * (gindex_x + BLOCK_SIZE)]; 
	}

	if (threadIdx.y < RADIUS ) {
		temp[lindex_x][lindex_y-RADIUS]=in[(gindex_y - RADIUS)+ DSIZE * gindex_x];
		temp[lindex_x][lindex_y + BLOCK_SIZE] = in[gindex_y + BLOCK_SIZE + DSIZE * gindex_x];
	}


	// Apply the stencil
	int result = 0;
	for (int offset = -RADIUS; offset <= RADIUS; offset++){
		__syncthreads(); //makes sure we have access to everything in temp accessed across multiple threads
		result += temp[lindex_x + offset][lindex_y];
		//avoid double-counting 
		if(offset!=0){
			result += temp[lindex_x][lindex_y + offset];
		}
	}

	// Store the result
	out[gindex_y+DSIZE*gindex_x] = result;
}

// Square matrix multiplication on CPU : C = A * B
void matrix_mul_cpu(const float *A, const float *B, float *C, int size) {
  //FIXME:
  // i iterates over rows of matrix A
  for (int i = 0; i<size; i++){
    // j iterates over columns of matrix B
    for (int j = 0; j<size; j++){
        float temp = 0;
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
__global__ void matrix_mul_gpu(const float *A, const float *B, float *C, int size) {

    //FIXME:
    // create thread x index
    // create thread y index
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idy = blockIdx.x * blockDim.x + threadIdx.x;;
    // Make sure we are not out of range
    if ((idx < size) && (idy < size)) {
        float temp = 0;
        for (int i = 0; i < size; i++){
            //FIXME : Add dot product of row and column
            temp += A [idx * size +idy] * B [idy * size +idx];
        }
        C[idx*size+idy] = temp;                    
    }

}

// error checking macro from matrix multiplication
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


int main(void) {

    int *h_A, *h_A_stencilled, *h_B, *h_B_stencilled, *h_C; //host copies
    int *d_A, *d_A_stencilled, *d_B, *d_B_stencilled, *d_C; //device copies

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

    // Allocate device memory 
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_A_stencilled, size);
    cudaMalloc(&d_B_stencilled, size);
    cudaMalloc(&d_C, size);
    cudaCheckErrors("After Memory Allocation");

    // Copy from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_stencilled, h_A_stencilled, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_stencilled, h_B_stencilled, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

	// Alloc space for host copies and setup values
	in = (int *)malloc(size); fill_ints(in, (DSIZE)*(DSIZE));
	out = (int *)malloc(size); fill_ints(out, (DSIZE)*(DSIZE));

	// Alloc space for device copies
	cudaMalloc((void **)&d_in, size);
	cudaMalloc((void **)&d_out, size);

	// Copy to device
	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

	// Launch stencil_2d() kernel on GPU
	int gridSize = DSIZE/BLOCK_SIZE; //from Asignment 2 mult_matrix.cu
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	
	// Properly set memory address for first element on which the stencil will be applied
	stencil_2d<<<grid,block>>>(d_in + RADIUS*(DSIZE) + RADIUS , d_out + RADIUS*(DSIZE) + RADIUS);

	// Copy result back to host
	cudaMemcpy(in, d_in, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

	// Error Checking
	for (int i = 0; i < DSIZE; ++i) {
		for (int j = 0; j < DSIZE; ++j) {

			if (i < RADIUS || DSIZE-i<= RADIUS) {
				if (out[j+i*DSIZE] != 1) {
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[j+i*(DSIZE)], 1);
					return -1;
				}
			}
			else if (j < RADIUS || DSIZE-j<= RADIUS) {
				if (out[j+i*(DSIZE)] != 1) {
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[j+i*(DSIZE)], 1);
					return -1;
				}
			}		 
			else {
				if (out[j+i*(DSIZE)] != 1 + 4 * RADIUS) {
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[j+i*(DSIZE)], 1 + 4*RADIUS);
					return -1;
				}
			}
		}
	}

	// Cleanup
	free(in);
	free(out);
	cudaFree(d_in);
	cudaFree(d_out);
	printf("Success!\n");

	return 0;
}


