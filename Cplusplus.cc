#include <iostream>
using namespace std;

const int DSIZE = 512;
int matrixA [DSIZE][DSIZE]= {};
int matrixA_stencilled [DSIZE][DSIZE]= {};
int matrixB [DSIZE][DSIZE]= {};
int matrixB_stencilled [DSIZE][DSIZE]= {};
int matrixC [DSIZE][DSIZE]= {};
int matrixD [DSIZE][DSIZE]= {};
const int RADIUS = 3;

void stencil2d(int in[DSIZE][DSIZE], int out[DSIZE][DSIZE]){
    //nested for loop for stencil
    for (int i=0; i<DSIZE; i++){
        for(int j=0; j<DSIZE; j++){
            out[i][j] = in[i][j];
            //fill border first
            if (i<RADIUS || j <RADIUS || i >=(DSIZE-RADIUS)|| j >= (DSIZE-RADIUS)){ //start at 0
                continue;
            }

            else{
                for (int k=1; k<=RADIUS; k++){
                    out[i][j]+=in[i+k][j];
                    out[i][j]+=in[i-k][j];
                    out[i][j]+=in[i][j+k];
                    out[i][j]+=in[i][j-k];
                }     
            }
        }
    }
}

void matrix_mul (int a[DSIZE][DSIZE], int b[DSIZE][DSIZE], int c[DSIZE][DSIZE]){
     for (int i=0; i<DSIZE; i++){
        for(int j=0; j<DSIZE; j++){
            int result=0;
            for (int k=0; k<DSIZE; k++){
                result += a[i][k]*b[k][j];
            }
            c[i][j]=result;
        }
     }
}

int error_stencil(int in[DSIZE][DSIZE], int out[DSIZE][DSIZE]){
    for (int i = 0; i < DSIZE; ++i) {
        for (int j = 0; j < DSIZE; ++j) {
            if (i < RADIUS || DSIZE-i<= RADIUS || j < RADIUS || DSIZE-j<= RADIUS) {
                if (out[i][j] != in[i][j]) {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[i][j], in[i][j]);
                    return -1;
                }
            }
		 
            else { 
                int expectedResult = in[i][j];
                for (int k=1; k<=RADIUS; k++){
                    expectedResult +=in[i+k][j];
                    expectedResult +=in[i-k][j];
                    expectedResult +=in[i][j+k];
                    expectedResult +=in[i][j-k];
                }
                if (out[i][j] != expectedResult) {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[i][j], expectedResult);
                    return -1;
                }
            }
        }
    }
    printf("no stencil error \n");
    return 0;
}

int error_matrix_mul(int a[DSIZE][DSIZE], int b[DSIZE][DSIZE], int c[DSIZE][DSIZE]){
     for (int i=0; i<DSIZE; i++){
        for(int j=0; j<DSIZE; j++){
            int expectedResult=0;
            for (int k=0; k<DSIZE; k++){
                expectedResult += a[i][k]*b[k][j];
            }
            if (c[i][j]!=expectedResult){
                printf("Multiplication wrong at [%d,%d], was: %d, should be: %d\n", i,j, c[i][j], expectedResult);
                    return -1;
            }
        }
     }
     printf("no multiplication error \n");
     return 0;
}

int main(){
    for (int i=0; i<DSIZE; i++){
        for(int j=0; j<DSIZE; j++){
            matrixA[i][j]= (rand() % 10); //upper limit on integers?
            matrixB[i][j]= (rand() % 10);
            matrixA_stencilled[i][j]=0;
            matrixB_stencilled[i][j]=0;
            matrixC[i][j]=0;
            matrixD[i][j]= 0;
        }
    }

    // cout<<"Matrix A: ";
    //     for (int i=0; i<DSIZE; i++){
    //         for (int j=0; j<DSIZE; j++){
    //             cout<<matrixA[i][j]<<" ";
    //         }
    //         cout<<"\n";
    //     }
    //     cout<<"\n";

    // cout<<"Matrix B: ";
    //     for (int i=0; i<DSIZE; i++){
    //         for (int j=0; j<DSIZE; j++){
    //             cout<<matrixB[i][j]<<" ";
    //         }
    //         cout<<"\n";
    //     }
    //     cout<<"\n";

    stencil2d(matrixA, matrixA_stencilled);
    stencil2d(matrixB, matrixB_stencilled);
    error_stencil(matrixA, matrixA_stencilled);
    error_stencil(matrixB, matrixB_stencilled);
    
    // cout<<"Matrix A stencilled: ";
    //     for (int i=0; i<DSIZE; i++){
    //         for (int j=0; j<DSIZE; j++){
    //             cout<<matrixA_stencilled[i][j]<<" ";
    //         }
    //         cout<<"\n";
    //     }
    //     cout<<"\n";

    // cout<<"Matrix B stencilled: ";
    //     for (int i=0; i<DSIZE; i++){
    //         for (int j=0; j<DSIZE; j++){
    //             cout<<matrixB_stencilled[i][j]<<" ";
    //         }
    //         cout<<"\n";
    //     }
    //     cout<<"\n";

    matrix_mul(matrixA, matrixB, matrixD);
    matrix_mul(matrixA_stencilled, matrixB_stencilled, matrixC);
    error_matrix_mul(matrixA, matrixB, matrixD);
    error_matrix_mul(matrixA_stencilled, matrixB_stencilled, matrixC);

    // cout<<"For debugging: Matrix A * Matrix B: ";
    //     for (int i=0; i<DSIZE; i++){
    //         for (int j=0; j<DSIZE; j++){
    //             cout<<matrixD[i][j]<<" ";
    //         }
    //         cout<<"\n";
    //     }
    //     cout<<"\n";

    // cout<<"Matrix A stenciled * Matrix B stencilled: ";
    //     for (int i=0; i<DSIZE; i++){
    //         for (int j=0; j<DSIZE; j++){
    //             cout<<matrixC[i][j]<<" ";
    //         }
    //         cout<<"\n";
    //     }
    //     cout<<"\n";
    
}
