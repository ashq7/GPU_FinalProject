#include <iostream>
using namespace std;

const int DSIZE = 8;
int matrixA [DSIZE][DSIZE]= {};
int matrixA_stencilled [DSIZE][DSIZE]= {};
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

int main(){
    for (int i=0; i<DSIZE; i++){
        for(int j=0; j<DSIZE; j++){
            matrixA[i][j]= (rand() % 10); //upper limit on integers?
            matrixA_stencilled[i][j]=0;
        }
    }

    cout<<"Matrix A: ";
        for (int i=0; i<DSIZE; i++){
            for (int j=0; j<DSIZE; j++){
                cout<<matrixA[i][j]<<" ";
            }
            cout<<"\n";
        }
        cout<<"\n";

    stencil2d(matrixA, matrixA_stencilled);
    
    cout<<"Matrix A stencilled: ";
        for (int i=0; i<DSIZE; i++){
            for (int j=0; j<DSIZE; j++){
                cout<<matrixA_stencilled[i][j]<<" ";
            }
            cout<<"\n";
        }
        cout<<"\n";

    
}
