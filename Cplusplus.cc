#include <iostream>
using namespace std;

const int DSIZE = 512;
int matrixA [DSIZE][DSIZE]= {};

int main(){
    for (int i=0; i<DSIZE; i++){
        for(int j=0; j<DSIZE; j++){
            matrixA[i][j]= (rand() % 10); //upper limit on integers?
        }
    }

    cout<<"Matrix A: ";
        for (int i=0; i<10; i++){
            for (int j=0; j<DSIZE; j++){
                cout<<matrixA[i][j]<<" ";
            }
            cout<<"\n";
        }
        cout<<"\n";


    //nested for loop for stencil
}
