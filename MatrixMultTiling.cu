#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <ctime>
#include "common.h"

using namespace std;

#define SIZEM 2000
#define TILE_SIZE 32

void fillMatrices(float * ip){

    srand(time(NULL));

    for (int i = 0; i < SIZEM*SIZEM; i++){
        ip[i] = (float)rand()/(RAND_MAX/ 10.0f);
        //ip[i] = i;
    }    
}

void checkResult(float *hostRef, float *h_C)
{
    double epsilon = SIZEM*SIZEM;
    bool match = 1;

    for (int i = 0; i < SIZEM*SIZEM; i++)
    {
        if (abs(hostRef[i] - h_C[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], h_C[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

__global__ void multMatrix(float *MatA, float *MatB, float *MatC, int nx, int ny)
{   
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int idx = ix * nx + iy;

    float auxiliar = 0;

    if (ix < nx && iy < ny){
        for(int i = 0; i < ny ; i++){
            auxiliar += MatA[ix * nx + i] * MatB[i * ny + iy];
        }
    }

    MatC[idx] = auxiliar;
}

void Mult(float * h_A, float * h_B, float * hostRef, int nx, int ny){
    
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            int sum = 0;
            for (int k = 0; k < ny; k++){
                sum = sum + h_A[i * nx + k] * h_B[k * nx + j];
            }
            hostRef[i * nx + j] = sum;
        }
    }
}

__global__ void multMatrixTile(float *MatA, float *MatB, float *MatC, int nx, int ny)
{   
    unsigned int tileix = threadIdx.x + blockIdx.x * TILE_SIZE;
    unsigned int tileiy = threadIdx.y + blockIdx.y * TILE_SIZE;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];


    float auxiliar = 0;

    for (int i = 0 ; i < (TILE_SIZE + nx -1)/TILE_SIZE; i++ ){

        if (i * TILE_SIZE + threadIdx.x < nx && tileix < nx){

            tileA[threadIdx.y][threadIdx.x] = MatA[tileiy * ny + i * TILE_SIZE + threadIdx.x];
        }
        else{
            tileA[threadIdx.y][threadIdx.x] = 0;
        }
        if (i * TILE_SIZE + threadIdx.y < nx && tileiy < nx ){

            tileB[threadIdx.y][threadIdx.x] = MatB[(i * TILE_SIZE + threadIdx.y) * nx + tileix];
        }
        else{
            tileB[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for(int j = 0; j < TILE_SIZE; j++){

            auxiliar += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (tileix < nx && tileiy < nx){

        MatC[ tileiy * nx + tileix] = auxiliar;
    }
}

void printMatrix(float *Mat, int size){

    int salto = 1; 

    for (int i = 0; i < size; i++){
        printf("    %f  ", Mat[i]);
        if(salto == 10){
            printf("\n");
            salto = 1;
        }else{
            salto++;
        }
    }  printf("\n");

}

int main (int argc, char ** argv){

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // Tamaño de la matriz
    int nx = SIZEM;
    int ny = SIZEM;

    int nxy = nx * ny;
    float nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // Apartar memoria 
    float *h_A, *h_B, *h_C, *h_CPU, *h_GPUTiles;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    h_CPU = (float *)malloc(nBytes);
    h_GPUTiles =(float *) malloc(nBytes);


    // Inicializar matrices
    fillMatrices(h_A);
    fillMatrices(h_B);

    memset(h_C, 0.0, nBytes);
    memset(h_CPU, 0.0, nBytes);

    // Apartar memoria en la GPU
    float *d_MatA, *d_MatB, *d_MatC, *d_MatTile;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");
    SAFE_CALL(cudaMalloc((void **)&d_MatTile, nBytes), "Error allocating d_MatTile");

    // Transferir informacion a la GPU
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // Invocar al kernel del lado del host
    int dimx = TILE_SIZE;
    int dimy = TILE_SIZE;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    auto average = 0;
    auto average2 = 0;
    auto average3 = 0;

    // GPU 
    auto start_cpu =  chrono::high_resolution_clock::now();
    multMatrix<<<grid, block>>>(d_MatA, d_MatB, d_MatTile, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    auto end_cpu =  chrono::high_resolution_clock::now();

    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    average = duration_ms.count();

    // GPU TILES
    start_cpu =  chrono::high_resolution_clock::now();
    multMatrixTile<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    
    duration_ms = end_cpu - start_cpu;
    average2 = duration_ms.count();

    // CPU
    /*start_cpu =  chrono::high_resolution_clock::now();
    Mult(h_A, h_B, h_CPU,nx, nx);
    end_cpu =  chrono::high_resolution_clock::now();
    
    duration_ms = end_cpu - start_cpu;
    average3 = duration_ms.count();*/
 

    printf("multMatrixGPU <<<(%d,%d), (%d,%d)>>> elapsed %d ms\n", grid.x,
           grid.y,
           block.x, block.y, average);

    printf("multMatrixTile <<<(%d,%d), (%d,%d)>>> elapsed %d ms\n", grid.x,
           grid.y,
           block.x, block.y, average2);

    //printf("Mult in CPU elapsed %d ms \n", average3);

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(h_C, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");
    SAFE_CALL(cudaMemcpy(h_GPUTiles, d_MatTile, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatTile");

    // Compare CPU and GPU results
    checkResult(h_C, h_CPU);

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatTile), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_CPU);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return 0;
}