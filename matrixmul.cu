/* matrixmul.cu
 *  
 * Jonathan Lehman
 * February 22, 2012
 */

#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

__global__
void mult( float*, float*, float*, int, int, int, int, int);
void buildArrays( int, int );
void checkArgs(int, char**);
void checkGPUCapabilities(int, int, int, int, int);
int nearestDivInt(int, int);

//set block size
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

//user input
int GRID_WIDTH;
int GRID_HEIGHT;
int MATRIX_A_HEIGHT;
int MATRIX_A_WIDTH;
int MATRIX_B_HEIGHT;
int MATRIX_B_WIDTH;

int TOTAL_ELEM;
int MAT_A_ELEM;
int MAT_B_ELEM;
int MAT_C_ELEM;

// Keep track of the time.
cudaEvent_t start, stop; 
float elapsedTime; 


//arrays
float* a;
float* b;
float* c;

int main( int argc, char *argv[] ){
  	
	float *dev_a, *dev_b, *dev_c;
	
	//check validity of arguments
	checkArgs(argc, argv);
  
	//assign variables
	GRID_WIDTH = atoi(argv[1]);
	GRID_HEIGHT = atoi(argv[2]);
	MATRIX_A_HEIGHT = atoi(argv[3]);
	MATRIX_A_WIDTH = atoi(argv[4]);
	MATRIX_B_HEIGHT = atoi(argv[5]);
	MATRIX_B_WIDTH = atoi(argv[6]);
	
	//check that multiplication is valid
	if(MATRIX_A_WIDTH != MATRIX_B_HEIGHT){
		fprintf(stderr, "\nmatrixmul: Matrix A width, %d, must equal Matrix B height, %d, otherwise these matrices cannot be multiplied\n", MATRIX_A_WIDTH, MATRIX_B_HEIGHT );
		exit(1);
	}
	
	//make sure dimensions of C matrix are divisible by block size
	if(nearestDivInt(MATRIX_A_WIDTH, BLOCK_SIZE) != MATRIX_A_WIDTH){
		MATRIX_A_WIDTH = nearestDivInt(MATRIX_A_WIDTH, BLOCK_SIZE);
		
		if(MATRIX_A_WIDTH == 0){
			MATRIX_A_WIDTH = BLOCK_SIZE;
			
		}
		
		MATRIX_B_HEIGHT = MATRIX_A_WIDTH; 
		
		printf("Matrix A width and Matrix B height must be divisible by the block dimension %d\nChanging the dimensions of Matrix A to %d x %d (HxW) and Matrix B to %d x % d (HxW)\n", BLOCK_SIZE, MATRIX_A_HEIGHT, MATRIX_A_WIDTH, MATRIX_B_HEIGHT, MATRIX_B_WIDTH);
	}

	
	MAT_A_ELEM = MATRIX_A_WIDTH * MATRIX_A_HEIGHT;
	MAT_B_ELEM = MATRIX_B_WIDTH * MATRIX_B_HEIGHT;
	
	
	//check that matrixA is divisible by block size, if not change dimensions
	if(nearestDivInt(MAT_A_ELEM, BLOCK_SIZE * BLOCK_SIZE) != MAT_A_ELEM){
	
		
		MATRIX_A_HEIGHT = nearestDivInt(MATRIX_A_HEIGHT, BLOCK_SIZE * BLOCK_SIZE);
		
		if(MATRIX_A_HEIGHT == 0){
			MATRIX_A_HEIGHT = BLOCK_SIZE * BLOCK_SIZE;
		}
		
		printf("Matrix A not divisible by the block size, %d\nChanging the dimensions of Matrix A to %d x %d (HxW)\n", BLOCK_SIZE * BLOCK_SIZE, MATRIX_A_HEIGHT, MATRIX_A_WIDTH);
	}
	
	//check that matrixB is divisible by block size, if not change dimensions
	if(nearestDivInt(MAT_B_ELEM, BLOCK_SIZE * BLOCK_SIZE) != MAT_B_ELEM){
	

		MATRIX_B_WIDTH = nearestDivInt(MATRIX_B_WIDTH, BLOCK_SIZE * BLOCK_SIZE);
		
		if(MATRIX_B_WIDTH == 0){
			MATRIX_B_WIDTH = BLOCK_SIZE * BLOCK_SIZE;
		}
		
		printf("Matrix B not divisible by the block size, %d\nChanging the dimensions of Matrix B to %d x %d (HxW)\n", BLOCK_SIZE * BLOCK_SIZE, MATRIX_B_HEIGHT, MATRIX_B_WIDTH);
	}
	
	//need to ensure that the gridwidth is the same as this value, to ensure that the multiplier will work in ALL instances
	if(MATRIX_B_WIDTH != GRID_WIDTH * BLOCK_SIZE){
		MATRIX_B_WIDTH = GRID_WIDTH * BLOCK_SIZE;
		printf("Matrix B width must equal the grid width, %d, times the block size, %d\nChanging the dimensions of Matrix B to %d x %d (HxW)\n", GRID_WIDTH, BLOCK_SIZE, MATRIX_B_HEIGHT, MATRIX_B_WIDTH);
	}
	
	MAT_A_ELEM = MATRIX_A_WIDTH * MATRIX_A_HEIGHT;
	MAT_B_ELEM = MATRIX_B_WIDTH * MATRIX_B_HEIGHT;
	MAT_C_ELEM = MATRIX_A_HEIGHT * MATRIX_B_WIDTH;
	
	TOTAL_ELEM = MAT_A_ELEM + MAT_B_ELEM + MAT_C_ELEM;
	
	//check that there are no more elements in the resultant matrix than threads to calculate them
	if(GRID_WIDTH * BLOCK_SIZE * GRID_HEIGHT * BLOCK_SIZE < MAT_C_ELEM){
		printf("There must be more threads in the grid, %d, than elements in the resulting matrix, %d\n", GRID_WIDTH * BLOCK_SIZE * GRID_HEIGHT * BLOCK_SIZE, MAT_C_ELEM);
		exit(1);
	}
	
	
	//check that GPU can handle arguments
	checkGPUCapabilities(GRID_WIDTH, GRID_HEIGHT, BLOCK_SIZE, BLOCK_SIZE, TOTAL_ELEM);
  
	/* Initialize the source arrays here. */
  	a = new float[MAT_A_ELEM];
  	b = new float[MAT_B_ELEM];
  	c = new float[MAT_C_ELEM];
  	
  
  	//fill array a and b with random doubles
  	buildArrays(MAT_A_ELEM, MAT_B_ELEM);
  	
  	/*printf( "The sequence:\n" );
	  for( int i = 0; i < MAT_A_ELEM; i++ ){ 
	    if(i % MATRIX_A_WIDTH == 0){
	  		printf("\n");
	  }
	    printf( "%f\t", a[i] );
	    }
	  printf( "\n" );
	  
	  printf( "The sequence:\n" );
	  for( int i = 0; i < MAT_B_ELEM; i++ ) {
	    if(i % MATRIX_B_WIDTH == 0){
	  		printf("\n");
	  }
	    printf( "%f\t",b[i] );
	    }
	  printf( "\n" );*/
    	
    	//check if there will be enough blocks to handle matrix size (if not some threads will take on more than one addition)
    	int reps = ceil((double)(MAT_C_ELEM) / (BLOCK_SIZE * BLOCK_SIZE * GRID_WIDTH * GRID_HEIGHT));   
  	
  	/* Allocate global device memory. */
  	cudaMalloc( (void **)&dev_a, sizeof(float) * MAT_A_ELEM );
  	cudaMalloc( (void **)&dev_b, sizeof(float) * MAT_B_ELEM );
  	cudaMalloc( (void **)&dev_c, sizeof(float) * MAT_C_ELEM );
  
  	/* Copy the host values to global device memory. */
  	cudaMemcpy( dev_a, a, sizeof(float) * MAT_A_ELEM, cudaMemcpyHostToDevice );
  	cudaMemcpy( dev_b, b, sizeof(float) * MAT_B_ELEM, cudaMemcpyHostToDevice);
  	
  	/* Start the timer. */
  	cudaEventCreate(&start); 
  	cudaEventCreate(&stop); 
  	cudaEventRecord( start, 0 ); 
  
  	/* Execute the kernel. */
  	dim3 block(BLOCK_SIZE, BLOCK_SIZE); //threads w x h
  	dim3 grid(GRID_WIDTH, GRID_HEIGHT); //blocks w x h
  	mult<<<grid, block>>>(dev_a, dev_b, dev_c, MATRIX_A_WIDTH, MATRIX_B_WIDTH, MATRIX_A_HEIGHT, reps, MAT_C_ELEM);

  	/* Wait for the kernel to complete. Needed for timing. */  
  	cudaThreadSynchronize();
  	
  	/* Stop the timer and print the resulting time. */
	  cudaEventRecord( stop, 0 ); 
	  cudaEventSynchronize( stop ); 
	  cudaEventElapsedTime( &elapsedTime, start, stop ); 
	  printf( "Time: %f secs\n", elapsedTime / 1000 );
	  
  
  	/* Get result from device. */
  	cudaMemcpy(c, dev_c, sizeof(float) * MAT_C_ELEM, cudaMemcpyDeviceToHost); 
  	
  	/*printf( "The sequence:\n" );
	  for( int i = 0; i < MAT_C_ELEM; i++ ){
	  	if(i % MATRIX_B_WIDTH == 0){
	  		printf("\n");
	  	}
	    printf( "%f\t", c[i] );
	    
	  }
	  printf( "\n" );*/
  	
  	//print any cuda error messages
  	const char* errorString = cudaGetErrorString(cudaGetLastError());
	printf("GPU Error: %s\n", errorString);

  	
  	
  	//destroy cuda event
  	cudaEventDestroy( start ); 
  	cudaEventDestroy( stop );
    	
  	/* Free the allocated device memory. */
  	cudaFree(dev_a);
  	cudaFree(dev_b);
  	cudaFree(dev_c);
  
  	//free allocated host memory
	free(a);
	free(b);
	free(c);
}

__global__
void mult( float *a, float *b, float *c, int wA , int wB, int hA, int reps, int size)
{   
	//grid dimensions (# blocks)
	int gridW = gridDim.x;   
	int gridH = gridDim.y;
	
        //block id
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;  
	
	//thread id
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	
	//float to store c subtotal
	float cTot = 0;
	
	//values to iterate through submatrix blocks
	int aStart;
	int aSize;
	int aStop;
	int bStart;
	int bSize;
	
	
	//shared memory for each block (A and B matrices)
	__shared__ float shA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float shB[BLOCK_SIZE][BLOCK_SIZE];
	
	 //loop through number of times matrix elements fill more than an entire grid
	for(int i = 0; i < reps; i++){
	
		//A blocks
		// index of first submatrix of A (account for if doesnt fit on initial grid)
		if(hA > gridH * BLOCK_SIZE){
			aStart = wA * BLOCK_SIZE * (blockY + gridW * i);
		}
		else{
			aStart = wA * BLOCK_SIZE * blockY;
		}
		
		// size of each submatrix of A
		aSize  = BLOCK_SIZE;
		
		// index of last submatrix of A
		aStop   = aStart + wA - 1;
		
		
		
		//B blocks
		// index of first submatrix of B (account for if doesnt fit on initial grid)
		if(wB > gridW * BLOCK_SIZE){
			bStart = BLOCK_SIZE * (blockX + gridH * i);
		}
		else{
			bStart = BLOCK_SIZE * blockX;
		}
		
		// size of each submatrix of B
		bSize  = BLOCK_SIZE * wB;
		
		
		// loop through submatrices for a and b by specified steps
		for (int aVal = aStart, bVal = bStart; aVal <= aStop; aVal += aSize, bVal += bSize){
		
			int aIndex = aVal + wA * threadY + threadX;
			int bIndex = bVal + wB * threadY + threadX;
			
		
			//load memory for matrices a and b into shared memory
			shA[threadX][threadY] = a[aIndex];
			shB[threadX][threadY] = b[bIndex];
			__syncthreads();
			
			for (int i = 0; i < BLOCK_SIZE; i++){
				cTot += shA[i][threadX] * shB[threadY][i];
			}
			__syncthreads();
			
					
		}
		
		
		//store values to correct index in c
		int cVal = wB * BLOCK_SIZE * blockY + BLOCK_SIZE * blockX;
		int index = cVal + wB * threadX + threadY + (gridW * gridH * BLOCK_SIZE * BLOCK_SIZE * i);
		if(index < size){
			c[index] = cTot;
		}
	
	}
}


void buildArrays( int mat_a_size, int mat_b_size ){
	/* Seed the random number generator. */
	srand( 200 );

	for(int i = 0; i < mat_a_size; i++){
		float val = rand() / (float(RAND_MAX));
		a[i] = val;
	}
  
	srand( 300 );
  
	for(int i = 0; i < mat_b_size; i++){ 
  		float val = rand() / (float(RAND_MAX));
  		b[i] = val;
  	}

}

void checkArgs(int argc, char *argv[]){
	
	//check number of arguments
	if(argc != 7){
		fprintf(stderr, "\nmatrixmul: Incorrect number of arguments. matrixmul requires 6 arguments not %d\nCorrect usage: \"matrixmul grid-width grid-height matA-height matA-width matB-height matB-width\"\n", argc - 1);
		exit(1);
	}
	
	
	char* invalChar;
	long arg;
	
	//check each argument
	for(int i = 1; i < 7; i++){
		//check for overflow of argument
		if((arg = strtol(argv[i], &invalChar, 10)) >= INT_MAX){
			fprintf(stderr, "\nmatrixmul: Overflow. Invalid argument %d for matrixmul, '%s'.\nThe argument must be a valid, positive, non-zero integer less than %d.\n", i, argv[i], INT_MAX);
			exit(1);
		}
	
		//check that argument is a valid positive integer and check underflow
		if(!(arg > 0) || (*invalChar)){
			fprintf(stderr, "\nmatrixmul: Invalid argument %d for matrixmul, '%s'.  The argument must be a valid, positive, non-zero integer.\n", i, argv[i]);
			exit(1);
		}
		
	}	
}

void checkGPUCapabilities(int gridW, int gridH, int blockW, int blockH, int size){
	//check what GPU is being used
	int devId;  
	cudaGetDevice( &devId );
	
	//get device properties for GPU being used
	cudaDeviceProp gpuProp;
	cudaGetDeviceProperties( &gpuProp, devId );
	
	//check if GPU has enough memory to handle the 3 arrays
	if(gpuProp.totalGlobalMem < (size * sizeof(float))){
		fprintf(stderr, "\nmatrixmul: Insufficient GPU. GPU does not have enough memory to handle the data size: %ld. It can only handle data sizes up to %ld.\n", (size * sizeof(float)) * 3, gpuProp.totalGlobalMem);
		exit(1);
	}
	
	//check if GPU can handle the number of threads per bloc
	if(gpuProp.maxThreadsPerBlock < (blockW * blockH)){
		fprintf(stderr, "\nmatrixmul: Insufficient GPU. GPU can only handle %d threads per block, not %d.\n", gpuProp.maxThreadsPerBlock, (blockW * blockH));
		exit(1);
	}
	
	//check that GPU can handle the number of threads in the block width
	if(gpuProp.maxThreadsDim[0] < blockW){
		fprintf(stderr, "\nmatrixmul: Insufficient GPU. GPU can only handle %d threads as the block width of each block, not %d.\n", gpuProp.maxThreadsDim[0], blockW );
		exit(1);
	}
	
	//check that GPU can handle the number of threads in the block height
	if(gpuProp.maxThreadsDim[1] < blockH){
		fprintf(stderr, "\nmatrixmul: Insufficient GPU. GPU can only handle %d threads as the block height of each block, not %d.\n", gpuProp.maxThreadsDim[1], blockH );
		exit(1);
	}
	
	//check that GPU can handle the number of blocks in the grid width
	if(gpuProp.maxGridSize[0] < gridW){
		fprintf(stderr, "\nmatrixmul: Insufficient GPU. GPU can only handle %d blocks as the grid width of each grid, not %d.\n", gpuProp.maxGridSize[0], gridW );
		exit(1);
	}
	
	//check that GPU can handle the number of blocks in the grid height
	if(gpuProp.maxGridSize[1] < gridH){
		fprintf(stderr, "\nmatrixmul: Insufficient GPU. GPU can only handle %d blocks as the grid height of each grid, not %d.\n", gpuProp.maxGridSize[1], gridH );
		exit(1);
	}
}

//returns nearest int to initVal divisible by divBy
int nearestDivInt(int initVal, int divBy){

	int attemptVal = initVal / divBy;
	
	return (abs(initVal - (attemptVal * divBy)) <= abs(initVal - ((attemptVal + 1) * divBy))) ? attemptVal * divBy : (attemptVal + 1) * divBy;

}

