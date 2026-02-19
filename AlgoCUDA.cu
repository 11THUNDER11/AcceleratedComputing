#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CHECK(call)                                       \
    {                                                     \
        const cudaError_t error = call;                   \
        if (error != cudaSuccess)                         \
        {                                                 \
            printf("Error: %s:%d, ", __FILE__, __LINE__); \
            printf("code:%d, reason: %s\n", error,        \
                   cudaGetErrorString(error));            \
            exit(1);                                      \
        }                                                 \
    }

double cpuSecond()
{
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

__global__ void NeiboredReduction(float *inputData, int blockValuesLeftToCompute,float *outputData, int global_index, int blockId){
	
	for (int stride = 1; blockValuesLeftToCompute > 1; stride *= 2) {
        //soltanto i thread con ID multiplo di 2stride restano attivi
        if ( (global_index + threadIdx.x) % (2 * stride) == 0) {
            inputData[global_index + threadIdx.x] += inputData[global_index + threadIdx.x + stride];
        }
        blockValuesLeftToCompute /= 2;
        __syncthreads();
    }

	//EFFETTUO LA RIDUZIONE PARALLELA (Interleaved Addressing)
	if (threadIdx.x == 0) {
        //salva il risultato finale
        outputData[blockId] = inputData[global_index];
    }
}

__global__ void InterleavedReduction(float *inputData, int strideI,float *outputData, int global_index, int blockId){
	
    //Utilizzo di una shared memory per la riduzione parallela
    extern __shared__ float s[];
	//Carico i dati in shared memory
    //printf("Child block %d - thread %d) loading the value s[%d] = %.2f\n", blockId, threadIdx.x, global_index + threadIdx.x, inputData[global_index + threadIdx.x]);
    s[threadIdx.x] = inputData[global_index + threadIdx.x];
    __syncthreads();
    for (int stride = strideI ;stride > 0; stride >>= 1) {
		// Ogni thread attivo somma due elementi separati dallo stride corrente
		if (threadIdx.x < stride) {
			// Il thread somma l'elemento alla sua posizione con quello a distanza 'stride'
			//printf("B%d - T%d) summing the value s[%d] = %.2f with s[%d] = %.2f\n", blockId, threadIdx.x, global_index, s[global_index], global_index + stride, s[global_index + stride]);
			s[threadIdx.x] += s[threadIdx.x  + stride];
		}
		__syncthreads(); // Sincronizzazione da parte dei thread che non lavorano
	}

	//EFFETTUO LA RIDUZIONE PARALLELA (Interleaved Addressing)
	if (threadIdx.x == 0) {
        //salva il risultato finale
        //printf("Child block %d - thread %d) saving the value s[%d] = %.2f\n", blockId, threadIdx.x, global_index, s[threadIdx.x]);
        outputData[blockId] = s[threadIdx.x];
    }
}
//Ipotesi : il numero di thread che vengono lanciati per blocco è sempre un multiplo di 32.
//I dati in ingresso sono sempre multipli di 32. In caso contrario, si aggiungono valori nulli.

__global__ void Algo(float* inputData, float* outputData, int numberOfElements) {

	int global_index = (blockIdx.x * numberOfElements) + threadIdx.x;
	//int blockValuesLeftToCompute = numberOfElements;
	//int sumsToComputeThisCyclePerThread;

	//int iteration;
	//int x;
	//int stride;
    extern __shared__ float s[];
    
    //Carico i dati in shared memory
    s[threadIdx.x] = 0;
    //__syncthreads();

    for(int i = 0; i < numberOfElements; i += blockDim.x){
        if(i + threadIdx.x < numberOfElements){
            s[threadIdx.x] += inputData[global_index + i];
        }
    }

    __syncthreads();

    //inputData[global_index] = s[threadIdx.x];

    /*
	//for che scorre le righe
	iteration = 1;
	for (stride = blockDim.x; blockValuesLeftToCompute > blockDim.x; stride *= 2) {
		sumsToComputeThisCyclePerThread = (blockValuesLeftToCompute / 2) / blockDim.x;

		//for che scorre le colonne
		x = 0;
		for (int sumsDone = 0; sumsDone < sumsToComputeThisCyclePerThread; sumsDone++) {
			//printf("B%d - T%d) summing the value s[%d] = %.2f with s[%d] = %.2f\n", blockIdx.x, threadIdx.x, global_index + x * blockDim.x, inputData[global_index + x * blockDim.x], global_index + x * blockDim.x + stride, inputData[global_index + x * blockDim.x + stride]);
			inputData[global_index + x * blockDim.x] += inputData[global_index + x * blockDim.x + stride];
			x += powf(2, iteration);
		}

		blockValuesLeftToCompute /= 2;
		iteration++;
		
	}
	//__syncthreads();
	*/
	//Spengo i thread che non servono più.
	//Se restano 16 valori, soltanto i thread 0...15 devono lavorare
    //if (threadIdx.x >= blockValuesLeftToCompute) return;

	//Effettuo la riduzione finale

    for (int stride = blockDim.x / 2 ;stride > 0; stride >>= 1) {
		// Ogni thread attivo somma due elementi separati dallo stride corrente
		if (threadIdx.x < stride) {
			// Il thread somma l'elemento alla sua posizione con quello a distanza 'stride'
			//printf("B%d - T%d) summing the value s[%d] = %.2f with s[%d] = %.2f\n", blockId, threadIdx.x, global_index, s[global_index], global_index + stride, s[global_index + stride]);
			s[threadIdx.x] += s[threadIdx.x  + stride];
		}
		__syncthreads(); // Sincronizzazione da parte dei thread che non lavorano
	}

	//EFFETTUO LA RIDUZIONE PARALLELA (Interleaved Addressing)
	if (threadIdx.x == 0) {
        //salva il risultato finale
        //printf("Child block %d - thread %d) saving the value s[%d] = %.2f\n", blockId, threadIdx.x, global_index, s[threadIdx.x]);
        outputData[blockIdx.x] = s[threadIdx.x];
    }

    /*
	if(threadIdx.x == 0){
        
        //printf("B%d - T%d) starting the reduction with %d values\n", blockIdx.x, threadIdx.x, blockDim.x);
		/*
        for(int i = 0; i < blockValuesLeftToCompute; i++){
            printf("B%d - T%d) s[%d] = %.2f\n", blockIdx.x, threadIdx.x, global_index + i, inputData[global_index + i]);
        }
        */
        //InterleavedReduction <<<1, blockDim.x, blockDim.x * sizeof(float)>>> (inputData, blockDim.x / 2	, outputData, global_index, blockIdx.x );
		//NeiboredReduction <<<1, blockValuesLeftToCompute / 2>>> (inputData, blockValuesLeftToCompute / 2, outputData, global_index, blockIdx.x);
	//}
    
}





__global__ void generateSerie(float* serie, int size, int numberOfSeries, float initialValue, float maxPercentage)
{
    //Calcolo ID del thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Controllo se il thread è valido
    //if (idx >= numberOfSeries) return;

    //Dichiarazione primo elemento della serie
    int startIndex = idx * size;
    int endIndex = startIndex + size;

    serie[startIndex] = initialValue;

    curandState local;
    curand_init((unsigned long long) clock() + idx, 0, 0, &local);


    for (int i = startIndex + 1; i < endIndex; i++)
    {
        float perc = curand_uniform(&local);

        float percentage = (perc * 2 * maxPercentage) - maxPercentage;
        serie[i] = serie[i - 1] + (serie[i - 1] * percentage);
    }

}

int main(int argc, char** argv) {
	//--------------------CUDA SETUP--------------------

	printf("Starting...\n");
	srand(time(NULL));

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	//--------------------FILE REPORT--------------------

	FILE *fileReport;
	const char *filename = "report.txt";
	fileReport = fopen(filename, "w");
	
	/*
	if (fileReport == NULL) {
		printf("Error opening file!\n");
		exit(1);
	}
	*/
	//--------------------VARIABLES--------------------

	int numberOfElements = 2 << 9;
	int numberOfSeries = 2 << 9;
	int blockSize = 1024;
	
	int numBlocks = numberOfSeries;

	float* series_h = (float*)malloc(numberOfSeries * numberOfElements * sizeof(float));
	float* results_h = (float*)malloc(numberOfSeries * sizeof(float));
	float* resFromGPU_h = (float*)malloc(numberOfSeries * sizeof(float));
	float* series_d, *results_d;

	CHECK(cudaMalloc((void**)&results_d, numberOfSeries * sizeof(float)));
	CHECK(cudaMalloc((void**)&series_d, numberOfSeries * numberOfElements * sizeof(float)));
	CHECK(cudaMemcpy(series_d, series_h, numberOfSeries * numberOfElements * sizeof(float), cudaMemcpyHostToDevice));
	

	//--------------------SERIES GENERATION--------------------

	int t = 32;
    int b = (numberOfSeries + blockSize - 1) / blockSize;
	generateSerie << <b, t >> > (series_d, numberOfElements,numberOfSeries, 100.0, 0.1);
	CHECK(cudaDeviceSynchronize());

	printf("Series generated\n");
	//fprintf(fileReport, "Series generated\n");
	CHECK(cudaMemcpy(series_h, series_d, numberOfSeries * numberOfElements * sizeof(float), cudaMemcpyDeviceToHost));
	

	//--------------------SERIES SUM ON CPU--------------------

	int index = 0;
	for (int i = 0; i < numberOfSeries * numberOfElements; i+= numberOfElements) {
		results_h[index] = 0;

		//printf("Series %d\n", index);

		for (int j = 0; j < numberOfElements; j++) {

			//printf("s[%d] = %.2f\n", i + j, series_h[i + j]);

			//series_h[i + j] = (float) (i+j) + ((i+j) % 2);
			results_h[index] += series_h[i + j];
			//printf("s[%d] = %.2f\n", i + j, series_h[i + j]);
		}
		//printf("----\n");
		index++;
	}
	

	/* 
	for (int i = 0; i < numberOfSeries; i++) {
		printf("Final sum is %.2f\n", results_h[i]);
	}
	*/

	//--------------------SERIES SUM ON GPU--------------------

	printf("Beginning computation for %d series of %d numbers each\n", numberOfSeries, numberOfElements);
	printf("It has been determined that %d blocks will be created, each containing %d threads\n", numBlocks, blockSize);

	//fprintf(fileReport, "Beginning computation for %d series of %d numbers each\n", numberOfSeries, numberOfElements);
	//fprintf(fileReport, "It has been determined that %d blocks will be created, each containing %d threads\n", numBlocks, blockSize);

	
	double iStart = cpuSecond();
	Algo << <numberOfSeries, MIN(blockSize,1024)  , MIN(blockSize,1024) * sizeof(float)>> > (series_d, results_d, numberOfElements);
	CHECK(cudaDeviceSynchronize());
	double iElaps = cpuSecond() - iStart;
	CHECK(cudaMemcpy(resFromGPU_h, results_d, numberOfSeries * sizeof(float), cudaMemcpyDeviceToHost));

	//--------------------COMPARISON--------------------

	//FILE *compareFile;
	//onst char *compareFilename = "compare.txt";
	//compareFile = fopen(compareFilename, "w");

	float epsilon = 1.0e-1;
	int counter = 0;
	for (int i = 0; i < numberOfSeries; i ++) {

		//fprintf(compareFile, "Series %d: %f VS %f\n", i, results_h[i], resFromGPU_h[i]);

		if (abs(results_h[i] - resFromGPU_h[i]) > epsilon) {
			//printf("ERROR at index %d: C: %.2f VS G: %.2f\n", i, results_h[i], results_d[i]);
			counter++;
			//fprintf(fileReport, "ERROR at index %d: C: %f VS G: %f\n", i, results_h[i], resFromGPU_h[i]);
		}
	}

	//fclose(compareFile);
	
	printf("Algo elapsed %f sec\n", iElaps);
	//fprintf(fileReport, "Algo elapsed %f sec\n", iElaps);
	
	printf("Total errors: %d\n", counter);
	//fprintf(fileReport, "Total errors: %d\n", counter);

	printf("Success rate: %.2f%\n", (float)(numberOfSeries - counter) / numberOfSeries * 100.0f);
	//fprintf(fileReport, "Success rate: %.2f\n", (float)(numberOfSeries - counter) / numberOfSeries * 100.0f);

	//--------------------FREE MEMORY--------------------

	//fclose(fileReport);
	cudaFree(series_d);
	free(series_h);
	return 0;
}
