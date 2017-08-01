//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <algorithm>
#include <cstring>
/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */



/// parallel histogram function on GPU
__global__ void hist_create(unsigned int* Hist,
			    unsigned int* const Vals,
			    unsigned int digit)
{
   // S_Hist is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
   //extern __shared__ unsigned int S_Hist[];
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int mask = 1 << digit;
    unsigned int bin = (Vals[tid] & mask) >> digit;
    atomicAdd(&(Hist[bin]), 1);
    //__syncthreads();  
    
}


/// parallel predicate on GPU
__global__ void predicate(unsigned int* const Vals,
			  unsigned int* Vals_saved0,
			  unsigned int* Vals_saved1,
			  unsigned int digit)
{
 
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    unsigned int mask = 1 << digit;
    unsigned int bin = (Vals[tid] & mask) >> digit;
    
    if (bin == 0) {
     Vals_saved0[tid] = 1;
     Vals_saved1[tid] = 0;
    }
  __syncthreads(); 

    if (bin == 1) {
     Vals_saved0[tid] = 0;
     Vals_saved1[tid] = 1;
    }
    
}

/// parallel exclusive sum on GPU
__global__ void exclus_sum(unsigned int*  Vals,
			   unsigned int* exc_sum,
			   int stepsize,
                           int whichblk  )
{

    int tid = threadIdx.x + blockIdx.x*blockDim.x + whichblk*520;
    unsigned int sum=0;
 
   exc_sum[tid] = Vals[tid];
   __syncthreads(); 
    
    if ( (tid-stepsize) >= 0 )  {
     sum  =  exc_sum[tid] + exc_sum[tid - stepsize];
    }
    else { 
     sum = exc_sum[tid];
    }
    __syncthreads(); 
   exc_sum[tid] = sum;  
  

}

/// parallel histogram function on GPU
__global__ void compact(unsigned int*  d_inputVals,
               		unsigned int*  d_inputPos,
               		unsigned int*  d_outputVals,
               		unsigned int*  d_outputPos,
               		unsigned int*  exc_sum0,
               		unsigned int*  exc_sum1,
               		unsigned int*  predicate0,
			unsigned int* Hist)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int digit = 1;
    if (digit == predicate0[tid] ) {
     unsigned int a = exc_sum0[tid];
     d_outputVals[a] = d_inputVals[tid];
     d_outputPos[a] = d_inputPos[tid];
    }
  __syncthreads(); 

    digit = 0;
    if ( digit == predicate0[tid] ) {
     unsigned int a = exc_sum1[tid] + Hist[0];
     d_outputVals[a] = d_inputVals[tid];
     d_outputPos[a] = d_inputPos[tid];
    }
    
}


/// parallel swap out to input on GPU
__global__ void swap_oi(unsigned int*  inputV,
			  unsigned int*  inputP,
			  unsigned int*  outputV,
			  unsigned int*  outputP)
{
    int myx = threadIdx.x + blockIdx.x*blockDim.x;
    int myy = threadIdx.y + blockIdx.y*blockDim.y;
    int myId = myx + myy*gridDim.x*blockDim.x;

     inputV[myId] = outputV[myId];
     inputP[myId] = outputP[myId];
    
}



void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
  unsigned int numBins = 2;
  unsigned int *d_binHistogram, *d_predicate0, *d_predicate1, *d_tempvals0, *d_tempvals1;
  unsigned int h_binHistogram[numBins],h_test[numElems],h_test2[numElems], h_inputVals[numElems],h_predicate0[numElems],h_predicate1[numElems];
  unsigned int n_dig=32; 
  checkCudaErrors(cudaMalloc((void**) &d_binHistogram,  numBins*sizeof(unsigned int) ) );
  checkCudaErrors(cudaMalloc((void**) &d_predicate0, numElems*sizeof(unsigned int) ) );
  checkCudaErrors(cudaMalloc((void**) &d_predicate1, numElems*sizeof(unsigned int) ) );
  checkCudaErrors(cudaMalloc((void**) &d_tempvals0, numElems*sizeof(unsigned int) ) );
  checkCudaErrors(cudaMalloc((void**) &d_tempvals1, numElems*sizeof(unsigned int) ) );
  const dim3 blockSize(520 ,1, 1);
  const dim3 gridSize(1, 1, 1);

//checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost));

  for (unsigned int digit = 0; digit < n_dig; ++digit) {
   checkCudaErrors(cudaMemset((void**) d_binHistogram, 0, numBins*sizeof(unsigned int) ) );
   checkCudaErrors(cudaMemset((void**) d_predicate0, 0, numElems*sizeof(unsigned int) ) );
   checkCudaErrors(cudaMemset((void**) d_tempvals0, 0, numElems*sizeof(unsigned int) ) );
   hist_create <<<gridSize, blockSize>>> (d_binHistogram, d_inputVals, digit);  
    //checkCudaErrors(cudaMemcpy(&h_binHistogram, d_binHistogram, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("i %d   \n", digit );
   predicate <<<gridSize, blockSize>>> ( d_inputVals,
			                 d_predicate0,
				         d_predicate1,
			                 digit);  

 //checkCudaErrors(cudaMemcpy(h_predicate0, d_predicate0, numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost));
   // printf("d_p %d   \n", h_predicate0[22] );

 //  for (unsigned int j = 0; j < numElems; ++j) { 
  //  h_predicate0[j] = 0;
  // }
  //   unsigned int mask = 1 << digit;
  //   unsigned int summ=0;
    //perform histogram of data & mask into bins
 //    for (unsigned int j = 0; j < numElems; ++j) {
 //       unsigned int bin = (h_inputVals[j] & mask) >> digit;
 //      if(bin == 0)  { 
 //      summ +=1; 
 //      h_predicate0[j] = summ;}
 //    }
   //printf("h_p %d   ", h_predicate0[29] );
// checkCudaErrors(cudaMemcpy(d_predicate0, h_predicate0, numElems*sizeof(unsigned int), cudaMemcpyHostToDevice));
// checkCudaErrors(cudaMemcpy(d_predicate1, h_predicate1, numElems*sizeof(unsigned int), cudaMemcpyHostToDevice));
 //  cudaDeviceSynchronize();
    for(int s = 1; s <= numElems; s *= 2) {
     for (int j = 0; j < 424 ; j++) {
      exclus_sum <<<gridSize, blockSize>>> (d_predicate0, d_tempvals0, s, j);
      cudaDeviceSynchronize();
     }
    }
 //  exclus_sum <<<gridSize, blockSize>>> (d_predicate1, d_tempvals1, numElems);


  checkCudaErrors(cudaMemcpy(h_test, d_tempvals0, numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost));

   //printf("d_p %d   \n", h_test[22] );
// checkCudaErrors(cudaMemcpy(h_test2, d_tempvals1, numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost));
/*

   compact<<<gridSize, blockSize>>> (d_inputVals,
               			      d_inputPos,
               			      d_outputVals,
               			      d_outputPos,
               			      d_tempvals0,
               		              d_tempvals1,
               		              d_predicate0,
			              d_binHistogram); 
    cudaDeviceSynchronize();
   if (digit < (n_dig-1) ){
     swap_oi<<<gridSize, blockSize>>> (d_inputVals,
               			       d_inputPos,
               			       d_outputVals,
               			       d_outputPos); 
   }
    cudaDeviceSynchronize();
  
  */

 }


 checkCudaErrors(cudaFree(d_binHistogram));
 checkCudaErrors(cudaFree(d_predicate0));
 checkCudaErrors(cudaFree(d_predicate1));
 checkCudaErrors(cudaFree(d_tempvals0));
 checkCudaErrors(cudaFree(d_tempvals1));

}
























