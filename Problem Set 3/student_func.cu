/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <iostream>
#include <stdio.h>

/// parallel MIN-1 reduce function on GPU
__global__ void min_reduce1(float* d_min,
                           const float* const d_in )
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = (blockDim.x / 2); s > 0; s >>= 1)
    {
            if ( (tid <=s ) && ((tid + s) < blockDim.x  )  )
        {
           // sdata[tid] += sdata[tid + s];
             if (sdata[tid + s] < sdata[tid] ) {
              sdata[tid] = sdata[tid + s];
             }
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    { 
        d_min[blockIdx.x] = sdata[0];
    }

}


/// parallel MIN-2 reduce function on GPU
__global__ void min_reduce2(float* d_min,
                            float* d_in )
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = (blockDim.x / 2) ; s > 0; s >>= 1)
    {
            if ( (tid <= s) && ((tid + s) < blockDim.x  )  )
        {
           // sdata[tid] += sdata[tid + s];
             if (sdata[tid + s] < sdata[tid] ) {
              sdata[tid] = sdata[tid + s];
             }
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_min[0] = sdata[0];
    }

}

/// parallel MAX-1 reduce function on GPU
__global__ void max_reduce1(float* d_max,
                           const float* const d_in )
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = (blockDim.x / 2); s > 0; s >>= 1)
    {
            if ( (tid <= s) && ((tid + s) < blockDim.x  )  )
        {
           // sdata[tid] += sdata[tid + s];
             if (sdata[tid + s] > sdata[tid] ) {
              sdata[tid] = sdata[tid + s];
             }
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_max[blockIdx.x] = sdata[0];
    }
}


/// parallel MAX-2 reduce function on GPU
__global__ void max_reduce2(float* d_max,
                            float*  d_in )
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = (blockDim.x / 2) ; s > 0; s >>= 1)
    {
            if ( (tid <= s) && ((tid + s) < blockDim.x  )  )
        {
           // sdata[tid] += sdata[tid + s];
             if (sdata[tid + s] > sdata[tid] ) {
              sdata[tid] = sdata[tid + s];
             }
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_max[0] = sdata[0];
    }
}


/// parallel histogram function on GPU
__global__ void hist_create(float d_min, 
                             const float* const d_in,
                              int* d_hist,
                              const size_t numBins,
                              float lumRange)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    float myItem = d_in[myId];
    unsigned int myBin  = ((myItem - d_min)/lumRange)*numBins ;
     if (myBin > numBins -1) {
       myBin = numBins -1;
     }
    atomicAdd(&(d_hist[myBin]), 1);

}

/// parallel cal histogram cdf function on GPU
__global__ void cdf_calc(int* d_hist,
                         unsigned int* const d_cdf,
 			 const size_t numBins,
                          int max_steps )
{
   // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ int isdata[];

    int tid  = threadIdx.x;

    // load shared mem from global mem
    isdata[tid] = d_hist[tid];
    __syncthreads();            // make sure entire block is loaded!

    int temp=1, sum=0;

  for (unsigned int i = 0; i < max_steps ; i++)
    {
    if (  (tid-temp) >= 0 )  {
     sum  =  isdata[tid] + isdata[tid - temp];
    }
    else 
    { sum = isdata[tid];}
    temp*=2;
    __syncthreads(); 
    isdata[tid] = sum;
  __syncthreads();
  }

  d_cdf[tid] = isdata[tid];

}


__global__ void map_intoex(unsigned int* d_in,
                         unsigned int* const d_cdf)
{
    int tid  = threadIdx.x;
    if (tid > 0)
    {  d_cdf[tid]=d_in[tid-1]; }
    else
    {d_cdf[tid]=0;}
 

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{


  //Step 1
  //first we find the minimum and maximum across the entire image


  float *d_min_lognum, *d_max_lognum,*d_min2,*d_max2;
  int  *d_hist_logLum;
  unsigned int *d_logtemp;

  checkCudaErrors(cudaMalloc((void **) &d_min_lognum,  sizeof(float) ) );
  checkCudaErrors(cudaMalloc((void **) &d_max_lognum,  sizeof(float) ) );
  checkCudaErrors(cudaMalloc((void **) &d_min2,  sizeof(float)*numRows) );
  checkCudaErrors(cudaMalloc((void **) &d_max2,  sizeof(float)*numRows) );
  checkCudaErrors(cudaMalloc((void **) &d_logtemp,  sizeof(int)*numBins) );

  const dim3 blockSize(numCols , 1, 1);
  const dim3 gridSize(numRows, 1, 1);
  const dim3 blockSize1(numRows , 1, 1);
  const dim3 gridSize1(1, 1, 1);
  const dim3 blockSize2(numBins , 1, 1);
  int threads= numCols;
  float lumRange;

  //TODO
  //Here are the steps you need to implement
   // 1) find the minimum and maximum value in the input logLuminance channel
    //   store in min_logLum and max_logLum

     min_reduce1 <<<gridSize, blockSize, threads * sizeof(float)>>>
                                (d_min2, d_logLuminance);
     min_reduce2 <<<gridSize1, blockSize1, numRows * sizeof(float)>>>
                                (d_min_lognum, d_min2);
     max_reduce1 <<<gridSize, blockSize, threads * sizeof(float)>>>
                                (d_max2, d_logLuminance);
     max_reduce2 <<<gridSize1, blockSize1, numRows * sizeof(float)>>>
                                (d_max_lognum, d_max2);
  
      checkCudaErrors(cudaMemcpy(&min_logLum, d_min_lognum,   sizeof(float), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(&max_logLum, d_max_lognum,   sizeof(float), cudaMemcpyDeviceToHost));


  //  2) subtract them to find the range
    lumRange = (max_logLum - min_logLum);

  //  3) generate a histogram of all the values in the logLuminance channel using
   //    the formula: bin = (lum[i] - lumMin) / lumRange * numBins

   checkCudaErrors(cudaMalloc((void **) &d_hist_logLum,  sizeof(int)*numBins ) );
    checkCudaErrors( cudaMemset((void**) d_hist_logLum, 0, sizeof(int)*numBins) );

   hist_create <<<gridSize, blockSize>>>
                                (min_logLum, d_logLuminance,d_hist_logLum,numBins,lumRange);
  
   // 4) Perform an exclusive scan (prefix sum) on the histogram to get
    //   the cumulative distribution of luminance values (this should go in the
   //    incoming d_cdf pointer which already has been allocated for you)       

  int max_steps = log2(1.0f*numBins);

            cdf_calc <<<gridSize1, blockSize2, numBins*sizeof(int)>>> ( d_hist_logLum, d_logtemp, numBins, max_steps );
    map_intoex <<<gridSize1, blockSize2>>> ( d_logtemp,d_cdf);



checkCudaErrors(cudaFree(d_min_lognum));
checkCudaErrors(cudaFree(d_max_lognum));
checkCudaErrors(cudaFree(d_hist_logLum));
checkCudaErrors(cudaFree(d_min2));
checkCudaErrors(cudaFree(d_max2));






}
