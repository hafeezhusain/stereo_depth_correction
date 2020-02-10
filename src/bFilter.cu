#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

#define TILE_X 32
#define TILE_Y 16


using namespace std;
using namespace cv;

// Initialize texture memory to store the input
texture<float, 2, cudaReadModeElementType> inTexture;

// Bilateral filter kernel
__global__ void gpuCalculation(float* input, float* output, int width, int height, size_t pitch, int r)
{
	// Initialize global Tile indices along x,y and xy
	int txIndex = __mul24(blockIdx.x, TILE_X) + threadIdx.x;
	int tyIndex = __mul24(blockIdx.y, TILE_Y) + threadIdx.y;

	// If within image size
	if ((txIndex < width-r) && (tyIndex < height-r) && (txIndex >= r) && (tyIndex >= r))
	{
		float r_c = 0;
		float Ws = 0;
		// Get the centre pixel value
		float centrePx = tex2D(inTexture, txIndex, tyIndex);

		// Iterate through filter size from centre pixel
		for (int dy = -r; dy <= r; dy++) {
			for (int dx = -r; dx <= r; dx++) {
				// Get the current pixel value
				float currPx = tex2D(inTexture, txIndex + dx, tyIndex + dy);
				if (currPx > 0){
					float dist = (float)sqrt(pow(dx, 2) + pow(dy, 2));
					float gs = 1 / (1 + dist);
					float gr = 1 / (1 + abs(centrePx - currPx));
					float w = gs * gr;
					r_c += w * currPx;
					Ws += w;
				}			
			}
		}
		float *row_out = (float *)((char*)output + tyIndex * pitch);
		row_out[txIndex] = (Ws == 0) ? 0 : r_c / Ws;
	}
}

void bilateralFilter_gpu(const Mat & input, Mat & output, int r)
{
	// Size of image
//	int im_size = input.cols*input.rows*sizeof(float);

	// Variables to allocate space for input and output GPU variables
	size_t pitch_in;                                                      // Avoids bank conflicts (Read documentation for further info)
	size_t pitch_out;
	float *d_input = NULL;
	float *d_output = NULL;

	//Allocate device memory
	cudaMallocPitch((void**)&d_input, &pitch_in, sizeof(float)*input.cols, input.rows); // Find pitch
	cudaMemcpy2D(d_input, pitch_in, input.ptr(), sizeof(float)*input.cols, sizeof(float)*input.cols, input.rows, cudaMemcpyHostToDevice); // create input padded with pitch
	cudaBindTexture2D(0, inTexture, d_input, input.cols, input.rows, pitch_in); // bind the new padded input to texture memory
	cudaMallocPitch((void**)&d_output, &pitch_out, sizeof(float)*output.cols, output.rows);
	cudaMemcpy2D(d_output, pitch_out, output.ptr(), sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice);

	//Creating the block size
	dim3 block(TILE_X, TILE_Y);

	//Calculate grid size to cover the whole image
	dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	// Kernel call
	gpuCalculation << <grid, block >> > (d_input, d_output, input.cols, input.rows, pitch_out, r);

	// Wait for the GPU to finish
	cudaDeviceSynchronize();

	// Copy output from device to host
	cudaMemcpy2D(output.ptr(), sizeof(float)*output.cols, d_output, pitch_out, sizeof(float)*output.cols, output.rows, cudaMemcpyDeviceToHost);

	//Unbind the texture
	cudaUnbindTexture(inTexture);	

	// Free GPU variables
	cudaFree(d_input);
	cudaFree(d_output);

}
