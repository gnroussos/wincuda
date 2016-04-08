//
//gpu kernel code

#include "WinCuda.h"

class complex
{
	float r;
	float i;

public:
	__device__ complex(float a, float b) : r(a), i(b)  {}

	__device__ float magnitude2()
	{
		return r*r + i*i;
	}
	__device__ complex operator *(const complex &a)
	{
		return complex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	__device__ complex operator +(const complex &a)
	{
		return complex(r + a.r, i + a.i);
	}
};

__device__ int julia(int x, int  y)
{
	const float scale = 1.4f;

	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	complex c(-0.8f, 0.156f);
	complex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++)
	{
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}
__global__ void kernel(unsigned char *ptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int  offset = x + y * gridDim.x * blockDim.x ;

	int value = julia(x, y);

	ptr[offset * 4 + 0] = 255 * value;
	ptr[offset * 4 + 1] = 255 * value;
	ptr[offset * 4 + 2] = 255 * value;
	ptr[offset * 4 + 3] = 255;
}

extern "C" void runCuda(unsigned char *dev_bitmap, unsigned char *mem_bitmap, unsigned int width, unsigned int height)
{
	//cudaError error = cudaSuccess;
	unsigned int size = width * height * 4;

	cudaMalloc((void**)&dev_bitmap, size);

	dim3 grid(DIM/16, DIM/16);
	dim3 threads(16, 16);

	kernel <<<grid, threads>>>(dev_bitmap);

	cudaMemcpy(mem_bitmap, dev_bitmap, size, cudaMemcpyDeviceToHost);

	/*error = cudaGetLastError();
	if (error != cudaSuccess)
	cout << "cuda failed to run, error= " << error <<endl;*/
}
 

