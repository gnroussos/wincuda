#include "WinCuda.h"

class complex
{
	float r;
	float i;

public:
	complex(float a, float b) : r(a), i(b)  {}

	float magnitude2()
	{
		return r*r + i*i;
	}
	complex operator *(const complex &a)
	{
		return complex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	complex operator +(const complex &a)
	{
		return complex(r + a.r, i + a.i);
	}
};

int juliaCPU(int x, int  y)
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

void runCPU(unsigned char *ptr, unsigned int width, unsigned int height)
{
	for (int y = 0; y < height ; y++)
		for (int x = 0; x < width; x++)
		{
			int offset = x + y * width;

			int value = juliaCPU(x, y);

			ptr[offset * 4 + 0] = 0;
			ptr[offset * 4 + 1] = 255 * value;
			ptr[offset * 4 + 2] = 0;
			ptr[offset * 4 + 3] = 255;
		}
}
