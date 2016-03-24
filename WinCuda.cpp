// WinCuda.cpp : Defines the entry point for the application.
//

#include "WinCuda.h"

using namespace std;

#define MAX_LOADSTRING 100

// Global Variables
HINSTANCE hInstance;							// current instance
TCHAR szTitle[MAX_LOADSTRING];					// The title bar text
TCHAR szWindowClass[MAX_LOADSTRING];			// the main window class name

TCHAR *barText[] = { "GPU", "CPU", "Time: " };	//status bar text

unsigned char *dev_bitmap = nullptr;			// gpu bitmap
unsigned char *mem_bitmap = nullptr;			// host bitmap

LARGE_INTEGER start;
LARGE_INTEGER stop;
double frequency = 0;							//Time for one clock tick in seconds

// windows functions 
ATOM				MyRegisterClass();
BOOL				InitInstance();
LRESULT CALLBACK	WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK	About(HWND, UINT, WPARAM, LPARAM);

void Clean();

// time function
double GetFrequency();
void StartTimer();
double StopTimer(void);						// return elapsed time in seconds

extern "C" void runCuda(unsigned char *dev_bitmap, unsigned char *mem_bitmap, unsigned int width, unsigned int height);
extern void runCPU(unsigned char *ptr, unsigned int width, unsigned int height);

int main()
{
	MSG msg;
	HACCEL hAccelTable;

	// Initialize global strings
	LoadString(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
	LoadString(hInstance, IDC_WINCUDA, szWindowClass, MAX_LOADSTRING);

	// register class
	MyRegisterClass();

	// application initialization
	if (!InitInstance ())
	{
		return FALSE;
	}

	hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_WINCUDA));
	ZeroMemory(&msg, sizeof(msg));

	// Main message loop
	while (GetMessage(&msg, NULL, 0, 0))
	{
		if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	return (int) msg.wParam;
}

ATOM MyRegisterClass()
{
	WNDCLASSEX wcex;
	
	hInstance = GetModuleHandle(NULL);

	wcex.cbSize			= sizeof(WNDCLASSEX);
	wcex.style			= CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc	= WndProc;
	wcex.cbClsExtra		= 0;
	wcex.cbWndExtra		= 0;
	wcex.hInstance		= hInstance;
	wcex.hIcon			= LoadIcon(hInstance, MAKEINTRESOURCE(IDI_WINCUDA));
	wcex.hCursor		= LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground	= NULL;
	wcex.lpszMenuName	= MAKEINTRESOURCE(IDR_MENU);
	wcex.lpszClassName	= szWindowClass;
	wcex.hIconSm		= LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

	return RegisterClassEx(&wcex);
}

BOOL InitInstance()
{
   HWND hWnd;

   hWnd = CreateWindow(szWindowClass, szTitle, WS_CAPTION | WS_SYSMENU,
	   0, 0, DIM, DIM, NULL, NULL, hInstance, NULL);

   if (!hWnd)
   {
      return FALSE;
   }

   ShowWindow(hWnd, SW_SHOWDEFAULT);
   //UpdateWindow(hWnd);

   return TRUE;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	int wmId, wmEvent;
	PAINTSTRUCT ps;
	HDC hdc;
	HMENU hMenu;
	TCHAR str[10];
	basic_string <TCHAR> msg;
	
	static bool enable_GPU;
	static HWND hBotBar;
	static int parts[2] = { 60, 250 };
	double time = 0.0;

	cudaError error = cudaSuccess;

	switch (message)
	{
	case WM_CREATE:
		// init global behaviour
		enable_GPU = false;
		
		frequency = 1.0 / GetFrequency();
				
		mem_bitmap = new unsigned char[(unsigned int) DIM * (unsigned int) DIM * 4]; 
		
		hBotBar = CreateWindow(STATUSCLASSNAME, NULL, WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, hWnd, NULL, hInstance, NULL);
		SendMessage(hBotBar, SB_SETPARTS, (WPARAM)2, (LPARAM)parts);
	
		SendMessage(hBotBar, SB_SETTEXT, 0, (LPARAM)barText[1]);
		break;
	case WM_COMMAND:
		wmId    = LOWORD(wParam);
		wmEvent = HIWORD(wParam);
		// Parse the menu selections: 
		switch (wmId)
		{
		case IDM_ABOUT:
			DialogBox(hInstance, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
			break;
		case IDM_EXIT:
			DestroyWindow(hWnd);
			break;
		case ID_SETTINGS_GPU:
			if (enable_GPU)
				break;
			else
			{
				enable_GPU = true;
				
				SendMessage(hBotBar, SB_SETTEXT, 0, (LPARAM)barText[0]);
				
				hMenu = GetMenu(hWnd);
				CheckMenuItem(hMenu, ID_SETTINGS_GPU, MF_CHECKED);
				CheckMenuItem(hMenu, ID_SETTINGS_CPU, MF_UNCHECKED);

				//
				//since we profile the performance, reset memory buffers & rerun instead of simply redraw
				//
				ZeroMemory(mem_bitmap, (unsigned int)DIM * (unsigned int)DIM * 4);
				cudaFree(dev_bitmap);

				InvalidateRect(hWnd, NULL, TRUE);
				UpdateWindow(hWnd);
			}
			break;
		case ID_SETTINGS_CPU:
			if (!enable_GPU)
				break;
			else
			{
				enable_GPU = false;
				SendMessage(hBotBar, SB_SETTEXT, 0, (LPARAM)barText[1]);
			
				hMenu = GetMenu(hWnd);
				CheckMenuItem(hMenu, ID_SETTINGS_CPU, MF_CHECKED);
				CheckMenuItem(hMenu, ID_SETTINGS_GPU, MF_UNCHECKED);

				ZeroMemory(mem_bitmap, (unsigned int)DIM * (unsigned int)DIM * 4);

				InvalidateRect(hWnd, NULL, TRUE);
				UpdateWindow(hWnd);
			}
			break;

		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
		break;
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		if (enable_GPU)
		{
			StartTimer();
			runCuda(dev_bitmap, mem_bitmap, (unsigned int)DIM, (unsigned int)DIM);
			time = StopTimer();

			error = cudaGetLastError();
			if (error != cudaSuccess)
			{
				MessageBox(NULL, "cuda failed", (LPCSTR)error, MB_OK);
				return error;
			}

			sprintf(str, "%.3f", time);
			msg.insert(0, barText[2]);
			msg.append(str);
			msg.append(_T(" seconds"));
		}
		else
		{
			StartTimer();
			runCPU(mem_bitmap, (unsigned int)DIM, (unsigned int)DIM);
			time = StopTimer();

			sprintf(str, "%.3f", time);
			msg.insert(0, barText[2]);
			msg.append(str);
			msg.append(_T(" seconds"));
		}
		
		BITMAPINFOHEADER bmInfoHdr;

		int		width, height;
		int		xStart, yStart;
		
		width = height = DIM;
		xStart = yStart = 0;

/////////////////////////////////////
//Windows draws BGR ordered
//swap to RGB
////////////////////////////////////
		//unsigned char	*argbPixelsPtr;
		//unsigned char	*rgbPixelsPtr;
		//int		i, j;

		//argbPixelsPtr = (unsigned char *)argbPixels;
		//rgbPixelsPtr = (unsigned char *)rgbPixels;

		//for (i = 0; i < DIM ;i++)
		//{
		//	for (j = 0; j < DIM; j++)
		//	{
		//		*rgbPixelsPtr++ = *argbPixelsPtr++;
		//		*rgbPixelsPtr++ = *argbPixelsPtr++;
		//		*rgbPixelsPtr++ = *argbPixelsPtr++;
		//		argbPixelsPtr++;						//advance alpha component
		//		rgbPixelsPtr++;
		//	}
		//}
		
		//setup device independent bitmap
		bmInfoHdr.biSize = sizeof(BITMAPINFOHEADER);
		bmInfoHdr.biWidth = width;
		bmInfoHdr.biHeight = height;
		bmInfoHdr.biPlanes = 1;
		bmInfoHdr.biBitCount = 32;
		bmInfoHdr.biCompression = BI_RGB;
		bmInfoHdr.biSizeImage = 0;
		bmInfoHdr.biXPelsPerMeter = 0;
		bmInfoHdr.biYPelsPerMeter = 0;
		bmInfoHdr.biClrUsed = 0;
		bmInfoHdr.biClrImportant = 0;

		//Display dib
		SetDIBitsToDevice(hdc, xStart, yStart, width, height,0, 0, 0, height, mem_bitmap, (BITMAPINFO *)&bmInfoHdr,DIB_RGB_COLORS);
		
		SendMessage(hBotBar, SB_SETTEXT, 1, (LPARAM)msg.c_str());
		EndPaint(hWnd, &ps);
		break;
	case WM_DESTROY:
			Clean();
			PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	cudaDeviceProp prop = { 0 };
	cudaError error = cudaSuccess;
	
	switch (message)
	{
	case WM_INITDIALOG:
		int count, dev;
		TCHAR str[40];

		// get cuda properties
		cudaGetDeviceCount(&count);
		cudaGetDevice(&dev);
		cudaGetDeviceProperties(&prop,dev);
		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			MessageBox(NULL, "cuda failed", (LPCSTR)error, MB_OK);
			return error;
		}
		// print details
		SetDlgItemInt(hDlg, IDC_STATIC_Devs, count, FALSE);
		SetDlgItemInt(hDlg, IDC_STATIC_Current, dev, FALSE);
		SetDlgItemText(hDlg, IDC_STATIC_Name, prop.name);
		sprintf(str, "%d.%d", prop.major, prop.minor);
		SetDlgItemText(hDlg, IDC_STATIC_Ver, str);
		SetDlgItemInt(hDlg, IDC_STATIC_Clock, prop.clockRate, FALSE);
		SetDlgItemInt(hDlg, IDC_STATIC_Memory, prop.totalGlobalMem, FALSE);
		SetDlgItemInt(hDlg, IDC_STATIC_Multi, prop.multiProcessorCount, FALSE);
		SetDlgItemInt(hDlg, IDC_STATIC_Async, prop.asyncEngineCount, FALSE);
		
		return (INT_PTR)TRUE;
	case WM_COMMAND:
		if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
		{
			EndDialog(hDlg, LOWORD(wParam));
			return (INT_PTR)TRUE;
		}
		break;
	}
	return (INT_PTR)FALSE;
}

void Clean()
{
	cudaFree(dev_bitmap);

	delete mem_bitmap;
}

double GetFrequency()
{
	LARGE_INTEGER proc_freq;
	QueryPerformanceFrequency(&proc_freq);
	return static_cast <double> (proc_freq.QuadPart);
}

void StartTimer()
{
	// Set thread to use processor 0
	DWORD_PTR oldmask = SetThreadAffinityMask(GetCurrentThread(), 0);
	QueryPerformanceCounter(&start);
	// Restore original
	SetThreadAffinityMask(GetCurrentThread(), oldmask);
}

double StopTimer()
{
	DWORD_PTR oldmask = SetThreadAffinityMask(GetCurrentThread(), 0);
	QueryPerformanceCounter(&stop);
	SetThreadAffinityMask(GetCurrentThread(), oldmask);
	return ((stop.QuadPart - start.QuadPart) * frequency);
}