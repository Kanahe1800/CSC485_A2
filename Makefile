all:
	nvcc -O3 -std=c++20 -arch=native main.cu -o a2.exe

run: all
	a2.exe

clean:
	-del /q a2.exe a2.exp a2.lib 2>nul