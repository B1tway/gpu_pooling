#include <iostream>
#include <CL/cl.hpp>
#include <utility>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <math.h>
#include <vector>
#define __CL_ENABLE_EXCEPTIONS

#define TILE_SIZE (32)
void reportError(cl_int err, const std::string& filename, int line)
{
	if (CL_SUCCESS == err)
		return;
	std::string message = "OpenCL error code " + std::to_string(err) + " encountered at " + filename + ":" + std::to_string(line);
	throw std::runtime_error(message);
}
#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


std::string read_kernel(std::string filename) {
	std::ifstream input_file(filename);
	if (!input_file.is_open()) {
		std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
		exit(EXIT_FAILURE);
	}
	return std::string((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
}

void run_kernel(std::string filename, std::string kernel_name, int32_t w, int32_t h, int32_t pw, int32_t ph, int32_t stride) {
	cl_int errcode;
	float* in = new float[w * h];
	for (size_t i = 0; i < h; i++)
	{
		for (size_t j = 0; j < w; j++)
		{
			in[i * w + j] = i * w + j;

		}
	}
	int32_t ow = ceil((double) w / stride);
	int32_t oh = ceil((double) h / stride);
	float* out = new float[ow * oh];
	for (size_t i = 0; i < oh; i++)
	{
		for (size_t j = 0; j < ow; j++)
		{
			out[i * ow + j] = 0;
		}
	}
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0)
	{
		std::cout << "No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform current_platform = platforms[1];
	std::cout << "Using platform: " << current_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
	std::vector<cl::Device> all_devices;
	errcode = current_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	OCL_SAFE_CALL(errcode);
	if (all_devices.size() == 0) { // Check for issues
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device current_device = all_devices[0];
	std::cout << "Using device: " << current_device.getInfo<CL_DEVICE_NAME>() << "\n";
	std::cout << current_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
	cl::Context context({ current_device });
	std::string kernel_source = read_kernel(filename);
	cl::Program program(context, kernel_source, true);
	if (program.build({ current_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(current_device) << "\n";
		getchar();
		exit(1);
	}
	cl::Event event;
	cl::Buffer buffer_in(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * w * h);
	cl::Buffer buffer_out(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * ow * oh);
	cl::CommandQueue queue(context, current_device, CL_QUEUE_PROFILING_ENABLE);
	errcode = queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(float) * w * h, in);
	OCL_SAFE_CALL(errcode);
	cl::Kernel kernel = cl::Kernel(program, kernel_name.c_str());
	errcode = kernel.setArg(0, buffer_in);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(1, buffer_out);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(2, sizeof(int32_t), &w);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(3, sizeof(int32_t), &h);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(4, sizeof(int32_t), &pw);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(5, sizeof(int32_t), &ph);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(6, sizeof(int32_t), &stride);
	OCL_SAFE_CALL(errcode);
	cl::NDRange global(ow * oh);
	errcode = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, NULL, &event);
	OCL_SAFE_CALL(errcode);
	errcode = queue.finish();
	OCL_SAFE_CALL(errcode);
	cl_ulong startTime;
	cl_ulong endTime;
	event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
	event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
	float milliseconds = (endTime - startTime) / 1000000.0;
	//std::cout << milliseconds << std::endl;
	std::cout << "Execution took " << milliseconds << " milliseconds\n";
	errcode = queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(float) * ow * oh, out);
	OCL_SAFE_CALL(errcode);
	for (size_t i = 0; i < oh; i++)
	{
		for (size_t j = 0; j < ow; j++)
		{

				std::cout << out[i * ow + j] << std::endl;
			/*	std::cout << "ERROR " << i << " " << j << std::endl;*/
			
		}
	}
	free(in);
	free(out);
}
void run_square_kernel(std::string filename, std::string kernel_name, int32_t w, int32_t h) {
	cl_int errcode;
	int* in = new int[w * h];
	int max = 0;
	for (size_t i = 0; i < h; i++)
	{
		for (size_t j = 0; j < w; j++)
		{
			in[i * w + j] = i * w + j;
			max = i * w + j;
		}
	}
	int32_t ow = ceil((double)w / 2);
	int32_t oh = ceil((double)h / 2);
	int* out = new int[ow * oh];
	for (size_t i = 0; i < oh; i++)
	{
		for (size_t j = 0; j < ow; j++)
		{
			out[i * ow + j] = 0;
		}
	}
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0)
	{
		std::cout << "No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform current_platform = platforms[1];
	std::cout << "Using platform: " << current_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
	std::vector<cl::Device> all_devices;
	errcode = current_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	OCL_SAFE_CALL(errcode);
	if (all_devices.size() == 0) { // Check for issues
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device current_device = all_devices[0];
	std::cout << "Using device: " << current_device.getInfo<CL_DEVICE_NAME>() << "\n";
	std::cout << current_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
	cl::Context context({ current_device });
	std::string kernel_source = read_kernel(filename);
	cl::Program program(context, kernel_source, true);
	if (program.build({ current_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(current_device) << "\n";
		getchar();
		exit(1);
	}
	cl::Event event;
	cl::Buffer buffer_in(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * w * h);
	cl::Buffer buffer_out(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * ow * oh);
	cl::CommandQueue queue(context, current_device, CL_QUEUE_PROFILING_ENABLE);
	errcode = queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(float) * w * h, in);
	OCL_SAFE_CALL(errcode);
	cl::Kernel kernel = cl::Kernel(program, kernel_name.c_str());
	errcode = kernel.setArg(0, buffer_in);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(1, buffer_out);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(2, sizeof(int32_t), &w);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(3, sizeof(int32_t), &h);
	OCL_SAFE_CALL(errcode);
	cl::NDRange global(ow, oh);
	cl::NDRange local(32, 32);
	errcode = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL, &event);
	OCL_SAFE_CALL(errcode);
	errcode = queue.finish();
	OCL_SAFE_CALL(errcode);
	cl_ulong startTime;
	cl_ulong endTime;
	event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
	event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
	float milliseconds = (endTime - startTime) / 1000000.0;
	//std::cout << milliseconds << std::endl;
	std::cout << "Execution took " << milliseconds << " milliseconds\n";
	errcode = queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(int) * ow * oh, out);
	OCL_SAFE_CALL(errcode);
	//for (size_t i = 0; i < oh; i++)
	//{
	//	for (size_t j = 0; j < ow; j++)
	//	{

	//		std::cout << out[i * ow + j] << "   ";
	//	/*		std::cout << "ERROR " << i << " " << j << std::endl;*/

	//	}
	//	std::cout << std::endl;
	//}
	free(in);
	free(out);
}
int main() {
	int32_t n, m;
	//std::cin >> n >> m;
	//run_kernel("polling.cl", "simple_polling", 4, 4, 2, 2, 2);
	run_square_kernel("polling.cl", "square_polling", 1024, 1024);
}
