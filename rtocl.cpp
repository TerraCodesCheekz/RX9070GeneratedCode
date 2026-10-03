// Ths one surprised me. This is an openCL based ray tracer that is supposed to make an apple. It doesn't really look like an apple, but it is GPU accelerated... however, I have heard that it did not work on my best friend's RX 6600, but it does on my 9070... so.

#define CL_TARGET_OPENCL_VERSION 300
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <CL/cl.h>

// -------------------------------------------------------------------------
// OPENCL KERNEL SOURCE (Runs on GPU - Strict OpenCL C Syntax)
// -------------------------------------------------------------------------
const char* kernelSource = R"(
typedef struct { float x, y, z; } Vec3;

Vec3 v_add(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 v_sub(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 v_mul(Vec3 a, float s) { return (Vec3){a.x * s, a.y * s, a.z * s}; }
float v_dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float v_norm(Vec3 a) { return sqrt(a.x*a.x + a.y*a.y + a.z*a.z); }
Vec3 v_normalize(Vec3 a) {
    float n = v_norm(a);
    return (n > 0.0f) ? (Vec3){a.x/n, a.y/n, a.z/n} : (Vec3){0,0,0};
}

typedef struct { Vec3 orig; Vec3 dir; } Ray;

bool intersect_apple(Ray r, Vec3 axes, float* t, Vec3* normal, Vec3* p) {
    Vec3 o = {r.orig.x / axes.x, r.orig.y / axes.y, r.orig.z / axes.z};
    Vec3 d = {r.dir.x / axes.x, r.dir.y / axes.y, r.dir.z / axes.z};
    float b = 2.0f * (o.x * d.x + o.y * d.y + o.z * d.z);
    float c = (o.x * o.x + o.y * o.y + o.z * o.z) + (d.x * d.x + d.y * d.y + d.z * d.z) - 1.0f;
    float disc = b * b - 4.0f * c;
    if (disc <= 0) return false;
    float hit_t = (-b - sqrt(disc)) / 2.0f;
    if (hit_t < 0.001f) return false;
    *t = hit_t;
    *p = v_add(r.orig, v_mul(r.dir, hit_t));
    *normal = v_normalize((Vec3){(*p).x / (axes.x * axes.x), (*p).y / (axes.y * axes.y), (*p).z / (axes.z * axes.z)});
    return true;
}

bool intersect_stem(Ray r, float* t, Vec3* normal, Vec3* p) {
    float a = 0.1f, h = 0.5f; Vec3 center = {0.0f, 0.8f, 0.0f};
    float dx = r.orig.x - center.x; float dz = r.orig.z - center.z;
    float A = r.dir.x * r.dir.x + r.dir.z * r.dir.z;
    float B = 2.0f * (dx * r.dir.x + dz * r.dir.z);
    float C = dx * dx + dz * dz - a * a;
    float disc = B * B - 4.0f * A * C;
    if (disc <= 0) return false;
    float hit_t = (-B - sqrt(disc)) / (2.0f * A);
    if (hit_t < 0.001f) return false;
    Vec3 hitP = v_add(r.orig, v_mul(r.dir, hit_t));
    if (hitP.y >= center.y && hitP.y <= center.y + h) {
        *t = hit_t; *p = hitP;
        *normal = v_normalize((Vec3){hitP.x - center.x, 0.0f, hitP.z - center.z});
        return true;
    }
    return false;
}

__kernel void raytrace_kernel(
    __global float* image_out,
    int width, int height, int samples,
    float camX, float camY, float camZ,
    float lightDirX, float lightDirY, float lightDirZ)
{
    int i = get_global_id(0);
    if (i >= width * height) return;

    int px = i % width; int py = i / width;
    Vec3 pixelColor = {0, 0, 0};
    Vec3 lDir = v_normalize((Vec3){lightDirX, lightDirY, lightDirZ});

    for (int s = 0; s < samples; ++s) {
        float jitterX = (float)(i % 100) / 1000.0f;
        float jitterY = (float)(py % 100) / 1000.0f;
        // FIXED: Use C-style casts (float) instead of float()
        float u = ((float)px + jitterX) / (float)width;
        float v = ((float)py + jitterY) / (float)height;
        float screenX = (2.0f * u - 1.0f) * ((float)width / (float)height);
        float screenY = 1.0f - 2.0f * v;

        Ray ray = {{camX, camY, camZ}, {screenX, screenY, -1.0f}};
        float t_apple = 1e20f, t_stem = 1e20f;
        Vec3 n_apple, n_stem, p_apple, p_stem;

        // FIXED: Use compound literal (Vec3){...} instead of {}
        Vec3 appleAxes = (Vec3){1.0f, 0.9f, 1.0f};
        bool hit_a = intersect_apple(ray, appleAxes, &t_apple, &n_apple, &p_apple);
        bool hit_s = intersect_stem(ray, &t_stem, &n_stem, &p_stem);

        Vec3 color = {0,0,0};
        if (hit_a && (!hit_s || t_apple < t_stem)) {
            color = v_mul((Vec3){0.7f, 0.05f, 0.05f}, fmax(0.1f, v_dot(n_apple, lDir)));
        } else if (hit_s) {
            color = v_mul((Vec3){0.4f, 0.3f, 0.1f}, fmax(0.1f, v_dot(n_stem, lDir)));
        }
        pixelColor = v_add(pixelColor, color);
    }

    image_out[i*3 + 0] = pixelColor.x / (float)samples;
    image_out[i*3 + 1] = pixelColor.y / (float)samples;
    image_out[i*3 + 2] = pixelColor.z / (float)samples;
}
)";

// -------------------------------------------------------------------------
// HOST CODE (Runs on CPU)
// -------------------------------------------------------------------------

void checkError(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) { std::cerr << "Error: " << msg << " (" << err << ")" << std::endl; exit(1); }
}

int main(int argc, char** argv) {
    int width = 800, height = 600, samples = 1;
    float camX = 0, camY = 0.5f, camZ = 3.0f;
    if (argc > 1) samples = std::stoi(argv[1]);
    if (argc > 2) camX = std::stof(argv[2]);
    if (argc > 3) camY = std::stof(argv[3]);
    if (argc > 4) camZ = std::stof(argv[4]);

    cl_platform_id platform; cl_device_id device; cl_int err;
    checkError(clGetPlatformIDs(1, &platform, NULL), "Platform");
    checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL), "Device");

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "Context");
    // Using the modern function to avoid deprecation warning
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    checkError(err, "Queue");

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    checkError(err, "Program Source");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096]; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        std::cerr << "Kernel Build Error:\n" << log << std::endl; exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "raytrace_kernel", &err);
    checkError(err, "Kernel");

    std::vector<float> h_image(width * height * 3);
    cl_mem d_image = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * width * height * 3, NULL, &err);
    checkError(err, "Buffer");

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_image);
    clSetKernelArg(kernel, 1, sizeof(int), &width);
    clSetKernelArg(kernel, 2, sizeof(int), &height);
    clSetKernelArg(kernel, 3, sizeof(int), &samples);
    clSetKernelArg(kernel, 4, sizeof(float), &camX);
    clSetKernelArg(kernel, 5, sizeof(float), &camY);
    clSetKernelArg(kernel, 6, sizeof(float), &camZ);
    float lDirX = 1.0f, lDirY = 1.0f, lDirZ = 1.0f;
    clSetKernelArg(kernel, 7, sizeof(float), &lDirX);
    clSetKernelArg(kernel, 8, sizeof(float), &lDirY);
    clSetKernelArg(kernel, 9, sizeof(float), &lDirZ);

    size_t global_work_size = width * height;
    std::cout << "Launching GPU kernel (" << width << "x" << height << ") with " << samples << " samples...\n";
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    checkError(err, "Launch Kernel");

    checkError(clEnqueueReadBuffer(queue, d_image, CL_TRUE, 0, sizeof(float) * width * height * 3, h_image.data(), 0, NULL, NULL), "Read Buffer");

    std::cout << "Writing output to apple_gpu.ppm...\n";
    std::ofstream out("apple_gpu.ppm");
    out << "P3\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        int r = std::min(255, (int)(h_image[i * 3 + 0] * 255));
        int g = std::min(255, (int)(h_image[i * 3 + 1] * 255));
        int b = std::min(255, (int)(h_image[i * 3 + 2] * 255));
        out << r << " " << g << " " << b << " ";
    }

    clReleaseMemObject(d_image); clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    std::cout << "Done!\n";
    return 0;
}
