#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define g 9.81f  // Gravity (m/s^2)
#define l 1.0f   // Pendulum length (m)
#define dt 0.02f // Time step (s)

// CUDA Kernel to update the pendulum state
__global__ void pendulum_step(float* theta, float* omega, float torque, int steps) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= steps) return;

    float dtheta = omega[idx];
    float domega = (-g / l) * sinf(theta[idx]) + torque;

    // Update state
    theta[idx + 1] = theta[idx] + dtheta * dt;
    omega[idx + 1] = omega[idx] + domega * dt;
}

int main() {
    int steps = 200;
    
    // Allocate memory on CPU
    float *h_theta = new float[steps + 1];
    float *h_omega = new float[steps + 1];

    // Initial condition
    h_theta[0] = 0.1f;
    h_omega[0] = 0.0f;

    // Allocate memory on GPU
    float *d_theta, *d_omega;
    cudaMalloc(&d_theta, (steps + 1) * sizeof(float));
    cudaMalloc(&d_omega, (steps + 1) * sizeof(float));

    // Copy initial state to GPU
    cudaMemcpy(d_theta, h_theta, (steps + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega, h_omega, (steps + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block, 256 threads (adjust as needed)
    int threadsPerBlock = 256;
    int blocks = (steps + threadsPerBlock - 1) / threadsPerBlock;
    pendulum_step<<<blocks, threadsPerBlock>>>(d_theta, d_omega, 0.0f, steps);
    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(h_theta, d_theta, (steps + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_omega, d_omega, (steps + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // Print first 10 values
    for (int i = 0; i < 10; i++) {
        std::cout << "Step " << i << ": Theta = " << h_theta[i] << ", Omega = " << h_omega[i] << std::endl;
    }

    // Free memory
    delete[] h_theta;
    delete[] h_omega;
    cudaFree(d_theta);
    cudaFree(d_omega);

    return 0;
}
