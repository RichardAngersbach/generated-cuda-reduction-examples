namespace kernels {
  __global__ __launch_bounds__(256) void PyCodegen_VDotProd_gen (const double * RESTRICT const _data_x, const double * RESTRICT const _data_y, const int32_t _size_x_0, const int32_t _size_x_1, const int32_t _size_x_2, const int32_t _stride_x_0, const int32_t _stride_x_1, const int32_t _stride_x_2, const int32_t _stride_y_0, const int32_t _stride_y_1, const int32_t _stride_y_2, double * const r)
  {
    double r_local = 0.0;
    const int32_t ctr_2 = (int32_t) blockIdx.z * (int32_t) blockDim.z + (int32_t) threadIdx.z;
    const int32_t ctr_1 = (int32_t) blockIdx.y * (int32_t) blockDim.y + (int32_t) threadIdx.y;
    const int32_t ctr_0 = (int32_t) blockIdx.x * (int32_t) blockDim.x + (int32_t) threadIdx.x;
    if(ctr_2 < _size_x_2 && ctr_1 < _size_x_1 && ctr_0 < _size_x_0)
    {
      r_local = r_local + _data_x[ctr_0 * _stride_x_0 + ctr_1 * _stride_x_1 + ctr_2 * _stride_x_2] * _data_y[ctr_0 * _stride_y_0 + ctr_1 * _stride_y_1 + ctr_2 * _stride_y_2];
    }
    {
      const uint32_t thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
      const uint32_t lane_id = thread_id % 32;
      r_local = r_local + __shfl_xor_sync(0xffffffff, r_local, 16u);
      r_local = r_local + __shfl_xor_sync(0xffffffff, r_local, 8u);
      r_local = r_local + __shfl_xor_sync(0xffffffff, r_local, 4u);
      r_local = r_local + __shfl_xor_sync(0xffffffff, r_local, 2u);
      r_local = r_local + __shfl_xor_sync(0xffffffff, r_local, 1u);
      if(r_local != 0.0 && lane_id == 0)
      {
        atomicAdd(r, r_local);
      }
    }
  }
} // namespace kernels

void PyCodegen_VDotProd(const double * RESTRICT const _data_x, const double * RESTRICT const _data_y, const int32_t _size_x_0, const int32_t _size_x_1, const int32_t _size_x_2, const int32_t _stride_x_0, const int32_t _stride_x_1, const int32_t _stride_x_2, const int32_t _stride_y_0, const int32_t _stride_y_1, const int32_t _stride_y_2, double * const r)
{
  {
    const dim3 __block_size { 128, 2, 1 };
    const std::tuple< uint32_t, uint32_t, uint32_t > __work_items { _size_x_0, _size_x_1, _size_x_2 };
    const dim3 __grid_size { (std::get< 0 >(__work_items) + __block_size.x - 1) / __block_size.x, (std::get< 1 >(__work_items) + __block_size.y - 1) / __block_size.y, (std::get< 2 >(__work_items) + __block_size.z - 1) / __block_size.z };
    /* clang-format off */
    /* [pystencils-sfg] Formatting may add illegal spaces between angular brackets in `<<< >>>` */
    kernels::PyCodegen_VDotProd_gen<<< __grid_size, __block_size, 0 >>>(_data_x, _data_y, _size_x_0, _size_x_1, _size_x_2, _stride_x_0, _stride_x_1, _stride_x_2, _stride_y_0, _stride_y_1, _stride_y_2, r);
    /* clang-format on */
  }
  cudaError_t err = cudaPeekAtLastError();
  if(err != cudaSuccess) {
    printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(1);
  }
}