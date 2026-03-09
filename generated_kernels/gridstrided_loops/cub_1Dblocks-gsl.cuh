namespace kernels {
  __global__ __launch_bounds__(512) void PyCodegen_VDotProd_gen (const double * RESTRICT const _data_x, const double * RESTRICT const _data_y, const int32_t _size_x_0, const int32_t _size_x_1, const int32_t _size_x_2, const int32_t _stride_x_0, const int32_t _stride_x_1, const int32_t _stride_x_2, const int32_t _stride_y_0, const int32_t _stride_y_1, const int32_t _stride_y_2, double * const r)
  {
    double r_local = 0.0;
    for(int32_t ctr_2__0 = 0; ctr_2__0 < _size_x_2; ctr_2__0 += (int32_t) gridDim.z * (int32_t) blockDim.z)
    {
      for(int32_t ctr_1__0 = 0; ctr_1__0 < _size_x_1; ctr_1__0 += (int32_t) gridDim.y * (int32_t) blockDim.y)
      {
        for(int32_t ctr_0__0 = 0; ctr_0__0 < _size_x_0; ctr_0__0 += (int32_t) gridDim.x * (int32_t) blockDim.x)
        {
          const int32_t ctr_2 = ctr_2__0 + ((int32_t) blockIdx.z * (int32_t) blockDim.z + (int32_t) threadIdx.z);
          const int32_t ctr_1 = ctr_1__0 + ((int32_t) blockIdx.y * (int32_t) blockDim.y + (int32_t) threadIdx.y);
          const int32_t ctr_0 = ctr_0__0 + ((int32_t) blockIdx.x * (int32_t) blockDim.x + (int32_t) threadIdx.x);
          if(ctr_2 < _size_x_2 && ctr_1 < _size_x_1 && ctr_0 < _size_x_0)
          {
            r_local = r_local + _data_x[ctr_0 * _stride_x_0 + ctr_1 * _stride_x_1 + ctr_2 * _stride_x_2] * _data_y[ctr_0 * _stride_y_0 + ctr_1 * _stride_y_1 + ctr_2 * _stride_y_2];
          }
        }
      }
    }
    {
      typedef cub::BlockReduce <double, 512, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 1, 1> BlockReduce;
      __shared__ typename BlockReduce::TempStorage shared_mem;
      const double block_accum = BlockReduce(shared_mem).Sum(r_local);
      if(block_accum != 0.0 && (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0))
      {
        atomicAdd(r, block_accum);
      }
    }
  }
} // namespace kernels

void PyCodegen_VDotProd(const double * RESTRICT const _data_x, const double * RESTRICT const _data_y, const int32_t _size_x_0, const int32_t _size_x_1, const int32_t _size_x_2, const int32_t _stride_x_0, const int32_t _stride_x_1, const int32_t _stride_x_2, const int32_t _stride_y_0, const int32_t _stride_y_1, const int32_t _stride_y_2, double * const r)
{
  const dim3 block_size { 512, 1, 1 };
  const dim3 grid_size { min( ( uint32_t ( _size_x_0 ) + 512 - 1 ) / 512, 1 ), min( ( uint32_t ( _size_x_1 ) + 1 - 1 ) / 1, 24 ), min( ( uint32_t ( _size_x_2 ) + 1 - 1 ) / 1, 36 ) };
  /* clang-format off */
  /* [pystencils-sfg] Formatting may add illegal spaces between angular brackets in `<<< >>>` */
  kernels::PyCodegen_VDotProd_gen<<< grid_size, block_size, 0 >>>(_data_x, _data_y, _size_x_0, _size_x_1, _size_x_2, _stride_x_0, _stride_x_1, _stride_x_2, _stride_y_0, _stride_y_1, _stride_y_2, r);
  /* clang-format on */
  cudaError_t err = cudaPeekAtLastError();
  if(err != cudaSuccess) {
    printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(1);
  }
}