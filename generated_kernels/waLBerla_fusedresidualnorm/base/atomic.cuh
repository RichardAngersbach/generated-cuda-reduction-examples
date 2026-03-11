namespace FusedResidualNorm_kernels {
  __global__ __launch_bounds__(256) void FusedResidualNorm (const double * RESTRICT const _data_c, double * RESTRICT const _data_res, const double * RESTRICT const _data_rhs, const int32_t _size_0, const int32_t _size_1, const int32_t _size_2, const int32_t _stride_0, const int32_t _stride_1, const int32_t _stride_2, double * const norm)
  {
    double norm_local = 0.0;
    const int32_t ctr_2 = (int32_t) blockIdx.z * (int32_t) blockDim.z + (int32_t) threadIdx.z;
    const int32_t ctr_1 = (int32_t) blockIdx.y * (int32_t) blockDim.y + (int32_t) threadIdx.y;
    const int32_t ctr_0 = (int32_t) blockIdx.x * (int32_t) blockDim.x + (int32_t) threadIdx.x;
    if(ctr_2 < _size_2 && ctr_1 < _size_1 && ctr_0 < _size_0)
    {
      double * RESTRICT const _data_res__0 = (double * RESTRICT) &_data_res[ctr_0 * _stride_0 + ctr_1 * _stride_1 + ctr_2 * _stride_2];
      const double * RESTRICT const _data_c__0 = (const double * RESTRICT) &_data_c[ctr_0 * _stride_0 + ctr_1 * _stride_1 + ctr_2 * _stride_2];
      const double * RESTRICT const _data_rhs__0 = (const double * RESTRICT) &_data_rhs[ctr_0 * _stride_0 + ctr_1 * _stride_1 + ctr_2 * _stride_2];
      _data_res__0[0] = -6.0 * _data_c__0[0] + _data_c__0[-1 * _stride_2] + _data_c__0[_stride_0] + _data_c__0[_stride_1] + _data_c__0[-1 * _stride_1] + _data_c__0[_stride_2] + _data_c__0[-1 * _stride_0] + _data_rhs__0[0];
      norm_local = norm_local + _data_res__0[0] * _data_res__0[0];
    }
    {
      if(norm_local != 0.0)
      {
        atomicAdd(norm, norm_local);
      }
    }
  }
} // namespace FusedResidualNorm_kernels