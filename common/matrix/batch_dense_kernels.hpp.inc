/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

template <typename ValueType>
__device__ __forceinline__ void add_scaled(
    const gko::batch_dense::BatchEntry<const ValueType> &alpha,
    const gko::batch_dense::BatchEntry<const ValueType> &x,
    const gko::batch_dense::BatchEntry<ValueType> &y)
{
    if (alpha.num_rhs == 1) {
        for (int i = threadIdx.x; i < x.num_rows; i += blockDim.x) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] +=
                    alpha.values[0] * x.values[i * x.stride + j];
            }
        }
    } else {
        for (int i = threadIdx.x; i < x.num_rows; i += blockDim.x) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] +=
                    alpha.values[j] * x.values[i * x.stride + j];
            }
        }
    }
}

template <typename ValueType>
__global__ __launch_bounds__(default_block_size, sm_multiplier) void add_scaled(
    const gko::batch_dense::UniformBatch<const ValueType> alpha,
    const gko::batch_dense::UniformBatch<const ValueType> x,
    const gko::batch_dense::UniformBatch<ValueType> y)
{
    for (size_type ibatch = blockIdx.x; ibatch < x.num_batch;
         ibatch += gridDim.x) {
        const auto alpha_b = gko::batch::batch_entry(alpha, ibatch);
        const auto x_b = gko::batch::batch_entry(x, ibatch);
        const auto y_b = gko::batch::batch_entry(y, ibatch);
        add_scaled(alpha_b, x_b, y_b);
    }
}

namespace {

/**
 * The answer is only available on thread 0 of the thread-block.
 */
template <typename ReduceType, typename LocalOp>
__device__ __forceinline__ ReduceType reduce_one_rhs(LocalOp local_op,
                                                     const int num_rows)
{
    __shared__ ReduceType sum;
    if (threadIdx.x == 0) {
        sum = zero<ReduceType>();
    }
    const int num_its = (num_rows - 1) / blockDim.x + 1;
    constexpr auto warp_size = config::warp_size;
    const auto tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const int tile_rank = threadIdx.x / warp_size;
    // const int num_tiles = (blockDim.x - 1) / warp_size + 1;
    constexpr int num_tiles = (default_block_size - 1) / warp_size + 1;
    for (int i = 0; i < num_its; i++) {
        const int row_offset = i * blockDim.x;
        const int row = row_offset + tile_rank * warp_size + tile.thread_rank();
        if (row >= num_rows) {
            break;
        }
        __shared__ ReduceType warp_sums[num_tiles];
        // const ValueType localdata = conj(x.values[row * x.stride + j])
        //     * y.values[row * y.stride + j];
        const ReduceType localdata = local_op(row);
        // reduce within each warp
        const ReduceType warpsum = reduce(
            tile, localdata,
            [](ReduceType local, ReduceType remote) { return local + remote; });
        if (tile.thread_rank() == 0) {
            warp_sums[tile_rank] = warpsum;
        }
        __syncthreads();
        // Reduce across warps using the first warp
        if (tile_rank == 0) {
            for (int it = tile.thread_rank(); it < num_tiles; it += warp_size) {
                const ReduceType sum2 =
                    reduce(tile, warp_sums[it],
                           [](ReduceType a, ReduceType b) { return a + b; });
                if (tile.thread_rank() == 0) {
                    sum += sum2;
                }
            }
        }
    }
    return sum;
}

}  // namespace

/**
 * Computes the dot products of some column vectors in global or shared memory.
 *
 * @param result  Holds dot product value for each pair of vectors.
 */
template <typename ValueType>
__device__ __forceinline__ void compute_dot(
    const gko::batch_dense::BatchEntry<const ValueType> &x,
    const gko::batch_dense::BatchEntry<const ValueType> &y,
    const gko::batch_dense::BatchEntry<ValueType> &result)
{
    // for (int j = threadIdx.x; j < x.num_rhs; j += blockDim.x) {
    //     result.values[j] = zero<ValueType>();
    // }
    // __syncthreads();
    // This is inefficient for non-unit stride
    for (int j = 0; j < x.num_rhs; j++) {
        const ValueType sum = reduce_one_rhs<ValueType>(
            [&](int row) {
                return conj(x.values[row * x.stride + j]) *
                       y.values[row * y.stride + j];
            },
            x.num_rows);
        if (threadIdx.x == 0) {
            result.values[j] = sum;
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < x.num_rhs; j += blockDim.x) {
        // result.values[j] = 2.0+sqrt(result.values[j]);
        printf("Nomr: %g ", result.values[j]);
    }
}

template <typename ValueType>
__global__
    __launch_bounds__(default_block_size, sm_multiplier) void compute_dot(
        const gko::batch_dense::UniformBatch<const ValueType> x,
        const gko::batch_dense::UniformBatch<const ValueType> y,
        const gko::batch_dense::UniformBatch<ValueType> result)
{
    for (size_type ibatch = blockIdx.x; ibatch < x.num_batch;
         ibatch += gridDim.x) {
        const auto x_b = gko::batch::batch_entry(x, ibatch);
        const auto y_b = gko::batch::batch_entry(y, ibatch);
        const auto r_b = gko::batch::batch_entry(result, ibatch);
        compute_dot(x_b, y_b, r_b);
    }
}


/**
 * Computes the 2-norms of some column vectors in global or shared memory.
 *
 * @param result  Holds norm value for each vector in x.
 */
template <typename ValueType>
__device__ __forceinline__ void compute_norm2(
    const gko::batch_dense::BatchEntry<const ValueType> &x,
    const gko::batch_dense::BatchEntry<remove_complex<ValueType>> &result)
{
    using real_type = remove_complex<ValueType>;
    for (int j = threadIdx.x; j < x.num_rhs; j += blockDim.x) {
        result.values[j] = zero<real_type>();
    }
    __syncthreads();
    // TODO: replace this with a parallel reduce below
    if (threadIdx.x == 0) {
        for (int i = 0; i < x.num_rows; i++) {
            for (int j = 0; j < x.num_rhs; ++j) {
                result.values[j] += squared_norm(x.values[i * x.stride + j]);
            }
        }
    }
    // for(int j = 0; j < x.num_rhs; ++j) {
    //     const real_type sum = reduce_one_rhs<real_type>([&x,j](int row) {
    //             return squared_norm(x.values[row*x.stride+j]);
    //         },
    //         x.num_rows);
    //     if(threadIdx.x == 0) {
    //         result.values[j] = sum;
    //     }
    // }
    __syncthreads();
    for (int j = threadIdx.x; j < x.num_rhs; j += blockDim.x) {
        result.values[j] = sqrt(result.values[j]);
    }
}

template <typename ValueType>
__global__
    __launch_bounds__(default_block_size, sm_multiplier) void compute_norm2(
        const gko::batch_dense::UniformBatch<const ValueType> x,
        const gko::batch_dense::UniformBatch<remove_complex<ValueType>> result)
{
    for (size_type ibatch = blockIdx.x; ibatch < x.num_batch;
         ibatch += gridDim.x) {
        const auto x_b = gko::batch::batch_entry(x, ibatch);
        const auto r_b = gko::batch::batch_entry(result, ibatch);
        compute_norm2(x_b, r_b);
    }
}


/**
 * 'Scales' a dense matrix or set of vectors with a dense vector.
 * It is equivalent to multiplication by a diagonal matrix whose diagonal
 * entries are given by the scaling vector.
 *
 * One thread is assigned to each entry.
 */
template <typename ValueType>
__device__ __forceinline__ void batch_scale(
    const gko::batch_dense::BatchEntry<const ValueType> &scale_vec,
    const gko::batch_dense::BatchEntry<ValueType> &a)
{
    for (int iz = threadIdx.x; iz < a.num_rows * a.num_rhs; iz += blockDim.x) {
        const int row = iz / a.num_rhs;
        const int col = iz % a.num_rhs;
        a.values[row * a.stride + col] *= scale_vec.values[row];
    }
}

/**
 * Copies the values of one multi-vector into another.
 *
 * Note that the output multi-vector should already have memory allocated
 * and stride set.
 */
template <typename ValueType>
__device__ __forceinline__ void copy(
    const gko::batch_dense::BatchEntry<const ValueType> &in,
    const gko::batch_dense::BatchEntry<ValueType> &out)
{
    for (int iz = threadIdx.x; iz < in.num_rows * in.num_rhs;
         iz += blockDim.x) {
        const int i = iz / in.num_rhs;
        const int j = iz % in.num_rhs;
        out.values[i * out.stride + j] = in.values[i * in.stride + j];
    }
}
