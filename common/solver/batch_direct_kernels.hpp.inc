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
__global__ void setup_batch_pointers(
    const size_type nbatch, const int nrows, const size_type mat_stride,
    ValueType *const __restrict__ entire_batch_mats,
    ValueType **const __restrict__ mat_batches, const int nrhs,
    const size_type vec_stride, ValueType *const __restrict__ entire_batch_vecs,
    ValueType **const __restrict__ vec_batches)
{
    for (size_type ib = thread::get_thread_id_flat(); ib < nbatch;
         ib += thread::get_thread_num_flat()) {
        mat_batches[ib] = gko::batch::batch_entry_ptr(entire_batch_mats,
                                                      mat_stride, nrows, ib);
        vec_batches[ib] = gko::batch::batch_entry_ptr(entire_batch_vecs,
                                                      vec_stride, nrhs, ib);
    }
}


template <typename ValueType>
__global__ void transpose_scale_copy(
    const size_type nbatch, const int nrows, const int nrhs,
    const size_type in_stride, const size_type out_stride,
    const ValueType *const __restrict__ scaling,
    const ValueType *const __restrict__ in, ValueType *const __restrict__ out)
{
    for (size_type ib = blockIdx.x; ib < nbatch; ib += gridDim.x) {
        auto outb = gko::batch::batch_entry_ptr(out, out_stride, nrows, ib);
        auto inb = gko::batch::batch_entry_ptr(in, in_stride, nrhs, ib);
        auto scaleb = gko::batch::batch_entry_ptr(scaling, 1, nrows, ib);
        for (int iz = threadIdx.x; iz < nrows * nrhs; iz += blockDim.x) {
            const int irow = iz / nrhs;
            const int jcol = iz % nrhs;
            outb[irow * out_stride + jcol] =
                inb[jcol * in_stride + irow] * scaleb[irow];
        }
    }
}