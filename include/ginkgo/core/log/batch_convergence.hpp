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

#ifndef GKO_PUBLIC_CORE_LOG_BATCH_CONVERGENCE_HPP_
#define GKO_PUBLIC_CORE_LOG_BATCH_CONVERGENCE_HPP_


#include <memory>


#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


namespace gko {
/**
 * @brief The logger namespace .
 * @ref log
 * @ingroup log
 */
namespace log {


/**
 * Logs the final residuals and iteration counts for a batch solver.
 *
 * The purpose of this logger is to give simple access to standard data
 * generated by the solver once it has converged.
 *
 * @ingroup log
 */
template <typename ValueType = default_precision>
class BatchConvergence : public Logger {
public:
    using real_type = remove_complex<ValueType>;

    /**
     * Copies arrays of iterations and residual norms into this.
     *
     * A deep copy is done so the arguments can be on any executor.
     *
     * @param num_iterations  Array (size number of matrices x number of
     *     right-hand sides) which stores the iteration count at which each RHS
     *     of each linear system converged. The convergence iteration count for
     *     the different RHS are stored contiguously.
     * @param residual_norm  A BatchDense matrix of size
     *     num_matrices x 1 x num_RHS, which stores the final residual norms.
     */
    void on_batch_solver_completed(
        const Array<int> &num_iterations,
        const BatchLinOp *residual_norm) const override;

    /**
     * Creates a convergence logger. This dynamically allocates the memory,
     * constructs the object and returns an std::unique_ptr to this object.
     *
     * @param exec  the executor
     * @param enabled_events  the events enabled for this logger. By default all
     *                        events.
     *
     * @return an std::unique_ptr to the the constructed object
     *
     * @internal here I cannot use EnableCreateMethod due to complex circular
     * dependencies. At the same time, this method is short enough that it
     * shouldn't be a problem.
     */
    static std::unique_ptr<BatchConvergence> create(
        std::shared_ptr<const Executor> exec,
        const mask_type &enabled_events = Logger::all_events_mask)
    {
        return std::unique_ptr<BatchConvergence>(
            new BatchConvergence(exec, enabled_events));
    }

    /**
     * @return  The number of iterations for entire batch
     */
    const Array<int> &get_num_iterations() const noexcept
    {
        return num_iterations_;
    }

    /**
     * @return  The residual norms for the entire batch.
     */
    const matrix::BatchDense<real_type> *get_residual_norm() const noexcept
    {
        return residual_norm_.get();
    }

protected:
    /**
     * Creates a Convergence logger.
     *
     * @param exec  the executor
     * @param enabled_events  the events enabled for this logger. By default all
     *                        events.
     */
    explicit BatchConvergence(
        std::shared_ptr<const gko::Executor> exec,
        const mask_type &enabled_events = Logger::all_events_mask)
        : Logger(exec, enabled_events),
          num_iterations_(exec),
          residual_norm_(matrix::BatchDense<real_type>::create(exec))
    {}

private:
    mutable Array<int> num_iterations_;
    mutable std::unique_ptr<matrix::BatchDense<real_type>> residual_norm_{};
};


}  // namespace log
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_LOG_BATCH_CONVERGENCE_HPP_
