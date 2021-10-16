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

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>


// @sect3{Type aliases for convenience}
// Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
// with one column/one row. The advantage of this concept is that using
// multiple vectors is a now a natural extension of adding columns/rows are
// necessary.
using value_type = double;
using real_type = gko::remove_complex<value_type>;
using index_type = int;
using size_type = gko::size_type;
using vec_type = gko::matrix::BatchDense<value_type>;
using real_vec_type = gko::matrix::BatchDense<real_type>;
// using mtx_type = gko::matrix::BatchCsr<value_type, index_type>;
using mtx_type = gko::matrix::BatchEll<value_type, index_type>;
using solver_type = gko::solver::BatchBicgstab<value_type>;


int main(int argc, char* argv[])
{
    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }

    // @sect3{Where do you want to run your solver ?}
    // The gko::Executor class is one of the cornerstones of Ginkgo. Currently,
    // we have support for an gko::OmpExecutor, which uses OpenMP
    // multi-threading in most of its kernels,
    // a gko::ReferenceExecutor, a single threaded specialization of
    // the OpenMP executor and a gko::CudaExecutor which runs the code on a
    // NVIDIA GPU if available.
    // @note With the help of C++, you see that you only ever need to change the
    // executor and all the other functions/ routines within Ginkgo should
    // automatically work and run on the executor with any other changes.
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                 true);
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    // @sect3{Read batch from files}
    // Name of the problem, which is also the directory under which all
    //  matrices are stored.
    const std::string problem_descr_str = argc >= 3 ? argv[2] : "pores_1";
    // Number of linear systems to read from files.
    const size_type num_systems = argc >= 4 ? std::atoi(argv[3]) : 2;
    // Max number of times to duplicate whatever systems are read from files.
    const size_type num_duplications = argc >= 5 ? std::atoi(argv[4]) : 100;
    const int num_picard = argc >= 6 ? std::atoi(argv[5]) : 2;
    const std::string out_file = argc >= 7 ? argv[6] : "timings.txt";
    // Whether to enable diagonal scaling of the matrices before solving.
    //  The scaling vectors need to be available in a 'S.mtx' file.
    const std::string batch_scaling = "none";
    const int nrepeats = 10;

    // @sect3{Create the batch solver factory}
    const real_type reduction_factor{1e-10};
    // Create a batched solver factory with relevant parameters.
    auto solver_gen =
        solver_type::build()
            .with_max_iterations(500)
            .with_residual_tol(reduction_factor)
            .with_tolerance_type(gko::stop::batch::ToleranceType::absolute)
            // .with_preconditioner(gko::preconditioner::batch::type::jacobi)
            .on(exec);

    // open file to write out timing data
    std::cout << "Writing to " << out_file << std::endl;
    std::ofstream outfile(out_file);
    outfile << "processor \"case name\" \"solver type\" \"matrix format\" "
            << "\"tolerance type\" \"batch size\" repetition iterations  "
            << "\"solve time (s)\"\n";

    const size_type num_total_systems = num_systems * num_duplications;
    std::cout << "Total number of systems = " << num_total_systems << std::endl;
    std::cout << num_picard << " Picard iterations.\n";
    double avg_total_time = 0.0;
    int avg_total_iters = 0;
    std::vector<double> total_times(nrepeats, 0.0);
    std::vector<int> total_iters(nrepeats, 0);
    for (int irpt = 0; irpt < nrepeats; irpt++) {
        for (int ipic = 0; ipic < num_picard; ipic++) {
            auto data = std::vector<gko::matrix_data<value_type>>(num_systems);
            std::vector<gko::matrix_data<value_type>> bdata(num_systems);
            auto scale_data =
                std::vector<gko::matrix_data<value_type>>(num_systems);
            auto xinit_data =
                std::vector<gko::matrix_data<value_type>>(num_systems);
            for (size_type i = 0; i < data.size(); ++i) {
                const std::string mat_str = "A.mtx";
                const std::string f_base = problem_descr_str + "/p_" +
                                           std::to_string(ipic) + "/" +
                                           std::to_string(i) + "/";
                std::string fname = f_base + mat_str;
                std::ifstream mtx_fd(fname);
                data[i] = gko::read_raw<value_type>(mtx_fd);
                mtx_fd.close();
                std::string bfname = f_base + "b.mtx";
                std::ifstream b_fd(bfname);
                bdata[i] = gko::read_raw<value_type>(b_fd);
                b_fd.close();

                std::string xfname = f_base + "x_init.mtx";
                std::ifstream x_fd(xfname);
                xinit_data[i] = gko::read_raw<value_type>(x_fd);
                x_fd.close();
                // If necessary, 'scaling vectors' can be provided to
                // diagonal-scale
                //  a system from the left and the right. For this example,
                //  no scaling vectors are provided.
                if (batch_scaling == "explicit") {
                    std::string scale_fname = f_base + "S.mtx";
                    std::ifstream scale_fd(scale_fname);
                    scale_data[i] = gko::read_raw<value_type>(scale_fd);
                }
            }
            auto single_batch = mtx_type::create(exec);
            single_batch->read(data);
            // We can duplicate the batch a few times if we wish.
            std::shared_ptr<mtx_type> A =
                mtx_type::create(exec, num_duplications, single_batch.get());
            // Create RHS
            auto temp_b = vec_type::create(exec);
            temp_b->read(bdata);
            auto b = vec_type::create(exec, num_duplications, temp_b.get());
            // Create initial guess as 0 and copy to device
            auto temp_x = vec_type::create(exec);
            temp_x->read(xinit_data);
            auto x = vec_type::create(exec, num_duplications, temp_x.get());

            // @sect3{Batch logger}
            // Create a logger to obtain the iteration counts and "implicit"
            // residual
            //  norms for every system after the solve.
            std::shared_ptr<const gko::log::BatchConvergence<value_type>>
                logger = gko::log::BatchConvergence<value_type>::create(exec);

            // @sect3{Generate and solve}
            // Generate the batch solver from the batch matrix
            auto solver = solver_gen->generate(A);

            // add the logger to the solver
            solver->add_logger(logger);

            std::chrono::steady_clock::time_point t1 =
                std::chrono::steady_clock::now();
            // Solve the batch system
            solver->apply(lend(b), lend(x));
            std::chrono::steady_clock::time_point t2 =
                std::chrono::steady_clock::now();
            // This is not necessary, but one might want to remove the
            // logger before
            //  the next solve using the same solver object.
            solver->remove_logger(logger.get());

            auto time_span =
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                          t1);
            total_times[irpt] += time_span.count();
            total_iters[irpt] +=
                logger->get_num_iterations().get_const_data()[0];
        }

        outfile << executor_string << " " << problem_descr_str
                << " bicgstab ELL absolute " << num_total_systems << " " << irpt
                << " " << total_iters[irpt] << " " << total_times[irpt] << "\n";
    }

    // for (int irpt = 0; irpt < nrepeats; irpt++) {
    //    avg_total_time += total_times[irpt];
    //    avg_total_iters += total_iters[irpt];
    //}
    // avg_total_time /= nrepeats;
    // const double davgiters = static_cast<double>(avg_total_iters) / nrepeats;
    // avg_total_iters = static_cast<int>(davgiters);


    outfile.close();

    return 0;
}
