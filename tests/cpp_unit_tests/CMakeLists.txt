set(TEST_NAMES
  test_distribution
  test_file_utils
  test_tally
  test_interpolate
  test_math
  test_mcpl_stat_sum
  test_mesh
  test_region
  test_tensor
  # Add additional unit test files here
)

foreach(test ${TEST_NAMES})
  add_executable(${test} ${test}.cpp)
  target_link_libraries(${test} Catch2::Catch2WithMain libopenmc)
  add_test(NAME ${test} COMMAND ${test} WORKING_DIRECTORY ${UNIT_TEST_BIN_OUTPUT_DIR})
endforeach()

# Shared-window tests: provide their own main (MPI_Init/Finalize around the
# Catch2 session) and run under multiple MPI ranks, so they are wired up
# separately from the single-process tests above. They exercise cross-process
# atomics, so they require both MPI and OpenMP (without OpenMP the atomic
# scoring is a no-op and the hammer test would legitimately detect lost
# updates).
if(OPENMC_USE_MPI AND OPENMC_USE_OPENMP)
  add_executable(test_shared_window test_shared_window.cpp)
  target_link_libraries(test_shared_window Catch2::Catch2 libopenmc
                        OpenMP::OpenMP_CXX)
  # 2 ranks (single node): the atomic hammer.
  add_test(NAME test_shared_window
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:test_shared_window> "[hammer]"
    WORKING_DIRECTORY ${UNIT_TEST_BIN_OUTPUT_DIR})
  # 4 ranks split into 2 artificial nodes: the multi-node end-of-batch protocol
  # (internode reduce + non-master-leader zero) that a single node cannot reach.
  add_test(NAME test_shared_window_multinode
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            $<TARGET_FILE:test_shared_window> "[multinode]"
    WORKING_DIRECTORY ${UNIT_TEST_BIN_OUTPUT_DIR})

  # Rma remote-scoring tests: drive a real Tally through the score_add seam so
  # the row coalescer, per-target double-buffered staging, batched hindexed
  # MPI_Accumulate, and end-of-batch drain are exercised across ranks/threads.
  # Also needs its own main (MPI_Init_thread + Catch2 session) run under MPI.
  add_executable(test_rma_window test_rma_window.cpp)
  target_link_libraries(test_rma_window Catch2::Catch2 libopenmc
                        OpenMP::OpenMP_CXX)
  # 2 ranks (single node): all three staging cases (hammer/wrap/drain).
  add_test(NAME test_rma_window
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:test_rma_window> "[rma]"
    WORKING_DIRECTORY ${UNIT_TEST_BIN_OUTPUT_DIR})
  # 4 ranks: uneven block ownership plus each thread staging to more targets.
  add_test(NAME test_rma_window_n4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            $<TARGET_FILE:test_rma_window> "[rma]"
    WORKING_DIRECTORY ${UNIT_TEST_BIN_OUTPUT_DIR})
endif()
