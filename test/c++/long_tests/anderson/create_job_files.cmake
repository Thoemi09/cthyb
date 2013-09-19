set(krylov_ncpus 256)
 
set(block_g 1)
set(use_qn 1)
set(krylov_log_file "krylov_block_qn.log")
configure_file(${SRC_DIR}/krylov.job.in krylov_block_qn.job)
set(block_g 1)
set(use_qn 0)
set(krylov_log_file "krylov_block.log")
configure_file(${SRC_DIR}/krylov.job.in krylov_block.job)
set(block_g 0)
set(use_qn 1)
set(krylov_log_file "krylov_qn.log")
configure_file(${SRC_DIR}/krylov.job.in krylov_qn.job)
set(block_g 0)
set(use_qn 0)
set(krylov_log_file "krylov.log")
configure_file(${SRC_DIR}/krylov.job.in krylov.job)
 
set(hyb_ncpus 64)
configure_file(${SRC_DIR}/hyb.job.in hyb.job)