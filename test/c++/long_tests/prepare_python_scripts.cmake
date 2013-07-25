set(ref_pytriqs_pythonpath $ENV{TRIQS_1_INSTALL_DIR}/lib/python2.7/site-packages)

set(preamble
"#!/bin/env python

import sys
sys.path = [\"${ref_pytriqs_pythonpath}\"] + sys.path
")

file(GLOB python_scripts ${SRC_DIR}/*.py)
foreach(f ${python_scripts})
	file(RELATIVE_PATH f ${SRC_DIR} ${f})
	configure_file(${SRC_DIR}/${f} ${f} @ONLY)
endforeach()
