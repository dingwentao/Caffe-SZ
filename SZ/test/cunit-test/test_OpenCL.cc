#include <vector>
#include <random>
#include <algorithm>

#include "CUnit/CUnit.h"
#include "CUnit/Basic.h"
#include "CUnit_Array.h"
#include "RegressionTest.hpp"

#include "sz.h"
#include "zlib.h"

namespace {
	struct sz_opencl_state* state = nullptr;
}

extern "C" {
int init_suite()
{
	int rc = sz_opencl_init(&state);
	rc |= sz_opencl_error_code(state);
	return rc;
}

int clean_suite()
{
	int rc = sz_opencl_release(&state);
	return rc;
}


void test_valid_opencl()
{
	int rc = sz_opencl_check(state);
	CU_ASSERT_EQUAL(rc, 0);
}

void test_identical_opencl_impl()
{
	auto num_random_test_cases = 32;
	auto opencl_compressor = sz_compress_float3d_opencl;
	auto existing_compressor = SZ_compress_float_3D_MDQ_decompression_random_access_with_blocked_regression;
	test_identical_output_random(num_random_test_cases, opencl_compressor, existing_compressor);
	test_identical_output_deterministic(opencl_compressor, existing_compressor);
}


int main(int argc, char *argv[])
{
	unsigned int num_failures = 0;
	if (CUE_SUCCESS != CU_initialize_registry())
	{
		return CU_get_error();
	}

	CU_pSuite suite = CU_add_suite("test_opencl_suite", init_suite, clean_suite);
	if(suite == nullptr) {
		goto error;
	}

	if(CU_add_test(suite, "test_valid_opencl", test_valid_opencl) == nullptr ||
			CU_add_test(suite, "test_identical_opencl_impl", test_identical_opencl_impl) == nullptr) {
		goto error;
	}

	CU_basic_set_mode(CU_BRM_VERBOSE);
	CU_basic_run_tests();
	CU_basic_show_failures(CU_get_failure_list());
	num_failures = CU_get_number_of_failures();

error:
	CU_cleanup_registry();
	return  num_failures ||  CU_get_error();
}

}
