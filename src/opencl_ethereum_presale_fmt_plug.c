/*
 * This software is Copyright (c) 2017 Dhiru Kholia <kholia at kth.se> and
 * Copyright (c) 2013 Lukas Odzioba <ukasz at openwall dot net> and it is
 * hereby released to the general public under the following terms:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 */
#ifdef HAVE_OPENCL
#if FMT_EXTERNS_H
extern struct fmt_main fmt_ethereum_presale;
#elif FMT_REGISTERS_H
john_register_one(&fmt_ethereum_presale);
#else
#include <stdint.h>
#include <string.h>

#include "misc.h"
#include "arch.h"
#include "common.h"
#include "formats.h"
#include "options.h"
#include "common-opencl.h"
#define FORMAT_LABEL            "ethereum-presale"
#define FORMAT_NAME            	"Ethereum presale wallet"
#define FORMAT_TAG              "$ethereum-presale$"
#define TAG_LENGTH              (sizeof(FORMAT_TAG) - 1)
#define ALGORITHM_NAME          "PBKDF2-SHA256 AES-256 OpenCL"
#define BENCHMARK_COMMENT       ""
#define BENCHMARK_LENGTH        -1001
#define MIN_KEYS_PER_CRYPT      1
#define MAX_KEYS_PER_CRYPT      1
#define BINARY_SIZE             0
#define BINARY_ALIGN            MEM_ALIGN_WORD
#define SALT_SIZE               sizeof(*cur_salt)
#define SALT_ALIGN              1
#define PLAINTEXT_LENGTH        55
#define KERNEL_NAME             "ethereum_presale_pbkdf2_sha256_kernel"
#define SPLIT_KERNEL_NAME       "pbkdf2_sha256_loop"
#define AES_KERNEL_NAME         "ethereum_presale_decrypt"
#define AES_LEN             1024

#define HASH_LOOPS              (2000) // factors 7 89 113 (for 70400)
#define ITERATIONS              70400

typedef struct {
	uint8_t length;
	uint8_t v[PLAINTEXT_LENGTH];
} pass_t;

typedef struct {
	uint32_t hash[8];
} crack_t;


typedef struct {
	uint8_t  aes_ct[AES_LEN]; /* ciphertext */
	uint32_t aes_len;         /* actual data length (up to AES_LEN) */
} salt_t;

typedef struct {
	uint32_t ipad[8];
	uint32_t opad[8];
	uint32_t hash[8];
	uint32_t W[8];
	uint32_t rounds;
} state_t;

static pass_t *host_pass;			      /** plain ciphertexts **/
static salt_t *host_salt;			      /** salt **/
static cl_int cl_error;
static cl_mem mem_in, mem_out, mem_salt, mem_state, mem_cracked;
static cl_kernel split_kernel, decrypt_kernel;
static struct fmt_main *self;

static unsigned int *cracked, cracked_size;
static salt_t *cur_salt;

#define STEP			0
#define SEED			1024

static const char * warn[] = {
        "xfer: ",  ", init: " , ", crypt: ", ", decrypt: ", ", res xfer: "
};

static int split_events[] = { 2, -1, -1 };

// This file contains auto-tuning routine(s). Has to be included after formats definitions.
#include "opencl-autotune.h"
#include "memdbg.h"

static void create_clobj(size_t kpc, struct fmt_main *self)
{
#define CL_RO CL_MEM_READ_ONLY
#define CL_WO CL_MEM_WRITE_ONLY
#define CL_RW CL_MEM_READ_WRITE
printf("create_clobj kpc %d\n", kpc);

#define CLCREATEBUFFER(_flags, _size, _string)\
	clCreateBuffer(context[gpu_id], _flags, _size, NULL, &cl_error);\
	HANDLE_CLERROR(cl_error, _string);

#define CLKERNELARG(kernel, id, arg, msg)\
	HANDLE_CLERROR(clSetKernelArg(kernel, id, sizeof(arg), &arg), msg);

	host_pass = mem_calloc(kpc, sizeof(pass_t));
	host_salt = mem_calloc(1, sizeof(salt_t));
	cracked_size = kpc * sizeof(*cracked);
	cracked = mem_calloc(cracked_size, 1);

	mem_in = CLCREATEBUFFER(CL_RO, kpc * sizeof(pass_t),
	                        "Cannot allocate mem in");
	mem_salt = CLCREATEBUFFER(CL_RO, sizeof(salt_t),
	                          "Cannot allocate mem salt");
	mem_out = CLCREATEBUFFER(CL_WO, kpc * sizeof(crack_t),
	                         "Cannot allocate mem out");
	mem_state = CLCREATEBUFFER(CL_RW, kpc * sizeof(state_t),
	                           "Cannot allocate mem state");
	mem_cracked = CLCREATEBUFFER(CL_RW, cracked_size,
	                           "Cannot allocate mem cracked");

	CLKERNELARG(crypt_kernel, 0, mem_in, "Error while setting mem_in");
	CLKERNELARG(crypt_kernel, 1, mem_state, "Error while setting mem_state");

	CLKERNELARG(split_kernel, 0, mem_state, "Error while setting mem_state");
	CLKERNELARG(split_kernel, 1 ,mem_out, "Error while setting mem_out");

	CLKERNELARG(decrypt_kernel, 0, mem_salt, "Error while setting mem_salt");
	CLKERNELARG(decrypt_kernel, 1 ,mem_out, "Error while setting mem_out");
	CLKERNELARG(decrypt_kernel, 2 ,mem_cracked, "Error setting mem_cracked");
}

/* ------- Helper functions ------- */
static size_t get_task_max_work_group_size()
{
	size_t s;

	s = autotune_get_task_max_work_group_size(FALSE, 0, crypt_kernel);
	s = MIN(s, autotune_get_task_max_work_group_size(FALSE, 0, split_kernel));
	s = MIN(s, autotune_get_task_max_work_group_size(FALSE, 0, decrypt_kernel));
	printf("get_task_max_work_group_size %d\n", s);
	return s;
}

static void release_clobj(void)
{
	printf("release_clobj\n");
	if (host_salt) {
		HANDLE_CLERROR(clReleaseMemObject(mem_cracked), "Release mem cracked");
		HANDLE_CLERROR(clReleaseMemObject(mem_in), "Release mem in");
		HANDLE_CLERROR(clReleaseMemObject(mem_salt), "Release mem salt");
		HANDLE_CLERROR(clReleaseMemObject(mem_out), "Release mem out");
		HANDLE_CLERROR(clReleaseMemObject(mem_state), "Release mem state");

		MEM_FREE(cracked);
		MEM_FREE(host_pass);
		MEM_FREE(host_salt);
	}
}

static void init(struct fmt_main *_self)
{
	printf("init %d\n", gpu_id);
	self = _self;
	opencl_prepare_dev(gpu_id);
}

static void reset(struct db_main *db)
{
	printf("reset %d\n");
	if (!autotuned) {
		char build_opts[64];

		snprintf(build_opts, sizeof(build_opts),
		         "-DHASH_LOOPS=%u -DPLAINTEXT_LENGTH=%u",
		         HASH_LOOPS, PLAINTEXT_LENGTH);
		opencl_init("$JOHN/kernels/ethereum_presale_kernel.cl",
		            gpu_id, build_opts);

		crypt_kernel =
			clCreateKernel(program[gpu_id], KERNEL_NAME, &cl_error);
		HANDLE_CLERROR(cl_error, "Error creating crypt kernel");

		split_kernel =
			clCreateKernel(program[gpu_id], SPLIT_KERNEL_NAME, &cl_error);
		HANDLE_CLERROR(cl_error, "Error creating split kernel");

		decrypt_kernel =
			clCreateKernel(program[gpu_id], AES_KERNEL_NAME, &cl_error);
		HANDLE_CLERROR(cl_error, "Error creating decrypt kernel");

		// Initialize openCL tuning (library) for this format.
		opencl_init_auto_setup(SEED, HASH_LOOPS, split_events, warn,
		                       2, self, create_clobj, release_clobj,
		                       sizeof(state_t), 0, db);

		// Auto tune execution from shared/included code.
		autotune_run(self, ITERATIONS, 0,
		             (cpu(device_info[gpu_id]) ?
		              1000000000 : 10000000000ULL));
	}
}

static void done(void)
{
	if (autotuned) {
		release_clobj();
		HANDLE_CLERROR(clReleaseKernel(crypt_kernel), "Release kernel 1");
		HANDLE_CLERROR(clReleaseKernel(split_kernel), "Release kernel 2");
		HANDLE_CLERROR(clReleaseKernel(decrypt_kernel), "Release kernel 3");
		HANDLE_CLERROR(clReleaseProgram(program[gpu_id]),
		               "Release Program");

		autotuned--;
	}
}

static void set_salt(void *salt)
{

	cur_salt = (salt_t*)salt;
	printf("set_salt ct len %d\n", cur_salt->aes_len);
	memcpy(host_salt->aes_ct, cur_salt->aes_ct, cur_salt->aes_len);
	host_salt->aes_len = cur_salt->aes_len;

	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], mem_salt,
		CL_FALSE, 0, sizeof(salt_t), host_salt, 0, NULL, NULL),
	    "Copy salt to gpu");
}

static int crypt_all(int *pcount, struct db_salt *salt)
{
	int i;
	const int count = *pcount;
	int loops = (2000 + HASH_LOOPS - 1) / HASH_LOOPS;
	size_t *lws = local_work_size ? &local_work_size : NULL;
	printf("crypt_all %d, loops: %d\n", count, loops);
	global_work_size = GET_MULTIPLE_OR_BIGGER(count, local_work_size);

	// Copy data to gpu
	BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], mem_in,
		CL_FALSE, 0, global_work_size * sizeof(pass_t), host_pass, 0,
		NULL, multi_profilingEvent[0]), "Copy data to gpu");

	// Run kernel
	BENCH_CLERROR(clEnqueueNDRangeKernel(queue[gpu_id], crypt_kernel,
		1, NULL, &global_work_size, lws, 0, NULL, multi_profilingEvent[1]), "Run kernel");

	for (i = 0; i < (ocl_autotune_running ? 1 : loops); i++) {
		BENCH_CLERROR(clEnqueueNDRangeKernel(queue[gpu_id], split_kernel,
			1, NULL, &global_work_size, lws, 0, NULL, multi_profilingEvent[2]), "Run split kernel");
		BENCH_CLERROR(clFinish(queue[gpu_id]), "clFinish");
		opencl_process_event();
	}

	// Run eth_ps decrypt/compare kernel
	BENCH_CLERROR(clEnqueueNDRangeKernel(queue[gpu_id], decrypt_kernel,
		1, NULL, &global_work_size, lws, 0, NULL, multi_profilingEvent[3]), "Run kernel");

	// Read the result back
	BENCH_CLERROR(clEnqueueReadBuffer(queue[gpu_id], mem_cracked,
		CL_TRUE, 0, cracked_size, cracked, 0,
		NULL, multi_profilingEvent[4]), "Copy result back");

	printf("crypt_all done %d\n", count);
	return count;
}

static int cmp_all(void *binary, int count)
{
	int i;
	printf("cmp_all %d \n", count);
	for (i = 0; i < count; i++)
		if (cracked[i])
			return 1;
	return 0;
}

static int cmp_one(void *binary, int index)
{
	printf("cmp_one %d \n", index);
	return cracked[index];
}

static int cmp_exact(char *source, int index)
{
	printf("cmp_exact %d \n", index);
	return 1;
}

static void set_key(char *key, int index)
{
	int saved_len = MIN(strlen(key), PLAINTEXT_LENGTH);
	printf("set_key %d, %s\n", index, key);

	memcpy(host_pass[index].v, key, saved_len);
	host_pass[index].length = saved_len;

}

static char *get_key(int index)
{
	static char ret[PLAINTEXT_LENGTH + 1];
	printf("get_key %d\n", index);

	memcpy(ret, host_pass[index].v, PLAINTEXT_LENGTH);
	ret[host_pass[index].length] = 0;

	return ret;
}

struct fmt_tests eth_ps_tests[] = {
	{"$ethereum-presale$1712024504cf3ac45c2e7bcbf3df62c958b5dc4c6e87f7491ef2c745502b67ce3e671a19a02a118a4634bb9d84e49a3ab7c92299424b2c68bfe11e26adac635b2d9895c5f727c78e6eb1eeef5283ee270d8ffdadff381a24e3a727277dc02537", "this-is-a-test"},
	{NULL}
};

int eth_ps_common_valid(char *ciphertext, struct fmt_main *self)
{
	char *ctcopy, *keeptr, *p;
	int value, extra;
	int salt_length;
	printf("eth_ps_common_valid %s\n", ciphertext);
	if (strncmp(ciphertext, FORMAT_TAG, TAG_LENGTH) != 0)
		return 0;

	ctcopy = strdup(ciphertext);
	keeptr = ctcopy;

	ctcopy += TAG_LENGTH;
	printf("eth_ps_common_valid ctcopy %s\n", ctcopy);
	if (hexlenl(ctcopy, &extra) < 100 || extra){
		printf("eth_ps_common_valid invalid hash %s\n", ctcopy);
		goto err;
}
	printf("eth_ps_common_valid VALID %s\n", ciphertext);
	MEM_FREE(keeptr);
	return 1;

err:
	printf("eth_ps_common_valid INVALID %s\n", ciphertext);
	MEM_FREE(keeptr);
	return 0;
}

void *ethereum_presale_get_salt(char *ciphertext)
{
	char *ctcopy = strdup(ciphertext);
	char *keeptr = ctcopy;
	int i;
	char *p;
	static salt_t *cs;

	printf("*****ethereum_presale_get_salt ctcopy %s\n", ciphertext);
	cs = mem_calloc_tiny(sizeof(salt_t), MEM_ALIGN_WORD);

	ctcopy += TAG_LENGTH;
	p = ctcopy;
	memset(cs->aes_ct, 0, sizeof(cs->aes_ct));
	for (i = 0; p[i * 2] && i < AES_LEN; i++){
		cs->aes_ct[i] = atoi16[ARCH_INDEX(p[i * 2])] * 16
			+ atoi16[ARCH_INDEX(p[i * 2 + 1])];
		//printf("%d => %x\n", i, cs->aes_ct[i]);
	}
	printf("cs->aes_len %d\n", i, cs->aes_len);
	cs->aes_len = i;

	MEM_FREE(keeptr);

	return (void *)cs;
}

unsigned int eth_ps_common_iteration_count(void *salt)
{
	return 2000;
}

struct fmt_main fmt_ethereum_presale = {
	{
		FORMAT_LABEL,
		FORMAT_NAME,
		ALGORITHM_NAME,
		BENCHMARK_COMMENT,
		BENCHMARK_LENGTH,
		0,
		PLAINTEXT_LENGTH,
		BINARY_SIZE,
		BINARY_ALIGN,
		SALT_SIZE,
		SALT_ALIGN,
		MIN_KEYS_PER_CRYPT,
		MAX_KEYS_PER_CRYPT,
		FMT_CASE | FMT_8_BIT,
		{
			"iteration count",
		},
		{ FORMAT_TAG },
		eth_ps_tests
	}, {
		init,
		done,
		reset,
		fmt_default_prepare,
		eth_ps_common_valid,
		fmt_default_split,
		fmt_default_binary,
		ethereum_presale_get_salt,
		{
			eth_ps_common_iteration_count,
		},
		fmt_default_source,
		{
			fmt_default_binary_hash
		},
		fmt_default_salt_hash,
		NULL,
		set_salt,
		set_key,
		get_key,
		fmt_default_clear_keys,
		crypt_all,
		{
			fmt_default_get_hash
		},
		cmp_all,
		cmp_one,
		cmp_exact
	}
};

#endif /* plugin stanza */

#endif /* HAVE_OPENCL */
