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
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "misc.h"
#include "arch.h"
#include "common.h"
#include "formats.h"
#include "options.h"
#include "common-opencl.h"
#include "KeccakHash.h"
#include "ethereum_common.h"

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
	uint8_t  data[AES_LEN]; /* the seed */
	uint32_t length;
	bool cracked;
} seed_t;

typedef struct {
	uint32_t ipad[8];
	uint32_t opad[8];
	uint32_t hash[8];
	uint32_t W[8];
	uint32_t rounds;
} state_t;

static pass_t *host_pass;			      /** plain ciphertexts **/
static custom_salt *host_salt;			      /** salt **/
static cl_int cl_error;
static cl_mem mem_in, mem_out, mem_salt, mem_state, mem_seed;
static cl_kernel split_kernel, decrypt_kernel;
static struct fmt_main *self;
static seed_t *seed;
static unsigned seed_size;
//static salt_t *cur_salt;
static custom_salt  *cur_salt;

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
	host_salt = mem_calloc(1, sizeof(custom_salt));
	memset(host_salt, 0, sizeof(custom_salt));
	seed_size = kpc * sizeof(seed_t);
	seed = mem_calloc(seed_size, 1);
	memset(seed, 0, seed_size);

	mem_in = CLCREATEBUFFER(CL_RO, kpc * sizeof(pass_t),
	                        "Cannot allocate mem in");
	mem_salt = CLCREATEBUFFER(CL_RO, sizeof(custom_salt),
	                          "Cannot allocate mem salt");
	mem_out = CLCREATEBUFFER(CL_WO, kpc * sizeof(crack_t),
	                         "Cannot allocate mem out");
	mem_state = CLCREATEBUFFER(CL_RW, kpc * sizeof(state_t),
	                           "Cannot allocate mem state");
	mem_seed = CLCREATEBUFFER(CL_RW, seed_size,
	                           "Cannot allocate mem seed");

	CLKERNELARG(crypt_kernel, 0, mem_in, "Error while setting mem_in");
	CLKERNELARG(crypt_kernel, 1, mem_state, "Error while setting mem_state");

	CLKERNELARG(split_kernel, 0, mem_state, "Error while setting mem_state");
	CLKERNELARG(split_kernel, 1 ,mem_out, "Error while setting mem_out");

	CLKERNELARG(decrypt_kernel, 0, mem_salt, "Error while setting mem_salt");
	CLKERNELARG(decrypt_kernel, 1 ,mem_out, "Error while setting mem_out");
	CLKERNELARG(decrypt_kernel, 2 ,mem_seed, "Error setting mem_seed");
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
		HANDLE_CLERROR(clReleaseMemObject(mem_seed), "Release mem seed");
		HANDLE_CLERROR(clReleaseMemObject(mem_in), "Release mem in");
		HANDLE_CLERROR(clReleaseMemObject(mem_salt), "Release mem salt");
		HANDLE_CLERROR(clReleaseMemObject(mem_out), "Release mem out");
		HANDLE_CLERROR(clReleaseMemObject(mem_state), "Release mem state");

		MEM_FREE(seed);
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
	printf("reset\n");
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
	cur_salt = (custom_salt*)salt;
	printf("set_salt ct len %d\n", cur_salt->eslen);
	memcpy(host_salt, cur_salt, sizeof(custom_salt));

	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], mem_salt,
		CL_FALSE, 0, sizeof(custom_salt), host_salt, 0, NULL, NULL),
	    "Copy salt to gpu");
}

static void computeBkp(count){
	//printf("computeBkp %d", count);

	for(int i = 0; i < count; i++){
		Keccak_HashInstance hash;
		unsigned char bkp[128];
		Keccak_HashInitialize(&hash, 1088, 512, 256, 0x01);
		//printf("bkp %d seed[i].data[0] 0x%x, len: %d\n", i, seed[i].data[0], seed[i].length);
		Keccak_HashUpdate(&hash, seed[i].data, seed[i].length * 8);
		Keccak_HashUpdate(&hash, (unsigned char*)"\x02", 1 * 8);
		Keccak_HashFinal(&hash, (unsigned char*)bkp);
		//printf("bkp %d bkp[0] %x\n", i, bkp[0]);
		//printf("bkp to match  %x\n", host_salt->bkp[0]);
		if(memcmp(bkp, host_salt->bkp, host_salt->bkplen) == 0){
			printf("DONE\n");
			seed[i].cracked = true;
		}
	}
}

static int crypt_all(int *pcount, struct db_salt *salt)
{
	int i;
	const int count = *pcount;
	int loops = (2000 + HASH_LOOPS - 1) / HASH_LOOPS;
	size_t *lws = local_work_size ? &local_work_size : NULL;
	//printf("crypt_all %d, loops: %d\n", count, loops);
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
	BENCH_CLERROR(clEnqueueReadBuffer(queue[gpu_id], mem_seed,
		CL_TRUE, 0, seed_size, seed, 0,
		NULL, multi_profilingEvent[4]), "Copy result back");

	computeBkp(count);
	//printf("crypt_all done %d\n", count);
	return count;
}


static int cmp_all(void *binary, int count)
{
	int i;
	//printf("cmp_all %d 0x%x\n", count, (unsigned char*)binary);
	if(seed_size <= count){
		printf("cmp_all error, count %d greater than seed_size %d\n", count, seed_size);
		return 0;
	}
	for (i = 0; i < count; i++){
		if (seed[i].cracked)
			return 1;
	}
	return 0;
}

static int cmp_one(void *binary, int index)
{
	printf("cmp_one %d \n", index);
	if(seed_size <= index){
		printf("cmp_one error, index %d greater than seed_size %d\n", index, seed_size);
		return 0;
	}
	return seed[index].cracked;
}

static int cmp_exact(char *source, int index)
{
	printf("cmp_exact %d \n", index);
	return 1;
}

static void set_key(char *key, int index)
{
	int saved_len = MIN(strlen(key), PLAINTEXT_LENGTH);
	//printf("set_key %d, %s\n", index, key);

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
	{"$ethereum-presale$w*9ace195a18b83b09da933f60f64fb4651e8413a5fdc12249724088d43dcd6d3c943f83d78d475e61177ca38965cb7794a2e1225250ae29134e60492871c79bf2*b9cfc3e87d22f37be858f97944844efbd32e5da5*ec97af5d7e240a559ee74f7a9e7312f2", "openwall"},
	//{"$ethereum-presale$w*4ab35fcb5c3101af70d5b3bf22829af3dbd48813273b17566ee364285c7bcfb2d52611a58e54d3e6be27e458073304e71e356afc4c97da0143910308f30563fe*c4dfdaa5288b477d0ff25a260ef32a24282cc4e2*97b3d2c2c8106507b87744f0b12f73c2", "password12345"},
	{NULL}
};

int ethereum_presale_valid(char *ciphertext, struct fmt_main *self)
{
	char *ctcopy, *keeptr, *p;
	int extra;
	printf("ethereum_presale_valid %s\n", ciphertext);
	if (strncmp(ciphertext, FORMAT_TAG, TAG_LENGTH) != 0)
		return 0;

	ctcopy = strdup(ciphertext);
	keeptr = ctcopy;

	ctcopy += TAG_LENGTH;
	if ((p = strtokm(ctcopy, "*")) == NULL) // type
		goto err;
	if (*p != 'p' && *p != 's' && *p != 'w')
		goto err;
	if (*p == 'p') {
		if ((p = strtokm(NULL, "*")) == NULL)   // iterations
			goto err;
		if (!isdec(p))
			goto err;
		if ((p = strtokm(NULL, "*")) == NULL)   // salt
			goto err;
		if (hexlenl(p, &extra) != 64 || extra)
			goto err;
		if ((p = strtokm(NULL, "*")) == NULL)   // ciphertext
			goto err;
		if (hexlenl(p, &extra) != 64 || extra)
			goto err;
		if ((p = strtokm(NULL, "*")) == NULL)   // mac
			goto err;
		if (hexlenl(p, &extra) != 64 || extra)
			goto err;
	} else if (*p == 's') {
		if ((p = strtokm(NULL, "*")) == NULL)   // N
			goto err;
		if (!isdec(p))
			goto err;
		if ((p = strtokm(NULL, "*")) == NULL)   // r
			goto err;
		if (!isdec(p))
			goto err;
		if ((p = strtokm(NULL, "*")) == NULL)   // p
			goto err;
		if (!isdec(p))
			goto err;
		if ((p = strtokm(NULL, "*")) == NULL)   // salt
			goto err;
		if (hexlenl(p, &extra) != 64 || extra)
			goto err;
		if ((p = strtokm(NULL, "*")) == NULL)   // ciphertext
			goto err;
		if (hexlenl(p, &extra) > 128 || extra)
			goto err;
		if ((p = strtokm(NULL, "*")) == NULL)   // mac
			goto err;
		if (hexlenl(p, &extra) != 64 || extra)
			goto err;
	} else if (*p == 'w') {
		if ((p = strtokm(NULL, "*")) == NULL)   // encseed
			goto err;
		if (hexlenl(p, &extra) >= 2048 * 2 || extra)
			goto err;
		if ((p = strtokm(NULL, "*")) == NULL)   // ethaddr
			goto err;
		if (hexlenl(p, &extra) > 128 || extra)
			goto err;
		if ((p = strtokm(NULL, "*")) == NULL)   // bkp
			goto err;
	}

	MEM_FREE(keeptr);
	return 1;

err:
	printf("ethereum_presale_valid error %s\n", ciphertext);
	MEM_FREE(keeptr);
	return 0;
}

void *ethereum_presale_get_salt(char *ciphertext)
{
	char *ctcopy = strdup(ciphertext);
	char *keeptr = ctcopy;
	int i;
	char *p;
	static custom_salt *cur_salt;
printf("ethereum_common_get_salt %s\n", ciphertext);
	cur_salt = mem_calloc_tiny(sizeof(custom_salt), MEM_ALIGN_WORD);

	ctcopy += TAG_LENGTH;
	p = strtokm(ctcopy, "*");
	if (*p == 'p')
		cur_salt->type = 0; // PBKDF2
	else if (*p == 's')
		cur_salt->type = 1; // scrypt
	else if (*p == 'w')
		cur_salt->type = 2; // PBKDF2, presale wallet

	p = strtokm(NULL, "*");
	if (cur_salt->type == 0) {
		cur_salt->iterations = atoi(p);
		p = strtokm(NULL, "*");
	} else if (cur_salt->type == 1) {
		cur_salt->N = atoi(p);
		p = strtokm(NULL, "*");
		cur_salt->r = atoi(p);
		p = strtokm(NULL, "*");
		cur_salt->p = atoi(p);
		p = strtokm(NULL, "*");
	} else if (cur_salt->type == 2) {

		cur_salt->eslen = strlen(p) / 2;
    //printf("ethereum_common_get_salt cur_salt->eslen %d\n", cur_salt->eslen);
		for (i = 0; i < cur_salt->eslen; i++)
			cur_salt->encseed[i] = atoi16[ARCH_INDEX(p[i * 2])] * 16
				+ atoi16[ARCH_INDEX(p[i * 2 + 1])];

		p = strtokm(NULL, "*");//Ethaddr
		p = strtokm(NULL, "*");//Bkp
		cur_salt->bkplen = strlen(p) / 2;
		//printf("ethereum_common_get_salt bkplen %d\n", cur_salt->bkplen);
		for (i = 0; i < cur_salt->bkplen; i++)
			cur_salt->bkp[i] = atoi16[ARCH_INDEX(p[i * 2])] * 16
				+ atoi16[ARCH_INDEX(p[i * 2 + 1])];
		//printf("ethereum_common_get_salt cur_salt->bkp[i] %x\n", cur_salt->bkp[i]);
	}
	if (cur_salt->type == 0 || cur_salt->type == 1) {
		cur_salt->saltlen = strlen(p) / 2;
		for (i = 0; i < cur_salt->saltlen; i++)
			cur_salt->salt[i] = atoi16[ARCH_INDEX(p[i * 2])] * 16
				+ atoi16[ARCH_INDEX(p[i * 2 + 1])];
		p = strtokm(NULL, "*");
		cur_salt->ctlen = strlen(p) / 2;
		for (i = 0; i < cur_salt->ctlen; i++)
			cur_salt->ct[i] = atoi16[ARCH_INDEX(p[i * 2])] * 16
				+ atoi16[ARCH_INDEX(p[i * 2 + 1])];
	}

	MEM_FREE(keeptr);

	return (void *)cur_salt;
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
		FMT_CASE | FMT_8_BIT | FMT_HUGE_INPUT,
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
		ethereum_presale_valid,
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
