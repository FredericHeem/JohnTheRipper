/*
 * This software is Copyright (c) 2017 magnum
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 */

#include "pbkdf2_hmac_sha256_kernel.cl"
#define OCL_AES_CBC_DECRYPT 1
#define AES_KEY_TYPE __global
#define AES_SRC_TYPE __constant
#include "opencl_aes.h"

#define AES_LEN             1024

typedef struct {
	uint8_t  aes_ct[AES_LEN]; /* ciphertext */
	uint32_t aes_len;         /* actual data length (up to AES_LEN) */
} eth_presale_salt_t;

__kernel void ethereum_presale_decrypt(__constant eth_presale_salt_t *salt,
                           __global crack_t *out,
                           __global uint32_t *cracked)
{
	uint32_t gid = get_global_id(0);
	AES_KEY akey;
	uchar iv[16];
	int i;
	uchar seed[AES_LEN];
	memset(&seed, 0, sizeof(seed));
	for (i = 0; i < 16; i++)
		iv[i] = salt->aes_ct[i];

	printf("ethereum_presale_decrypt aes_ct[0] = %x\n", salt->aes_ct[0]);
	printf("aes_ct[salt->aes_len-1] = %x\n", salt->aes_ct[salt->aes_len-1]);
	printf("aes_ct[16] = %x\n", salt->aes_ct[16]);
	printf("aes_len = %d\n", salt->aes_len);
	printf("hash[0] = %x\n", ((__global uchar*)out[gid].hash)[0]);
	printf("hash[15] = %x\n", ((__global uchar*)out[gid].hash)[15]);

	AES_set_decrypt_key((__global uchar*)out[gid].hash, 128, &akey);

	printf("akey[0] = %x\n", akey.rd_key[0]);
	printf("akey[1] = %x\n", akey.rd_key[1]);
	printf("akey.rounds = %d\n", akey.rounds);
	AES_cbc_decrypt(salt->aes_ct + 16, seed, salt->aes_len - 16, &akey, iv);

	for(int i = 0; i < salt->aes_len; i++){
		printf("%x ", (uchar*)seed[i]);
	}
	printf("\n");
	printf("seed[0] = %x\n", (uchar*)seed[0]);
	printf("seed[1] = %x\n", seed[1]);
	printf("seed[2] = %x\n", seed[2]);
	printf("seed[last - 1] = %x\n", seed[salt->aes_len - 2]);
	printf("seed[last] = %x\n", seed[salt->aes_len - 1]);
	printf("seed[last + 1] = %x\n", seed[salt->aes_len]);

	
	cracked[gid] = false;
}
