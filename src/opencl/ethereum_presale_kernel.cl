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

#define BLOBLEN                 24
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
	uchar plaintext[AES_LEN];
	for (i = 0; i < 16; i++)
		iv[i] = salt->aes_ct[i];
	printf("ethereum_presale_decrypt aes_ct[0] = %x\n", salt->aes_ct[0]);
	printf("aes_ct[0] = %x\n", salt->aes_ct[1]);
	printf("aes_len = %d\n\n", salt->aes_len);
	printf("hash = %#v4hhx\n", out[gid].hash);

	AES_set_decrypt_key((__global uchar*)out[gid].hash, 16, &akey);
	AES_cbc_decrypt(salt->aes_ct, plaintext, salt->aes_len, &akey, iv);
printf("plaintext = %#v4hhx\n\n", plaintext);
	cracked[gid] = false;
}
