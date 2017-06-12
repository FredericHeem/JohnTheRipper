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

//ethereum_common.h
typedef struct {
	uint32_t type;
	int N, r, p;
	int iterations;
	uint8_t salt[64];
	int saltlen;
	uint8_t ct[256];
	int ctlen;
	// following fields are only required for handling presale wallets
	uint8_t encseed[2048];
	int eslen;
	uint8_t ethadd[128];
	int ealen;
  uint8_t bkp[128];
	int bkplen;
} custom_salt;

typedef struct {
	uint8_t  data[AES_LEN]; /* the seed */
	uint32_t length;
	bool cracked;
} seed_t;

__kernel void ethereum_presale_decrypt(__constant custom_salt *salt,
                           __global crack_t *out,
                           __global seed_t *seed_out)
{
	uint32_t gid = get_global_id(0);
	AES_KEY akey;
	uchar iv[16];
	int i;
	uchar seed[AES_LEN];
  #define BKP_SIZE 128
  uchar bkp[BKP_SIZE];
  int padbyte;
  int seed_length;

	memset(&seed, 0, sizeof(seed));

  if(salt->eslen < 32){
    printf("invalid cypher length %d\n", salt->eslen);
    return;
  }
	for (i = 0; i < 16; i++){
		iv[i] = salt->encseed[i];
	}
  /*
	printf("ethereum_presale_decrypt aes_ct[0] = %x\n", salt->encseed[0]);
	printf("aes_ct[salt->eslen-1] = %x\n", salt->encseed[salt->eslen-1]);
	printf("aes_ct[16] = %x\n", salt->encseed[16]);
	printf("eslen = %d\n", salt->eslen);
	printf("hash[0] = %x\n", ((__global uchar*)out[gid].hash)[0]);
	printf("hash[15] = %x\n", ((__global uchar*)out[gid].hash)[15]);
*/
	AES_set_decrypt_key((__global uchar*)out[gid].hash, 128, &akey);
/*
	printf("akey[0] = %x\n", akey.rd_key[0]);
	printf("akey[1] = %x\n", akey.rd_key[1]);
	printf("akey.rounds = %d\n", akey.rounds);
  */
	AES_cbc_decrypt(salt->encseed + 16, seed, salt->eslen - 16, &akey, iv);
/*
	for(int i = 0; i < salt->eslen; i++){
		//printf("%x ", (uchar*)seed[i]);
	}
	printf("\n");
	printf("seed[0] = %x\n", (uchar*)seed[0]);
	printf("seed[1] = %x\n", seed[1]);
	printf("seed[2] = %x\n", seed[2]);
	printf("seed[last - 1] = %x\n", seed[salt->eslen - 2]);
	printf("seed[last] = %x\n", seed[salt->eslen - 1]);
	printf("seed[last + 1] = %x\n", seed[salt->eslen]);
*/
  padbyte = seed[salt->eslen - 16 - 1];
  //printf("padbyte = %x\n", padbyte);
  seed_length = salt->eslen - 16 - padbyte;
  //printf("seed_length = %d\n", seed_length);
  if(seed_length < 0 ) seed_length = 0;
  seed_out[gid].length = seed_length;
  for(int i = 0; i < seed_length; i++){
		seed_out[gid].data[i] = seed[i];
	}
}
