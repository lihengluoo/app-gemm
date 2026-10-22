/*****************************************************************************
Copyright (c) 2011-2014, The OpenBLAS Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.

   3. Neither the name of the OpenBLAS project nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************/

/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the            */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above       */
/*      copyright notice, this list of conditions and the            */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE  PROVIDED  BY  THE  UNIVERSITY  OF  TEXAS  AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS  OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT NOT LIMITED TO,  THE IMPLIED  WARRANTIES  OF     */
/*    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE       */
/*    DISCLAIMED.  IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT      */
/*    AUSTIN OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,     */
/*    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES        */
/*    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE       */
/*    GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS   */
/*    INTERRUPTION)  HOWEVER CAUSED  AND ON ANY THEORY OF LIABILITY,  */
/*    WHETHER IN  CONTRACT, STRICT  LIABILITY, OR TORT  (INCLUDING    */
/*    NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY  OUT OF THE USE OF */
/*    THIS  SOFTWARE,  EVEN  IF ADVISED  OF THE POSSIBILITY  OF SUCH   */
/*    DAMAGE.                                                        */
/*                                                                   */
/* The views and conclusions contained in the software and            */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

#include "common.h"

#ifdef SMP_SERVER

/* Minimal memory layer for the create/join executor.  The whole file
   deliberately touches only this external surface:

     pthread_mutex_lock / pthread_mutex_unlock   pool critical section
     aligned_alloc / free                        block storage
     abort                                        allocation failure

   There is no environment reading, no CPU probing, no thread-local
   buffer table, no atfork registration, and no mmap backend: every
   worker takes its packing scratch from a mutex-protected free pool of
   fixed-size blocks and returns it there.  Fork safety is out of scope
   for this single-process extraction. */

/* ---- thread count: programmatic only, no detection ----------------- */

/* blas_num_threads is the remembered floor for goto_set_num_threads;
   blas_cpu_number is the count the interface consults. */
int blas_num_threads = DEFAULT_CPU_NUMBER;
_Atomic int blas_cpu_number = 0;

int blas_get_cpu_number(void) {

  int current = atomic_load(&blas_cpu_number);

  if (current == 0) {
    current = DEFAULT_CPU_NUMBER;
    atomic_store(&blas_cpu_number, current);
  }

  return current;
}

OPENBLAS_EXPORT
int openblas_get_num_threads(void) {

  return atomic_load(&blas_cpu_number);
}

/* Blocking stays at the compiled defaults; there is no override. */
int openblas_block_factor(void) {

  return 0;
}

/* ---- buffer pool: mutex-guarded free list of fixed blocks ----------- */

static void *buffer_pool[NUM_BUFFERS];
static pthread_mutex_t buffer_pool_mutex = PTHREAD_MUTEX_INITIALIZER;

/* One block covers the sa/sb packing layout the queue drivers assume:
   the same BUFFER_SIZE the previous allocator handed out, rounded up
   to the GEMM_ALIGN boundary.  GEMM_ALIGN is a rounding mask in the
   OpenBLAS conventions (0xFFFF), so the allocation alignment is the
   boundary it rounds to. */
static void *alloc_buffer_block(void) {

  BLASULONG mask = (BLASULONG)GEMM_ALIGN;
  size_t align = (size_t)(mask + 1);
  size_t bytes = (size_t)((BUFFER_SIZE + mask) & ~mask);
  void *block = aligned_alloc(align, bytes);

  if (block == NULL) abort();
  if (((BLASULONG)block & mask) != 0) abort();

  return block;
}

void *blas_memory_alloc(int procpos) {

  void *buffer = NULL;
  int position;

  (void)procpos;

  pthread_mutex_lock(&buffer_pool_mutex);

  for (position = 0; position < NUM_BUFFERS; position++) {
    if (buffer_pool[position] != NULL) {
      buffer = buffer_pool[position];
      buffer_pool[position] = NULL;
      break;
    }
  }

  pthread_mutex_unlock(&buffer_pool_mutex);

  if (buffer == NULL) buffer = alloc_buffer_block();

  return buffer;
}

void blas_memory_free(void *buffer) {

  int position;

  if (buffer == NULL) return;

  pthread_mutex_lock(&buffer_pool_mutex);

  for (position = 0; position < NUM_BUFFERS; position++) {
    if (buffer_pool[position] == NULL) {
      buffer_pool[position] = buffer;
      buffer = NULL;
      break;
    }
  }

  pthread_mutex_unlock(&buffer_pool_mutex);

  /* A pool with every slot live just returns the block to the heap. */
  if (buffer != NULL) free(buffer);
}

#endif
