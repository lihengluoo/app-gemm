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
   3. Neither the name of the OpenBLAS project nor the names of 
      its contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

#include "common.h"

#ifdef SMP_SERVER

/* Trivial fork-join executor for parallel BLAS calls.  Every exec_blas */
/* call creates one pthread per queued job, runs the first job on the   */
/* calling thread, and joins the workers before returning.  There is no */
/* persistent worker pool: the caller fully writes each queue node     */
/* before pthread_create, each node is read only by its own worker, and */
/* pthread_join publishes the results to the caller.                   */

/* Kept for the initialization and shutdown call sites in memory.c.     */
/* Without persistent threads there is nothing to start or stop; the   */
/* flag only records that the thread count was initialized.            */
int blas_server_avail = 0;

int blas_omp_threads_local = 1;

int blas_thread_init(void) {

  if (blas_cpu_number == 0) blas_get_cpu_number();
  blas_server_avail = 1;

  return 0;
}

int BLASFUNC(blas_thread_shutdown)(void) {

  return 0;
}

/* The four serial level-3 drivers this extraction builds (declared in
   common_level3.h), selected by the transpose mode bits the interface
   stores in every queue node.  Real types only produce N and T modes. */
static void call_queue_driver(blas_queue_t *queue, void *sa, void *sb) {

  switch (queue -> mode & (BLAS_TRANSA | BLAS_TRANSB)) {

    case BLAS_TRANSA_N | BLAS_TRANSB_N :
      i8gemm_nn(queue -> args, queue -> range_m, queue -> range_n, sa, sb, queue -> position);
      break;

    case BLAS_TRANSA_T | BLAS_TRANSB_N :
      i8gemm_tn(queue -> args, queue -> range_m, queue -> range_n, sa, sb, queue -> position);
      break;

    case BLAS_TRANSA_N | BLAS_TRANSB_T :
      i8gemm_nt(queue -> args, queue -> range_m, queue -> range_n, sa, sb, queue -> position);
      break;

    default :
      i8gemm_tt(queue -> args, queue -> range_m, queue -> range_n, sa, sb, queue -> position);
      break;
  }
}

/* Thread body for one queued job.  Packing scratch comes from a freshly
   allocated buffer, laid out exactly like the former pool workers'
   scratch: sa at GEMM_OFFSET_A, sb after the packed A area of
   DGEMM_P * DGEMM_Q doubles (this extraction tags the queue mode
   BLAS_DOUBLE so the scratch is sized from that shared blocking). */
static void *blas_job_thread(void *arg) {

  blas_queue_t *queue = arg;
  void *buffer = blas_memory_alloc(2);
  void *sa = queue -> sa;
  void *sb = queue -> sb;

  if (sa == NULL) sa = (void *)((BLASLONG)buffer + GEMM_OFFSET_A);
  if (sb == NULL) sb = (void *)(((BLASLONG)sa + ((DGEMM_P * DGEMM_Q * sizeof(double)
                                           + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);

  queue -> worker_sb = sb;

  call_queue_driver(queue, sa, sb);

  blas_memory_free(buffer);

  return NULL;
}

/* Execute every queued job and return once all of them finished.  The
   first node runs on the calling thread in parallel with the workers.
   pthread_create is assumed to succeed, matching the rest of this
   extraction, which treats resource exhaustion as out of scope. */
int exec_blas(BLASLONG num, blas_queue_t *queue) {

  pthread_t threads[MAX_CPU_NUMBER];
  blas_queue_t *job;
  BLASLONG created = 0, i;

  if ((num <= 0) || (queue == NULL)) return 0;

  for (job = queue -> next; job != NULL && created < num - 1; job = job -> next) {

    job -> position = created + 1;

    pthread_create(&threads[created], NULL, &blas_job_thread, job);

    created ++;
  }

  queue -> position  = 0;
  queue -> worker_sb = queue -> sb;
  call_queue_driver(queue, queue -> sa, queue -> sb);

  for (i = 0; i < created; i ++) pthread_join(threads[i], NULL);

  return 0;
}

/* Without persistent workers the thread count is only a stored value
   that num_cpu_avail() consults when a call decides how many jobs to
   queue. */
void goto_set_num_threads(int num_threads) {

  if (num_threads < 1) num_threads = blas_num_threads;
  if (num_threads > MAX_CPU_NUMBER) num_threads = MAX_CPU_NUMBER;

  blas_num_threads = num_threads;
  blas_cpu_number  = num_threads;
}

OPENBLAS_EXPORT
void openblas_set_num_threads(int num_threads) {

	goto_set_num_threads(num_threads);
}

#endif
