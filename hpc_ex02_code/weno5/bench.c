#include <stdio.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifndef WENOEPS
#define WENOEPS 1.e-6
#endif

#include "weno.h"

float * myalloc(const int NENTRIES, const int verbose )
{
	const int initialize = 1;
	enum { alignment_bytes = 32 } ;
	float * tmp = NULL;

	const int result = posix_memalign((void **)&tmp, alignment_bytes, sizeof(float) * NENTRIES);
	assert(result == 0);

	if (initialize)
	{
		for(int i=0; i<NENTRIES; ++i)
			tmp[i] = drand48();

		if (verbose)
		{
			for(int i=0; i<NENTRIES; ++i)
				printf("tmp[%d] = %f\n", i, tmp[i]);
			printf("==============\n");
		}
	}
	return tmp;
}

double get_wtime()
{
	struct timeval t;
	gettimeofday(&t,  NULL);
	return t.tv_sec + t.tv_usec*1e-6;
}

void check_error(const double tol, float ref[], float val[], const int N)
{
	static const int verbose = 0;

	for(int i=0; i<N; ++i)
	{
		assert(!isnan(ref[i]));
		assert(!isnan(val[i]));

		const double err = ref[i] - val[i];
		const double relerr = err/fmaxf(FLT_EPSILON, fmaxf(fabs(val[i]), fabs(ref[i])));

		if (verbose) printf("+%1.1e,", relerr);

		if (fabs(relerr) >= tol && fabs(err) >= tol)
			printf("\n%d: %e %e -> %e %e\n", i, ref[i], val[i], err, relerr);

		assert(fabs(relerr) < tol || fabs(err) < tol);
	}

	if (verbose) printf("\t");
}


void benchmark(int argc, char *argv[], const int NENTRIES_, const int NTIMES, const int verbose, char *benchmark_name)
{
	const int NENTRIES = 8 * (NENTRIES_ / 8);

	/*
	 * FLOP count per element for weno_minus_core:
	 *   3 smoothness indicators (is0,is1,is2): 13 each = 39
	 *   3 epsilon additions:                           =  3
	 *   3 alpha computations (mul,div,mul):             =  9
	 *   alphasum (2 adds):                             =  2
	 *   inv_alpha (1 div):                             =  1
	 *   3 omega weights (2 muls + 2 subs):             =  4
	 *   3 reconstruction polynomials: 5 each           = 15
	 *   final weighted sum (3 muls + 2 adds):          =  5
	 *   TOTAL:                                         = 78
	 */
	const double FLOPS_PER_ELEMENT = 78.0;

	/* Memory traffic per element: read 5 floats + write 1 float = 6 * 4 bytes = 24 bytes */
	const double BYTES_PER_ELEMENT = 6.0 * sizeof(float);

	printf("nentries set to %d (%s)\n", NENTRIES, benchmark_name);

	float * const a = myalloc(NENTRIES, verbose);
	float * const b = myalloc(NENTRIES, verbose);
	float * const c = myalloc(NENTRIES, verbose);
	float * const d = myalloc(NENTRIES, verbose);
	float * const e = myalloc(NENTRIES, verbose);
	float * const f = myalloc(NENTRIES, verbose);
	float * const gold = myalloc(NENTRIES, verbose);
	float * const result = myalloc(NENTRIES, verbose);

	/* Compute reference solution */
	weno_minus_reference(a, b, c, d, e, gold, NENTRIES);

	const double tol = 1e-5;
	const double tol_intrinsics = 1e-4;

	/* ---- Benchmark: Scalar reference ---- */
	{
		/* Warmup */
		weno_minus_reference(a, b, c, d, e, result, NENTRIES);
		check_error(tol, gold, result, NENTRIES);

		double t0 = get_wtime();
		for (int t = 0; t < NTIMES; ++t)
			weno_minus_reference(a, b, c, d, e, result, NENTRIES);
		double t1 = get_wtime();

		double elapsed = t1 - t0;
		double total_flops = FLOPS_PER_ELEMENT * NENTRIES * NTIMES;
		double total_bytes = BYTES_PER_ELEMENT * NENTRIES * NTIMES;
		printf("  %-20s  time: %10.6f s  |  %8.3f GFLOP/s  |  %8.3f GB/s\n",
		       "Scalar", elapsed, total_flops / elapsed * 1e-9, total_bytes / elapsed * 1e-9);
	}

	/* ---- Benchmark: OpenMP SIMD ---- */
	{
		weno_minus_vectorized(a, b, c, d, e, result, NENTRIES);
		check_error(tol, gold, result, NENTRIES);

		double t0 = get_wtime();
		for (int t = 0; t < NTIMES; ++t)
			weno_minus_vectorized(a, b, c, d, e, result, NENTRIES);
		double t1 = get_wtime();

		double elapsed = t1 - t0;
		double total_flops = FLOPS_PER_ELEMENT * NENTRIES * NTIMES;
		double total_bytes = BYTES_PER_ELEMENT * NENTRIES * NTIMES;
		printf("  %-20s  time: %10.6f s  |  %8.3f GFLOP/s  |  %8.3f GB/s\n",
		       "OMP SIMD", elapsed, total_flops / elapsed * 1e-9, total_bytes / elapsed * 1e-9);
	}

	/* ---- Benchmark: SSE intrinsics ---- */
	{
		weno_minus_sse(a, b, c, d, e, result, NENTRIES);
		check_error(tol_intrinsics, gold, result, NENTRIES);

		double t0 = get_wtime();
		for (int t = 0; t < NTIMES; ++t)
			weno_minus_sse(a, b, c, d, e, result, NENTRIES);
		double t1 = get_wtime();

		double elapsed = t1 - t0;
		double total_flops = FLOPS_PER_ELEMENT * NENTRIES * NTIMES;
		double total_bytes = BYTES_PER_ELEMENT * NENTRIES * NTIMES;
		printf("  %-20s  time: %10.6f s  |  %8.3f GFLOP/s  |  %8.3f GB/s\n",
		       "SSE intrinsics", elapsed, total_flops / elapsed * 1e-9, total_bytes / elapsed * 1e-9);
	}

	/* ---- Benchmark: AVX intrinsics ---- */
	{
		weno_minus_avx(a, b, c, d, e, result, NENTRIES);
		check_error(tol_intrinsics, gold, result, NENTRIES);

		double t0 = get_wtime();
		for (int t = 0; t < NTIMES; ++t)
			weno_minus_avx(a, b, c, d, e, result, NENTRIES);
		double t1 = get_wtime();

		double elapsed = t1 - t0;
		double total_flops = FLOPS_PER_ELEMENT * NENTRIES * NTIMES;
		double total_bytes = BYTES_PER_ELEMENT * NENTRIES * NTIMES;
		printf("  %-20s  time: %10.6f s  |  %8.3f GFLOP/s  |  %8.3f GB/s\n",
		       "AVX intrinsics", elapsed, total_flops / elapsed * 1e-9, total_bytes / elapsed * 1e-9);
	}

	printf("\n");

	free(a);
	free(b);
	free(c);
	free(d);
	free(e);
	free(gold);
	free(result);
}

int main (int argc, char *  argv[])
{
	printf("=================================================================\n");
	printf("  WENO5 Benchmark — 78 FLOPs per element, 24 bytes per element\n");
	printf("=================================================================\n\n");

	const int debug = 0;

	if (debug)
	{
		benchmark(argc, argv, 4, 1, 1, "debug");
		return 0;
	}

	/* PEAK-LIKE: data fits in L1/L2 cache → measures compute throughput */
	{
		const int nentries = 16 * (int)(pow(32 + 6, 2) * 4);  /* ~92k elements ≈ 2.1 MB total */
		const int ntimes = (int)floor(2. / (1e-7 * nentries)); /* enough iterations for ~2s */

		for(int i = 0; i < 4; ++i)
		{
			printf("*************** PEAK-LIKE BENCHMARK (RUN %d) **************************\n", i);
			benchmark(argc, argv, nentries, ntimes, 0, "cache");
		}
	}

	/* STREAM-LIKE: data exceeds last-level cache → measures memory bandwidth */
	{
		const double desired_mb = 128 * 4;
		const int nentries = (int)floor(desired_mb * 1024. * 1024. / 7 / sizeof(float));
		const int ntimes = 3;  /* fewer iterations since each is expensive */

		for(int i = 0; i < 4; ++i)
		{
			printf("*************** STREAM-LIKE BENCHMARK (RUN %d) **************************\n", i);
			benchmark(argc, argv, nentries, ntimes, 0, "stream");
		}
	}

    return 0;
}
