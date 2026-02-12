#pragma once

#include <xmmintrin.h>  /* SSE intrinsics */
#include <immintrin.h>  /* AVX intrinsics */

float weno_minus_core(const float a, const float b, const float c, const float d, const float e)
{
		const float is0 = a*(a*(float)(4./3.)  - b*(float)(19./3.)  + c*(float)(11./3.)) + b*(b*(float)(25./3.)  - c*(float)(31./3.)) + c*c*(float)(10./3.);
		const float is1 = b*(b*(float)(4./3.)  - c*(float)(13./3.)  + d*(float)(5./3.))  + c*(c*(float)(13./3.)  - d*(float)(13./3.)) + d*d*(float)(4./3.);
		const float is2 = c*(c*(float)(10./3.) - d*(float)(31./3.)  + e*(float)(11./3.)) + d*(d*(float)(25./3.)  - e*(float)(19./3.)) + e*e*(float)(4./3.);

		const float is0plus = is0 + (float)WENOEPS;
		const float is1plus = is1 + (float)WENOEPS;
		const float is2plus = is2 + (float)WENOEPS;

		const float alpha0 = (float)(0.1)*((float)1/(is0plus*is0plus));
		const float alpha1 = (float)(0.6)*((float)1/(is1plus*is1plus));
		const float alpha2 = (float)(0.3)*((float)1/(is2plus*is2plus));
		const float alphasum = alpha0+alpha1+alpha2;
		const float inv_alpha = ((float)1)/alphasum;

		const float omega0 = alpha0 * inv_alpha;
		const float omega1 = alpha1 * inv_alpha;
		const float omega2 = 1-omega0-omega1;

		return omega0*((float)(1.0/3.)*a - (float)(7./6.)*b + (float)(11./6.)*c) +
					 omega1*(-(float)(1./6.)*b + (float)(5./6.)*c + (float)(1./3.)*d) +
					 omega2*((float)(1./3.)*c  + (float)(5./6.)*d - (float)(1./6.)*e);
}

void weno_minus_reference(const float * const a, const float * const b, const float * const c,
			  const float * const d, const float * const e, float * const out,
			  const int NENTRIES)
{
		for (int i=0; i<NENTRIES; ++i)
			out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
}

void weno_minus_vectorized(const float * const a, const float * const b, const float * const c,
			   const float * const d, const float * const e, float * const out,
			   const int NENTRIES)
{
		#pragma omp simd
		for (int i=0; i<NENTRIES; ++i)
			out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
}

void weno_minus_sse(const float * const a, const float * const b, const float * const c,
		    const float * const d, const float * const e, float * const out,
		    const int NENTRIES)
{
	/* Broadcast all constants into SSE registers */
	const __m128 eps   = _mm_set1_ps((float)WENOEPS);
	const __m128 one   = _mm_set1_ps(1.0f);

	/* Smoothness indicator coefficients */
	const __m128 c4_3  = _mm_set1_ps(4.f/3.f);
	const __m128 c19_3 = _mm_set1_ps(19.f/3.f);
	const __m128 c11_3 = _mm_set1_ps(11.f/3.f);
	const __m128 c25_3 = _mm_set1_ps(25.f/3.f);
	const __m128 c31_3 = _mm_set1_ps(31.f/3.f);
	const __m128 c10_3 = _mm_set1_ps(10.f/3.f);
	const __m128 c13_3 = _mm_set1_ps(13.f/3.f);
	const __m128 c5_3  = _mm_set1_ps(5.f/3.f);

	/* Ideal weights */
	const __m128 w0    = _mm_set1_ps(0.1f);
	const __m128 w1    = _mm_set1_ps(0.6f);

	/* Reconstruction coefficients */
	const __m128 c1_3  = _mm_set1_ps(1.f/3.f);
	const __m128 c7_6  = _mm_set1_ps(7.f/6.f);
	const __m128 c11_6 = _mm_set1_ps(11.f/6.f);
	const __m128 c1_6  = _mm_set1_ps(1.f/6.f);
	const __m128 c5_6  = _mm_set1_ps(5.f/6.f);

	/* Process 4 floats per iteration (128-bit SSE register) */
	for (int i = 0; i < NENTRIES; i += 4)
	{
		/* Load 4 elements from each input array (aligned) */
		const __m128 va = _mm_load_ps(a + i);
		const __m128 vb = _mm_load_ps(b + i);
		const __m128 vc = _mm_load_ps(c + i);
		const __m128 vd = _mm_load_ps(d + i);
		const __m128 ve = _mm_load_ps(e + i);

		/* is0 = a*(a*4/3 - b*19/3 + c*11/3) + b*(b*25/3 - c*31/3) + c*c*10/3 */
		const __m128 is0 = _mm_add_ps(
			_mm_add_ps(
				_mm_mul_ps(va, _mm_add_ps(_mm_sub_ps(
					_mm_mul_ps(va, c4_3),
					_mm_mul_ps(vb, c19_3)),
					_mm_mul_ps(vc, c11_3))),
				_mm_mul_ps(vb, _mm_sub_ps(
					_mm_mul_ps(vb, c25_3),
					_mm_mul_ps(vc, c31_3)))),
			_mm_mul_ps(_mm_mul_ps(vc, vc), c10_3));

		/* is1 = b*(b*4/3 - c*13/3 + d*5/3) + c*(c*13/3 - d*13/3) + d*d*4/3 */
		const __m128 is1 = _mm_add_ps(
			_mm_add_ps(
				_mm_mul_ps(vb, _mm_add_ps(_mm_sub_ps(
					_mm_mul_ps(vb, c4_3),
					_mm_mul_ps(vc, c13_3)),
					_mm_mul_ps(vd, c5_3))),
				_mm_mul_ps(vc, _mm_sub_ps(
					_mm_mul_ps(vc, c13_3),
					_mm_mul_ps(vd, c13_3)))),
			_mm_mul_ps(_mm_mul_ps(vd, vd), c4_3));

		/* is2 = c*(c*10/3 - d*31/3 + e*11/3) + d*(d*25/3 - e*19/3) + e*e*4/3 */
		const __m128 is2 = _mm_add_ps(
			_mm_add_ps(
				_mm_mul_ps(vc, _mm_add_ps(_mm_sub_ps(
					_mm_mul_ps(vc, c10_3),
					_mm_mul_ps(vd, c31_3)),
					_mm_mul_ps(ve, c11_3))),
				_mm_mul_ps(vd, _mm_sub_ps(
					_mm_mul_ps(vd, c25_3),
					_mm_mul_ps(ve, c19_3)))),
			_mm_mul_ps(_mm_mul_ps(ve, ve), c4_3));

		/* Add epsilon to smoothness indicators */
		const __m128 is0p = _mm_add_ps(is0, eps);
		const __m128 is1p = _mm_add_ps(is1, eps);
		const __m128 is2p = _mm_add_ps(is2, eps);

		/* alpha_k = d_k / (is_k + eps)^2 */
		const __m128 alpha0 = _mm_mul_ps(w0, _mm_div_ps(one, _mm_mul_ps(is0p, is0p)));
		const __m128 alpha1 = _mm_mul_ps(w1, _mm_div_ps(one, _mm_mul_ps(is1p, is1p)));
		const __m128 alpha2 = _mm_mul_ps(_mm_set1_ps(0.3f), _mm_div_ps(one, _mm_mul_ps(is2p, is2p)));

		const __m128 alphasum = _mm_add_ps(_mm_add_ps(alpha0, alpha1), alpha2);
		const __m128 inv_alpha = _mm_div_ps(one, alphasum);

		/* Nonlinear weights */
		const __m128 omega0 = _mm_mul_ps(alpha0, inv_alpha);
		const __m128 omega1 = _mm_mul_ps(alpha1, inv_alpha);
		const __m128 omega2 = _mm_sub_ps(_mm_sub_ps(one, omega0), omega1);

		/* Reconstruction polynomials */
		const __m128 r0 = _mm_add_ps(_mm_sub_ps(
			_mm_mul_ps(c1_3, va),
			_mm_mul_ps(c7_6, vb)),
			_mm_mul_ps(c11_6, vc));
		const __m128 r1 = _mm_add_ps(_mm_sub_ps(
			_mm_mul_ps(c5_6, vc),
			_mm_mul_ps(c1_6, vb)),
			_mm_mul_ps(c1_3, vd));
		const __m128 r2 = _mm_sub_ps(_mm_add_ps(
			_mm_mul_ps(c1_3, vc),
			_mm_mul_ps(c5_6, vd)),
			_mm_mul_ps(c1_6, ve));

		/* Weighted combination and store (aligned) */
		const __m128 res = _mm_add_ps(_mm_add_ps(
			_mm_mul_ps(omega0, r0),
			_mm_mul_ps(omega1, r1)),
			_mm_mul_ps(omega2, r2));

		_mm_store_ps(out + i, res);
	}
}

void weno_minus_avx(const float * const a, const float * const b, const float * const c,
		    const float * const d, const float * const e, float * const out,
		    const int NENTRIES)
{
	/* Broadcast all constants into AVX 256-bit registers */
	const __m256 eps   = _mm256_set1_ps((float)WENOEPS);
	const __m256 one   = _mm256_set1_ps(1.0f);

	/* Smoothness indicator coefficients */
	const __m256 c4_3  = _mm256_set1_ps(4.f/3.f);
	const __m256 c19_3 = _mm256_set1_ps(19.f/3.f);
	const __m256 c11_3 = _mm256_set1_ps(11.f/3.f);
	const __m256 c25_3 = _mm256_set1_ps(25.f/3.f);
	const __m256 c31_3 = _mm256_set1_ps(31.f/3.f);
	const __m256 c10_3 = _mm256_set1_ps(10.f/3.f);
	const __m256 c13_3 = _mm256_set1_ps(13.f/3.f);
	const __m256 c5_3  = _mm256_set1_ps(5.f/3.f);

	/* Ideal weights */
	const __m256 w0    = _mm256_set1_ps(0.1f);
	const __m256 w1    = _mm256_set1_ps(0.6f);
	const __m256 w2    = _mm256_set1_ps(0.3f);

	/* Reconstruction coefficients */
	const __m256 c1_3  = _mm256_set1_ps(1.f/3.f);
	const __m256 c7_6  = _mm256_set1_ps(7.f/6.f);
	const __m256 c11_6 = _mm256_set1_ps(11.f/6.f);
	const __m256 c1_6  = _mm256_set1_ps(1.f/6.f);
	const __m256 c5_6  = _mm256_set1_ps(5.f/6.f);

	/* Process 8 floats per iteration (256-bit AVX register) */
	for (int i = 0; i < NENTRIES; i += 8)
	{
		/* Load 8 elements from each input array (aligned) */
		const __m256 va = _mm256_load_ps(a + i);
		const __m256 vb = _mm256_load_ps(b + i);
		const __m256 vc = _mm256_load_ps(c + i);
		const __m256 vd = _mm256_load_ps(d + i);
		const __m256 ve = _mm256_load_ps(e + i);

		/* is0 = a*(a*4/3 - b*19/3 + c*11/3) + b*(b*25/3 - c*31/3) + c*c*10/3 */
		const __m256 is0 = _mm256_add_ps(
			_mm256_add_ps(
				_mm256_mul_ps(va, _mm256_add_ps(_mm256_sub_ps(
					_mm256_mul_ps(va, c4_3),
					_mm256_mul_ps(vb, c19_3)),
					_mm256_mul_ps(vc, c11_3))),
				_mm256_mul_ps(vb, _mm256_sub_ps(
					_mm256_mul_ps(vb, c25_3),
					_mm256_mul_ps(vc, c31_3)))),
			_mm256_mul_ps(_mm256_mul_ps(vc, vc), c10_3));

		/* is1 = b*(b*4/3 - c*13/3 + d*5/3) + c*(c*13/3 - d*13/3) + d*d*4/3 */
		const __m256 is1 = _mm256_add_ps(
			_mm256_add_ps(
				_mm256_mul_ps(vb, _mm256_add_ps(_mm256_sub_ps(
					_mm256_mul_ps(vb, c4_3),
					_mm256_mul_ps(vc, c13_3)),
					_mm256_mul_ps(vd, c5_3))),
				_mm256_mul_ps(vc, _mm256_sub_ps(
					_mm256_mul_ps(vc, c13_3),
					_mm256_mul_ps(vd, c13_3)))),
			_mm256_mul_ps(_mm256_mul_ps(vd, vd), c4_3));

		/* is2 = c*(c*10/3 - d*31/3 + e*11/3) + d*(d*25/3 - e*19/3) + e*e*4/3 */
		const __m256 is2 = _mm256_add_ps(
			_mm256_add_ps(
				_mm256_mul_ps(vc, _mm256_add_ps(_mm256_sub_ps(
					_mm256_mul_ps(vc, c10_3),
					_mm256_mul_ps(vd, c31_3)),
					_mm256_mul_ps(ve, c11_3))),
				_mm256_mul_ps(vd, _mm256_sub_ps(
					_mm256_mul_ps(vd, c25_3),
					_mm256_mul_ps(ve, c19_3)))),
			_mm256_mul_ps(_mm256_mul_ps(ve, ve), c4_3));

		/* Add epsilon to smoothness indicators */
		const __m256 is0p = _mm256_add_ps(is0, eps);
		const __m256 is1p = _mm256_add_ps(is1, eps);
		const __m256 is2p = _mm256_add_ps(is2, eps);

		/* alpha_k = d_k / (is_k + eps)^2 */
		const __m256 alpha0 = _mm256_mul_ps(w0, _mm256_div_ps(one, _mm256_mul_ps(is0p, is0p)));
		const __m256 alpha1 = _mm256_mul_ps(w1, _mm256_div_ps(one, _mm256_mul_ps(is1p, is1p)));
		const __m256 alpha2 = _mm256_mul_ps(w2, _mm256_div_ps(one, _mm256_mul_ps(is2p, is2p)));

		const __m256 alphasum = _mm256_add_ps(_mm256_add_ps(alpha0, alpha1), alpha2);
		const __m256 inv_alpha = _mm256_div_ps(one, alphasum);

		/* Nonlinear weights */
		const __m256 omega0 = _mm256_mul_ps(alpha0, inv_alpha);
		const __m256 omega1 = _mm256_mul_ps(alpha1, inv_alpha);
		const __m256 omega2 = _mm256_sub_ps(_mm256_sub_ps(one, omega0), omega1);

		/* Reconstruction polynomials */
		const __m256 r0 = _mm256_add_ps(_mm256_sub_ps(
			_mm256_mul_ps(c1_3, va),
			_mm256_mul_ps(c7_6, vb)),
			_mm256_mul_ps(c11_6, vc));
		const __m256 r1 = _mm256_add_ps(_mm256_sub_ps(
			_mm256_mul_ps(c5_6, vc),
			_mm256_mul_ps(c1_6, vb)),
			_mm256_mul_ps(c1_3, vd));
		const __m256 r2 = _mm256_sub_ps(_mm256_add_ps(
			_mm256_mul_ps(c1_3, vc),
			_mm256_mul_ps(c5_6, vd)),
			_mm256_mul_ps(c1_6, ve));

		/* Weighted combination and store (aligned) */
		const __m256 res = _mm256_add_ps(_mm256_add_ps(
			_mm256_mul_ps(omega0, r0),
			_mm256_mul_ps(omega1, r1)),
			_mm256_mul_ps(omega2, r2));

		_mm256_store_ps(out + i, res);
	}
}
