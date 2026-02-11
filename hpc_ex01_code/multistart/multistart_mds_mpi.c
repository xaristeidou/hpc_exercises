#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi/mpi.h>

#define MAXVARS		(250)	/* max # of variables	     */
#define EPSMIN		(1E-6)	/* ending value of stepsize  */

/* prototype of local optimization routine, code available in torczon.c */
extern void mds(double *startpoint, double *endpoint, int n, double *val, double eps, int maxfevals, int maxiter,
         double mu, double theta, double delta, int *ni, int *nf, double *xl, double *xr, int *term);

struct Info {
	int best_trial;
	int best_nt;
	int best_nf;
	int best_fx;
	double best_pt[MAXVARS];
};

/* global variables */
unsigned long funevals = 0;

/* Rosenbrock classic parabolic valley ("banana") function */
double f(double *x, int n)
{
    double fv;
    int i;

    funevals++;
    fv = 0.0;
    for (i=0; i<n-1; i++)   /* rosenbrock */
        fv = fv + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);

		usleep(100);	/* do not remove, introduces some artificial work */

    return fv;
}


double get_wtime(void)
{
    struct timeval t;

    gettimeofday(&t, NULL);

    return (double)t.tv_sec + (double)t.tv_usec*1.0e-6;
}


int main(int argc, char *argv[])
{
	/* initialize MPI */
	MPI_Init(&argc, &argv);
	int size;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* problem parameters */
	int nvars = 4;		/* number of variables (problem dimension) */
	int ntrials = 64;	/* number of trials */
	double lower[MAXVARS], upper[MAXVARS];	/* lower and upper bounds */

	/* mds parameters */
	double eps = EPSMIN;
	int maxfevals = 10000;
	int maxiter = 10000;
	double mu = 1.0;
	double theta = 0.25;
	double delta = 0.25;

	double startpt[MAXVARS], endpt[MAXVARS];	/* initial and final point of mds */
	double fx;	/* function value at the final point of mds */
	int nt, nf;	/* number of iterations and function evaluations used by mds */

	/* information about the best point found by multistart */
	double best_pt[MAXVARS];
	double best_fx = 1e10;
	int best_trial = -1;
	int best_nt = -1;
	int best_nf = -1;

	/* local variables */
	int trial, i;
	double t0, t1;

	/* initialization of lower and upper bounds of search space */
	for (i = 0; i < MAXVARS; i++) lower[i] = -2.0;	/* lower bound: -2.0 */
	for (i = 0; i < MAXVARS; i++) upper[i] = +2.0;	/* upper bound: +2.0 */

	t0 = get_wtime();

	srand48(rank);

	/* starting guess for rosenbrock test function, search space in [-2, 2) */
	for (i = 0; i < nvars; i++) {
		startpt[i] = lower[i] + (upper[i]-lower[i])*drand48();
	}

	int term = -1;
	mds(startpt, endpt, nvars, &fx, eps, maxfevals, maxiter, mu, theta, delta,
		&nt, &nf, lower, upper, &term);


	#if DEBUG
	printf("\n\n\nMDS %d USED %d ITERATIONS AND %d FUNCTION CALLS, AND RETURNED\n", rank, nt, nf);
	for (i = 0; i < nvars; i++)
		printf("x[%3d] = %15.7le \n", i, endpt[i]);

	printf("f(x) = %15.7le\n", fx);
	#endif

	// TODO: collect all fx from each rank
	// MPI_Reduce(&fx, &best_fx, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

	if (rank==0)
	{		
		for (int i=1; i<size; i++)
		{
			printf("I am in iteration: %d\n\n", i);
			MPI_Recv(&fx, 1, MPI_DOUBLE, i, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&nt, 1, MPI_INT, i, 43, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&nf, 1, MPI_INT, i, 44, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&endpt, MAXVARS, MPI_DOUBLE, i, 45, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			/* keep the best solution */
			if (fx < best_fx)
			{
				best_trial = i;
				best_nt = nt;
				best_nf = nf;
				best_fx = fx;
				for (i = 0; i < nvars; i++)
					best_pt[i] = endpt[i];
			}
		}
		t1 = get_wtime();
		printf("\n\nFINAL RESULTS:\n");
		printf("Elapsed time = %.3lf s\n", t1-t0);
		printf("Total number of trials = %d\n", ntrials);
		printf("Total number of function evaluations = %ld\n", funevals);
		printf("Best result at trial %d used %d iterations, %d function calls and returned\n", best_trial, best_nt, best_nf);
		for (i = 0; i < nvars; i++) {
			printf("x[%3d] = %15.7le \n", i, best_pt[i]);
		}
		printf("f(x) = %15.7le\n", best_fx);
	}
	else
	{
		printf("I am sending data my ID is: %d: ", rank);
		MPI_Ssend(&fx, 1, MPI_DOUBLE, 0, 42, MPI_COMM_WORLD);
		MPI_Ssend(&nt, 1, MPI_INT, 0, 43, MPI_COMM_WORLD);
		MPI_Ssend(&nf, 1, MPI_INT, 0, 44, MPI_COMM_WORLD);
		MPI_Ssend(&endpt, MAXVARS, MPI_INT, 0, 45, MPI_COMM_WORLD);
	}


	MPI_Finalize();
	return 0;
}

