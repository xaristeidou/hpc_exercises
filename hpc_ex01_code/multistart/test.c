// #include <stdio.h>
// #include <omp.h>
// #include <unistd.h>

// int main(int argc,char *argv[]) {

//     #pragma omp parallel
//     {
//         #pragma omp single
//         {
//             for (int i=0; i<20; i++)
//             {
//                 #pragma omp task
//                 sleep(2);
//             }
//             printf("Hello from me: %i\n", omp_get_thread_num());
//             #pragma omp taskwait
//         }
//     }
//     return 0;
// }


#include <stdio.h>
// #include <omp.h>
#include <unistd.h>
#include <mpi/mpi.h>

int main(int argc,char *argv[]) {

    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello from me: %i\n", rank);

    MPI_Finalize();
    return 0;
}