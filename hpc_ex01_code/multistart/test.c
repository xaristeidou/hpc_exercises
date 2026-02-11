#include <stdio.h>
#include <unistd.h>
#include <mpi/mpi.h>

int main(int argc,char *argv[]) {

    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int id=rank;


    if (rank==0)
    {
        int tmp_id;
        MPI_Status status;
        for (int i=1; i<size; i++)
        {
            MPI_Recv(&tmp_id, 1, MPI_INT, i, 42, MPI_COMM_WORLD, &status);
            printf("\nReceived: %i\n", tmp_id);
        }
    }
    else
    {
        MPI_Send(&rank, 1, MPI_INT, 0, 42, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}