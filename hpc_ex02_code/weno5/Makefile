CC=gcc
CFLAGS=-O3 -march=native -fopenmp
LIBS=-lm

all: bench

bench: bench.c weno.h
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS)

clean:
	rm -f bench

