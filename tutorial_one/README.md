#Tutorial One: Nearest Neighbors with FLANN

[FLANN](http://www.cs.ubc.ca/research/flann/) is the de-facto library for computing nearest neighbor search. In this tutorial, we will characterize the FLANN library using: (a) its linear brute-force algorithm, (b) randomized kd-tree data structure, and (c) priority-search k-means.
All of the code is setup and written. All you have to do is the data collection.

## Running the Code

Make sure you login to a machine that has Docker. Instructions for this tutorial is below.

~~~bash 
docker pull cdel/viral_tutorial
docker run -ti --net=host cdel/viral_tutorial
mkdir -p /scratch/ ; cd /scratch/
wget 'http://homes.cs.washington.edu/~cdel/files/features.lldb.tgz'
tar -zxvf features.lldb.tgz
git clone git@github.com:carlodelmundo/VIRALTutorials.git
cd VIRALTutorials/tutorial_one/
make -j4
./NNSearch /scratch/features.lldb/
~~~

## Your Tasks

Under main.cpp, lines 74-81, contain the following calls.

~~~cpp
std::cout << "Starting NN Search using Linear Brute-Force" << std::endl;
compute_linear(dataset, query, nn, indices, dists);

std::cout << "Starting NN Search using KD Trees" << std::endl;
compute_kdtree(dataset, query, nn, indices, dists);

std::cout << "Starting NN Search using KMeans" << std::endl;
compute_kmeans(dataset, query, nn, indices, dists);
~~~

Each compute_* function calls the NN search using its respective data-structure. Your task is to do the following.

1. *Performance Analysis*. Quantify the execution time of each compute_* function call using a performance counter analysis tool such as [Google Performance Tools](https://github.com/gperftools/gperftools) or [GNU Profiler](https://sourceware.org/binutils/docs/gprof/). Use num_db_items = 512 and num_query_items = 512 (defaults). What is the speedups of the kd-tree vs. linear? k-means vs. linear? Next, vary the `num_db_items` to generate a graph of the execution time (y-axis) over num_db_items (x-axis).
