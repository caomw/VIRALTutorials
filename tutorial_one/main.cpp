#include <FeatureReader.hpp>
#include <iostream>
#include <flann/flann.hpp>
#include <FLANNHelper.hpp>
#include <stdio.h>

using namespace flann;
void printUsage(char **argv)
{
	std::cout << argv[0] << ": <LMDB database>" << std::endl;
}

void compute_linear(Matrix<float> &dataset, Matrix<float> &query, int nn, Matrix<int> &indices, Matrix<float> &dists)
{
    Index<L2<float> > index(dataset, flann::LinearIndexParams());
    index.buildIndex();                                                                                               
    index.knnSearch(query, indices, dists, nn, FLANN_CHECKS_UNLIMITED);
}

void compute_kdtree(Matrix<float> &dataset, Matrix<float> &query, int nn, Matrix<int> &indices, Matrix<float> &dists)
{
    // construct an randomized kd-tree index using 4 kd-trees
    Index<L2<float> > index(dataset, flann::KDTreeIndexParams(4));
    index.buildIndex();                                                                                               

	// The num_checks parameter is the number of leaf nodes to visit in the database. The higher the num_checks, the higher accuracy, but lower performance.
	// Think of each leaf node as containing one data item in the search database.
	int num_checks = 128;
    index.knnSearch(query, indices, dists, nn, num_checks);
}

void compute_kmeans(Matrix<float> &dataset, Matrix<float> &query, int nn, Matrix<int> &indices, Matrix<float> &dists)
{
    Index<L2<float> > index(dataset, flann::KMeansIndexParams());
    index.buildIndex();                                                                                               
	// The num_checks parameter is the number of leaf nodes to visit in the database. The higher the num_checks, the higher accuracy, but lower performance.
	// Think of each leaf node as containing one data item in the search database.
	int num_checks = 128;	
    index.knnSearch(query, indices, dists, nn, num_checks);
}

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cout << "ERROR! Expecting at least one command line argument!\n" << std::endl;
		printUsage(argv);
	}

	// Load data	
	std::string LMDB_DATABASE_NAME = std::string(argv[1]);
	std::vector<caffe::Datum> datums = readDatums(LMDB_DATABASE_NAME);
	int max_items = datums.size();
	int max_dims = datums[0].channels();
	std::cout << "There are " << datums.size() << " vectors of size " << max_dims << "-dimensionality in " << LMDB_DATABASE_NAME << std::endl;

	// Change these parameters below.
	int num_db_items = 512;		
	int num_query_items = 512;
	int num_dims = max_dims;
	int nn = 4;

	assert(num_db_items <= max_items);
	assert(num_dims <= max_dims);

	Matrix<float> dataset;	
	Matrix<float> query;
	copy(dataset, datums, num_db_items, max_dims);
	copy(query, datums, num_query_items, max_dims);

	Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
	Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

	std::cout << "Starting NN Search using Linear Brute-Force" << std::endl;
	compute_linear(dataset, query, nn, indices, dists);
	
	std::cout << "Starting NN Search using KD Trees" << std::endl;
	compute_kdtree(dataset, query, nn, indices, dists);
	
	std::cout << "Starting NN Search using KMeans" << std::endl;
	compute_kmeans(dataset, query, nn, indices, dists);

    delete[] dataset.ptr();
    delete[] query.ptr();
    delete[] indices.ptr();
    delete[] dists.ptr();
}
