#ifndef FLANN_HELPER_HPP
#define FLANN_HELPER_HPP

#include <FeatureReader.hpp>
#include <iostream>
#include <flann/flann.hpp>
#include <stdio.h>

__inline__ void copy_to_ary(float *& ary, caffe::Datum &d, int size)
{
        assert (size % 4 == 0); // Make sure that size is divisible by our loop unroll factor.
        for (int i = 0; i < size; i += 4)
        {
                ary[i+0] = d.float_data(i+0);
                ary[i+1] = d.float_data(i+1);
                ary[i+2] = d.float_data(i+2);
                ary[i+3] = d.float_data(i+3);
        }
}

void copy(flann::Matrix<float> &dst, std::vector<caffe::Datum> src, int N, int dim)
{
	dst = flann::Matrix<float>(new float[N*dim], N, dim);
	float *dst_ptr = (float *) &dst.ptr()[0];
	for (int i = 0; i < N; ++i)
	{
		copy_to_ary(dst_ptr, src[i], dim);
		dst_ptr += dim;
	}
}

#endif
