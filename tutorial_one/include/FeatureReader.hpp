#ifndef FEATURE_READER_HPP
#define FEATURE_READER_HPP

#include <lmdb++.hpp>
#include <caffe.pb.hpp>
#include <vector>

std::vector<caffe::Datum> readDatums(std::string db_name_)
{
        std::vector<caffe::Datum> datums;

        /* Create and open the LMDB environment: */
        auto env = lmdb::env::create();
        env.open(db_name_.c_str(), 0, 0664);

        /* Fetch key/value pairs in a read-only transaction: */
        auto rtxn = lmdb::txn::begin(env, nullptr, MDB_RDONLY);
        auto dbi = lmdb::dbi::open(rtxn, nullptr);
        auto cursor = lmdb::cursor::open(rtxn, dbi);

        std::string key, value;
        while (cursor.get(key, value, MDB_NEXT))
        {
                caffe::Datum datum;
                datum.ParseFromString(value);
                datums.push_back(datum);
        }
        cursor.close();
        rtxn.abort();

        return datums;
}

#endif
