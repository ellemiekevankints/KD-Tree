//////////////////////////////////////////////////////////////////////////////////////////
//                                                                                      //    
//                                  License Agreement                                   // 
//                      For Open Source Computer Vision Library                         //
//                                                                                      //    
//              Copyright (C) 2000-2008, Intel Corporation, all rights reserved.        //
//              Copyright (C) 2009, Willow Garage Inc., all rights reserved.            //
//              Copyright (C) 2013, OpenCV Foundation, all rights reserved.             //
//              Copyright (C) 2015, Itseez Inc., all rights reserved.                   //
//              Third party copyrights are property of their respective owners.         //    
//                                                                                      //
//  Redistribution and use in source and binary forms, with or without modification,    //
//  are permitted provided that the following conditions are met:                       //
//                                                                                      //
//   * Redistribution's of source code must retain the above copyright notice,          //
//     this list of conditions and the following disclaimer.                            //    
//                                                                                      //
//   * Redistribution's in binary form must reproduce the above copyright notice,       //
//     this list of conditions and the following disclaimer in the documentation        //
//     and/or other materials provided with the distribution.                           //
//                                                                                      //
//   * The name of the copyright holders may not be used to endorse or promote products //
//     derived from this software without specific prior written permission.            //
//                                                                                      //    
// This software is provided by the copyright holders and contributors "as is" and      //
// any express or implied warranties, including, but not limited to, the implied        //
// warranties of merchantability and fitness for a particular purpose are disclaimed.   //
// In no event shall the Intel Corporation or contributors be liable for any direct,    //
// indirect, incidental, special, exemplary, or consequential damages                   //
// (including, but not limited to, procurement of substitute goods or services;         //
// loss of use, data, or profits; or business interruption) however caused              //
// and on any theory of liability, whether in contract, strict liability,               //
// or tort (including negligence or otherwise) arising in any way out of                //
// the use of this software, even if advised of the possibility of such damage.         //
//                                                                                      //
//////////////////////////////////////////////////////////////////////////////////////////

#pragma once 

#ifndef KDTREE_CUH
#define KDTREE_CUH

#include <vector>
#include <algorithm>
#include <cstdio>
#include <iostream>

#include "Feature.cuh"
#include "Logger.hpp"
#include "Unity.cuh"

using namespace std;

const int K = 128; // dimensions
const int N = 20; // number of features
const int MAX_TREE_DEPTH = 32; // upper bound for tree level, equivalent to 4 billion generated features 

/****************
* KD TREE CLASS *
*****************/
template <typename T>
class KDTree {

public:
    
    // the node of the search tree.
    struct Node {
        Node() : idx(-1), left(-1), right(-1), boundary(0.f) {}
        Node(int _idx, int _left, int _right, float _boundary)
            : idx(_idx), left(_left), right(_right), boundary(_boundary) {}

        // split dimension; >=0 for nodes (dim), < 0 for leaves (index of the point)
        int idx;
        // node indices of the left and the right branches
        int left, right;
        // go to the left if query[node.idx]<=node.boundary, otherwise go to the right
        float boundary;
    };

    // priority queue used to search the tree
    struct PQueueElem {
        PQueueElem() : dist(0), idx(0) {}
        PQueueElem(float _dist, int _idx) : dist(_dist), idx(_idx) {}
        float dist; // distance of the query point from the node
        int idx; // current tree position
    };

    /****************************************
    * BELOW STRUCTS BELONG IN MATCH FACTORY *
    *****************************************/

    struct KeyPoint{
        int parentId;
        float2 loc;
    };

    struct Match{
        bool invalid;
        KeyPoint keyPoints[2];
    };
    
    struct DMatch: Match{
        float distance;
    };

    /* ************************************ */

    // constructors
    KDTree();
    KDTree(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> points);
    KDTree(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> points, vector<T> _labels);
    
    // builds the search tree
    void build(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> points);
    void build(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> points, vector<int> labels);
    
    // finds the K nearest neighbors of "feature" while looking at emax (at most) leaves
    DMatch findNearest(KDTree<T> kdtree, ssrlcv::Feature<T> queryFeature, int k, int emax) const;
    
    // return a vector with the specified index
    const float2 getPoint(int ptidx, int *label = 0) const;

    // print the kd tree
    void printKDTree();

    vector<Node> nodes; // all the tree nodes
    ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> points; // all the points 
    vector<int> labels; // the parallel array of labels
    int maxDepth;

};

/********************
* AUTO BUFFER CLASS *
*********************/
template<typename _Tp, size_t fixed_size = 1024/sizeof(_Tp)+8> 
class AutoBuffer {

public:
    typedef _Tp value_type;

    // the default constructor
    AutoBuffer();
    // constructor taking the real buffer size
    explicit AutoBuffer(size_t _size);

    // the assignment operator
    AutoBuffer<_Tp, fixed_size>& operator = (const AutoBuffer<_Tp, fixed_size>& buf);

    // destructor, calls deallocate()
    ~AutoBuffer();

    // allocates the new buffer of size _size. if the _size is small enough, stack-allocated buffer is used
    void allocate(size_t _size);
    // deallocates the buffer if it was dynamically allocated
    void deallocate();
    // returns pointer to the real buffer, stack-allocated or heap-allocated
    inline _Tp* data() { return ptr; }
    // returns read-only pointer to the real buffer, stack-allocated or heap-allocated
    inline const _Tp* data() const { return ptr; }

protected:
    // pointer to the real buffer, can point to buf if the buffer is small enough
    _Tp* ptr;
    // size of the real buffer
    size_t sz;
    // pre-allocated buffer. At least 1 element to confirm C++ standard requirements
    _Tp buf[(fixed_size > 0) ? fixed_size : 1];
};

/*****************************
* AUTO BUFFER IMPLEMENTATION *
******************************/
template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::AutoBuffer() {
    ptr = buf;
    sz = fixed_size;
}

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::AutoBuffer(size_t _size) {
    ptr = buf;
    sz = fixed_size;
    allocate(_size);
}

template<typename _Tp, size_t fixed_size> inline AutoBuffer<_Tp, fixed_size>&
AutoBuffer<_Tp, fixed_size>::operator = (const AutoBuffer<_Tp, fixed_size>& abuf) {
    if( this != &abuf )
    {
        deallocate();
        allocate(abuf.size());
        for( size_t i = 0; i < sz; i++ )
            ptr[i] = abuf.ptr[i];
    }
    return *this;
}

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::~AutoBuffer() { deallocate(); }

template<typename _Tp, size_t fixed_size> inline void
AutoBuffer<_Tp, fixed_size>::allocate(size_t _size) {
    if(_size <= sz)
    {
        sz = _size;
        return;
    }
    deallocate();
    sz = _size;
    if(_size > fixed_size)
    {
        ptr = new _Tp[_size];
    }
}

template<typename _Tp, size_t fixed_size> inline void
AutoBuffer<_Tp, fixed_size>::deallocate() {
    if( ptr != buf )
    {
        delete[] ptr;
        ptr = buf;
        sz = fixed_size;
    }
}

// validate matches function

#endif