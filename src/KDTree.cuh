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

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include <vector>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Feature.cuh"
#include "Logger.hpp"
#include "Unity.cuh"

using namespace std;

const int K = 128; // dimensions
const int N = 20; // number of features
const int MAX_TREE_DEPTH = 32; // upper bound for tree level, equivalent to 4 billion generated features 

namespace ssrlcv {

    /****************************************
    * BELOW STRUCTS BELONG IN MATCH FACTORY *
    *****************************************/

    // delete below code when transfering to SSRLCV

/* ************************************************************************************************************************************************************************************************************************************************************************** */

    struct uint2_pair{
        uint2 a;
        uint2 b;
    };

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

    namespace {
        /**
        * structs used with thrust::remove_if on GPU arrays
        */
        struct validate{
            __host__ __device__ bool operator()(const Match &m){
                return m.invalid;
            }
            __host__ __device__ bool operator()(const uint2_pair &m){
                return m.a.x == m.b.x && m.a.y == m.b.y;
            }
        };

        struct match_above_cutoff{
            __host__ __device__
            bool operator()(DMatch m){
                return m.distance > 0.0f;
            }
        };

        struct match_dist_thresholder{
            float threshold;
            match_dist_thresholder(float threshold) : threshold(threshold){};
            __host__ __device__
            bool operator()(DMatch m){
                return (m.distance > threshold);
            }
        };

        /**
        * struct for comparison to be used with thrust::sort on GPU arrays
        */
        struct match_dist_comparator{
            __host__ __device__
            bool operator()(const DMatch& a, const DMatch& b){
                return a.distance < b.distance;
            }
        };

    } // namespace

/* ************************************************************************************************************************************************************************************************************************************************************************** */

    /****************
    * KD-TREE CLASS *
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

        // constructors
        KDTree();
        KDTree(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> points);
        KDTree(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> points, vector<T> _labels);
    
        // builds the search tree
        void build(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> points);
        void build(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> points, vector<int> labels);

        // return a point with the specified index
        const float2 getPoint(int ptidx, int *label = 0) const;

        // print the kd tree
        void printKDTree();

        thrust::host_vector<Node> nodes; // all the tree nodes
        ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> points; // all the points 
        vector<int> labels; // the parallel array of labels
        int maxDepth;

    }; // KD Tree class

    /************************
    * PRIORITY QUEUE STRUCT *
    *************************/

    // priority queue used to search the tree
    struct PQueueElem {
        CUDA_CALLABLE_MEMBER PQueueElem() : dist(0), idx(0) {}
        CUDA_CALLABLE_MEMBER PQueueElem(float _dist, int _idx) : dist(_dist), idx(_idx) {}
        float dist; // distance of the query point from the node
        int idx; // current tree position
    };
  
    // put the code below in MatchFactroy.cu

/* ************************************************************************************************************************************************************************************************************************************************************************** */

    /*
     * \brief finds the k nearest neighbors to a point while looking at emax (at most) leaves
     * \param kdtree the KD-Tree to search through
     * \param queryFeature the query feature point
     * \param emax the max number of leaf nodes to search. a value closer to the total number feature points correleates to a higher accuracy macth
     * \param absoluteThreshold the maximum distance between two matched points
     * \param k the number of nearest neighbors 
    */ 
    template<typename T> 
    __device__ DMatch findNearest(ssrlcv::KDTree<T>* kdtree, typename KDTree<T>::Node* nodes, ssrlcv::Feature<T>* treeFeatures, ssrlcv::PQueueElem* pqueue, 
    ssrlcv::Feature<T> queryFeature, int emax, float absoluteThreshold, int k = 1);

    template<typename T>    
    class MatchFactory {
        private:
            ssrlcv::ptr::value<Unity<Feature<T>>> seedFeatures;
        public:
            float absoluteThreshold;
            float relativeThreshold;
            MatchFactory(float relativeThreshold, float absoluteThreshold);
            void validateMatches(ssrlcv::ptr::value<ssrlcv::Unity<DMatch>> matches); 
            ssrlcv::ptr::value<ssrlcv::Unity<DMatch>> generateDistanceMatches(int queryID, ssrlcv::ptr::value<Unity<Feature<T>>> queryFeatures, int targetID,
            ssrlcv::KDTree<T> kdtree);
    }; // MatchFactory class

    template<typename T>
    __global__ void matchFeaturesKDTree(unsigned int queryImageID, unsigned long numFeaturesQuery, Feature<T>* featuresQuery, 
    unsigned int targetImageID, unsigned long numFeaturesTarget, KDTree<T>* kdtree, typename KDTree<T>::Node* nodes, ssrlcv::Feature<T>* treeFeatures, ssrlcv::PQueueElem* pqueue, DMatch* matches, float absoluteThreshold);

    // __global__ void matchFeaturesKDTree(unsigned int queryImageID, unsigned long numFeaturesQuery,
    //     Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
    //     KDTree<T>* kdtree, DMatch* matches, float* seedDistances, float relativeThreshold, float absoluteThreshold);

/* ************************************************************************************************************************************************************************************************************************************************************************** */

} // namepsace ssrlcv


// delete below code when transfering to SSRLCV

void getGrid(unsigned long numElements, dim3 &grid, int device = 0){
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  grid = {(unsigned int)prop.maxGridSize[0],(unsigned int)prop.maxGridSize[1],(unsigned int)prop.maxGridSize[2]};
  if(numElements < grid.x){
    grid.x = numElements;
    grid.y = 1;
    grid.z = 1;
  }
  else{
    grid.x = 65536;
    if(numElements < grid.x*grid.y){
      grid.y = numElements/grid.x;
      grid.y++;
      grid.z = 1;
    }
    else if(numElements < grid.x*grid.y*grid.z){
      grid.z = numElements/(grid.x*grid.y);
      grid.z++;
    }
  }
} // getGrid

#endif