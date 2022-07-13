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
     * \param nodes the nodes of the KD-Tree
     * \param treeFeatures the KD-Tree feature points 
     * \param pqueue the priority queue used to search the tree 
     * \param queryFeature the query feature point
     * \param emax the max number of leaf nodes to search. a value closer to the total number feature points correleates to a higher accuracy macth
     * \param absoluteThreshold the maximum distance between two matched points
     * \param k the number of nearest neighbors. by default this value finds the 2 closest features to a given feature point
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
    unsigned int targetImageID, KDTree<T>* kdtree, typename KDTree<T>::Node* nodes, ssrlcv::Feature<T>* featuresTree, ssrlcv::PQueueElem* pqueue, DMatch* matches, float absoluteThreshold);

/* ************************************************************************************************************************************************************************************************************************************************************************** */

} // namepsace ssrlcv


// delete below code when transfering to SSRLCV


/**
* \brief Method for getting grid and block for a 1D kernel.
* \details This method calculates a grid and block configuration
* in an attempt to achieve high levels of CUDA occupancy as well
* as ensuring there will be enough threads for a specified number of elements.
* Methods for determining globalThreadID's from the returned grid and block
* can be found at the bottom of cuda_util.h but must be placed in the same
* compilational unit.
* \param numElements - number of elements that will be threaded in kernel
* \param grid - dim3 grid argument to be set withint this function
* \param block - dim3 block argument to be set within this function
* \param kernel - function pointer to the kernel that is going to use the grid and block
* \param dynamicSharedMem - size of dynamic shared memory used in kernel (optional parameter - will default to 0)
* \param device - the NVIDIA GPU device ID (optional parameter - will default to 0)
* \warning This creates grid and block dimensions that guarantee coverage of numElements.
* This likely means that there will be more threads that necessary, so make sure to check that
* globalThreadID < numElements in you kernel. Otherwise there will be an illegal memory access.
*/
template<typename... Types>
void getFlatGridBlock(unsigned long numElements, dim3 &grid, dim3 &block, void (*kernel)(Types...), size_t dynamicSharedMem = 0, int device = 0){
  grid = {1,1,1};
  block = {1,1,1};
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  int blockSize;
  int minGridSize;
  cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    kernel,
    dynamicSharedMem,
    numElements
  );
  block = {(unsigned int)blockSize,1,1};
  unsigned int gridSize = (numElements + (unsigned int)blockSize - 1) / (unsigned int)blockSize;
  if(gridSize > prop.maxGridSize[0]){
    if(gridSize >= 65535L*65535L*65535L){
      grid = {65535,65535,65535};
    }
    else{
      gridSize = (gridSize/65535L) + 1;
      grid.x = 65535;
      if(gridSize > 65535){
        grid.z = (grid.y/65535) + 1;
        grid.y = 65535;
      }
      else{
        grid.y = 65535;
        grid.z = 1;
      }
    }
  }
  else{
    grid = {gridSize,1,1};
  }
}

#endif