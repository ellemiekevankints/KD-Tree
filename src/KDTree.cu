#include "KDTree.cuh"

/******************
* KD TREE METHODS *
*******************/

template<typename T>
ssrlcv::KDTree<T>::KDTree() {}

template<typename T>
ssrlcv::KDTree<T>::KDTree(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> _points) {
    build(_points);
}

template<typename T>
ssrlcv::KDTree<T>::KDTree(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> _points, vector<T> _labels) {
    build(_points, _labels);
}

struct SubTree {
    SubTree() : first(0), last(0), nodeIdx(0), depth(0) {}
    SubTree(int _first, int _last, int _nodeIdx, int _depth)
        : first(_first), last(_last), nodeIdx(_nodeIdx), depth(_depth) {}
    int first;
    int last;
    int nodeIdx;
    int depth;
};

static float medianPartition(size_t* ofs, int a, int b, const unsigned char* vals) {
    int k, a0 = a, b0 = b;
    int middle = (a + b)/2;
    while( b > a ) {
        int i0 = a, i1 = (a+b)/2, i2 = b;
        float v0 = vals[ofs[i0]], v1 = vals[ofs[i1]], v2 = vals[ofs[i2]];
        int ip = v0 < v1 ? (v1 < v2 ? i1 : v0 < v2 ? i2 : i0) :
                 v0 < v2 ? (v1 == v0 ? i2 : i0): (v1 < v2 ? i2 : i1);
        float pivot = vals[ofs[ip]];
        swap(ofs[ip], ofs[i2]);

        for( i1 = i0, i0--; i1 <= i2; i1++ ) {
            if( vals[ofs[i1]] <= pivot ) {
                i0++;
                swap(ofs[i0], ofs[i1]);
            }
        } // for
        if( i0 == middle )
            break;
        if( i0 > middle )
            b = i0 - (b == i0);
        else
            a = i0;
    } // while

    float pivot = vals[ofs[middle]];
    for( k = a0; k < middle; k++ ) {
        if( !(vals[ofs[k]] <= pivot) ) {
           logger.err<<"ERROR: median partition unsuccessful"<<"\n"; 
        }
    }
    for( k = b0; k > middle; k-- ) {
       if( !(vals[ofs[k]] >= pivot) ) {
           logger.err<<"ERROR: median partition unsuccessful"<<"\n"; 
        } 
    }

    return vals[ofs[middle]];
} // medianPartition

template<typename T>
static void computeSums(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> points, int start, int end, unsigned char *sums) {
   
    int i, j, dims = K; 
    ssrlcv::Feature<T> data; 

    // initilize sums array with 0
    for(j = 0; j < dims; j++)
        sums[j*2] = sums[j*2+1] = 0;

    // compute the square of each element in the values array 
    for(i = start; i <= end; i++) {
        data = points->host[i];
        for(j = 0; j < dims; j++) {
            double t = data.descriptor.values[j], s = sums[j*2] + t, s2 = sums[j*2+1] + t*t;
            sums[j*2] = s; sums[j*2+1] = s2;
        }
    }
} // computeSums

template<typename T>
void ssrlcv::KDTree<T>::build(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> _points) {
    vector<int> labels;
    build(_points, labels);
} // build

template<typename T>
void ssrlcv::KDTree<T>::build(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> _points, vector<int> _labels) {

    if (_points->size() == 0) {
        logger.err<<"ERROR: number of features in image must be greater than zero"<<"\n";
    }
    
    // initilize nodes of KD Tree
    vector<KDTree::Node>().swap(nodes);
    points = _points;

    int i, j, n = _points->size(), top = 0;
    const unsigned char *data = _points->host->descriptor.values;
    unsigned char *dstdata = points->host->descriptor.values;

    // size of object in memory 
    size_t step = sizeof(ssrlcv::Feature<T>);

    labels.resize(n); // labels and points array will share same size 
    const int *_labels_data = 0;

    if( !_labels.empty() ) {
        int nlabels = N*K;
        if ( !(nlabels==n) ) {
            logger.err<<"ERROR: labels size must be equal to points size"<<"\n";
        } 
        _labels_data = _labels.data(); 
    }

    // will hold the SIFT_Descriptor values array AND its squares
    unsigned char sumstack[MAX_TREE_DEPTH*2][K*2];
    SubTree stack[MAX_TREE_DEPTH*2]; 

    vector<size_t> _ptofs(n);
    size_t *ptofs = &_ptofs[0];

    for( i = 0; i < n; i++ ) 
        ptofs[i] = i*step;

    nodes.push_back(Node());
    computeSums<T>(points, 0, n-1, sumstack[top]);
    stack[top++] = SubTree(0, n-1, 0, 0);
    int _maxDepth = 0;

    while(--top >= 0) {
        int first = stack[top].first, last = stack[top].last;
        int depth = stack[top].depth, nidx = stack[top].nodeIdx;
        int count = last - first + 1, dim = -1;
        const unsigned char *sums = sumstack[top]; // points to the first element in uchar array
        double invCount = 1./count, maxVar = -1.;

        if(count == 1) {
            int idx0 = (int)(ptofs[first]/step);
            int idx = idx0; // the dimension
            nodes[nidx].idx = ~idx;
            
            labels[idx] = _labels_data ? _labels_data[idx0] : idx0;
            _maxDepth = std::max(_maxDepth, depth);
            continue;
        }

        // find the dimensionality with the biggest variance
        for( j = 0; j < K; j++ ) {
            unsigned char m = sums[j*2]*invCount;
            unsigned char varj = sums[j*2+1]*invCount - m*m;
            if( maxVar < varj ) {
                maxVar = varj;
                dim = j;
            }
        }

        int left = (int)nodes.size(), right = left + 1;
        nodes.push_back(Node());
        nodes.push_back(Node());
        nodes[nidx].idx = dim;
        nodes[nidx].left = left;
        nodes[nidx].right = right;
        nodes[nidx].boundary = medianPartition(ptofs, first, last, data + dim);

        int middle = (first + last)/2;
        unsigned char *lsums = (unsigned char*)sums, *rsums = lsums + K*2;
        computeSums(points, middle+1, last, rsums);
        for(j = 0; j < K*2; j++)
            lsums[j] = sums[j] - rsums[j];
        stack[top++] = SubTree(first, middle, left, depth+1);
        stack[top++] = SubTree(middle+1, last, right, depth+1);
    } // while
    maxDepth = _maxDepth;
} // build

// The below algorithm is from:
// J.S. Beis and D.G. Lowe. Shape Indexing Using Approximate Nearest-Neighbor Search
// in High-Dimensional Spaces. In Proc. IEEE Conf. Comp. Vision Patt. Recog.,
// pages 1000--1006, 1997. https://www.cs.ubc.ca/~lowe/papers/cvpr97.pdf
template<typename T> 
ssrlcv::DMatch ssrlcv::KDTree<T>::findNearest(ssrlcv::KDTree<T> kdtree, ssrlcv::Feature<T> feature, int k, int emax) const {

    T desc = feature.descriptor;
    const unsigned char *vec = desc.values; // descriptor values[128] from query

    ssrlcv::AutoBuffer<unsigned char> _buf((k+1)*(sizeof(float) + sizeof(int)));

    int *idx = (int*)_buf.data(); // holds the node indices 
    float *dist = (float*)(idx + k + 1); // holds the euclidean distances
    int i, j, ncount = 0, e = 0;

    int qsize = 0, maxqsize = 1 << 10;
    ssrlcv::AutoBuffer<unsigned char> _pqueue(maxqsize*sizeof(PQueueElem)); 
    PQueueElem *pqueue = (PQueueElem*)_pqueue.data(); 
    emax = std::max(emax, 1);

    for (e = 0; e < emax;) {
        float d, alt_d = 0.f; 
        int nidx; // node index

        if (e == 0)
            nidx = 0;
        else {
            // take the next node from the priority queue
            if (qsize == 0)
                break;
            nidx = pqueue[0].idx; // current tree position
            alt_d = pqueue[0].dist; // distance of the query point from the node
            if (--qsize > 0) {
                std::swap(pqueue[0], pqueue[qsize]);
                d = pqueue[0].dist;
                for (i = 0;;) {
                    int left = i*2 + 1, right = i*2 + 2;
                    if (left >= qsize)
                        break;
                    if (right < qsize && pqueue[right].dist < pqueue[left].dist)
                        left = right;
                    if (pqueue[left].dist >= d)
                        break;
                    std::swap(pqueue[i], pqueue[left]);
                    i = left;
                } // for
            } // if
            if (ncount == k && alt_d > dist[ncount-1])
                continue;
        } // if-else

        for (;;) {
            if (nidx < 0) 
                break;
            const Node& n = kdtree.nodes[nidx];

            if (n.idx < 0) { // if it is a leaf node
                i = ~n.idx; 
                const unsigned char *row = kdtree.points->host[i].descriptor.values; // descriptor values[128] from target

                // euclidean distance
                for (j = 0, d = 0.f; j < K; j++) {
                    float t = vec[j] - row[j];
                    // printf("\nvec[%d] = %f\n", j, vec[j]);
                    // printf("\nrow[%d] = %f\n", j, row[j]);
                    // printf("\nt = %f\n", t);
                    d += t*t;
                }
                dist[ncount] = d;
                printf("\ndist[%d] = %f\n", ncount, dist[ncount]);
                idx[ncount] = i;

                for (i = ncount-1; i >= 0; i--) {
                    if (dist[i] <= d)
                        break;
                    std::swap(dist[i], dist[i+1]);
                    std::swap(idx[i], idx[i+1]);
                } // for
                ncount += ncount < k;
                e++;
                break;   
            } // if

            int alt;
            if (vec[n.idx] <= n.boundary) {
                nidx = n.left;
                alt = n.right;
            } else {
                nidx = n.right;
                alt = n.left;
            }

            d = vec[n.idx] - n.boundary;
            d = d*d + alt_d; // euclidean distance

            // subtree prunning
            if (ncount == k && d > dist[ncount-1])
                continue;
            // add alternative subtree to the priority queue
            pqueue[qsize] = PQueueElem(d, alt);
            for (i = qsize; i > 0;) {
                int parent = (i-1)/2;
                if (parent < 0 || pqueue[parent].dist <= d)
                    break;
                std::swap(pqueue[i], pqueue[parent]);
                i = parent;
            }
            qsize += qsize+1 < maxqsize;
        } // for
    } // for

    DMatch match;
    match.distance = dist[0]; // smallest distance
    int matchIndex = idx[0]; // index of corresponding node/point

    if (match.distance >= FLT_MAX) { match.invalid = true; } // (!!!) change FLT_MAX to absoluteThreshold
    else {
      match.invalid = false;
      match.keyPoints[0].loc = kdtree.points->host[matchIndex].loc; // img1 
      match.keyPoints[1].loc = feature.loc; // img2
      
      // we do not have Image class implemented so no image id
      // match.keyPoints[0].parentId = queryImageID;  
      // match.keyPoints[1].parentId = targetImageID;
    }

    // will need to change this to return dist[0];
    return match;
} // findNearest

/* ************************************************************************************************************************************************************************************************************************************************************************** */

/* THINGS TO DO WHEN TRANSFERRING generateDistanceMatch TO SSRLCV
* change parameters int queryID and targetID back to ssrlcv::ptr::value<ssrlcv::Image> query and target
* change queryID and targetID back to query->id and target->id in kernel call
* remove MatchFactory stuff from KDTree.cuh + getGrid 
* validateMatches can be deleted from this file, we just need it for testing purposes
* put generateDistanceMatches and matchFeaturesKDTree in MatchFactory.cu + .cuh 
*/

template<typename T>
ssrlcv::MatchFactory<T>::MatchFactory(float relativeThreshold, float absoluteThreshold) :
relativeThreshold(relativeThreshold), absoluteThreshold(absoluteThreshold)
{
  this->seedFeatures = nullptr;
}

template<typename T>
void ssrlcv::MatchFactory<T>::validateMatches(ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::DMatch>> matches) {
  MemoryState origin = matches->getMemoryState();
  if (origin != gpu) matches->setMemoryState(gpu);
  
  thrust::device_ptr<DMatch> needsValidating(matches->device.get());
  thrust::device_ptr<DMatch> new_end = thrust::remove_if(needsValidating,needsValidating+matches->size(),validate());
  cudaDeviceSynchronize();
  CudaCheckError();
  int numMatchesLeft = new_end - needsValidating;
  if (numMatchesLeft == 0) {
    std::cout<<"No valid matches found"<<"\n";
    matches.clear();
    return;
  }
  
  printf("%d valid matches found out of %lu original matches\n",numMatchesLeft,matches->size());

  ssrlcv::ptr::device<DMatch> validatedMatches_device(numMatchesLeft);
  CudaSafeCall(cudaMemcpy(validatedMatches_device.get(),matches->device.get(),numMatchesLeft*sizeof(DMatch),cudaMemcpyDeviceToDevice));

  matches->setData(validatedMatches_device,numMatchesLeft,gpu);

  if(origin != gpu) matches->setMemoryState(origin);
} // validateMatches

template<typename T>
ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::DMatch>> ssrlcv::MatchFactory<T>::generateDistanceMatches(int targetID, ssrlcv::KDTree<T> kdtree, int queryID, ssrlcv::ptr::value<ssrlcv::Unity<Feature<T>>> queryFeatures, ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances) {
  MemoryState origin[2] = {queryFeatures->getMemoryState(), targetFeatures->getMemoryState()};

  if(origin[0] != gpu) queryFeatures->setMemoryState(gpu);
  if(origin[1] != gpu) targetFeatures->setMemoryState(gpu);

  unsigned int numPossibleMatches = queryFeatures->size();

  ssrlcv::ptr::value<ssrlcv::Unity<DMatch>> matches = ssrlcv::ptr::value<ssrlcv::Unity<DMatch>>(nullptr, numPossibleMatches, ssrlcv::gpu);

  dim3 grid = {1,1,1};
  dim3 block = {32,1,1}; // IMPROVE
  getGrid(matches->size(),grid);

  clock_t timer = clock();

  if (seedDistances == nullptr) {
    matchFeaturesKDTree<T><<<grid, block>>>(queryID, queryFeatures->size(), queryFeatures->device.get(),
    targetID, kdtree.points->size(), kdtree, matches->device.get(), this->absoluteThreshold);
  }
  else if (seedDistances->size() != queryFeatures->size()) {
    logger.err<<"ERROR: seedDistances should have come from matching a seed image to queryFeatures"<<"\n";
    exit(-1);
  }
  else{
    // MemoryState seedOrigin = seedDistances->getMemoryState();
    // if(seedOrigin != gpu) seedDistances->setMemoryState(gpu);
    // matchFeaturesBruteForce<T><<<grid, block>>>(queryID, queryFeatures->size(), queryFeatures->device.get(),
    // targetID, targetFeatures->size(), targetFeatures->device.get(), matches->device.get(),seedDistances->device.get(),
    // this->relativeThreshold,this->absoluteThreshold);
    // if(seedOrigin != gpu) seedDistances->setMemoryState(seedOrigin);
  }
  cudaDeviceSynchronize();
  CudaCheckError();

  this->validateMatches(matches);

  printf("done in %f seconds.\n\n",((float) clock() -  timer)/CLOCKS_PER_SEC);

  if(origin[0] != gpu) queryFeatures->setMemoryState(origin[0]);
  if(origin[1] != gpu) targetFeatures->setMemoryState(origin[1]);

  return matches;
} // generateDistanceMatches

template<typename T>
__global__ void ssrlcv::matchFeaturesKDTree(unsigned int queryImageID, unsigned long numFeaturesQuery,
ssrlcv::Feature<T>* featuresQuery, unsigned int targetImageID, unsigned long numFeaturesTarget,
ssrlcv::KDTree<T> kdtree, DMatch* matches, float absoluteThreshold) {
  
  unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  
  if (blockId < numFeaturesQuery) {
    Feature<T> feature = featuresQuery[blockId];
    __shared__ int localMatch[32]; // not sure if i need this
    __shared__ float localDist[32]; // same here
    localMatch[threadIdx.x] = -1;
    localDist[threadIdx.x] = absoluteThreshold;
    __syncthreads();

    float currentDist = 0.0f;
    unsigned long numFeaturesTarget_register = numFeaturesTarget;
    for(int f = threadIdx.x; f < numFeaturesTarget_register; f += 32){
      currentDist = feature.descriptor.distProtocol(featuresTarget[f].descriptor,localDist[threadIdx.x]);
      if(localDist[threadIdx.x] > currentDist){
        localDist[threadIdx.x] = currentDist;
        localMatch[threadIdx.x] = f;
      }
    } // for 
    __syncthreads();
    
    if(threadIdx.x != 0) return;
    currentDist = absoluteThreshold;
    int matchIndex = -1;
    for(int i = 0; i < 32; ++i){
      if(currentDist > localDist[i]){
        currentDist = localDist[i];
        matchIndex = localMatch[i];
      }
    }
    DMatch match;
    match.distance = currentDist;
    if(match.distance >= absoluteThreshold){
      match.invalid = true;
    }
    else{
      match.invalid = false;
      match.keyPoints[0].loc = feature.loc;
      match.keyPoints[1].loc = featuresTarget[matchIndex].loc;
      match.keyPoints[0].parentId = queryImageID;
      match.keyPoints[1].parentId = targetImageID;
    }
    matches[blockId] = match;
  } // if
} // matchFeaturesKDTree

/* ************************************************************************************************************************************************************************************************************************************************************************** */

template<typename T>
const float2 ssrlcv::KDTree<T>::getPoint(int ptidx, int *label) const {
    if ( !((unsigned)ptidx < (unsigned)points->size()) ) {
        logger.err<<"ERROR: point index is out of range"<<"\n";
    } 
    if (label) { *label = labels[ptidx]; }
    return points->host[ptidx].loc;
} // getPoint

template<typename T>
void ssrlcv::KDTree<T>::printKDTree() {
    printf("\nPRINTING KD TREE...\n\n");
    
    printf("NODES: \n\n");
    vector<Node> nodes = this->nodes;
    for (size_t i = 0; i < nodes.size(); i ++) {
        printf("Node %zu\n", i);
        printf("\tIndex: %d\n", nodes[i].idx);
        printf("\tIndex of Left Branch: %d\n", nodes[i].left);
        printf("\tIndex of Right Branch: %d\n", nodes[i].right);
    }
    printf("\n");

    printf("POINTS: \n\n");
    ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<T>>> points = this->points;
    for (size_t i = 0; i < points->size(); i++) {
        points->host[i].descriptor.print(); 
    }
    printf("\n");

    printf("LABELS: \n\n");
    vector<int> labels = this->labels;
    for (size_t i = 0; i < labels.size(); i++) {
        printf("%d\n", labels[i]);
    }
    printf("\n");
    
    printf("...DONE PRINTING\n\n");
} // printKDTree

/***************
* DEBUG METHOD *
****************/
ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> generateFeatures() {

    ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> features = ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>(nullptr,N,ssrlcv::cpu); 
    ssrlcv::Feature<ssrlcv::SIFT_Descriptor>* featureptr = features->host.get();

    for (int i = 0; i < N; i++) {
        featureptr[i].loc = {(float)i, -1.0f}; 
        // fill descriptor with 128 random nums
        for (int j = 0; j < K; j++) {
            unsigned char r = (unsigned char) rand();
            featureptr[i].descriptor.values[j] = r;
        }
        featureptr[i].descriptor.theta = 0.0f;
        featureptr[i].descriptor.sigma = 0.0f;
    } // for

    return features;
} // generatePoints

/**************
* MAIN METHOD *
***************/
int main() {

    ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor> matchFactory = ssrlcv::MatchFactory<ssrlcv::SIFT_Descriptor>(0.6f,200.0f*200.0f);

    std::vector<ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>>> allFeatures; 

    ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> img1 = generateFeatures(); 
    ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::Feature<ssrlcv::SIFT_Descriptor>>> img2 = generateFeatures();  
    allFeatures.push_back(img1);
    allFeatures.push_back(img2);

    // build a kd tree using the first image
    ssrlcv::KDTree<ssrlcv::SIFT_Descriptor> kdtree = ssrlcv::KDTree<ssrlcv::SIFT_Descriptor>(allFeatures[0]);

    // print descriptors
    // printf("\nIMAGE 1 DESCRIPTORS\n");
    // for (size_t i = 0; i < img1->size(); i++) {
    //     cout << "(x, y) = " << "(" << img1->host[i].loc.x << ", " << img1->host[i].loc.y << ")" << endl;
    //     img1->host[i].descriptor.print();  
    // } 
    // printf("\nIMAGE 2 DESCRIPTORS\n");
    // for (size_t i = 0; i < img2->size(); i++) {
    //     cout << "(x, y) = " << "(" << img1->host[i].loc.x << ", " << img1->host[i].loc.y << ")" << endl;
    //     img2->host[i].descriptor.print(); 
    // }

    // array to hold match pairs
    // change cpu to gpu when doing kenrel function
    //unsigned int numPossibleMatches = img2->size();
    //ssrlcv::Unity<ssrlcv::DMatch>* dmatches = new ssrlcv::Unity<ssrlcv::DMatch>(nullptr, numPossibleMatches, ssrlcv::cpu);

    //int nn = 1; // k nearest neighbors
    //int emax = 5; // the max number of leaf nodes to search

    ssrlcv::ptr::value<ssrlcv::Unity<float>> seedDistances = nullptr;
    ssrlcv::ptr::value<ssrlcv::Unity<ssrlcv::DMatch>> distanceMatches = matchFactory.generateDistanceMatches(0,kdtree,1,allFeatures[1],seedDistances);

    // do this on N separate CUDA threads
    for (int i = 0; i < N; i++) { 
        dmatches->host[i] = kdtree.findNearest(kdtree, img2->host[i], nn, emax);
        printf("\nBEST MATCH\n");
        printf("\tdist = %f\n", dmatches->host[i].distance);
        printf("\tlocation of point on img1 = {%f, %f}\n", dmatches->host[i].keyPoints[0].loc.x, dmatches->host[i].keyPoints[0].loc.y);
        printf("\tlocation of point on img2 = {%f, %f}\n", dmatches->host[i].keyPoints[1].loc.x, dmatches->host[i].keyPoints[1].loc.y);
    }

    // const unsigned char *vec;
    // const unsigned char *row;
    // for (int i = 0; i < N; i++) { 
    //     vec = img2->host[i].descriptor.values;

    //     for (int j = 0; j < N; j ++) {
    //         row = img1->host[j].descriptor.values;
    //         float d = 0.f; // reset d
    //         int k = 0;
    //         for (k = 0, d = 0.f; k < K; k++) {
    //             float t = vec[k] - row[k];
    //             d += t*t;
    //         }
    //         cout << "DISTANCE BETWEEN img1[" << j << "] and img2[" << i << "] = " << d << endl; 
    //     } 
 
    // } 

    // validate matches

    //delete img1, img2;
    //delete dmatches;
    return 0;
} // main