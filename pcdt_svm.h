/*
 * pcdt_svm.h
 *
 *  Created on: 02-Feb-2016
 *      Author: kd
 */

#ifndef SRC_PCDT_SVM_H_
#define SRC_PCDT_SVM_H_
#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <math.h>
#include <armadillo>
#include <sstream>
#include <pthread.h>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#define BIN_SIZE 10
#define MAX_LEVELS 5
#define print_log false
#define EPOCHS 100

const float epsilon = 0.00001;

//Compute Dominant Eigen Vector
void domEV(arma::fmat*, arma::fmat*);

/*PCDT SVM*/
extern uint m; // Train Data points
extern uint n; // Dimension of Train Data point
extern uint m_test; // Number of Test Data Points

//Node for decision tree
class node{
public:
    // Index of data point at this node
    arma::Col<arma::uword> index;
    arma::Col<arma::uword> test_index;
    node* child[BIN_SIZE]; //Children of node
    arma::Mat<float>* dom_eigen;
    CvSVM *svm; //SVM for the node
    CvSVMParams* params; //Parameters for SVM
    bool isLeaf; //Is this node a leaf?
    bool isEmpty; //Is this node a empty?
    bool all_positive; //All data points at this node has label +1
    bool all_negative; //All data points at this node has label -1
    float min;
    float max;
    node(); //Constructor
    ~node(); //Destructor
}typedef node;
extern std::vector<node*> svm_nodes;
struct custom{
  node* current_node;
  arma::fmat* data;
  arma::fmat* labels;
};

void* train_svm(void*);
void create_decision_tree(node*, int, arma::fmat*,arma::fvec*);
void traverse_decision_tree(node*, int, arma::fmat*, arma::fvec*);
void decision_tree_populate(node*, int, arma::fmat*);
float compute_accuracy(arma::fvec*, arma::fvec*, int);
float compute_precision(arma::fvec*, arma::fvec*, int);
float compute_recall(arma::fvec*, arma::fvec*, int);
float compute_fScore(float, float);


#endif /* SRC_PCDT_SVM_H_ */
