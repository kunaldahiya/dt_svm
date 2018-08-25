/*Working File
  Developed by : Kunal
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <unistd.h>
#include <ctime>
#include <pthread.h>
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
/*Compile
$ g++ decision_tree.cpp -o decision_tree -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml
Run
$ ./decision_tree
*/
#define print_log false
#define BIN_SIZE 10 //Data to splitted in
#define NUM_THREADs 5
#define BLOCK_SIZE 16
#define PI 3.141592654
#define MEGEXTRA 1000000
#define MAX_LEVELS 4
using namespace cv;
using namespace std;
/*Data Matrix mxn
Label mx1*/
int m;
int n_red;
int n;
int m_test;
int n_test;
int n_red_test;
Mat* x;
Mat* y;
class node{
public:
    // Index of data point at this node
    vector <int> index;
    vector <int> test_index;
    node* child[BIN_SIZE];
    CvSVM *svm;
    bool isLeaf;
    bool all_positive;
    bool all_negative;
    node(){
      all_positive = false;
      all_negative = false;
      isLeaf = false;
      svm = NULL;
      for(int i=0; i<BIN_SIZE; i++){
          child[i] = NULL;
      }
    }
    ~node(){
      delete[] child;
      delete svm;
    }
    /*Check if node is leaf node
    Not checking via level but childs */

    /* data */
};

class sortable_data{
public:
    int value;
    int level;
};

bool myfunction (sortable_data a,sortable_data b) {
   return (x->at<float>(a.value,a.level)<x->at<float>(b.value,b.level));
}

bool myfunction2 (sortable_data a,sortable_data b) {
   return (y->at<float>(a.value,a.level)<y->at<float>(b.value,b.level));
}


//Train SVM
void train_data(vector<int> * v, CvSVM* SVM, CvSVMParams params, Mat* train_labels,Mat* data_train){
  int size = (*v).size() ;
  float labels[size] ;
  for(int i=0;i<size;i++){
    labels[i] = train_labels->at<int>((*v)[i],0) ;
  }
  Mat labelsMat(size, 1, CV_32S, labels);
  Mat trainingData(size,n,CV_32F);
  for(int i=0;i<size;i++){
    for(int j=0;j<n;j++){
      trainingData.at<float>(i,j) = data_train->at<float>((*v)[i],j);
    }
  }
    // Train the SVM
  SVM->train(trainingData, labelsMat, Mat(), Mat(), params);
  //delete [] trainingData;
  return ;
}


void create_decision_tree(node* root, int level, Mat* train_labels,Mat* data_train_red,Mat* data_train){
     int size = (root->index).size();
     //Return if highest number of levels is reached
     if(level==MAX_LEVELS){
       return;
     }
     // Train SVM
     else if(level==(MAX_LEVELS-1)){
       root->svm = new CvSVM();
       CvSVMParams params;
       params.svm_type    = CvSVM::C_SVC;
       params.kernel_type = CvSVM::RBF;
       params.gamma = 2; // for poly/rbf/sigmoid
       params.C = 32; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
       params.term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
       params.term_crit.max_iter = 100000;
       params.term_crit.epsilon = 1e-6;
       //float trainingData[49000][n];
       train_data(&(root->index), root->svm, params, train_labels, data_train);
       if(print_log){
         cout << "\nSVM Trained for level : " << level << endl;
       }
       root->isLeaf = true;
       return;
     }
     vector<sortable_data> myvec;
     myvec.resize(size);

     //Sort values based on pca component value
     for(int i=0;i<size;i++){
         myvec[i].value = (root->index)[i];
         myvec[i].level = level;
     }
     sort (myvec.begin(), myvec.end(), myfunction);
     for(int i=0;i<size;i++){
          //sorted values at node
          //level remains same
         (root->index)[i] = myvec[i].value;
     }
     //Divide data in bins
     int count[BIN_SIZE];
     //data samples per bin
     int base_count = size/BIN_SIZE;
     for(int i=0;i<BIN_SIZE;i++){
        count[i]=base_count;
     }
     int remaining = size -(base_count*BIN_SIZE);
     for(int i=0;i<(remaining);i++){
         count[i]++;
     }
     int temp_count_sum = 0;
     for(int i=0;i<BIN_SIZE;i++){
         temp_count_sum+=count[i];
     }
     if(temp_count_sum!=size){
         if(print_log){
           cout << "Unable to split into bins" <<endl;
         }
     }
     int k=0;
     int neg_count =0;
     int pos_count =0;
     for(int i = 0; i < BIN_SIZE; ++i){
        neg_count = 0;
        pos_count = 0;
        root->child[i] = new node;
        for(int j=0;j<count[i];j++){
           ((root->child[i])->index).push_back((root->index)[k]);
           if(train_labels->at<int>((root->index)[k],0)==-1){
              neg_count++;
           }
           else if(train_labels->at<int>((root->index)[k],0)==1){
              pos_count++;
           }
           k++;
         }
          if(neg_count==0){
             if(print_log){
                printf("neg ending decision_tree at level %d branch %d\n",level+1,i);
             }
             (root->child[i])->all_positive = true;
             (root->child[i])->isLeaf = true;
          }
          else if(pos_count == 0){
            if(print_log){
               printf("pos ending decision_tree at level %d branch %d\n",level+1,i);
          }
             (root->child[i])->all_negative = true;
             (root->child[i])->isLeaf = true;
          }
          else{
            if(print_log){
              cout << "Creating child : " << i << " at level : " << level+1<<endl;
            }
             create_decision_tree(root->child[i],level+1,train_labels,data_train_red,data_train);
          }
        }
    return ;
}

void traverse_decision_tree(node* root, int level, Mat* predicted_labels, Mat* data_test){
  if(print_log){
    cout << "Traversing level : " <<level<<endl;
  }
  int size = root->test_index.size();

  if(root->isLeaf){
    //All Positive
    if(root->all_positive){
      if(print_log){
        cout << "All Positive, labelling all positive" <<endl;
      }
      for(int i=0;i<size;i++){
        predicted_labels->at<int>(root->test_index[i],0) = 1;
      }
    }
    //All Negative
    else if(root->all_negative){
      if(print_log){
        cout << "All Negative, labelling all negative" <<endl;
      }
      for(int i=0;i<size;i++){
        predicted_labels->at<int>(root->test_index[i],0) = -1;
      }
    }
    //SVM
    else{
      if(print_log){
        cout << "Hit the leaf, using SVM" <<endl;
        /*if(root->svm==NULL){
          cout <<"WTF!! This shouldn't happen" <<endl;
          return;
        }*/
      }
       int current_index;
       for(int i=0;i<size;i++){
         float temp[n_test];
         current_index = root->test_index[i];
         for(int j=0;j<n_test;j++){
           temp[j] = data_test->at<float>(current_index,j);
         }
         Mat testVector(n_test,1,CV_32F,temp) ;
         predicted_labels->at<int>(current_index,0) = root->svm->predict(testVector);
       }
       return;
     }//SVM
  }//leaf node
  for(int i=0;i<BIN_SIZE;i++){
     node* current_node = root->child[i];
     if(current_node!=NULL){
       traverse_decision_tree(current_node, level+1, predicted_labels, data_test);
     }
  }
     return;
}


/*---------------------------------------------------------------------------------*/
void decision_tree_pop(node* root, int level){
     if(print_log){
        cout << "At level (): " << level << endl;
      }
     /*Check if need to continue
     Break if node is leaf*/
     if(root->isLeaf){
        return;
     }

     int size = (root->test_index).size();
     vector<sortable_data> myvec;
     myvec.resize(size);

     //Sort values based on pca component value
     for(int i=0;i<size;i++){
         myvec[i].value = (root->test_index)[i];
         myvec[i].level = level;
     }
     sort (myvec.begin(), myvec.end(), myfunction2);
     for(int i=0;i<size;i++){
          //sorted values at node
          //level remains same
         (root->test_index)[i] = myvec[i].value;
     }
     //Divide data in bins
     int count[BIN_SIZE];
     //data samples per bin
     int base_count = size/BIN_SIZE;
     for(int i=0;i<BIN_SIZE;i++){
        count[i]=base_count;
     }
     int remaining = size -(base_count*BIN_SIZE);
     for(int i=0;i<(remaining);i++){
         count[i]++;
     }
     int temp_count_sum = 0;
     for(int i=0;i<BIN_SIZE;i++){
         temp_count_sum+=count[i];
     }
     if(temp_count_sum!=size){
         if(print_log){
           cout << "Unable to split into bins" <<endl;
         }
     }
     int k=0;
     for(int i = 0; i < BIN_SIZE; ++i){
        node* current_node = root->child[i];
        for(int j=0;j<count[i];j++){
           (current_node->test_index).push_back((root->test_index)[k]);
           k++;
         }
            decision_tree_pop(root->child[i],level+1);
     }
    return ;
}
/*----------------------------------------------------------------------------------*/
int main(){
    CvMLData dataFile,dataFile2,dataFile3,dataFile4;
    clock_t begin_reading,end_reading,begin_training,end_training,begin_testing,end_testing,begin_total,end_total;
    double reading_time = 0, training_time = 0, testing_time =0, elapsed_time = 0;
    begin_total = clock();
    begin_reading = clock();
    /*------------Read PCA Train Data & labels-----------------------------------------*/
     if (dataFile.read_csv("/home/kd/PCDT_SVM/HIGGS/train_red.csv") != 0){
        fprintf(stderr, "Can't read csv file %s\n", "PCA Train");
        return -1;
    }
    Mat data_train_red(dataFile.get_values());
    n_red = data_train_red.cols;
    m = data_train_red.rows;
    x = &data_train_red;
    cout << "Training Data Size (with PCA) : " << m << " " <<n_red<<endl;
    /*-----------------Data File (Without PCA)-------------------------------*/
    if (dataFile2.read_csv("/home/kd/PCDT_SVM/HIGGS/train.csv") != 0){
        fprintf(stderr, "Can't read csv file %s\n", "Train");
        return -1;
    }
    Mat data_train(dataFile2.get_values());
    n = data_train.cols;
    dataFile2.set_response_idx(n-1);
    data_train = data_train.colRange(0,n-1).rowRange(0,m);
    Mat train_labels(dataFile2.get_responses());
    train_labels.convertTo(train_labels, CV_32S);
    cout << "Training Data Size : " << m << " " <<n<<endl;
    /*-----------------------------------------------------------------------*/
    end_reading = clock();
    reading_time = end_reading-begin_reading;

    begin_training = clock();
    node* root;
    root = new node;
    (root->index).resize(m);
    for(int i=0; i<m; i++){
        root->index[i]=i;
    }
    //Count number of positive and negative examples
    int neg_count =0;
    int pos_count =0;
    for( int i=0;i<m;i++){
        if(train_labels.at<int>((root->index)[i],0)==-1){
              neg_count++;
        }
        else if(train_labels.at<int>((root->index)[i],0)==1){
              pos_count++;
        }
    }
    //Check if all points has same label at root node
    //if not create a decision tree with this as root
    if(neg_count == 0){
      if(print_log){
        cout << "Neg ending decision_tree at level 0 branch 0" <<endl;
      }
      root->all_positive = true;
      root->isLeaf = true;
    }
    else if(pos_count == 0){
      if(print_log){
        cout << "Pos ending decision_tree at level 0 branch 0" <<endl;
      }
      root->all_negative = true;
      root->isLeaf = true;
    }
    else{
      cout << "Decision Tree started" << endl;
      create_decision_tree(root,0,&train_labels,&data_train_red,&data_train);
    }
    cout << "Decision Tree Completed !!" <<endl;
    end_training = clock();
    training_time = end_training - begin_training;
/* --------------------Read test file-------------------------------------*/
/*------------Read PCA Test Data & labels-----------------------------------------*/
    begin_reading = clock();
    if (dataFile3.read_csv("/home/kd/PCDT_SVM/HIGGS/test_red.csv") != 0){
        fprintf(stderr, "Can't read csv file %s\n", "PCA Train");
        return -1;
    }
    Mat data_test_red(dataFile3.get_values());
    n_red_test = data_test_red.cols;
    m_test = data_test_red.rows;
    cout << "Test Data Size (with PCA) : " << m_test << " " <<n_red_test<<endl;
    y = &data_test_red;
    end_reading = clock();
    reading_time = reading_time + end_reading - begin_reading ;
    begin_testing = clock();
    (root->test_index).resize(m_test);
    for(int i=0; i<m_test; i++){
        root->test_index[i]=i;
    }
    //Check if all points has same label at root node
    //if not create a decision tree with this as root
    decision_tree_pop(root,0);
    end_testing = clock();
    testing_time = end_testing - begin_testing;
    cout << "Populating test data completed" <<endl;

    /*-----------------Test Data File (Without PCA)---------------------------*/
    if (dataFile4.read_csv("/home/kd/PCDT_SVM/HIGGS/test.csv") != 0){
        fprintf(stderr, "Can't read csv file %s\n", "Test");
        return -1;
    }
    Mat data_test(dataFile4.get_values()); // Default data type is float
    n_test = data_test.cols;
    m_test = data_test.rows;
    cout << "Test Data Size : " << m_test << " " <<n_test<<endl;
    dataFile4.set_response_idx(n-1);
    data_test = data_test.colRange(0,n_test-1).rowRange(0,m_test);
    Mat test_labels(dataFile4.get_responses());
    //train_labels.convertTo(train_labels, CV_32S);
    end_reading = clock();
    reading_time = reading_time + end_reading - begin_reading;
    /*------------------------------------------------------------------------*/
    Mat predicted_labels(m_test,1,CV_32S);
    begin_testing = clock();
    traverse_decision_tree(root,0,&predicted_labels,&data_test);
    end_testing = clock();
    testing_time = testing_time + end_testing - begin_testing;
    cout << "Prediction completed" <<endl;
    int accurate_count = 0;
    for(int i=0;i<m_test;i++){
      if (test_labels.at<float>(i,0) == predicted_labels.at<float>(i,0)){
        accurate_count ++;
      }
    }
    cout << "Accuracy is : " << float(accurate_count)/m_test <<endl;
    end_total = clock();
    elapsed_time = double(end_total - begin_total) / CLOCKS_PER_SEC;
    training_time = double(training_time) / CLOCKS_PER_SEC;
    testing_time = double(testing_time) / CLOCKS_PER_SEC;
    reading_time = double(reading_time) / CLOCKS_PER_SEC;
    cout << "Elapsed Time : " <<elapsed_time <<endl;
    cout << "Reading Time : " <<reading_time <<endl;
    cout << "Training Time : " <<training_time <<endl;
    cout << "Testing Time : " <<testing_time <<endl;
    return 0;
}

