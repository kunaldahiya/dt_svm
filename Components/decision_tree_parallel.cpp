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
$ g++ decision_tree.cpp -o decision_tree -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lpthread
Run
$ ./decision_tree
*/
#define print_log false
#define BIN_SIZE 10 //Data to splitted in
#define NUM_THREADS 150
#define BLOCK_SIZE 16
#define MEGEXTRA 1000000
#define MAX_LEVELS 3
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

class node{
public:
    // Index of data point at this node
    vector <int> index;
    vector <int> test_index;
    node* child[BIN_SIZE];
    CvSVM *svm;
    CvSVMParams* params;
    bool isLeaf;
    float min;
    float max;
    bool isEmpty;
    bool all_positive;
    bool all_negative;
    node(){
      max = 0.0;
      min = 0.0;
      all_positive = false;
      all_negative = false;
      isLeaf = false;
      params = NULL;
      svm = NULL;
      isEmpty = true;
      for(int i=0; i<BIN_SIZE; i++){
          child[i] = NULL;
      }
    }
    ~node(){
      delete[] child;
      delete svm;
      delete params;
    }
    /*Check if node is leaf node
    Not checking via level but childs */

    /* data */
};
vector<node*> svm_nodes;
struct custom{
  node* current_node;
  Mat* data;
  Mat* labels;
};

//Train SVM
void* train_svm(void * r){
  custom* t = (struct custom*) r;
  int size = ((t->current_node)->index).size() ;
  float labels[size] ;
  for(int i=0;i<size;i++){
    labels[i] = (t->labels)->at<int>(((t->current_node)->index)[i],0) ;
  }
  Mat labelsMat(size, 1, CV_32S, labels);
  Mat trainingData(size,n,CV_32F);
  for(int i=0;i<size;i++){
    for(int j=0;j<n;j++){
      trainingData.at<float>(i,j) = (t->data)->at<float>((t->current_node)->index[i],j);
    }
  }
  ((t->current_node)->svm)->train(trainingData, labelsMat, Mat(), Mat(), *((t->current_node)->params));
  pthread_exit(NULL);
}


void create_decision_tree(node* root, int level, Mat* train_labels,Mat* data_train_red,Mat* data_train){
     int size = (root->index).size();
     cout << "Level : " << level<<" ,Points : "<<size <<endl;
     //Check if node is empty
     if(size==0){
       root->isLeaf = true;
       return;
     } else{
       root->isEmpty = false;
     }
     //Return if highest number of levels is reached
     if(level==MAX_LEVELS){
       return;
     }
     // Train SVM
     else if(level==(MAX_LEVELS-1) || size <=1000){
       root->svm = new CvSVM();
       //CvSVMParams params;
       root->params = new CvSVMParams();
       (root->params)->svm_type    = CvSVM::C_SVC;
       (root->params)->kernel_type = CvSVM::RBF;
       (root->params)->gamma = 32; // forpoly/rbf/sigmoid
       (root->params)->C = 32; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
       (root->params)->term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
       (root->params)->term_crit.max_iter = 100000;
       (root->params)->term_crit.epsilon = 1e-3;
       svm_nodes.push_back(root);
       //float trainingData[49000][n];
       //train_data(&(root->index), root->svm, root->params, train_labels, data_train);
       if(print_log){
         cout << "\nSVM Trained for level : " << level << endl;
       }
       root->isLeaf = true;
       return;
     }
     float min,max,temp;
     min = data_train_red->at<float>(level,root->index[0]);
     min = data_train_red->at<float>(level,root->index[0]);
     for(int i=0;i<size;i++){
       temp = data_train_red->at<float>(level,root->index[i]);
       if(temp>max){
         max = temp;
       }
       if(temp<min){
         min = temp;
       }
     }
     root->max = max;
     root->min = min;
     //Divide data in bins
      vector<vector<int> > matrix(BIN_SIZE, vector<int>(size));
     int count[BIN_SIZE];
     for(int i=0;i<BIN_SIZE;i++){
       count[i] = 0;
     }
     int bin;
     for(int i=0;i<size;i++){
       //Get the bin for this index
        temp = data_train_red->at<float>(level,(root->index)[i]);
        temp = (temp-(min))*BIN_SIZE/(max-min);
        bin = ceil(temp);
        if(bin<=0){
          bin = 1;
        }else if(bin>BIN_SIZE){
          bin = BIN_SIZE;
        }
        matrix[bin-1][count[bin-1]++] = (root->index)[i];
     }
     int neg_count =0;
     int pos_count =0;
     vector<int> v;
     for(int i = 0; i < BIN_SIZE; ++i){
        neg_count = 0;
        pos_count = 0;
        root->child[i] = new node;
        v.resize(count[i]);
        for(int j=0;j<count[i];j++){
           v[j] = matrix[i][j];
           if(train_labels->at<int>(v[j],0)==-1){
              neg_count++;
           }
           else if(train_labels->at<int>(v[j],0)==1){
              pos_count++;
           }
         }
         ((root->child[i])->index).insert(((root->child[i])->index).end(),v.begin(),v.end());
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
          v.clear();
        }
    return ;
}

void traverse_decision_tree(node* root, int level, Mat* predicted_labels, Mat* data_test){
  if(print_log){
    cout << "Traversing level : " <<level<<endl;
  }
  int size = root->test_index.size();

  if(root->isLeaf){
    //If node is empty
    //Can't predict, assigning zero
    if(root->isEmpty){
      if(print_log){
        cout << "Hit empty node, labelling all as zero" <<endl;
      }
      for(int i=0;i<size;i++){
        predicted_labels->at<int>(root->test_index[i],0) = 0;
      }
    }
    //All Positive
    if(root->all_positive){
      if(print_log){
        cout << "All Positive, labelling all as positive" <<endl;
      }
      for(int i=0;i<size;i++){
        predicted_labels->at<int>(root->test_index[i],0) = 1;
      }
    }
    //All Negative
    else if(root->all_negative){
      if(print_log){
        cout << "All Negative, labelling all as negative" <<endl;
      }
      for(int i=0;i<size;i++){
        predicted_labels->at<int>(root->test_index[i],0) = -1;
      }
    }
    // Use SVM to predict
    else{
      if(print_log){
        cout << "Hit the leaf, using SVM" <<endl;
        if((root->svm==NULL)){
          cout <<"WTF!! This shouldn't happen" <<endl;
          return;
        }
      }
       int current_index;
       for(int i=0;i<size;i++){
         float temp[n_test];
         current_index = root->test_index[i];
         for(int j=0;j<n_test;j++){
           temp[j] = data_test->at<float>(current_index,j);
         }
         Mat testVector(1,n_test,CV_32F,temp) ;
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
void decision_tree_pop(node* root, int level, Mat* test_red){
     if(print_log){
        cout << "At level (): " << level << endl;
      }
     /*Check if need to continue
     Break if node is leaf*/
     if(root->isLeaf){
        return;
     }
     int size = (root->test_index).size();
     // Matrix BIN_SIZE*size
     // A row(vector) for each bin
     vector<vector<int> > matrix(BIN_SIZE, vector<int>(size));
     //Elements in each bin
     int count[BIN_SIZE];
     for(int i=0;i<BIN_SIZE;i++){
       count[i] = 0;
     }
     float temp;
     int bin;
     for(int i=0;i<size;i++){
       //Get the bin for this index
        temp = test_red->at<float>(level,(root->test_index)[i]);
        temp = (temp-(root->min))*BIN_SIZE/(root->max-root->min);
        bin = ceil(temp);
        if(bin<=0){
          bin = 1;
        }else if(bin>BIN_SIZE){
          bin = BIN_SIZE;
        }
        matrix[bin-1][count[bin-1]++] = (root->test_index)[i];
     }
     vector<int> v;
     for(int i = 0; i < BIN_SIZE; i++){
        node* current_node = root->child[i];
        v.resize(count[i]);
        for(int j=0;j<count[i];j++){
           v[j] = matrix[i][j];
         }
         current_node->test_index.insert(current_node->test_index.end(), v.begin(), v.end());
         decision_tree_pop(current_node,level+1,test_red);
         v.clear();
     }
    return ;
}

/*----------------------------------------------------------------------------------*/
int main(){
    CvMLData dataFile,dataFile2,dataFile3,dataFile4;
    pthread_t threads[NUM_THREADS];
    pthread_attr_t attr;
    void *status;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
    clock_t begin_reading,end_reading,begin_training,end_training,begin_testing,end_testing,begin_total,end_total;
    double reading_time = 0, training_time = 0, testing_time =0, elapsed_time = 0;
    begin_total = clock();
    begin_reading = clock();
    /*------------Read PCA Train Data & labels-----------------------------------------*/
     if (dataFile.read_csv("/home/kd/PCDT_SVM/covtype/train_red.csv") != 0){
        fprintf(stderr, "Can't read csv file %s\n", "PCA Train");
        return -1;
    }
    Mat data_train_red(dataFile.get_values());
    n_red = data_train_red.cols;
    m = data_train_red.rows;
    cout << "Training Data Size (with PCA) : " << m << " " <<n_red<<endl;
    /*-----------------Data File (Without PCA)-------------------------------*/
    if (dataFile2.read_csv("/home/kd/PCDT_SVM/covtype/train.csv") != 0){
        fprintf(stderr, "Can't read csv file %s\n", "Train");
        return -1;
    }
    Mat data_train(dataFile2.get_values());
    n = data_train.cols;
    dataFile2.set_response_idx(n-1);
    n--;
    data_train = data_train.colRange(0,n).rowRange(0,m);
    Mat train_labels(dataFile2.get_responses());
    train_labels.convertTo(train_labels, CV_32S);
    cout << "Training Data Size : " << m << " " <<n<<endl;

    //Test if data is empty
    if(m==0 || n==0 || n_red==0){
      cout << "Empty train data, Exiting!!"<<endl;
      return -1;
    }
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
    root->isEmpty = false;
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
    cout << "Decision Tree Completed !!" << endl;
    cout << "Training of SVM started, Number of SVM's: " <<svm_nodes.size()<<endl;
    pthread_attr_destroy(&attr);
    struct custom t[NUM_THREADS];
    for(int i=0;i<svm_nodes.size();i+=NUM_THREADS){
      for(int j=0;j<NUM_THREADS && i+j<svm_nodes.size();j++){
          t[j].current_node = svm_nodes[i+j];
          t[j].data = &data_train;
          t[j].labels = &train_labels;
          pthread_create(&threads[j],NULL,train_svm,(void *) &t[j]);
      }
      for(int j=0;j<NUM_THREADS && i+j<svm_nodes.size();j++){
          pthread_join(threads[j],&status);
      }
      cout << "Completed Batch : " << i <<endl;
    }
    cout << "Training of SVM finished" << endl;
    //return 0;
    end_training = clock();
    training_time = end_training - begin_training;
/* --------------------Read test file-------------------------------------*/
/*------------Read PCA Test Data & labels-----------------------------------------*/
    begin_reading = clock();
    if (dataFile3.read_csv("/home/kd/PCDT_SVM/covtype/test_red.csv") != 0){
        fprintf(stderr, "Can't read csv file %s\n", "PCA Train");
        return -1;
    }
    Mat data_test_red(dataFile3.get_values());
    n_red_test = data_test_red.cols;
    m_test = data_test_red.rows;
    cout << "Test Data Size (with PCA) : " << m_test << " " <<n_red_test<<endl;
    end_reading = clock();
    reading_time = reading_time + end_reading - begin_reading ;
    //Test if data is empty
    if(m_test==0 || n_red_test==0){
      cout << "Empty reduced test data, Exiting!!"<<endl;
      return -1;
    }
    begin_testing = clock();
    (root->test_index).resize(m_test);
    for(int i=0; i<m_test; i++){
        root->test_index[i]=i;
    }
    //Check if all points has same label at root node
    //if not create a decision tree with this as root
    decision_tree_pop(root,0,&data_test_red);
    end_testing = clock();
    testing_time = end_testing - begin_testing;
    cout << "Populating test data completed" <<endl;

    /*-----------------Test Data File (Without PCA)---------------------------*/
    if (dataFile4.read_csv("/home/kd/PCDT_SVM/covtype/test.csv") != 0){
        fprintf(stderr, "Can't read csv file %s\n", "Test");
        return -1;
    }
    Mat data_test(dataFile4.get_values()); // Default data type is float
    n_test = data_test.cols;
    m_test = data_test.rows;
    dataFile4.set_response_idx(n_test-1);
    n_test--;
    cout << "Test Data Size : " << m_test << " " <<n_test<<endl;
    data_test = data_test.colRange(0,n_test).rowRange(0,m_test);
    Mat test_labels(dataFile4.get_responses());
    end_reading = clock();
    reading_time = reading_time + end_reading - begin_reading;
    /*------------------------------------------------------------------------*/
    //Test if data is empty
    if(m_test==0 || n_test==0){
      cout << "Empty test data, Exiting!!"<<endl;
      return -1;
    }
    Mat predicted_labels(m_test,1,CV_32S);
    begin_testing = clock();
    traverse_decision_tree(root,0,&predicted_labels,&data_test);
    end_testing = clock();
    testing_time = testing_time + end_testing - begin_testing;
    cout << "Prediction completed" <<endl;
    int accurate_count = 0;
    int assignment_failed = 0;
    for(int i=0;i<m_test;i++){
      if(predicted_labels.at<float>(i,0) == 0){
        assignment_failed++;
      }
      if (test_labels.at<float>(i,0) == predicted_labels.at<float>(i,0)){
        accurate_count ++;
      }
    }
    cout << "Accuracy is : " << float(accurate_count)/m_test <<endl;
    cout << "Couldn't classify : " << assignment_failed << " test samples" <<endl;
    end_total = clock();
    elapsed_time = double(end_total - begin_total) / CLOCKS_PER_SEC;
    training_time = double(training_time) / CLOCKS_PER_SEC;
    testing_time = double(testing_time) / CLOCKS_PER_SEC;
    reading_time = double(reading_time) / CLOCKS_PER_SEC;
    cout << "Elapsed Time : " <<elapsed_time <<endl;
    cout << "Reading Time : " <<reading_time <<endl;
    cout << "Training Time : " <<training_time <<endl;
    cout << "Testing Time : " <<testing_time <<endl;
    pthread_exit(NULL);
    return 0;
}
