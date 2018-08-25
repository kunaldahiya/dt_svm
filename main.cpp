#include "pcdt_svm.h"
#define NUM_THREADS 50
using namespace arma;
using namespace std;
vector<node*> svm_nodes;
uint m,n,m_test;
int main(int argc, char *argv[]){
    /*if ( argc != 3 ) {
      cout << "Correct usage : <pcdt_svm> <train_file> <test_file> <tune_params(Y:Yes,N:No/Use Default)>" << endl;
      exit(-1);
    }*/
    if(MAX_LEVELS==0){
    	cout << "Invalid number of levels. Exiting.." << endl;
    	exit(-1);
    }
    pthread_t threads[NUM_THREADS];
    time_t start_time, end_time;
    pthread_attr_t attr;
    void *status;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
    /* Training time : Training SVM
       Partitioning time : Creating the decision tree
    */
    clock_t begin_reading,end_reading,begin_training,end_training,begin_partitioning,end_partitioning,
            begin_testing,end_testing,begin_total,end_total;
    double reading_time = 0, training_time = 0, testing_time =0, elapsed_time = 0, partitioning_time = 0;
    fmat train_data, test_data;
    begin_total = clock();
    begin_reading = clock();
    //Read Training File
    train_data.load(argv[1]);
    end_reading = clock();
    m = train_data.n_rows;
    n = train_data.n_cols;
    fvec train_labels = train_data.col(n-1);
    train_data.shed_col(n-1);
    n--;
    cout << "Training Data Size : " << m << ", " << n <<endl;
    reading_time = end_reading-begin_reading;
  /*-----------------------------------------------------------------------*/
    begin_partitioning = clock();
    time(&start_time);
    node* root;
    root = new node;
    (root->index).set_size(m);
    for(int i=0; i<m; i++){
        root->index(i)=i;
    }
    root->isEmpty = false;
    //Count number of positive and negative examples
    int neg_count =0;
    int pos_count =0;
    for( int i=0;i<m;i++){
        if(train_labels((root->index)(i))==-1){
              neg_count++;
        }
        else if(train_labels((root->index)(i))==1){
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
      create_decision_tree(root, 0, &train_data, &train_labels);
    }
    end_partitioning = clock();
    cout << "Decision Tree Completed !!" << endl;
    cout << "Training of SVM started, Number of SVM's: " <<svm_nodes.size()<<endl;
    begin_training = clock();
    pthread_attr_destroy(&attr);
    struct custom t[NUM_THREADS];
    for(int i=0;i<svm_nodes.size();i+=NUM_THREADS){
      for(int j=0;j<NUM_THREADS && i+j<svm_nodes.size();j++){
          t[j].current_node = svm_nodes[i+j];
          t[j].data = &train_data;
          t[j].labels = &train_labels;
          pthread_create(&threads[j],NULL,train_svm,(void *) &t[j]);
      }
      for(int j=0;j<NUM_THREADS && i+j<svm_nodes.size();j++){
          pthread_join(threads[j],&status);
      }
      cout << "Completed Batch : " << i <<endl;
    }
    cout << "Training of SVM finished" << endl;
    end_training = clock();
    training_time = end_training - begin_training;
    partitioning_time = end_partitioning - begin_partitioning;
    //Get total number of support vectors
    int num_sv = 0;
    for(int i = 0; i < svm_nodes.size(); i++){
      num_sv = num_sv + (svm_nodes[i]->svm)->get_support_vector_count();
    }
    cout << "Total number of SV's : " << num_sv << endl;
    // --------------------Read test file-------------------------------------
    begin_reading = clock();
    test_data.load(argv[2]);
    end_reading = clock();
    reading_time = reading_time+end_reading-begin_reading;
    m_test = test_data.n_rows;
    //Handle the wrong data
    if(m_test==0 || test_data.n_cols ==0){
        cout << "There are no data points" <<endl;
        exit(-1);
    }else if(n!=test_data.n_cols-1){
    	cout << "Dimension mismatch" <<endl;
    	exit(-1);
    }
    fvec test_labels = test_data.col(n);
    test_data.shed_col(n);

/*------------Read PCA Test Data & labels-----------------------------------------*/
    begin_testing = clock();
    (root->test_index).set_size(m_test);
    for(int i=0; i<m_test; i++){
        root->test_index(i)=i;
    }
    //Check if all points has same label at root node
    //if not create a decision tree with this as root
    decision_tree_populate(root,0,&test_data);
    end_testing = clock();
    testing_time = end_testing - begin_testing;
    cout << "Populating test data completed" <<endl;
    fvec predicted_labels(m_test);
    /*-----------------Test Data File (Without PCA)---------------------------*/
    begin_testing = clock();
    traverse_decision_tree(root, 0, &test_data, &predicted_labels);
    end_testing = clock();
    end_total = clock();
    testing_time = testing_time + end_testing - begin_testing;
    cout << "Prediction completed" <<endl;

    //Compute different metrices
    float accuracy = compute_accuracy(&test_labels, &predicted_labels, m_test);
    float precision = compute_precision(&test_labels, &predicted_labels, m_test);
    float recall = compute_recall(&test_labels, &predicted_labels, m_test);
    float fScore = compute_fScore(precision, recall);

    elapsed_time = double(end_total - begin_total) / CLOCKS_PER_SEC;
    training_time = double(training_time) / CLOCKS_PER_SEC;
    partitioning_time = double(partitioning_time)/ CLOCKS_PER_SEC;
    testing_time = double(testing_time) / CLOCKS_PER_SEC;
    reading_time = double(reading_time) / CLOCKS_PER_SEC;
    cout << "Elapsed Time : " <<elapsed_time <<endl;
    cout << "Reading Time : " <<reading_time <<endl;
    cout << "Partitioning Time : " <<partitioning_time <<endl;
    cout << "Training Time : " <<training_time <<endl;
    cout << "Testing Time : " <<testing_time <<endl;
    return 0;
}
