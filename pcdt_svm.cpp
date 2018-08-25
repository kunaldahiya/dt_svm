#include "pcdt_svm.h"

using namespace std;
using namespace cv;
using namespace arma;
node::node(){
  max = 0.0;
  min = 0.0;
  dom_eigen = NULL;
  all_positive = false;
  all_negative = false;
  isLeaf = false;
  //params = NULL;
  //svm = NULL;
  isEmpty = true;
  for(int i=0; i<BIN_SIZE; i++){
      child[i] = NULL;
  }
}

node::~node(){
  delete[] child;
  delete svm;
  delete params;
}

void domEV(fmat* cov, fmat* v){
	fmat temp_mat, temp_mat2;
	float temp;
	for(int i=0;i<EPOCHS;i++){
		temp_mat = (*cov) * (*v);
		temp_mat2 = arma::sqrt(temp_mat.t()*temp_mat);
		temp = temp_mat2(0,0);
		temp_mat = temp_mat/temp;
		temp_mat2 = arma::abs(temp_mat - *v);
		*v = temp_mat;
		if(!any(vectorise(temp_mat2)>epsilon)){
			break;
		}

	}
}

void create_decision_tree(node* root, int level ,fmat* train_data, fvec* train_labels){
     int size = (root->index).n_rows;
     if(print_log){
     	cout << "Level : " << level<<", Points : "<<size <<endl;

     }
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
       CvSVMParams params;
       root->params = new CvSVMParams();
       (root->params)->svm_type = CvSVM::C_SVC;
       (root->params)->kernel_type = CvSVM::RBF;
       //(root->params)->degree =64;
       (root->params)->gamma = 4.768e-7; // forpoly/rbf/sigmoid
       (root->params)->C =1; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
       (root->params)->term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
       (root->params)->term_crit.max_iter = 100000;
       (root->params)->term_crit.epsilon = 1e-3;
       svm_nodes.push_back(root);
       if(print_log){
         cout << "\nSVM Trained for level : " << level << endl;
       }
       root->isLeaf = true;
       return;
     }
     float temp;
     //Compute Covariance Matrix
     fmat cov_mat, current_data;
     root->dom_eigen = new fmat();
     if(level!=0){
        arma::uvec indices(root->index);
     	current_data = (*train_data).rows(indices);
     	cov_mat = cov(current_data,current_data);
     }else{
     	cov_mat = cov(*train_data,*train_data);
     }
     root->dom_eigen->ones(cov_mat.n_cols,1);
     domEV(&cov_mat,root->dom_eigen);
     /*Don't need it further
      * Free memory
     */
     cov_mat = fmat(0,0);
     //Compute value of each data point in direction of max variance
     fmat projected_values;
     if(level!=0){
     	projected_values = current_data * (*root->dom_eigen);
     }else{
     	projected_values = (*train_data) * (*root->dom_eigen);
     }

     root->max = arma::max(arma::max(projected_values));
     root->min = arma::min(arma::min(projected_values));

     //Divide data in bins
     vector<vector<int> > matrix(BIN_SIZE, vector<int>(size));
     int count[BIN_SIZE];
     for(int i=0;i<BIN_SIZE;i++){
       count[i] = 0;
     }
     int bin;
     for(int i=0;i<size;i++){
       //Get the bin for this index
        temp = projected_values(i,0);
        temp = (temp-root->min)*BIN_SIZE/(root->max-root->min);
        bin = ceil(temp);
        if(bin<=0){
          bin = 1;
        }else if(bin>BIN_SIZE){
          bin = BIN_SIZE;
        }
        matrix[bin-1][count[bin-1]++] = (root->index)(i);
     }
     int neg_count =0;
     int pos_count =0;
     uvec v;
     for(int i = 0; i < BIN_SIZE; ++i){
        neg_count = 0;
        pos_count = 0;
        root->child[i] = new node;
        v.set_size(count[i]);
        for(int j=0;j<count[i];j++){
           v(j) = matrix[i][j];
           if((*train_labels)(v(j),0)==-1){
              neg_count++;
           }
           else if((*train_labels)(v(j),0)==1){
              pos_count++;
           }
         }
         ((root->child[i])->index) = v;
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
             create_decision_tree(root->child[i], level+1, train_data, train_labels);
          }
          //v.clear();
        }
    return ;
}

void decision_tree_populate(node* root, int level, fmat* test_data){
     if(print_log){
        cout << "At level (): " << level << endl;
      }
     /*Check if need to continue
     Break if node is leaf*/
     if(root->isLeaf){
        return;
     }
     int size = (root->test_index).n_rows;
     // Matrix BIN_SIZE*size
     // A row(vector) for each bin
     vector<vector<int> > matrix(BIN_SIZE, vector<int>(size));
     //Elements in each bin
     int count[BIN_SIZE];
     for(int i=0;i<BIN_SIZE;i++){
       count[i] = 0;
     }
     fvec projected_values;
     if(level!=0){
        arma::uvec indices(root->test_index);
     	fmat current_data = (*test_data).rows(indices);
     	projected_values = current_data * (*root->dom_eigen);
     }else{
     	projected_values = (*test_data) * (*root->dom_eigen);
     }
     float temp;
     int bin;
     for(int i=0;i<size;i++){
       //Get the bin for this index
        temp = projected_values(i);
        temp = (temp-(root->min))*BIN_SIZE/(root->max-root->min);
        bin = ceil(temp);
        if(bin<=0){
          bin = 1;
        }else if(bin>BIN_SIZE){
          bin = BIN_SIZE;
        }
        matrix[bin-1][count[bin-1]++] = (root->test_index)(i);
     }
     uvec v;
     for(int i = 0; i < BIN_SIZE; i++){
        node* current_node = root->child[i];
        v.set_size(count[i]);
        for(int j=0;j<count[i];j++){
           v(j) = matrix[i][j];
         }
         current_node->test_index = v;
         decision_tree_populate(current_node,level+1,test_data);
         //v.clear();
     }
    return ;
}

void traverse_decision_tree(node* root, int level, fmat* test_data, fvec* predicted_labels){
  if(print_log){
    cout << "Traversing level : " <<level<<endl;
  }
  int size = root->test_index.n_rows;

  if(root->isLeaf){
    //If node is empty
    //Can't predict, assigning zero
    if(root->isEmpty){
      if(print_log){
        cout << "Hit empty node, labeling all as zero" <<endl;
      }
      for(int i=0;i<size;i++){
        (*predicted_labels)(root->test_index(i)) = 0;
      }
    }
    //All Positive
    if(root->all_positive){
      if(print_log){
        cout << "All Positive, labeling all as positive" <<endl;
      }
      for(int i=0;i<size;i++){
        (*predicted_labels)(root->test_index(i)) = 1;
      }
    }
    //All Negative
    else if(root->all_negative){
      if(print_log){
        cout << "All Negative, labeling all as negative" <<endl;
      }
      for(int i=0;i<size;i++){
        (*predicted_labels)(root->test_index(i)) = -1;
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
         float temp[n];
         current_index = root->test_index(i);
         for(int j=0;j<n;j++){
           temp[j] = (*test_data)(current_index,j);
         }
         cv::Mat testVector(1,n,CV_32F,temp) ;
         //cout << testVector <<endl;

         (*predicted_labels)(current_index) = root->svm->predict(testVector);
         //cout << (int)(root->svm->predict(testVector)) <<endl;
       }
       return;
     }//SVM
  }//leaf node
  for(int i=0;i<BIN_SIZE;i++){
     node* current_node = root->child[i];
     if(current_node!=NULL){
       traverse_decision_tree(current_node, level+1, test_data, predicted_labels);
     }
  }
     return;
}

//Train SVM
void* train_svm(void * r){
  custom* t = (struct custom*) r;
  int size = ((t->current_node)->index).n_rows;
  int labels[size] ;
  for(int i=0;i<size;i++){
    labels[i] = (*(t->labels))(((t->current_node)->index)(i)) ;
  }
  cv::Mat labelsMat(size, 1, CV_32S, labels);
  cv::Mat trainingData(size, n, CV_32F);
  //cout << labelsMat;
  for(int i=0;i<size;i++){
    for(int j=0;j<n;j++){
      trainingData.at<float>(i,j) = (*(t->data))((t->current_node)->index(i),j);
    }
  }
  ((t->current_node)->svm)->train(trainingData, labelsMat, cv::Mat(), cv::Mat(), *((t->current_node)->params));
  pthread_exit(NULL);
}
