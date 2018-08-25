#include "pcdt_svm.h"
using namespace std;
using namespace arma;

float compute_accuracy(fvec* test_labels, fvec* predicted_labels, int m_test){
  int accurate_count = 0;
  int assignment_failed = 0;
  for(int i=0;i<m_test;i++){
    if((*predicted_labels)(i) == 0){
      assignment_failed++;
    }
    if ((*test_labels)(i) == (*predicted_labels)(i)){
      accurate_count ++;
    }
  }
  cout << "Accuracy is : " << float(accurate_count)/m_test <<endl;
  cout << "Couldn't classify : " << assignment_failed << " test samples" <<endl;
  return float(accurate_count)/m_test;
}

float compute_fScore(float precision, float recall){
  cout << "F-Score is : " << (2*precision*recall)/(precision+recall) << endl;
  return (2*precision*recall)/(precision+recall);
}

float compute_precision(fvec* test_labels, fvec* predicted_labels, int m_test){
  int val = 0;
  int retrieved = 0;
  int assignment_failed = 0;
  for(int i=0;i<m_test;i++){
    if((*predicted_labels)(i) == 0){
      assignment_failed++;
    }
    if ((*predicted_labels)(i) == 1){
		retrieved++;
		if((*test_labels)(i) == 1)
			val ++;
    }
  }
  cout << "Precision is : " << float(val)/retrieved << endl;
  return float(val)/m_test;
}

float compute_recall(fvec* test_labels, fvec* predicted_labels, int m_test){
  int val = 0;
  int assignment_failed = 0;
  int relevant = 0;
  for(int i=0;i<m_test;i++){
    if((*predicted_labels)(i) == 0){
      assignment_failed++;
    }
    if ((*test_labels)(i) == 1 ){
      relevant++;
      if((*predicted_labels)(i) == 1)
        val ++;
    }
  }
  cout << "Recall is : " << float(val)/relevant <<endl;
  return float(val)/relevant;
}
