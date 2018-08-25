// ParallelPCA.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <math.h>
#include <sstream>
#include <pthread.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

struct cov_data{
  int id;
  Mat* X; //nxd
  Mat* A; //dxd
  double *mean; //dx1
  int start;
  int end;
  int n;
  int d;
};

Mat* cov(double**, int, int, double*);
double* get_mean(double**, int, int);
double* domEV(double**, int, int, double);
double* threadMatMul(double**,double*, int, double*);
void *covThread(void *);
void *meanThread(void *);

int main(int argc, char **argv){
  int nepochs=100;
  double epsilon=0.00001;
  time_t st,et;
	time(&st);

  //string filename="/home/ds/Desktop/ijcnn/ijcnn1.csv";
  //Read File
  string filename="/home/kd/PCDT_SVM/mnist8m/train.csv";
  if (dataFile.read_csv(filename) != 0){
     fprintf(stderr, "Can't read csv file %s\n", filename );
     return -1;
  }
  Mat X(dataFile.get_values());
  int d = data.cols;
  int n = data.rows;
  dataFile.set_response_idx(--d);
  cout << "Data Size : " << n << " " <<d<<endl;


  double *mean=get_mean(&X, n, d);
  for(int i=0;i<d;i++){
    mean[i]/=n;
  }
  cout<<mean[0]<<" : "<<mean[2]<<" <--> "<<mean[d-2]<<" : "<<mean[d-1]<<endl;

  Mat* A=cov(&X, n, d,mean);
  for(int i=0;i<d;i++){
    for(int j=0;j<d;j++){
      A->at<float>(i,j)=A.at<float>(i,j)/(n-1);
    }
  }
  cout<<A[0][0]<<" : "<<A[0][1]<<" <--> "<<A[0][d-2]<<" : "<<A[0][d-1]<<endl;
  cout<<A[1][0]<<" : "<<A[1][1]<<" <--> "<<A[1][d-2]<<" : "<<A[1][d-1]<<endl;
  cout<<A[d-2][0]<<" : "<<A[d-2][1]<<" <--> "<<A[d-2][d-2]<<" : "<<A[d-2][d-1]<<endl;
  cout<<A[d-1][0]<<" : "<<A[d-1][1]<<" <--> "<<A[d-1][d-2]<<" : "<<A[d-1][d-1]<<endl;
  double *v=domEV(A,d,nepochs,epsilon);
  cout<<v[0]<<" : "<<v[2]<<" <--> "<<v[d-2]<<" : "<<v[d-1]<<endl;
  time(&et);
  cout<<" Time : "<<difftime(et,st)<<endl;
  return 0;
}

double* domEV(Mat* A,int d,int nepochs, double epsilon){
  time_t st,et;
  time(&st);
  double *b=(double*)malloc(d*sizeof(double));
  for(int j=0;j<d;j++){
    b[j]=1.0;
  }

  for(int epoch=0;epoch<nepochs;epoch++){
    cout<<"epoch: "<<epoch<<endl;
    double *tmp=(double*)malloc(d*sizeof(double));
    //tmp=threadMatMul(A,b,d,tmp);
    for(int i=0; i<d; i++){
      tmp[i] = 0;
      //cout<<"ok"<<endl;
      for (int j=0; j<d; j++){
        tmp[i] += A->at<float>(i,j) * b[j];
      }
    }

    double norm_sq=0.0;
    for (int k=0; k<d; k++){
      norm_sq += tmp[k]*tmp[k];
    }
    double norm = sqrt(norm_sq);

    int flag=1;
    for(int j=0;j<d;j++){
      double t=tmp[j]/norm;
      if(abs(b[j]-t)>epsilon)
        flag=0;
      b[j] = t;
    }
    if(flag)
      break;
  }
  time(&et);
  cout<<" A generated. Time : "<<difftime(et,st)<<endl;
  return b;
}

Mat*cov(Mat* X, int n, int d, double *mean){
  Mat A = Mat::zeros(d, d, CV_32F);
  int NUM_THREADS=50;
  int rc;
  pthread_t threads[NUM_THREADS];
  struct cov_data td[NUM_THREADS];
  pthread_attr_t attr;
  void *status;

  int chunks=d/NUM_THREADS;
  int r=d%NUM_THREADS;

  // Initialize and set thread joinable
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  //cout<<"ok"<<endl;
  for(int i=0; i < NUM_THREADS; i++ ){
    td[i].id=i;
    td[i].X=X;
    td[i].A=&A;
    td[i].mean=mean;
    td[i].d=d;
    td[i].n=n;
    if(i<r)
      td[i].start=i*chunks+i;
    else
      td[i].start=i*chunks+r;
    if(i<r)
      td[i].end=(i+1)*chunks+i+1;
    else
      td[i].end=(i+1)*chunks+r;

    rc = pthread_create(&threads[i], NULL, covThread, (void *)&td[i] );
    if (rc){
       cout << "Error:unable to create thread," << rc << endl;
       exit(-1);
    }
  }

  // free attribute and wait for the other threads
  pthread_attr_destroy(&attr);
  for(int i=0; i < NUM_THREADS; i++ )
  {
    rc = pthread_join(threads[i], &status);
    if (rc)
    {
       cout << "Error:unable to join," << rc << endl;
       exit(-1);
    }
  }

  //pthread_exit(NULL);
  return &A;
}

void *covThread(void *temp){
  struct cov_data *data;
  data = (struct cov_data *)temp;

  for(int l=1;l<data->n;l++)
  for(int i=data->start;i<data->end;i++){
    for(int j=0;j<data->d;j++){
      (data->A)->at<float>(i,j)+=((data->X)->at<float>(l,i)-data->mean[i])*((data->X)->at<float>(l,j)-data->mean[j]);
    }
  }
}

double* get_mean(Mat* X, int n, int d){
  double *mean=(double*)malloc(d*sizeof(double));
  int NUM_THREADS=d;
  int rc;
  pthread_t threads[NUM_THREADS];
  struct cov_data td[NUM_THREADS];
  pthread_attr_t attr;
  void *status;

  // Initialize and set thread joinable
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  for(int j=0;j<d;j++){
    mean[j]=0;
  }
  //cout<<"ok"<<endl;
  for(int i=0; i < NUM_THREADS; i++ ){
    td[i].id=i;
    td[i].X=X;
    td[i].mean=mean;
    td[i].d=d;
    td[i].n=n;

    rc = pthread_create(&threads[i], NULL, meanThread, (void *)&td[i] );
    if (rc){
       cout << "Error:unable to create thread," << rc << endl;
       exit(-1);
    }
  }

  // free attribute and wait for the other threads
  pthread_attr_destroy(&attr);
  for(int i=0; i < NUM_THREADS; i++ ){
    rc = pthread_join(threads[i], &status);
    if (rc){
       cout << "Error:unable to join," << rc << endl;
       exit(-1);
    }
  }
  return mean;
}

void *meanThread(void *temp){
  struct cov_data *data;
  data=(struct cov_data *)temp;
  for(int l=1;l<data->n;l++)
      data->mean[d->id]+=(data->X)->at<float>(l, data->id);
}
