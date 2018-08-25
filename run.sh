# Script to run decision tree program
# Clear old output file first
clear
rm out.txt
rm decision_tree
#Compile file
g++ -std=c++11 main.cpp pcdt_svm.h pcdt_svm.cpp metrices.cpp -o decision_tree -O2 -larmadillo -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lpthread -lopenblas

# Run and save output in out.txt

time ./decision_tree /home/kd/PCDT_SVM/mnist8m/train.csv /home/kd/PCDT_SVM/mnist8m/test.csv> out.txt
