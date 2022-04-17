#include <iostream>
#include <opencv2/opencv.hpp>
#include </usr/include/opencv2/calib3d/calib3d_c.h>
#include "omp.h"
#include <thread>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
using namespace cv;

#define my_NUM 16
Mat Pic_per_L[my_NUM];
Mat Pic_per_R[my_NUM];
Mat PIc_end[my_NUM];
int sem[my_NUM];

cv::Ptr<cv::StereoSGBM> Sgbm_Arr[my_NUM];
vector<thread> mythreads;

     Mat im3;
     Mat grayLeft,grayRight,grayLeft_h,grayRight_h;



int   setNumDisparities=165;//最大视差
int   setblock=1;//窗口大小
int   setUniquenessRatio=10,setSpeckleWindowSize=100,setSpeckleRange=32,setDisp12MaxDiff=100,p1=8,p2=32;

void my_thread2(int num){
  //等待信号
  while(!sem[num]){   
    cout<<"waiting"<<num<<endl; 
  }
  //得到信号
  Mat tmp;
      Sgbm_Arr[num]->setBlockSize(setblock);
            Sgbm_Arr[num]->setNumDisparities(setNumDisparities);
            Sgbm_Arr[num]->setP1(p1 * 1*setblock*setblock);
            Sgbm_Arr[num]->setP2(p2 * 1*setblock*setblock);  
            Sgbm_Arr[num]->setUniquenessRatio(setUniquenessRatio);
            Sgbm_Arr[num]->setSpeckleWindowSize(setSpeckleWindowSize);
            Sgbm_Arr[num]->setSpeckleRange(setSpeckleRange);
            Sgbm_Arr[num]->setDisp12MaxDiff(setDisp12MaxDiff);

            Sgbm_Arr[num]->compute(Pic_per_L[num], Pic_per_R[num], tmp);
            tmp.convertTo(tmp, CV_16S); 
            tmp.convertTo(tmp,CV_8UC1,255 / (setNumDisparities*16.0));//归一化  十分重要
            PIc_end[num]=tmp.clone();
}
//启动线程  准备sgbm
   void prepare(){
      for(int i=0;i<my_NUM;i++){ 
        Sgbm_Arr[i]= cv::StereoSGBM::create(0,9, setblock);     
        mythreads.push_back(thread(my_thread2, i)); 
       }   
   };

//分份
void pic_per(Mat pic,Mat * mat_arr,int NUM){

  int del=(pic.rows)%NUM;
  int per=(pic.rows-del)/NUM;
  Mat pic_del=pic.rowRange(del,pic.rows);

  for(int tmp=0;tmp<NUM;tmp++){
    *(mat_arr+tmp)=pic_del.rowRange(tmp*per,per*(1+tmp));
  }      
};

/* void my_thread(string name,cv::Ptr<cv::StereoSGBM> sgbm){
    Mat im3;
  Mat grayLeft,grayRight,grayLeft_h,grayRight_h;

   //l
  Mat im0 = cv::imread("../0.jpg_l.jpg");
  //r
  Mat im1 = cv::imread("../0.jpg_r.jpg");

   cvtColor(im1,grayLeft,COLOR_BGR2GRAY);
   cvtColor(im0,grayRight,COLOR_BGR2GRAY);
   grayLeft_h=grayLeft;
    grayRight_h=grayRight;
    setUniquenessRatio=10;
    setSpeckleWindowSize=100;
    setSpeckleRange=32;
    setDisp12MaxDiff=1;
    p1=8;
    p2=32;
      sgbm->setBlockSize(setblock);
            sgbm->setNumDisparities(setNumDisparities);
            sgbm->setP1(p1 * 1*setblock*setblock);
            sgbm->setP2(p2 * 1*setblock*setblock);  
            sgbm->setUniquenessRatio(setUniquenessRatio);
            sgbm->setSpeckleWindowSize(setSpeckleWindowSize);
            sgbm->setSpeckleRange(setSpeckleRange);
            sgbm->setDisp12MaxDiff(setDisp12MaxDiff);

            sgbm->compute(grayLeft_h, grayRight_h, im3);
            im3.convertTo(im3, CV_16S); 
            im3.convertTo(im3,CV_8UC1,255 / (setNumDisparities*16.0));//归一化  十分重要
            namedWindow(name,WINDOW_FREERATIO); 
            imshow(name,im3);
}

void thread_test(){
                 double start = getTickCount();
  thread th1(my_thread,"th1",sgbm);
  thread th2(my_thread,"th2",sgbm2); 
 
  th1.join();
  th2.join();
  double time = ((double)getTickCount() - start) / getTickFrequency();
  cout << "所用时间为：" << time << "秒" << endl;
} */

int main()
{

  //读取图片 预处理
   //l
  Mat im0 = cv::imread("../0.jpg_l.jpg");
  //r
  Mat im1 = cv::imread("../0.jpg_r.jpg");
   cvtColor(im1,grayLeft,COLOR_BGR2GRAY);
   cvtColor(im0,grayRight,COLOR_BGR2GRAY);
   for(int i=0;i<my_NUM;i++){
     sem[i]=0;
   }
   prepare(); 

   pic_per(grayRight,Pic_per_R,my_NUM);
   pic_per(grayLeft,Pic_per_L,my_NUM);
   for(int i=0;i<my_NUM;i++){
     sem[i]=1;
   }
    for(int i=0;i<my_NUM;i++){
        mythreads[i].join();
      };
      for(int i=0;i<my_NUM;i++){
        imshow(to_string(i),PIc_end[i]);
      }
 
    waitKey(0);

  return 0;
}
