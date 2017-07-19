//
//  main.cpp
//  texture-matching
//
//  Created by AtsuyaSato on 2017/06/29.
//  Copyright © 2017年 Atsuya Sato. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

using namespace std;
using namespace cv;

#define FIXED_SIZE 512

typedef struct {
    int width;
    int height;
    int stride;
    int *data;
}HuMat;

void create_humat(HuMat &mat, int width, int height){
    mat.width = width;
    mat.height = height;
    mat.stride = ((width + 3) >> 2) << 2;
    mat.data = (int*)malloc(sizeof(int) * mat.stride * mat.height);
}

void free_humat(HuMat &mat){
    free(mat.data);
    mat.data = NULL;
}

void show_humat(const char *winName, HuMat &src){
    int width = src.width;
    int height = src.height;
    
    Mat res(height, width, CV_8UC1, Scalar(0));
    
    int *ptrSrc = src.data;
    
    uchar *ptrRes = res.data;
    
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++)
            ptrRes[x] = abs(ptrSrc[x]);
        
        ptrSrc += src.stride;
        ptrRes += res.step;
    }
    
    imshow(winName, res);
    waitKey();
}

void haar_wavelet_transform(HuMat &img, HuMat &res){
    int height = img.height;
    int width = img.width;
    
    HuMat centerMat;
    
    create_humat(centerMat, width, height);
    
    int *ptrSrc = img.data;
    int *ptrRes = centerMat.data;
    
    int stride1 = img.stride;
    int stride2 = centerMat.stride;
    
    int len = width / 2;
    
    assert(height % 2 == 0 && width % 2 == 0);
    
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x += 2){
            int a = ptrSrc[x];
            int b = ptrSrc[x + 1];
            
            int idx = x >> 1;
            
            ptrRes[idx] = (a + b) / 2;
            ptrRes[idx + len] = (a - b) / 2;
        }
        ptrSrc += stride1;
        ptrRes += stride2;
    }
    
    create_humat(res, width, height);
    
    len = height / 2;
    
    ptrSrc = centerMat.data;
    ptrRes = res.data;
    
    stride1 = centerMat.stride;
    stride2 = res.stride;
    
    for(int y = 0; y < height; y += 2){
        for(int x = 0; x < width; x++){
            int a = ptrSrc[x];
            int b = ptrSrc[stride1 + x];
            
            ptrRes[x] = (a + b) / 2;
            ptrRes[len * stride2 + x] = (a - b) / 2;
        }
        ptrSrc += 2 * stride1;
        ptrRes += stride2;
    }
    
    free_humat(centerMat);
}

int main(int argc, char **argv){
    
    if(argc < 2) return -1;
    
    // Load an image
    Mat inputImage = imread(argv[1], 0);
    if(inputImage.empty()) return -1;
    
    resize(inputImage, inputImage, Size(FIXED_SIZE, FIXED_SIZE));
    
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    
    HuMat src;
    create_humat(src, cols, rows);
    
    uchar *ptrImg = inputImage.data;
    int *ptrSrc = src.data;
    
    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++)
            ptrSrc[x] = ptrImg[x];
        
        ptrSrc += src.stride;
        ptrImg += inputImage.step;
    }
    HuMat dst;
    haar_wavelet_transform(src, dst);
    show_humat("transformed", dst);
    
    free_humat(src);
    free_humat(dst);
    
    return 0;
}
