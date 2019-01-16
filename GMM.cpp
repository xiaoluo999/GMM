// GMM.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "MOG_BGS.h"
#include<opencv2/opencv.hpp>
#include "my_background_segm.h"   //自己定义的头文件，默认的是直接调用opencv自带的GMM有关的函数，所以本文中重新定义一个不同的类

using namespace std;
using namespace cv;
void my_opencv_GMM()
{
	int count_frame = 0;
	{
		VideoCapture capture("F:\\data\\fire\\93.mp4");
		if (!capture.isOpened())
		{
			cout << "读取视频失败" << endl;
			//return ;
		}
		//获取整个帧数
		long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
		cout << "整个视频共" << totalFrameNumber << "帧" << endl;

		//设置开始帧()
		long frameToStart = 1;
		capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
		cout << "从第" << frameToStart << "帧开始读" << endl;

		//设置结束帧
		int frameToStop = 650;

		if (frameToStop < frameToStart)
		{
			cout << "结束帧小于开始帧，程序错误，即将退出！" << endl;
			return ;
		}
		else
		{
			cout << "结束帧为：第" << frameToStop << "帧" << endl;
		}

		double rate = capture.get(CV_CAP_PROP_FPS);
		int delay = 1000 / rate;

		Mat frame;
		//前景图片
		Mat foreground;
		Mat GMM_gray;
		Mat GMM_canny;


		//使用默认参数调用混合高斯模型
		my_BackgroundSubtractorMOG mog;   //使用自己定义的高斯混合模型类
		bool stop(false);
		//currentFrame是在循环体中控制读取到指定的帧后循环结束的变量
		long currentFrame = frameToStart;
		while (!stop)
		{
			count_frame++;
			if (!capture.read(frame))
			{
				cout << "从视频中读取图像失败或者读完整个视频" << endl;
				return ;
			}
			cvtColor(frame, GMM_gray, CV_RGB2GRAY);
			//Canny(GMM_gray,GMM_canny,50,150,3); 
			//imshow("GMM_canny",GMM_canny);
			imshow("输入视频", frame);
			//参数为：输入图像、输出图像、学习速率
			//mog(GMM_canny,foreground,0.01);
			mog(GMM_gray, foreground, 0.01);
			//cout<<mog.nframes<<"  ";
			imshow("前景", foreground);
			medianBlur(foreground, foreground, 3);
			imshow("中值滤波后的前景", foreground);

			//按ESC键退出，按其他键会停止在当前帧

			int c = waitKey(delay);

			if ((char)c == 27 || currentFrame >= frameToStop)
			{
				stop = true;
			}
			if (c >= 0)
			{
				waitKey(0);
			}
			currentFrame++;

		}

		waitKey(0);
	}

}

int test_GMM()
{

	{
		Mat frame, gray, mask;
		VideoCapture capture;
		capture.open("F:\\car2.mp4");

		if (!capture.isOpened())
		{
			cout << "No camera or video input!\n" << endl;
			return -1;
		}

		MOG_BGS Mog_Bgs;
		int count = 0;

		while (1)
		{
			count++;
			capture >> frame;
			if (frame.empty())
				break;
			cvtColor(frame, gray, CV_RGB2GRAY);

			if (count == 1)
			{
				Mog_Bgs.init(gray);
				Mog_Bgs.processFirstFrame(gray);
				cout << " Using " << TRAIN_FRAMES << " frames to training GMM..." << endl;
			}
			else if (count < TRAIN_FRAMES)
			{
				Mog_Bgs.trainGMM(gray);
			}
			else if (count == TRAIN_FRAMES)
			{
				Mog_Bgs.getFitNum(gray);
				cout << " Training GMM complete!" << endl;
			}
			else
			{
				Mog_Bgs.testGMM(gray);
				mask = Mog_Bgs.getMask();
				morphologyEx(mask, mask, MORPH_OPEN, Mat());
				erode(mask, mask, Mat(7, 7, CV_8UC1), Point(-1, -1));  // You can use Mat(5, 5, CV_8UC1) here for less distortion
				dilate(mask, mask, Mat(7, 7, CV_8UC1), Point(-1, -1));
				imshow("mask", mask);
			}

			imshow("input", frame);

			if (cvWaitKey(10) == 'q')
				break;
		}

		return 0;
	}
}
void cv_GMM()
{
	cv::VideoCapture video;
	video.open("F:\\data\\fire\\93.mp4");
	Mat frame, mask, thresholdImage, output, background;
	video.read(frame);
	BackgroundSubtractorMOG bgSubtractor(200,5,0.7);
	while (true){
		video >> frame;
		bgSubtractor(frame, mask, 0.001);
		bgSubtractor.getBackgroundImage(background); // 返回当前背景图像
		cv::imshow("background", background);
		cv::imshow("mask", mask);
		cv::waitKey(10);
	}
	return ;
}
int main()
{
	test_GMM();
	//my_opencv_GMM();
	//cv_GMM();
	return 0;

	CvCapture*capture = cvCreateFileCapture("F:\\car2.mp4");//读取视频
	IplImage*mframe = cvQueryFrame(capture);//读取视频中的一帧
	cvNamedWindow("cr");
	cvShowImage("cr", mframe);
	int height = mframe->height;
	int width = mframe->width;
	int C = 4;//number of gaussian components
	int M = 4;//number of background components
	int std_init = 6;//initial standard deviation
	double D = 2.5;
	double T = 0.7;
	double alpha = 0.01;
	double p = alpha / (1 / C);
	double thresh = 0.25;
	int min_index = 0;
	int*rank_ind = 0;
	int i, j, k, m;
	int rand_temp = 0;
	int rank_ind_temp = 0;
	CvRNG state;
	IplImage*current = cvCreateImage(cvSize(mframe->width, mframe->height), IPL_DEPTH_8U, 1);
	IplImage*test = cvCreateImage(cvSize(mframe->width, mframe->height), IPL_DEPTH_8U, 1);
	IplImage*frg = cvCreateImage(cvSize(mframe->width, mframe->height), IPL_DEPTH_8U, 1);

	double*mean = (double*)malloc(sizeof(double)*width*height*C);//pixelmeans
	double*std = (double*)malloc(sizeof(double)*width*height*C);//pixel standard deviations
	double*w = (double*)malloc(sizeof(double)*width*height*C);//权值
	double*u_diff = (double*)malloc(sizeof(double)*width*height*C);//存放像素值与每一个单高斯模式的均值的差值
	int*bg_bw = (int*)malloc(sizeof(int)*width*height);
	double*rank = (double*)malloc(sizeof(double)* 1 * C);

	//初始化
	for (i = 0; i < height; i++)//对于每一个像素
	{
		for (j = 0; j < width; j++)
		{
			for (k = 0; k < C; k++)//对于每一个单高斯模型，初始化它的均值，标准差，权值
			{
				mean[i*width*C + j*C + k] = cvRandReal(&state) * 255;//产生0-255之间的随机数
				w[i*width*C + j*C + k] = (double)1 / C;//每一个单高斯模型的权值系数
				std[i*width*C + j*C + k] = std_init;
			}
		}
	}

	while (1)
	{
		rank_ind = (int*)malloc(sizeof(int)*C);

		cvCvtColor(mframe, current, CV_RGB2GRAY);//灰度化

		//对于每一个像素，分别计算它和每一个单高斯模型的均值的差值
		for (i = 0; i < height; i++)//对于每一个像素
		{
			for (j = 0; j < width; j++)
			{
				for (k = 0; k < C; k++)
				{
					u_diff[i*width*C + j*C + k] = abs((uchar)current->imageData[i*width + j] - mean[i*width*C + j*C + k]);

				}
			}
		}

		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				int match = 0;
				double temp = 0;
				double single_temp = 0;
				//遍历所有的单高斯模式，如果此像素满足任一单高斯模式，则匹配；如果此像素不满足任何的单高斯模式，则不匹配
				for (k = 0; k < C; k++)
				{
					if (abs(u_diff[i*width*C + j*C + k]) < D*std[i*width*C + j*C + k])//如果像素匹配某单个高斯模式，则对其权值、均值和标准差进行更新
					{
						match = 1;
						w[i*width*C + j*C + k] += alpha*(1 - w[i*width*C + j*C + k]);//更新权值
						p = alpha / w[i*width*C + j*C + k];
						mean[i*width*C + j*C + k] = (1 - p)*mean[i*width*C + j*C + k] + p*(uchar)current->imageData[i*width + j];//更新均值
						std[i*width*C + j*C + k] = sqrt((1 - p)*(std[i*width*C + j*C + k] * std[i*width*C + j*C + k]) + p*(pow((uchar)current->imageData[i*width + j] - mean[i*width*C + j*C + k], 2)));//更新标准差

					}
					else
					{
						w[i*width*C + j*C + k] = (1 - alpha)*w[i*width*C + j*C + k];//如果像素不符合某单个高斯模型，则将此单高斯模型的权值降低
					}
				}

				if (match == 1)//如果和任一单高斯模式匹配，则将权值归一化
				{
					for (k = 0; k < C; k++)
					{
						temp += w[i*width*C + j*C + k];//计算四个单高斯模式权值的和
					}
					for (k = 0; k < C; k++)
					{
						w[i*width*C + j*C + k] = w[i*width*C + j*C + k] / temp;//权值归一化，使得所有权值和为1
					}
				}
				else//如果和所有单高斯模式都不匹配，则寻找权值最小的高斯模式并删除，然后增加一个新的高斯模式
				{
					single_temp = w[i*width*C + j*C];
					for (k = 0; k < C; k++)
					{
						if (w[i*width*C + j*C + k] < single_temp)
						{
							min_index = k;//寻找权值最小的高斯模式
							single_temp = w[i*width*C + j*C + k];
						}

					}
					mean[i*width*C + j*C + min_index] = (uchar)current->imageData[i*width + j];//建立一个新的高斯模式，均值为当前像素值
					std[i*width*C + j*C + min_index] = std_init;//标准差为初始值

					for (k = 0; k < C; k++)
					{
						temp += w[i*width*C + j*C + k];//计算四个单高斯模式权值的和
					}
					for (k = 0; k < C; k++)
					{
						w[i*width*C + j*C + k] = w[i*width*C + j*C + k] / temp;//权值归一化，使得所有权值和为1
					}

				}

				for (k = 0; k < C; k++)//计算每个单高斯模式的重要性
				{
					rank[k] = w[i*width*C + j*C + k] / std[i*width*C + j*C + k];
					rank_ind[k] = k;
				}

				for (k = 1; k<C; k++)//对重要性排序
				{
					for (m = 0; m<k; m++)
					{
						if (rank[k] > rank[m])
						{
							//swap max values  
							rand_temp = rank[m];
							rank[m] = rank[k];
							rank[k] = rand_temp;
							//swap max index values  
							rank_ind_temp = rank_ind[m];
							rank_ind[m] = rank_ind[k];
							rank_ind[k] = rank_ind_temp;
						}
					}
				}

				bg_bw[i*width + j] = 0;
				for (k = 0; k < C; k++)//如果前几个单高斯模式的重要性之和大于T，则将这前几个单高斯模式认为为背景模型
				{
					temp += w[i*width*C + j*C + rank_ind[k]];
					bg_bw[i*width + j] += mean[i*width*C + j*C + rank_ind[k]] * w[i*width*C + j*C + rank_ind[k]];
					if (temp >= T)
					{
						M = k;
						break;
					}
				}

				test->imageData[i*width + j] = (uchar)bg_bw[i*width + j];//背景图像

				match = 0; k = 0;
				while ((match == 0) && (k <= M))//如果某像素不符合背景模型中任一单高斯模型，则此像素为前景像素
				{
					if (abs(u_diff[i*width*C + j*C + rank_ind[k]]) <= D*std[i*width*C + j*C + rank_ind[k]])
					{
						frg->imageData[i*width + j] = 0;
						match = 1;
					}
					else
						frg->imageData[i*width + j] = (uchar)current->imageData[i*width + j];

					k += 1;

				}


			}
		}
		mframe = cvQueryFrame(capture);
		if (mframe == NULL)
			return -1;
		cvNamedWindow("frg");
		cvShowImage("frg", frg);
		cvNamedWindow("back");
		cvShowImage("back", test);

		char s = cvWaitKey(10);
		//if (s == 27)
		//  break;
		free(rank_ind);

	}
	cvWaitKey();
	return 0;
}

