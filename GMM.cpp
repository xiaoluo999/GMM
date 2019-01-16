// GMM.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "MOG_BGS.h"
#include<opencv2/opencv.hpp>
#include "my_background_segm.h"   //�Լ������ͷ�ļ���Ĭ�ϵ���ֱ�ӵ���opencv�Դ���GMM�йصĺ��������Ա��������¶���һ����ͬ����

using namespace std;
using namespace cv;
void my_opencv_GMM()
{
	int count_frame = 0;
	{
		VideoCapture capture("F:\\data\\fire\\93.mp4");
		if (!capture.isOpened())
		{
			cout << "��ȡ��Ƶʧ��" << endl;
			//return ;
		}
		//��ȡ����֡��
		long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
		cout << "������Ƶ��" << totalFrameNumber << "֡" << endl;

		//���ÿ�ʼ֡()
		long frameToStart = 1;
		capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
		cout << "�ӵ�" << frameToStart << "֡��ʼ��" << endl;

		//���ý���֡
		int frameToStop = 650;

		if (frameToStop < frameToStart)
		{
			cout << "����֡С�ڿ�ʼ֡��������󣬼����˳���" << endl;
			return ;
		}
		else
		{
			cout << "����֡Ϊ����" << frameToStop << "֡" << endl;
		}

		double rate = capture.get(CV_CAP_PROP_FPS);
		int delay = 1000 / rate;

		Mat frame;
		//ǰ��ͼƬ
		Mat foreground;
		Mat GMM_gray;
		Mat GMM_canny;


		//ʹ��Ĭ�ϲ������û�ϸ�˹ģ��
		my_BackgroundSubtractorMOG mog;   //ʹ���Լ�����ĸ�˹���ģ����
		bool stop(false);
		//currentFrame����ѭ�����п��ƶ�ȡ��ָ����֡��ѭ�������ı���
		long currentFrame = frameToStart;
		while (!stop)
		{
			count_frame++;
			if (!capture.read(frame))
			{
				cout << "����Ƶ�ж�ȡͼ��ʧ�ܻ��߶���������Ƶ" << endl;
				return ;
			}
			cvtColor(frame, GMM_gray, CV_RGB2GRAY);
			//Canny(GMM_gray,GMM_canny,50,150,3); 
			//imshow("GMM_canny",GMM_canny);
			imshow("������Ƶ", frame);
			//����Ϊ������ͼ�����ͼ��ѧϰ����
			//mog(GMM_canny,foreground,0.01);
			mog(GMM_gray, foreground, 0.01);
			//cout<<mog.nframes<<"  ";
			imshow("ǰ��", foreground);
			medianBlur(foreground, foreground, 3);
			imshow("��ֵ�˲����ǰ��", foreground);

			//��ESC���˳�������������ֹͣ�ڵ�ǰ֡

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
		bgSubtractor.getBackgroundImage(background); // ���ص�ǰ����ͼ��
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

	CvCapture*capture = cvCreateFileCapture("F:\\car2.mp4");//��ȡ��Ƶ
	IplImage*mframe = cvQueryFrame(capture);//��ȡ��Ƶ�е�һ֡
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
	double*w = (double*)malloc(sizeof(double)*width*height*C);//Ȩֵ
	double*u_diff = (double*)malloc(sizeof(double)*width*height*C);//�������ֵ��ÿһ������˹ģʽ�ľ�ֵ�Ĳ�ֵ
	int*bg_bw = (int*)malloc(sizeof(int)*width*height);
	double*rank = (double*)malloc(sizeof(double)* 1 * C);

	//��ʼ��
	for (i = 0; i < height; i++)//����ÿһ������
	{
		for (j = 0; j < width; j++)
		{
			for (k = 0; k < C; k++)//����ÿһ������˹ģ�ͣ���ʼ�����ľ�ֵ����׼�Ȩֵ
			{
				mean[i*width*C + j*C + k] = cvRandReal(&state) * 255;//����0-255֮��������
				w[i*width*C + j*C + k] = (double)1 / C;//ÿһ������˹ģ�͵�Ȩֵϵ��
				std[i*width*C + j*C + k] = std_init;
			}
		}
	}

	while (1)
	{
		rank_ind = (int*)malloc(sizeof(int)*C);

		cvCvtColor(mframe, current, CV_RGB2GRAY);//�ҶȻ�

		//����ÿһ�����أ��ֱ��������ÿһ������˹ģ�͵ľ�ֵ�Ĳ�ֵ
		for (i = 0; i < height; i++)//����ÿһ������
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
				//�������еĵ���˹ģʽ�����������������һ����˹ģʽ����ƥ�䣻��������ز������κεĵ���˹ģʽ����ƥ��
				for (k = 0; k < C; k++)
				{
					if (abs(u_diff[i*width*C + j*C + k]) < D*std[i*width*C + j*C + k])//�������ƥ��ĳ������˹ģʽ�������Ȩֵ����ֵ�ͱ�׼����и���
					{
						match = 1;
						w[i*width*C + j*C + k] += alpha*(1 - w[i*width*C + j*C + k]);//����Ȩֵ
						p = alpha / w[i*width*C + j*C + k];
						mean[i*width*C + j*C + k] = (1 - p)*mean[i*width*C + j*C + k] + p*(uchar)current->imageData[i*width + j];//���¾�ֵ
						std[i*width*C + j*C + k] = sqrt((1 - p)*(std[i*width*C + j*C + k] * std[i*width*C + j*C + k]) + p*(pow((uchar)current->imageData[i*width + j] - mean[i*width*C + j*C + k], 2)));//���±�׼��

					}
					else
					{
						w[i*width*C + j*C + k] = (1 - alpha)*w[i*width*C + j*C + k];//������ز�����ĳ������˹ģ�ͣ��򽫴˵���˹ģ�͵�Ȩֵ����
					}
				}

				if (match == 1)//�������һ����˹ģʽƥ�䣬��Ȩֵ��һ��
				{
					for (k = 0; k < C; k++)
					{
						temp += w[i*width*C + j*C + k];//�����ĸ�����˹ģʽȨֵ�ĺ�
					}
					for (k = 0; k < C; k++)
					{
						w[i*width*C + j*C + k] = w[i*width*C + j*C + k] / temp;//Ȩֵ��һ����ʹ������Ȩֵ��Ϊ1
					}
				}
				else//��������е���˹ģʽ����ƥ�䣬��Ѱ��Ȩֵ��С�ĸ�˹ģʽ��ɾ����Ȼ������һ���µĸ�˹ģʽ
				{
					single_temp = w[i*width*C + j*C];
					for (k = 0; k < C; k++)
					{
						if (w[i*width*C + j*C + k] < single_temp)
						{
							min_index = k;//Ѱ��Ȩֵ��С�ĸ�˹ģʽ
							single_temp = w[i*width*C + j*C + k];
						}

					}
					mean[i*width*C + j*C + min_index] = (uchar)current->imageData[i*width + j];//����һ���µĸ�˹ģʽ����ֵΪ��ǰ����ֵ
					std[i*width*C + j*C + min_index] = std_init;//��׼��Ϊ��ʼֵ

					for (k = 0; k < C; k++)
					{
						temp += w[i*width*C + j*C + k];//�����ĸ�����˹ģʽȨֵ�ĺ�
					}
					for (k = 0; k < C; k++)
					{
						w[i*width*C + j*C + k] = w[i*width*C + j*C + k] / temp;//Ȩֵ��һ����ʹ������Ȩֵ��Ϊ1
					}

				}

				for (k = 0; k < C; k++)//����ÿ������˹ģʽ����Ҫ��
				{
					rank[k] = w[i*width*C + j*C + k] / std[i*width*C + j*C + k];
					rank_ind[k] = k;
				}

				for (k = 1; k<C; k++)//����Ҫ������
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
				for (k = 0; k < C; k++)//���ǰ��������˹ģʽ����Ҫ��֮�ʹ���T������ǰ��������˹ģʽ��ΪΪ����ģ��
				{
					temp += w[i*width*C + j*C + rank_ind[k]];
					bg_bw[i*width + j] += mean[i*width*C + j*C + rank_ind[k]] * w[i*width*C + j*C + rank_ind[k]];
					if (temp >= T)
					{
						M = k;
						break;
					}
				}

				test->imageData[i*width + j] = (uchar)bg_bw[i*width + j];//����ͼ��

				match = 0; k = 0;
				while ((match == 0) && (k <= M))//���ĳ���ز����ϱ���ģ������һ����˹ģ�ͣ��������Ϊǰ������
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

