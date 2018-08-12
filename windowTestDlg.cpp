
// windowTestDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "windowTest.h"
#include "windowTestDlg.h"
#include "afxdialogex.h"
#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"


#include <string>
#include <iostream>
using namespace std;
using namespace cv;


extern "C"
{
#include "imgfeatures.h"
#include "kdtree.h"
#include "minpq.h"
#include "sift.h"
#include "utils.h"
#include "xform.h"
}




//在k-d树上进行BBF搜索的最大次数
/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200
#define NN_SQ_DIST_RATIO_THR 0.5

//窗口名字符串
#define IMG1 "图1"
#define IMG2 "图2"
#define IMG1_FEAT "图1特征点"
#define IMG2_FEAT "图2特征点"
#define IMG_MATCH1 "距离比值筛选后的匹配结果"
#define IMG_MATCH2 "RANSAC筛选后的匹配结果"
#define IMG_MOSAIC_TEMP "临时拼接图像"
#define IMG_MOSAIC_SIMPLE "简易拼接图"
#define IMG_MOSAIC_BEFORE_FUSION "重叠区域融合前"
#define IMG_MOSAIC_PROC12 "拼接图1-2"



#ifdef _DEBUG
#define new DEBUG_NEW
#endif



// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();
	

// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void On32771();

	afx_msg void bblocation();
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
	ON_COMMAND(ID_32771, &CAboutDlg::On32771)
	ON_COMMAND(ID_32786, &CAboutDlg::bblocation)
END_MESSAGE_MAP()


// CwindowTestDlg 对话框




CwindowTestDlg::CwindowTestDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CwindowTestDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CwindowTestDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CwindowTestDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
//	ON_BN_CLICKED(IDC_MFCMENUBUTTON1, &CwindowTestDlg::OnBnClickedMfcmenubutton1)
ON_COMMAND(ID_32771, &CwindowTestDlg::On32771)
ON_COMMAND(ID_32774, &CwindowTestDlg::On32774)
ON_COMMAND(ID_32775, &CwindowTestDlg::canny)
ON_COMMAND(ID_Menu, &CwindowTestDlg::LaLace)
ON_COMMAND(ID_32778, &CwindowTestDlg::MaximumConnectedarea)
ON_COMMAND(ID_SIFT32779, &CwindowTestDlg::siftStitching)
ON_UPDATE_COMMAND_UI(ID_32781, &CwindowTestDlg::Fengshuilin)
ON_COMMAND(ID_32782, &CwindowTestDlg::weather)
ON_COMMAND(ID_32784, &CwindowTestDlg::Color)
ON_COMMAND(ID_32785, &CwindowTestDlg::weather2)
ON_COMMAND(ID_32786, &CwindowTestDlg::BBlocation)
ON_COMMAND(ID_32787, &CwindowTestDlg::Median_filter)
ON_COMMAND(ID_HSV_HSV32788, &CwindowTestDlg::hsv_color)
ON_COMMAND(ID_32789, &CwindowTestDlg::lunkuo)
END_MESSAGE_MAP()


// CwindowTestDlg 消息处理程序

BOOL CwindowTestDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CwindowTestDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CwindowTestDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CwindowTestDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CwindowTestDlg::OnBnClickedMfcmenubutton1()
{

}


void CAboutDlg::On32771()
{
	
}

IplImage* SrcImg = NULL;
string mPath_pd ;
Mat src,dst;
Mat bsSrc;
bool flage=true;
int spatialRad=10,colorRad=10,maxPryLevel=1;

//计算图2的四个角经矩阵H变换后的坐标
void CalcFourCorner(CvMat* &H,CvPoint& leftTop,CvPoint& leftBottom, CvPoint& rightTop,CvPoint& rightBottom,IplImage* img2)
{
	//计算图2的四个角经矩阵H变换后的坐标
	double v2[]={0,0,1};//左上角
	double v1[3];//变换后的坐标值
	CvMat V2 = cvMat(3,1,CV_64FC1,v2);
	CvMat V1 = cvMat(3,1,CV_64FC1,v1);
	cvGEMM(H,&V2,1,0,1,&V1);//矩阵乘法
	leftTop.x = cvRound(v1[0]/v1[2]);
	leftTop.y = cvRound(v1[1]/v1[2]);
	//cvCircle(xformed,leftTop,7,CV_RGB(255,0,0),2);

	//将v2中数据设为左下角坐标
	v2[0] = 0;
	v2[1] = img2->height;
	V2 = cvMat(3,1,CV_64FC1,v2);
	V1 = cvMat(3,1,CV_64FC1,v1);
	cvGEMM(H,&V2,1,0,1,&V1);
	leftBottom.x = cvRound(v1[0]/v1[2]);
	leftBottom.y = cvRound(v1[1]/v1[2]);
	//cvCircle(xformed,leftBottom,7,CV_RGB(255,0,0),2);

	//将v2中数据设为右上角坐标
	v2[0] = img2->width;
	v2[1] = 0;
	V2 = cvMat(3,1,CV_64FC1,v2);
	V1 = cvMat(3,1,CV_64FC1,v1);
	cvGEMM(H,&V2,1,0,1,&V1);
	rightTop.x = cvRound(v1[0]/v1[2]);
	rightTop.y = cvRound(v1[1]/v1[2]);
	//cvCircle(xformed,rightTop,7,CV_RGB(255,0,0),2);

	//将v2中数据设为右下角坐标
	v2[0] = img2->width;
	v2[1] = img2->height;
	V2 = cvMat(3,1,CV_64FC1,v2);
	V1 = cvMat(3,1,CV_64FC1,v1);
	cvGEMM(H,&V2,1,0,1,&V1);
	rightBottom.x = cvRound(v1[0]/v1[2]);
	rightBottom.y = cvRound(v1[1]/v1[2]);
	//cvCircle(xformed,rightBottom,7,CV_RGB(255,0,0),2);

}

int detectionFeature(IplImage* img,struct feature*& feat)
{
	int n  = sift_features( img, &feat);//检测图img中的SIFT特征点,n是图的特征点个数
	//export_features("feature.txt",feat,n);//将特征向量数据写入到文件
	return n;
}

IplImage* spliceImage(IplImage* img1,IplImage* img2)
{
	struct feature *feat1, *feat2;//feat1：图1的特征点数组，feat2：图2的特征点数组
	int n1, n2;//n1:图1中的特征点个数，n2：图2中的特征点个数
	struct feature *feat;//每个特征点
	struct kd_node *kd_root;//k-d树的树根
	struct feature **nbrs;//当前特征点的最近邻点数组
	CvMat * H = NULL;//RANSAC算法求出的变换矩阵
	struct feature **inliers;//精RANSAC筛选后的内点数组
	int n_inliers;//经RANSAC算法筛选后的内点个数,即feat2中具有符合要求的特征点的个数

	IplImage *xformed = NULL,*xformed_proc = NULL;//xformed临时拼接图，即只将图2变换后的图,xformed_proc是最终合成的图

	//图2的四个角经矩阵H变换后的坐标
	CvPoint leftTop,leftBottom,rightTop,rightBottom;
	cvSetImageROI(img1, cvRect(0, 0, 1920, 600));
		IplImage *img11 = cvCreateImage(cvGetSize(img1),img1->depth,img1->nChannels);
		cvCopy(img1, img11, NULL);
		cvSetImageROI(img2, cvRect(0, 0, 1920, 600));
		IplImage *img22 = cvCreateImage(cvGetSize(img2),img1->depth,img1->nChannels);
		cvCopy(img2, img22, NULL);
	///////////////////////////////////////////////////////////////////

	//特征点检测
	n1 = detectionFeature( img11,feat1 );//检测图1中的SIFT特征点,n1是图1的特征点个数
	//提取并显示第2幅图片上的特征点
	n2 = detectionFeature( img22, feat2 );//检测图2中的SIFT特征点，n2是图2的特征点个数
	
	//特征匹配
	//方式一：水平排列
	//将2幅图片合成1幅图片,img1在左，img2在右
	//stacked = stack_imgs_horizontal(img1, img2);//合成图像，显示经距离比值法筛选后的匹配结果
	//根据图1的特征点集feat1建立k-d树，返回k-d树根给kd_root
	kd_root = kdtree_build( feat1, n1 );
	CvPoint pt1,pt2;//连线的两个端点
	double d0,d1;//feat2中每个特征点到最近邻和次近邻的距离
	int matchNum = 0;//经距离比值法筛选后的匹配点对的个数
	//遍历特征点集feat2，针对feat2中每个特征点feat，选取符合距离比值条件的匹配点，放到feat的fwd_match域中
	for(int i = 0; i < n2; i++ )
	{
		feat = feat2+i;//第i个特征点的指针
		//在kd_root中搜索目标点feat的2个最近邻点，存放在nbrs中，返回实际找到的近邻点个数
		int k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
		if( k == 2 )
		{
			d0 = descr_dist_sq( feat, nbrs[0] );//feat与最近邻点的距离的平方
			d1 = descr_dist_sq( feat, nbrs[1] );//feat与次近邻点的距离的平方
			//若d0和d1的比值小于阈值NN_SQ_DIST_RATIO_THR，则接受此匹配，否则剔除
			if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
			{   //将目标点feat和最近邻点作为匹配点对
				pt2.x = cvRound(feat->x);pt2.y = cvRound(feat->y);
				pt1.x = cvRound(nbrs[0]->x); pt1.y = cvRound(nbrs[0]->y);
				pt2.x += img1->width;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点
				//cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );//画出连线
				matchNum++;//统计匹配点对的个数
				feat2[i].fwd_match = nbrs[0];//使点feat的fwd_match域指向其对应的匹配点
			}
		}
		free( nbrs );//释放近邻数组
		
	}
	
	//利用RANSAC算法筛选匹配点,计算变换矩阵H，
	//无论img1和img2的左右顺序，H永远是将feat2中的特征点变换为其匹配点，即将img2中的点变换为img1中的对应点
	H = ransac_xform(feat2,n2,FEATURE_FWD_MATCH,lsq_homog,4,0.01,homog_xfer_err,3.0,&inliers,&n_inliers);

	//若能成功计算出变换矩阵，即两幅图中有共同区域
	IplImage* stacked_ransac;
	
	///stacked_ransac = stack_imgs(img1, img2);
	cvResetImageROI(img1);
	cvResetImageROI(img2);
	stacked_ransac = stack_imgs_horizontal(img1, img2);

	if( H )
	{
      int invertNum = 0;//统计pt2.x > pt1.x的匹配点对的个数，来判断img1中是否右图  
  
      //遍历经RANSAC算法筛选后的特征点集合inliers，找到每个特征点的匹配点，画出连线  
        for(int i=0; i<n_inliers; i++)  
          {  
             feat = inliers[i];//第i个特征点  
             pt2 = cvPoint(cvRound(feat->x), cvRound(feat->y));//图2中点的坐标  
             pt1 = cvPoint(cvRound(feat->fwd_match->x), cvRound(feat->fwd_match->y));//图1中点的坐标(feat的匹配点)  
        
             //统计匹配点的左右位置关系，来判断图1和图2的左右位置关系  
            if(pt2.x > pt1.x)  
               invertNum++;  
  
			 // pt2.y += img1->height;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点 
              pt2.x += img1->width;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点  
              cvLine(stacked_ransac,pt1,pt2,CV_RGB(255,0,255),1,8,0);//在匹配图上画出连线  
           }  
          cvNamedWindow(IMG_MATCH2,CV_WINDOW_NORMAL);//创建窗口  
          cvShowImage(IMG_MATCH2,stacked_ransac);//显示经RANSAC算法筛选后的匹配图 
		  cvSaveImage("ransac.bmp",stacked_ransac);
		  fprintf( stderr, "Found ransac %d total matches\n", n_inliers );
	 }

	if( H )
	{
		//全景拼接
		//若能成功计算出变换矩阵，即两幅图中有共同区域，才可以进行全景拼接
		//拼接图像，img1是左图，img2是右图
		CalcFourCorner(H,leftTop,leftBottom,rightTop,rightBottom,img2);//计算图2的四个角经变换后的坐标
		//为拼接结果图xformed分配空间,高度为图1图2高度的较小者，根据图2右上角和右下角变换后的点的位置决定拼接图的宽度
		xformed = cvCreateImage(cvSize(MIN(rightTop.x,rightBottom.x),MIN(img1->height,img2->height)),IPL_DEPTH_8U,3);
		//用变换矩阵H对右图img2做投影变换(变换后会有坐标右移)，结果放到xformed中
		cvWarpPerspective(img2,xformed,H,CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,cvScalarAll(0));

		//处理后的拼接图，克隆自xformed
		xformed_proc = cvCloneImage(xformed);

		//重叠区域左边的部分完全取自图1
		cvSetImageROI(img1,cvRect(0,0,MIN(leftTop.x,leftBottom.x),xformed_proc->height));
		cvSetImageROI(xformed,cvRect(0,0,MIN(leftTop.x,leftBottom.x),xformed_proc->height));
		cvSetImageROI(xformed_proc,cvRect(0,0,MIN(leftTop.x,leftBottom.x),xformed_proc->height));
		cvAddWeighted(img1,1,xformed,0,0,xformed_proc);
		cvResetImageROI(img1);
		cvResetImageROI(xformed);
		cvResetImageROI(xformed_proc);

		////////////////////////////////////////////////////////////
		//图像融合
		//采用加权平均的方法融合重叠区域
		int start = MIN(leftTop.x,leftBottom.x) ;//开始位置，即重叠区域的左边界
		double processWidth = img1->width - start;//重叠区域的宽度
		double alpha = 1;//img1中像素的权重
		for(int i=0; i<xformed_proc->height; i++)//遍历行
		{
			const uchar * pixel_img1 = ((uchar *)(img1->imageData + img1->widthStep * i));//img1中第i行数据的指针
			const uchar * pixel_xformed = ((uchar *)(xformed->imageData + xformed->widthStep * i));//xformed中第i行数据的指针
			uchar * pixel_xformed_proc = ((uchar *)(xformed_proc->imageData + xformed_proc->widthStep * i));//xformed_proc中第i行数据的指针
			for(int j=start; j<img1->width; j++)//遍历重叠区域的列
			{
				//如果遇到图像xformed中无像素的黑点，则完全拷贝图1中的数据
				if(pixel_xformed[j*3] < 50 && pixel_xformed[j*3+1] < 50 && pixel_xformed[j*3+2] < 50 )
				{
					alpha = 1;
				}
				else
				{   //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比
					alpha = (processWidth-(j-start)) / processWidth ;
				}
				pixel_xformed_proc[j*3] = pixel_img1[j*3] * alpha + pixel_xformed[j*3] * (1-alpha);//B通道
				pixel_xformed_proc[j*3+1] = pixel_img1[j*3+1] * alpha + pixel_xformed[j*3+1] * (1-alpha);//G通道
				pixel_xformed_proc[j*3+2] = pixel_img1[j*3+2] * alpha + pixel_xformed[j*3+2] * (1-alpha);//R通道
			}
		}

	}
	else //无法计算出变换矩阵，即两幅图中没有重合区域
	{
		return NULL;
	}
	///////////////////////////////////////////////////////////////////////////
	kdtree_release(kd_root);//释放kd树
	//只有在RANSAC算法成功算出变换矩阵时，才需要进一步释放下面的内存空间
	if(H)
	{
		cvReleaseMat(&H);//释放变换矩阵H
		free(inliers);//释放内点数组
	}
	if (NULL != xformed)
	{
		cvReleaseImage(&xformed);
	}
	return xformed_proc;
}
void CwindowTestDlg::On32771()
{
	// TODO: 在此添加命令处理程序代码
    CFileDialog dlg(  
        TRUE, _T("*.bmp;*.jpg;*.tif"), NULL,  
        OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY,  
        "image files All Files (*.*) |*.*||", NULL  
        );// 选项图片的约定;    
    dlg.m_ofn.lpstrTitle = _T("打开图片");// 打开文件对话框的标题名;(*.bmp; *.jpg) |*.bmp; *.jpg |    
    if (dlg.DoModal() == IDOK)// 判断是否获得图片;    
    {  
  
        if (dlg.GetFileExt() != "bmp" && dlg.GetFileExt() != "JPG"&&dlg.GetFileExt() != "tif"&&dlg.GetFileExt() != "jpg")  
        {  
            AfxMessageBox(_T("请选择正确的图片格式！"), MB_OK);  
            return;  
        }  
  
        CString mPath = dlg.GetPathName();// 获取图片路径;
		string mpath=dlg.GetPathName();
		mPath_pd=mpath;
  
        SrcImg = cvLoadImage(mPath);//读取图片、缓存到一个局部变量ipl中;
		bsSrc=imread(mpath, IMREAD_COLOR);
        if (!SrcImg)// 判断是否成功载入图片;    
            return;  
    }  

	cvNamedWindow("source1",CV_WINDOW_NORMAL);  
    cvShowImage("source1", SrcImg);
}


void CwindowTestDlg::On32774()
{
	// TODO: 在此添加命令处理程序代码
	
	cv::Mat matimg;
	Mat dst1;
    matimg = cv::Mat(SrcImg);
	Mat grad_x, grad_y;  
    Mat abs_grad_x, abs_grad_y;  
  
    //【3】求 X方向梯度  
    Sobel( matimg, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT );  
    convertScaleAbs( grad_x, abs_grad_x );    
  
    //【4】求Y方向梯度  
    Sobel( matimg, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT );  
    convertScaleAbs( grad_y, abs_grad_y );     
  
    //【5】合并梯度(近似)  
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst1 ); 
	namedWindow("Sobel", CV_WINDOW_NORMAL);
    imshow("Sobel", dst1);   
    waitKey(0);   
       
}


void CwindowTestDlg::canny()
{
	// TODO: 在此添加命令处理程序代码
	cv::Mat matimg;
    matimg = cv::Mat(SrcImg);
	Mat dst1,edge,gray; 
	Mat src1=matimg.clone();
  
    // 【1】创建与src同类型和大小的矩阵(dst)  
    dst1.create( src1.size(), src1.type() );  
  
    // 【2】将原图像转换为灰度图像  
    cvtColor( src1, gray, CV_BGR2GRAY );  
  
    // 【3】先用使用 3x3内核来降噪  
    blur( gray, edge, Size(3,3) );  
  
    // 【4】运行Canny算子  
    Canny( edge, edge, 3,9,3 );  
  
    //【5】将g_dstImage内的所有元素设置为0   
    dst1 = Scalar::all(0);  
  
    //【6】使用Canny算子输出的边缘图g_cannyDetectedEdges作为掩码，来将原图g_srcImage拷到目标图g_dstImage中  
   // src1.copyTo( dst, edge);  
  
    //【7】显示效果图  
	namedWindow("Canny", CV_WINDOW_NORMAL);
    imshow("Canny", edge);   
    waitKey(0);   
}


void CwindowTestDlg::LaLace()
{
	// TODO: 在此添加命令处理程序代码
	Mat src_gray,dst1, abs_dst;  
    cv::Mat matimg;
    matimg = cv::Mat(SrcImg);
  
    //【3】使用高斯滤波消除噪声  
    GaussianBlur( matimg, matimg, Size(3,3), 0, 0, BORDER_DEFAULT );  
  
    //【4】转换为灰度图  
    cvtColor( matimg, src_gray, CV_RGB2GRAY );  
  
    //【5】使用Laplace函数  
    Laplacian( src_gray, dst1, CV_16S, 3, 1, 0, BORDER_DEFAULT );  
  
    //【6】计算绝对值，并将结果转换成8位  
    convertScaleAbs( dst1, abs_dst );  
  
    //【7】显示效果图 
	namedWindow("Laplace", CV_WINDOW_NORMAL);
    imshow( "Laplace", abs_dst );  
  
    waitKey(0);   
}


void CwindowTestDlg::MaximumConnectedarea()
{
	// TODO: 在此添加命令处理程序代码
	IplImage* src1 = cvCreateImage(cvGetSize(SrcImg), IPL_DEPTH_8U, 1);
	cvCvtColor(SrcImg, src1, CV_RGB2GRAY);
	IplImage* src2 = SrcImg; 
    IplImage* dst1 = cvCreateImage(cvGetSize(src1), 8, 3);  
    CvMemStorage* storage = cvCreateMemStorage(0);  
    CvSeq* contour = 0;  
    cvThreshold(src1, src1,120, 255, CV_THRESH_BINARY);   // 二值化  
    cvNamedWindow("Source",CV_WINDOW_NORMAL);  
    cvShowImage("Source", src1);  
    // 提取轮廓  
    int contour_num = cvFindContours(src1, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);  
    cvZero(dst1);        // 清空数组  
    CvSeq *_contour = contour;   
    double maxarea = 0;  
    double minarea = 100;  
    int m = 0;  
    for( ; contour != 0; contour = contour->h_next )    
    {    
  
        double tmparea = fabs(cvContourArea(contour));  
        if(tmparea < minarea)     
        {    
            cvSeqRemove(contour, 0); // 删除面积小于设定值的轮廓  
            continue;  
        }    
        CvRect aRect = cvBoundingRect( contour, 0 );   
        if ((aRect.width/aRect.height)<1)    
        {    
            cvSeqRemove(contour, 0); //删除宽高比例小于设定值的轮廓  
            continue;  
        }    
        if(tmparea > maxarea)    
        {    
            maxarea = tmparea;  
        }    
        m++;  
        // 创建一个色彩值  
        CvScalar color = CV_RGB( 0, 255, 255 );   
        cvDrawContours(dst1, contour, color, color, -1, 1, 8);   //绘制外部和内部的轮廓 
		cvDrawContours(src2, contour, color, color, -1, 1, 8);
    }    
    contour = _contour;  
    int count = 0;  
    for(; contour != 0; contour = contour->h_next)  
    {    
        count++;  
        double tmparea = fabs(cvContourArea(contour));  
        if (tmparea == maxarea)    
        {    
            CvScalar color = CV_RGB( 255, 0, 0);  
            cvDrawContours(dst1, contour, color, color, -1, 1, 8);
			cvDrawContours(src2, contour, color, color, -1, 1, 8);
        }    
    }    
    printf("The total number of contours is:%d", count);  
    cvNamedWindow("Components", CV_WINDOW_NORMAL);  
    cvShowImage("Components", dst1);
	cvNamedWindow("source2",CV_WINDOW_NORMAL);  
    cvShowImage("source2", src2);
    cvWaitKey(0);  
    cvDestroyWindow("Source");  
    cvReleaseImage(&src1);  
    cvDestroyWindow("Components");  
    cvReleaseImage(&dst1); 
}


void CwindowTestDlg::siftStitching()
{
	// TODO: 在此添加命令处理程序代码
	string name,name1,name2,name3,name4,name5,name6,name7,name8,name9,name10,name11,name12,name13,name14,name15,name16;
	IplImage *img1=NULL, *img2=NULL;/*,*img3=NULL, *img4=NULL,*img5=NULL, *img6=NULL,*img7=NULL, *img8=NULL,*img9=NULL, *img14=NULL,
		*img10=NULL, *img11=NULL,*img12=NULL, *img13=NULL,*img15=NULL, *img16=NULL;*/
	IplImage* imgtotal= NULL;

	int n1,n2;
	struct feature* feat1, * feat2,* feat3,* feat4,* feat5,* feat6,* feat7,* feat8,* feat9,* feat10,* feat11,* feat12,* feat13,* feat14,* feat15,* feat16, * feat; 
	IplImage *img1_Feat=NULL, *img2_Feat=NULL,*img3_Feat=NULL,*img4_Feat=NULL,*img5_Feat=NULL,*img6_Feat=NULL,*img7_Feat=NULL,*img8_Feat=NULL,*img9_Feat=NULL,
		*img10_Feat=NULL,*img11_Feat=NULL,*img12_Feat=NULL,*img13_Feat=NULL,*img14_Feat=NULL,*img15_Feat=NULL,*img16_Feat=NULL;
	
	IplImage* stacked;IplImage* stacked_ransac;
	struct feature** nbrs; 
	struct kd_node* kd_root; 

	///////////////////////////////////////////////////

	//加载图片
	 TCHAR szFilter[] = _T("JPEG文件(*.jpg)|*.jpg|bmp文件(*.bmp)|*.bmp||");   
    //文件类型说明和扩展名间用 | 分隔，每种文件类型间用 | 分隔，末尾用 || 指明。
    CFileDialog fileDlg(TRUE, _T("*.bmp;*.jpg;*.tif"),NULL,OFN_ALLOWMULTISELECT,"image files All Files (*.*) |*.*||",this);
    if(fileDlg.DoModal() != IDOK)     //没有点确定按钮
        return;
    POSITION pos = fileDlg.GetStartPosition();
   
        
        CString szPathName = fileDlg.GetNextPathName(pos);  
    //CString CFileDialog::GetNextPathName( POSITION& pos ) 对于选择了多个文件的情况得到下一个文件位置，并同时返回当前文件名。但必须已经调用过POSITION CFileDialog::GetStartPosition( )来得到最初的POSITION变量。
        TRACE( _T("%s/n"), szPathName);    
        img1 = cvLoadImage(szPathName);
		 szPathName = fileDlg.GetNextPathName(pos);  
    //CString CFileDialog::GetNextPathName( POSITION& pos ) 对于选择了多个文件的情况得到下一个文件位置，并同时返回当前文件名。但必须已经调用过POSITION CFileDialog::GetStartPosition( )来得到最初的POSITION变量。
        TRACE( _T("%s/n"), szPathName);    
        img2 = cvLoadImage(szPathName);
       // imgs.push_back(imgg);
	
	
		//img1 = cvLoadImage( name1.c_str());//打开图1，强制读取为三通道图像
		cvSetImageROI(img1, cvRect(0, 0, 1920, 600));
		IplImage *img11 = cvCreateImage(cvGetSize(img1),img1->depth,img1->nChannels);
		cvCopy(img1, img11, NULL);

		//img2 = cvLoadImage(name2.c_str());
		cvSetImageROI(img2, cvRect(0, 0, 1920, 600));
		IplImage *img22 = cvCreateImage(cvGetSize(img2),img2->depth,img2->nChannels);
		cvCopy(img2, img22, NULL);
		

	//sift特征提取
	cvResetImageROI(img1);
	cvResetImageROI(img2);
	img1_Feat = cvCloneImage(img1);//复制图1，深拷贝，用来画特征点
    img2_Feat = cvCloneImage(img2);//复制图2，深拷贝，用来画特征点
    //默认提取的是LOWE格式的SIFT特征点
    //提取并显示第1幅图片上的特征点
    n1 = sift_features( img11, &feat1 );//检测图1中的SIFT特征点,n1是图1的特征点个数
    export_features("featureb1.txt",feat1,n1);//将特征向量数据写入到文件
    draw_features( img1_Feat, feat1, n1 );//画出特征点
	/*img1 = cvCloneImage(img11);
	cvResetImageROI(img1);*/
    cvNamedWindow(IMG1_FEAT,CV_WINDOW_NORMAL);//创建窗口
    cvShowImage(IMG1_FEAT,img1_Feat);//显示
	cvSaveImage("1_feature.bmp",img1_Feat);

    //提取并显示第2幅图片上的特征点
    n2 = sift_features( img22, &feat2 );//检测图2中的SIFT特征点，n2是图2的特征点个数
    export_features("featureb2.txt",feat2,n2);//将特征向量数据写入到文件
    draw_features( img2_Feat, feat2, n2 );//画出特征点
	/*img2 = cvCloneImage(img22);
	cvResetImageROI(img2);*/
    cvNamedWindow(IMG2_FEAT,CV_WINDOW_NORMAL);//创建窗口
    cvShowImage(IMG2_FEAT,img2_Feat);//显示
	cvSaveImage("2_feature.bmp",img2_Feat);
	/*cvResetImageROI(img1);
	cvResetImageROI(img2);*/

	//根据图1的特征点集feat1建立k-d树，返回k-d树根给kd_root
    kd_root = kdtree_build( feat1, n1 );

    CvPoint pt1,pt2;//连线的两个端点
    double d0,d1;//feat2中每个特征点到最近邻和次近邻的距离
    int matchNum = 0;//经距离比值法筛选后的匹配点对的个数

	//stacked = stack_imgs(img1, img2);
	
	stacked = stack_imgs_horizontal(img1, img2);

    //遍历特征点集feat2，针对feat2中每个特征点feat，选取符合距离比值条件的匹配点，放到feat的fwd_match域中
    for(int i = 0; i < n2; i++ )
    {
        feat = feat2+i;//第i个特征点的指针
        //在kd_root中搜索目标点feat的2个最近邻点，存放在nbrs中，返回实际找到的近邻点个数
        int k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
        if( k == 2 )
        {
            d0 = descr_dist_sq( feat, nbrs[0] );//feat与最近邻点的距离的平方
            d1 = descr_dist_sq( feat, nbrs[1] );//feat与次近邻点的距离的平方
            //若d0和d1的比值小于阈值NN_SQ_DIST_RATIO_THR，则接受此匹配，否则剔除
            if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
            {   //将目标点feat和最近邻点作为匹配点对
                pt2 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );//图2中点的坐标
                pt1 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );//图1中点的坐标(feat的最近邻点)

                pt2.x += img1->width;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点
				//pt2.y += img1->height;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点

                cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );//画出连线
                matchNum++;//统计匹配点对的个数
                feat2[i].fwd_match = nbrs[0];//使点feat的fwd_match域指向其对应的匹配点
            }
        }
        free( nbrs );//释放近邻数组
    }
    fprintf( stderr, "Found %d total matches\n", matchNum ); 
    //显示并保存经距离比值法筛选后的匹配图
    cvNamedWindow(IMG_MATCH1,CV_WINDOW_NORMAL);//创建窗口
    cvShowImage(IMG_MATCH1,stacked);//显示
	cvSaveImage("pipei.bmp",stacked);
	////////////////////////////////////////////////////////

	//此处应统一计算特征点，进行匹配，然后统一进行拼接，直接拼接出大图

	///////////////////////////////////////////////////////

	imgtotal = spliceImage(img1,img2);
	

	////////////////////////////////////////////
	if (NULL != imgtotal)
	{
		cvNamedWindow("拼接后", CV_WINDOW_NORMAL);//创建窗口
		cvShowImage("拼接后",imgtotal);//显示处理后的拼接图
		cvSaveImage("gg12.bmp",imgtotal);
		cvWaitKey(10);
	}
	
	//////////////////////////////////////////////////////////////////////////////

	cvWaitKey(0);

	if(NULL != img1)
	{
		cvReleaseImage(&img1);
	}

	if(NULL != img2)
	{
		cvReleaseImage(&img2);
	}
	
	if (NULL != imgtotal)
	{
		cvReleaseImage(&imgtotal);
		cvDestroyWindow(IMG_MOSAIC_PROC12);//显示处理后的拼接图
	}
}


Vec3b RandomColor(int value)    //<span style="line-height: 20.8px; font-family: sans-serif;">//生成随机颜色函数</span>  
{  
    value=value%255;  //生成0~255的随机数  
    RNG rng;  
    int aa=rng.uniform(0,value);  
    int bb=rng.uniform(0,value);  
    int cc=rng.uniform(0,value);  
    return Vec3b(aa,bb,cc);  
}   
void CwindowTestDlg::Fengshuilin(CCmdUI *pCmdUI)
{
	// TODO: 在此添加命令更新用户界面处理程序代码
	
	 Mat image= cv::Mat(SrcImg);    //载入RGB彩色图像  
	 namedWindow("Source Image", CV_WINDOW_NORMAL);
    imshow("Source Image",image);  
  
    //灰度化，滤波，Canny边缘检测  
    Mat imageGray;  
    cvtColor(image,imageGray,CV_RGB2GRAY);//灰度转换  
    GaussianBlur(imageGray,imageGray,Size(5,5),2);   //高斯滤波      
    Canny(imageGray,imageGray,80,150);    
 
  
    //查找轮廓  
    vector<vector<Point>> contours;    
    vector<Vec4i> hierarchy;    
    findContours(imageGray,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point());    
    Mat imageContours=Mat::zeros(image.size(),CV_8UC1);  //轮廓     
    Mat marks(image.size(),CV_32S);   //Opencv分水岭第二个矩阵参数  
    marks=Scalar::all(0);  
    int index = 0;  
    int compCount = 0;  
    for( ; index >= 0; index = hierarchy[index][0], compCount++ )   
    {  
        //对marks进行标记，对不同区域的轮廓进行编号，相当于设置注水点，有多少轮廓，就有多少注水点  
        drawContours(marks, contours, index, Scalar::all(compCount+1), 1, 8, hierarchy);  
        drawContours(imageContours,contours,index,Scalar(255),1,8,hierarchy);    
    }  
  
    //我们来看一下传入的矩阵marks里是什么东西  
    Mat marksShows;  
    convertScaleAbs(marks,marksShows); 
	namedWindow("marksShow", CV_WINDOW_NORMAL);
	namedWindow("轮廓", CV_WINDOW_NORMAL);
    imshow("marksShow",marksShows);  
    imshow("轮廓",imageContours);  
    watershed(image,marks);  
  
    //我们再来看一下分水岭算法之后的矩阵marks里是什么东西  
    Mat afterWatershed;  
    convertScaleAbs(marks,afterWatershed);
	namedWindow("After Watershed", CV_WINDOW_NORMAL); 
    imshow("After Watershed",afterWatershed);  
  
    //对每一个区域进行颜色填充  
    Mat PerspectiveImage=Mat::zeros(image.size(),CV_8UC3);  
    for(int i=0;i<marks.rows;i++)  
    {  
        for(int j=0;j<marks.cols;j++)  
        {  
            int index=marks.at<int>(i,j);  
            if(marks.at<int>(i,j)==-1)  
            {  
                PerspectiveImage.at<Vec3b>(i,j)=Vec3b(255,255,255);  
            }              
            else  
            {  
                PerspectiveImage.at<Vec3b>(i,j) =RandomColor(index);  
            }  
        }  
    }  
	namedWindow("After ColorFill", CV_WINDOW_NORMAL); 
    imshow("After ColorFill",PerspectiveImage);  
  
    //分割并填充颜色的结果跟原始图像融合  
    Mat wshed;  
    addWeighted(image,0.4,PerspectiveImage,0.6,0,wshed);  
	namedWindow("AddWeighted Image", CV_WINDOW_NORMAL); 
    imshow("AddWeighted Image",wshed);  
  
    waitKey();  
}


void CwindowTestDlg::weather()
{
	

}

int BinarizeImageByOTSU (IplImage * src)  
{   
    assert(src != NULL);  
     
    //get the ROI   
    CvRect rect = cvGetImageROI(src);  
     
    //information of the source image   
    int x = rect.x;  
    int y = rect.y;  
    int width = rect.width;   
    int height = rect.height;  
    int ws = src->widthStep;  
     
    int thresholdValue=1;//阈值   
    int ihist [256] ; // 图像直方图, 256个点   
    int i, j, k,n, n1, n2, Color=0;  
    double m1, m2, sum, csum, fmax, sb;  
    memset (ihist, 0, sizeof (ihist)) ; // 对直方图置 零...   
     
    for (i=y;i< y+height;i++) // 生成直方图   
    {   
        int mul =  i*ws;  
        for (j=x;j<x+width;j++)  
        {   
            //Color=Point (i,j) ;   
            Color = (int)(unsigned char)*(src->imageData + mul+ j);  
            ihist [Color] +=1;  
        }  
    }  
    sum=csum=0.0;  
    n=0;  
    for (k = 0; k <= 255; k++)  
    {   
        sum+= (double) k* (double) ihist [k] ; // x*f (x) 质量矩   
        n +=ihist [k]; //f (x) 质量   
    }  
    // do the otsu global thresholding method   
    fmax = - 1.0;  
    n1 = 0;  
    for (k=0;k<255;k++)   
    {  
        n1+=ihist [k] ;  
        if (! n1)  
        {   
            continue;   
        }  
        n2=n- n1;  
        if (n2==0)   
        {  
            break;  
        }  
        csum+= (double) k*ihist [k] ;  
        m1=csum/ n1;  
        m2= (sum- csum) /n2;  
        sb = ( double) n1* ( double) n2* ( m1 - m2) * (m1- m2) ;  
         
        if (sb>fmax)   
        {  
            fmax=sb;  
            thresholdValue=k;  
        }  
    }  
     
    //binarize the image    
    cvThreshold( src, src ,thresholdValue, 255, CV_THRESH_BINARY );   
    return 0;  
}  

void CwindowTestDlg::Color()
{
	// TODO: 在此添加命令处理程序代码
	IplImage* img0; 
    IplImage* img;
    IplImage* src;
    IplImage* dst_gray;
	img0=SrcImg;
    if(img0)//载入图像   
    {  
		
		//cvSmooth(img0,smooth,CV_GAUSSIAN,3,3);
        img = cvCreateImage( cvSize(img0->width,img0->height), 8, 3);
        cvResize(img0,img);
        src = cvCreateImage( cvSize(img0->width,img0->height), 8, 3);
        cvResize(img0,src);
		//cvSetImageROI(src, cvRect(0, 0, 1920, 600));
		cvSetImageROI(src, cvRect(0, 600, 2000, 3000));
		cvSmooth(src,src,CV_GAUSSIAN,17,17);
		cvResetImageROI(src);
		cvErode(src,src);
		cvErode(src,src);
		cvErode(src,src);
		//cvErode(src,src);
		cvNamedWindow("src", CV_WINDOW_NORMAL);  
        cvShowImage( "src", src);
        cvNamedWindow("img0", CV_WINDOW_NORMAL);
        cvNamedWindow( "img0", 1 );  
        cvShowImage( "img0", img );
        //为轮廓显示图像申请空间,3通道图像，以便用彩色显示   
        IplImage* dst = cvCreateImage( cvGetSize(src), 8, 3);   
        CvMemStorage* storage = cvCreateMemStorage(0);  
         
        CvScalar s;
        for(int i = 0;i < src->height;i++)
        {
            for(int j = 0;j < src->width;j++)
            {
                 
                s = cvGet2D(src,i,j); // get the (i,j) pixel value
                if(s.val[0]<240&&s.val[1]<255&&s.val[2]>120)
                {
                    s.val[0]=235;
                    s.val[1]=206;
                    s.val[2]=135;
                }
				else if(s.val[0]<10&&s.val[1]<240&&s.val[2]<30)
                {
                    s.val[0]=0;
                    s.val[1]=0;
                    s.val[2]=0;
                }
				else if(s.val[0]<10&&s.val[1]<240&&s.val[2]<800&&s.val[2]>30)
                {
                    s.val[0]=0;
                    s.val[1]=252;
                    s.val[2]=124;
                }
                else
                {
                    s.val[0]=0;
                    s.val[1]=0;
                    s.val[2]=0;
                }
 
                cvSet2D(src,i,j,s);   //设置像素
            }
        }
		cvSetImageROI(src, cvRect(0, 300, 1000, 100));
		cvErode(src,src);
		cvResetImageROI(src);
		cvSetImageROI(src, cvRect(0, 500, 2000, 100));
		cvErode(src,src);
		cvErode(src,src);
		cvErode(src,src);
		cvErode(src,src);
		cvErode(src,src);
		cvErode(src,src);
		cvResetImageROI(src);
		cvSetImageROI(src, cvRect(0, 600, 2000, 3000));
		cvDilate(src,src);
		cvDilate(src,src);
		cvDilate(src,src);
		cvDilate(src,src);
		cvResetImageROI(src);
        cvNamedWindow( "image", CV_WINDOW_NORMAL ); 
        cvShowImage("image",src);
 
        dst_gray=cvCreateImage(cvGetSize(src),8,1);
 
        cvCvtColor(src,dst_gray,CV_BGR2GRAY);
        cvNamedWindow( "灰度图",CV_WINDOW_NORMAL ); 
        cvShowImage("灰度图",dst_gray);
 
 
        //可动态增长元素序列   
        CvSeq* contour = 0;  
        //对图像进行自适二值化   
        BinarizeImageByOTSU(dst_gray);  
		cvNamedWindow("Source", CV_WINDOW_NORMAL);
        cvNamedWindow( "Source", 1 );  
        cvShowImage( "Source", dst_gray );  
        //在二值图像中寻找轮廓   
        cvFindContours( dst_gray, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );  
 
 
        cvZero( dst );//清空数组   
        cvCvtColor(dst_gray,dst,CV_GRAY2BGR);  
        //目标轮廓最小下限   
        int mix_area = 25000;  
        //目标轮廓最大上限   
        int max_area = 350000;  
        //可存放在1-，2-，3-，4-TUPLE类型的捆绑数据的容器   
        CvScalar color = CV_RGB( 255, 0, 0);  
        //在图像中绘制外部和内部的轮廓   
        for( ; contour != 0; contour = contour->h_next)  
        {  
            //取得轮廓的最小矩形   
            CvRect aRect = cvBoundingRect( contour, 1 );  
            //取得矩形的面积   
            int tmparea=aRect.height*aRect.height;    
            if (((double)aRect.width/(double)aRect.height>1.2)  
                && ((double)aRect.width/(double)aRect.height<2.65)&&((double)aRect.width>30)&&((double)aRect.width<133))  
            {  
                cvRectangle(img,cvPoint(aRect.x,aRect.y),cvPoint(aRect.x+aRect.width ,aRect.y+aRect.height),color,2);  
            }  
        }  
        //cvNamedWindow( "img", CV_WINDOW_NORMAL );  
        //cvShowImage( "img", img );
        cvWaitKey(0); 
        cvDestroyWindow("img0");
        cvDestroyWindow("img");
        cvDestroyWindow("灰度图");
        cvReleaseImage(&dst);  
        cvDestroyWindow("Source");  
        cvReleaseImage(&dst_gray);
        cvReleaseImage(&src);
        cvReleaseImage(&img);
        cvReleaseImage(&img0);
         
       
    }     
   

}


void CwindowTestDlg::weather2()
{
	
	MessageBox(_T("晚疫病")); 
	
	
	/*Mat a=cv::Mat(SrcImg);
	Mat d;
    flip(a,d,0);

	
	Mat b=d(Range(1,235), Range(1,a.cols));
	Mat c;
	int num=0 ;
	cvtColor(b,c,CV_BGR2GRAY);
	for(int i=0;i<c.rows;i++)
	{
		for(int j=0;j<c.cols;j++)
		    {
				if(c.at<uchar>(i,j)>=240)
                     num++;
		     }
	}
	
	if (num<500)
	{  
		MessageBox(_T("晴天！"));  
	} 
	else
	{
	  MessageBox(_T("多云！"));
	}*/
}





 

 











static void help()  
{  
    cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"  
        "and then grabcut will attempt to segment it out.\n"  
        "Call:\n"  
        "./grabcut <image_name>\n"  
        "\nSelect a rectangular area around the object you want to segment\n" <<  
        "\nHot keys: \n"  
        "\tESC - quit the program\n"  
        "\tr - restore the original image\n"  
        "\tn - next iteration\n"  
        "\n"  
        "\tleft mouse button - set rectangle\n"  
        "\n"  
        "\tCTRL+left mouse button - set GC_BGD pixels\n"  
        "\tSHIFT+left mouse button - set CG_FGD pixels\n"  
        "\n"  
        "\tCTRL+right mouse button - set GC_PR_BGD pixels\n"  
        "\tSHIFT+right mouse button - set CG_PR_FGD pixels\n" << endl;  
}  
  
const Scalar RED = Scalar(0,0,255);  
const Scalar PINK = Scalar(230,130,255);  
const Scalar BLUE = Scalar(255,0,0);  
const Scalar LIGHTBLUE = Scalar(255,255,160);  
const Scalar GREEN = Scalar(0,255,0);  
  
const int BGD_KEY = CV_EVENT_FLAG_CTRLKEY;  //Ctrl键  
const int FGD_KEY = CV_EVENT_FLAG_SHIFTKEY; //Shift键  
  
static void getBinMask( const Mat& comMask, Mat& binMask )  
{  
    if( comMask.empty() || comMask.type()!=CV_8UC1 )  
        CV_Error( CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );  
    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )  
        binMask.create( comMask.size(), CV_8UC1 );  
    binMask = comMask & 1;  //得到mask的最低位,实际上是只保留确定的或者有可能的前景点当做mask  
}  
class GCApplication  
{  
public:  
	enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };  
    static const int radius = 2;  
    static const int thickness = -1;  
  
    void reset();  
    void setImageAndWinName( const Mat& _image, const string& _winName );  
    void showImage() const;  
    void mouseClick( int event, int x, int y, int flags, void* param );  
    int nextIter();  
    int getIterCount() const { return iterCount; }  
private:  
    void setRectInMask();  
    void setLblsInMask( int flags, Point p, bool isPr );  
  
    const string* winName;  
    const Mat* image;  
    Mat mask;  
    Mat bgdModel, fgdModel;  
  
    uchar rectState, lblsState, prLblsState;  
    bool isInitialized;  
  
    Rect rect;  
    vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;  
    int iterCount;  
}; 
void GCApplication::reset()  
{  
    if( !mask.empty() )  
        mask.setTo(Scalar::all(GC_BGD));  
    bgdPxls.clear(); fgdPxls.clear();  
    prBgdPxls.clear();  prFgdPxls.clear();  
  
    isInitialized = false;  
    rectState = NOT_SET;    //NOT_SET == 0  
    lblsState = NOT_SET;  
    prLblsState = NOT_SET;  
    iterCount = 0;  
}  
  
/*给类的成员变量赋值而已*/  
void GCApplication::setImageAndWinName( const Mat& _image, const string& _winName  )  
{  
    if( _image.empty() || _winName.empty() )  
        return;  
    image = &_image;  
    winName = &_winName;  
    mask.create( image->size(), CV_8UC1);  
    reset();  
}  
  
/*显示4个点，一个矩形和图像内容，因为后面的步骤很多地方都要用到这个函数，所以单独拿出来*/  
void GCApplication::showImage() const  
{  
    if( image->empty() || winName->empty() )  
        return;  
  
    Mat res;  
    Mat binMask;  
    if( !isInitialized )  
        image->copyTo( res );  
    else  
    {  
        getBinMask( mask, binMask );  
        image->copyTo( res, binMask );  //按照最低位是0还是1来复制，只保留跟前景有关的图像，比如说可能的前景，可能的背景  
    }  
  
    vector<Point>::const_iterator it;  
    /*下面4句代码是将选中的4个点用不同的颜色显示出来*/  
    for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )  //迭代器可以看成是一个指针  
        circle( res, *it, radius, BLUE, thickness );  
    for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )  //确定的前景用红色表示  
        circle( res, *it, radius, RED, thickness );  
    for( it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it )  
        circle( res, *it, radius, LIGHTBLUE, thickness );  
    for( it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it )  
        circle( res, *it, radius, PINK, thickness );  
  imwrite("Grabcut.jpg",res);
  imshow("Grabcut",res);
    /*画矩形*/  
    if( rectState == IN_PROCESS || rectState == SET )  
        rectangle( res, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), GREEN, 2);  
  
    imshow( *winName, res ); 
	
}  
  
/*该步骤完成后，mask图像中rect内部是3，外面全是0*/  
void GCApplication::setRectInMask()  
{  
    assert( !mask.empty() );  
    mask.setTo( GC_BGD );   //GC_BGD == 0  
    rect.x = max(0, rect.x);  
    rect.y = max(0, rect.y);  
    rect.width = min(rect.width, image->cols-rect.x);  
    rect.height = min(rect.height, image->rows-rect.y);  
    (mask(rect)).setTo( Scalar(GC_PR_FGD) );    //GC_PR_FGD == 3，矩形内部,为可能的前景点  
}  
  
void GCApplication::setLblsInMask( int flags, Point p, bool isPr )  
{  
    vector<Point> *bpxls, *fpxls;  
    uchar bvalue, fvalue;  
    if( !isPr ) //确定的点  
    {  
        bpxls = &bgdPxls;  
        fpxls = &fgdPxls;  
        bvalue = GC_BGD;    //0  
        fvalue = GC_FGD;    //1  
    }  
    else    //概率点  
    {  
        bpxls = &prBgdPxls;  
        fpxls = &prFgdPxls;  
        bvalue = GC_PR_BGD; //2  
        fvalue = GC_PR_FGD; //3  
    }  
    if( flags & BGD_KEY )  
    {  
        bpxls->push_back(p);  
        circle( mask, p, radius, bvalue, thickness );   //该点处为2  
    }  
    if( flags & FGD_KEY )  
    {  
        fpxls->push_back(p);  
        circle( mask, p, radius, fvalue, thickness );   //该点处为3  
    }  
}  
  
/*鼠标响应函数，参数flags为CV_EVENT_FLAG的组合*/  
void GCApplication::mouseClick( int event, int x, int y, int flags, void* )  
{  
    // TODO add bad args check  
    switch( event )  
    {  
    case CV_EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels  
        {  
            bool isb = (flags & BGD_KEY) != 0,  
                isf = (flags & FGD_KEY) != 0;  
            if( rectState == NOT_SET && !isb && !isf )//只有左键按下时  
            {  
                rectState = IN_PROCESS; //表示正在画矩形  
                rect = Rect( x, y, 1, 1 );  
            }  
            if ( (isb || isf) && rectState == SET ) //按下了alt键或者shift键，且画好了矩形，表示正在画前景背景点  
                lblsState = IN_PROCESS;  
        }  
        break;  
    case CV_EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels  
        {  
            bool isb = (flags & BGD_KEY) != 0,  
                isf = (flags & FGD_KEY) != 0;  
            if ( (isb || isf) && rectState == SET ) //正在画可能的前景背景点  
                prLblsState = IN_PROCESS;  
        }  
        break;  
    case CV_EVENT_LBUTTONUP:  
        if( rectState == IN_PROCESS )  
        {  
            rect = Rect( Point(rect.x, rect.y), Point(x,y) );   //矩形结束  
            rectState = SET;  
            setRectInMask();  
            assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );  
            showImage();  
        }  
        if( lblsState == IN_PROCESS )   //已画了前后景点  
        {  
            setLblsInMask(flags, Point(x,y), false);    //画出前景点  
            lblsState = SET;  
            showImage();  
        }  
        break;  
    case CV_EVENT_RBUTTONUP:  
        if( prLblsState == IN_PROCESS )  
        {  
            setLblsInMask(flags, Point(x,y), true); //画出背景点  
            prLblsState = SET;  
            showImage();  
        }  
        break;  
    case CV_EVENT_MOUSEMOVE:  
        if( rectState == IN_PROCESS )  
        {  
            rect = Rect( Point(rect.x, rect.y), Point(x,y) );  
            assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );  
            showImage();    //不断的显示图片  
        }  
        else if( lblsState == IN_PROCESS )  
        {  
            setLblsInMask(flags, Point(x,y), false);  
            showImage();  
        }  
        else if( prLblsState == IN_PROCESS )  
        {  
            setLblsInMask(flags, Point(x,y), true);  
            showImage();  
        }  
        break;  
    }  
}  
  
/*该函数进行grabcut算法，并且返回算法运行迭代的次数*/  
int GCApplication::nextIter()  
{  
    if( isInitialized )  
        //使用grab算法进行一次迭代，参数2为mask，里面存的mask位是：矩形内部除掉那些可能是背景或者已经确定是背景后的所有的点，且mask同时也为输出  
        //保存的是分割后的前景图像  
        grabCut( *image, mask, rect, bgdModel, fgdModel, 1 );  
    else  
    {  
        if( rectState != SET )  
            return iterCount;  
  
        if( lblsState == SET || prLblsState == SET )  
            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK );  
        else  
            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT );  
  
        isInitialized = true;  
    }  
    iterCount++;  
  
    bgdPxls.clear(); fgdPxls.clear();  
    prBgdPxls.clear(); prFgdPxls.clear();  
	
  
    return iterCount;  
}  
  
GCApplication gcapp;  
  
static void on_mouse( int event, int x, int y, int flags, void* param )  
{  
    gcapp.mouseClick( event, x, y, flags, param );  
}  

void CAboutDlg::bblocation()
{
    

}


void CwindowTestDlg::BBlocation()
{
	// TODO: 在此添加命令处理程序代码
	Mat image;
	if(flage==true){
	image= cv::Mat(SrcImg);
	}
	else{
	string filename = "HSV.jpg";  
    image = imread( filename, 1 );
	}
	cout<<flage;
	
	//HSV();
	//Mat image= cv::Mat(SrcImg);
    if( image.empty() )  
    {  
        cout << "\n Durn, couldn't read image filename " << endl;  

    }  
  
    help();  
  
    const string winName = "image";  
    cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );  
    cvSetMouseCallback( winName.c_str(), on_mouse, 0 );  
  
    gcapp.setImageAndWinName( image, winName );  
    gcapp.showImage();
  
    for(;;)  
    {  
        int c = cvWaitKey(0);  
        switch( (char) c )  
        {  
        case '\x1b':  
            cout << "Exiting ..." << endl;  
            goto exit_main;  
        case 'r':  
            cout << endl;  
            gcapp.reset();  
            gcapp.showImage();  
            break;  
        case 'n':  
            int iterCount = gcapp.getIterCount();  
            cout << "<" << iterCount << "... ";  
            int newIterCount = gcapp.nextIter(); 
            if( newIterCount > iterCount )  
            {  
                gcapp.showImage();  
                cout << iterCount << ">" << endl; 
				

            }  
            else  
                cout << "rect must be determined" << endl;  
            break;  
        }  
    }  
	
  
exit_main:  
    cvDestroyWindow( winName.c_str() );  
}


void CwindowTestDlg::Median_filter()
{
	
	Mat image= cv::Mat(SrcImg);
	//imshow("005.jpg",image);
	Mat Salt_Image;  
    image.copyTo(Salt_Image);
	Mat image3, image4;  
	medianBlur(Salt_Image, image4, 3);    
    imshow("中值滤波", image4);
	imwrite("中值滤波.jpg", image4);
	SrcImg=&(IplImage) image4;
    waitKey();  

}



Mat img;  
//灰度值归一化  
Mat bgr;  
//HSV图像  
Mat hsv;  
//色相  
int hmin = 0;  
int hmin_Max = 360;  
int hmax = 78;  
int hmax_Max = 360;  
//饱和度  
int smin = 32;  
int smin_Max = 255;  
int smax = 255;  
int smax_Max = 255;  
//亮度  
int vmin = 36;  
int vmin_Max = 255;  
int vmax = 255;  
int vmax_Max = 255;  
//显示原图的窗口  
string windowName = "src";  
//输出图像的显示窗口  
string dstName = "dst";  
//输出图像  
Mat dst_HSV;  
//回调函数  
void callBack()  
{  
    //输出图像分配内存  
    dst_HSV = Mat::zeros(img.size(), CV_32FC3);  
    //掩码  
    Mat mask;  
    inRange(hsv, Scalar(hmin, smin / float(smin_Max), vmin / float(vmin_Max)), Scalar(hmax, smax / float(smax_Max), vmax / float(vmax_Max)), mask);  
    //只保留  
    for (int r = 0; r < bgr.rows; r++)  
    {  
        for (int c = 0; c < bgr.cols; c++)  
        {  
            if (mask.at<uchar>(r, c) == 255)  
            {  
                dst_HSV.at<Vec3f>(r, c) = bgr.at<Vec3f>(r, c);  
            }  
        }  
    }  
    //输出图像  
	//Mat element = getStructuringElement(MORPH_RECT, Size(3, 3)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
    //腐蚀操作
	//Mat out;
    //morphologyEx(dst, out, MORPH_OPEN,element);
    imshow(dstName, dst_HSV); 
	//imshow("fushihou", out);
    //保存图像  
    dst_HSV.convertTo(dst_HSV, CV_8UC3, 255.0, 0);  
    imwrite("HSV.jpg", dst_HSV);
	//SrcImg=&(IplImage) dst_HSV;
}  

void CwindowTestDlg::hsv_color()
{
	flage=false;
	// TODO: 在此添加命令处理程序代码 
	//img = imread("中值滤波.jpg", IMREAD_COLOR); 
	 img= cv::Mat(SrcImg);
	// img= bsSrc;
    if (!img.data || img.channels() != 3)  
        cout<<"出错"<<endl;  
    //imshow(windowName, img);  
    //彩色图像的灰度值归一化  
    img.convertTo(bgr, CV_32FC3, 1.0 / 255, 0);  
    //颜色空间转换  
    cvtColor(bgr, hsv, COLOR_BGR2HSV);  
    //定义输出图像的显示窗口  
    //namedWindow(dstName,CV_WINDOW_NORMAL);  
    //调节色相 H  
    //createTrackbar("hmin", dstName, &hmin, hmin_Max, callBack);  
    //createTrackbar("hmax", dstName, &hmax, hmax_Max, callBack);  
    ////调节饱和度 S  
    //createTrackbar("smin", dstName, &smin, smin_Max, callBack);  
    //createTrackbar("smax", dstName, &smax, smax_Max, callBack);  
    ////调节亮度 V  
    //createTrackbar("vmin", dstName, &vmin, vmin_Max, callBack);  
    //createTrackbar("vmax", dstName, &vmax, vmax_Max, callBack);  
    //callBack(0,0);
	callBack();
    waitKey(0);  
}


void CwindowTestDlg::lunkuo()
{
	Mat src_image = imread("Grabcut.jpg");  
    if(!src_image.data)  
    {  
        cout << "src image load failed!" << endl;  
    }  
    namedWindow("src image", WINDOW_NORMAL);  
    imshow("src image", src_image);  
  
    /*此处高斯去燥有助于后面二值化处理的效果*/  
    Mat blur_image;  
    GaussianBlur(src_image, blur_image, Size(15, 15), 0, 0);  
    imshow("GaussianBlur", blur_image); 
	imwrite("高斯去噪.jpg",blur_image);
  
    /*灰度变换与二值化*/  
    Mat gray_image, binary_image;  
    cvtColor(blur_image, gray_image, COLOR_BGR2GRAY);  
    threshold(gray_image, binary_image, 0, 255, THRESH_BINARY);  
    imshow("binary", binary_image);
	imwrite("二值变换.jpg",binary_image);
  
    /*形态学闭操作*/  
    Mat morph_image;  
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));  
    morphologyEx(binary_image, morph_image, MORPH_CLOSE, kernel, Point(-1, -1), 2);  
    imshow("morphology", morph_image); 
	imwrite("形态学闭操作.jpg",morph_image);
  
    /*查找外轮廓*/  
    vector< vector<Point> > contours;  
    vector<Vec4i> hireachy;  
    findContours(morph_image, contours, hireachy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());  
    Mat result_image = Mat::zeros(src_image.size(), CV_8UC3);  
    for(size_t t = 0; t < contours.size(); t++)  
    {  
        /*过滤掉小的干扰轮廓*/  
        Rect rect = boundingRect(contours[t]);  
        if(rect.width < src_image.cols/2)  
            continue;  
        if(rect.width > (src_image.cols - 20))  
            continue;  
  
        /*计算面积与周长*/  
        double area = contourArea(contours[t]);  
        double len = arcLength(contours[t], true);  
  
        drawContours(result_image, contours, static_cast<int>(t), Scalar(0, 0, 255), 1, 8, hireachy);  
        cout << "area of start cloud: " << area << endl;  
        cout << "len of start cloud: " << len << endl;  
    }  
  
    imshow("result image", result_image); 
	imwrite("轮廓.jpg",result_image);
}
