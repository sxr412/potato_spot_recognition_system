
// windowTestDlg.h : 头文件
//

#pragma once


// CwindowTestDlg 对话框
class CwindowTestDlg : public CDialogEx
{
// 构造
public:
	CwindowTestDlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
	enum { IDD = IDD_WINDOWTEST_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedMfcmenubutton1();
	afx_msg void On32771();
	afx_msg void On32774();
	afx_msg void canny();
	afx_msg void LaLace();
	afx_msg void MaximumConnectedarea();
	afx_msg void siftStitching();
	afx_msg void Fengshuilin(CCmdUI *pCmdUI);
	afx_msg void weather();
	afx_msg void Color();
	afx_msg void weather2();
	afx_msg void BBlocation();
	afx_msg void Median_filter();
	afx_msg void hsv_color();
	afx_msg void lunkuo();
};
