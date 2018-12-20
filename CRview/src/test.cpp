#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <Windows.h>
#include <ctime>
#include "../include/data_struct.h"
#include "../include/csv_reader.h"
#include "../include/step_modifier.h"

float PI = 3.1415;

using namespace std;

int main() {
	CSVReader loader("data.csv");
	vector<Data> dataList = loader.getData();

	StepModifier mod;

	State test_state;
	test_state.icp.r = 0.056;
	test_state.icp.th = 0.0;
	test_state.swf.r = 0.158;
	test_state.swf.th = (90.0)*PI / 180.0;

	PolarCoord test_swf;
	test_swf.r = 0.152;
	test_swf.th = 2.406;

	mod.setCurrent(test_state, test_swf);
	mod.findCaptureRegion(&dataList);
	Input temp = mod.modifier();

	mod.initPlotting("file", "right");
	int count = 0;

	while (count < 19)
	{
		test_state.icp.th += ((360.0)*PI / 180.0) / 20;
		mod.setCurrent(test_state, test_swf);
		mod.findCaptureRegion(&dataList);
		mod.plotCR();
		Sleep(500);
		count++;
	}

	//State test_state;
	//test_state.icp.r = 0.056;
	//test_state.icp.th = 2.65;
	//test_state.swf.r = 0.158;
	//test_state.swf.th = (159.0)*PI / 180.0;

	//PolarCoord test_swf;
	//test_swf.r = 0.152;
	//test_swf.th = 2.406;

	//mod.setCurrent(test_state, test_swf);
	//mod.findCaptureRegion(&dataList);
	//Input temp = mod.modifier();

	////cout << temp.dsf.r << ", " << temp.dsf.th << endl;
	//mod.initPlotting("file", "right");
	//int count = 0;

	//while ((20.0)*PI / 180.0 < test_state.swf.th
	//	&& test_state.swf.th < (160.0)*PI / 180.0)
	//{
	//	test_state.swf.th -= ((140.0)*PI / 180.0) / 20;
	//	mod.setCurrent(test_state, test_swf);
	//	mod.findCaptureRegion(&dataList);
	//	mod.plotCR();
	//	Sleep(500);
	//}


	return 0;
}