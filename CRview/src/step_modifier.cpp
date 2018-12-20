#include "../include/step_modifier.h"


using namespace std;

StepModifier::StepModifier()
{
  readTable();
}

StepModifier::~StepModifier()
{
}

//void StepModifier::plotCR_pended()
//{
//  FILE *fp_1;
//  FILE *fp_2;
//  FILE *fp_3;
//  FILE *fp_current;
//
//  fopen_s(&fp_1, "1step.csv", "w");
//  fopen_s(&fp_2, "2step.csv", "w");
//  fopen_s(&fp_3, "3step.csv", "w");
//  fopen_s(&fp_current, "current_state.csv", "w");
//
//  fprintf(fp_current, "%f, %f, %f, %f\n",
//          current_state.icp.r, current_state.icp.th,
//          current_state.swf.r, current_state.swf.th);
//
//  for (size_t i = 0; i < captureRegion.size(); i++) {
//    switch (captureRegion[i].n) {
//    case 1:
//      fprintf(fp_1, "%f, %f\n", captureRegion[i].dsf.r, captureRegion[i].dsf.th);
//    case 2:
//      fprintf(fp_2, "%f, %f\n", captureRegion[i].dsf.r, captureRegion[i].dsf.th);
//    case 3:
//      fprintf(fp_2, "%f, %f\n", captureRegion[i].dsf.r, captureRegion[i].dsf.th);
//    }
//  }
//
//  fclose(fp_1);
//  fclose(fp_2);
//  fclose(fp_3);
//  fclose(fp_current);
//
//  p("set xrange [-0.22:0.22]");
//  p("set yrange [0.0:0.22]");
//  p("set size ratio -1");
//  p("set polar");
//  p("set grid polar 30");
//
//  p("plot \"3step.csv\" u 2:1 t '3step' with points pointsize 1.5 pointtype 7 lc 3, \
//       \"2step.csv\" u 2:1 t '2step' with points pointsize 1.5 pointtype 7 lc 2, \
//       \"1step.csv\" u 2:1 t '1step' with points pointsize 1.5 pointtype 7 lc 1, \
//       \"current_state.csv\" u 2:1 t 'Capture Point' with points pointsize 2.0 pointtype 7 lc \"orange\", \
//       \"current_state.csv\" u 4:3 t 'Current Swing Foot' with points pointsize 2.0 pointtype 7 lc \"green\"");
// }

void StepModifier::initPlotting(string output, string supportfoot)
{
	p("set encoding utf8");
	p("set size ratio -1");
	p("set polar");
	p("set grid polar 0");

	if (output == "file")
	{
		p("set terminal gif animate optimize delay 50 size 600,900");
		p("set output 'plot.gif'");
	}

	p("set yrange [-0.22:0.22]");
	if (supportfoot=="right")
	{
		p("set xrange [0.22:-0.1]");
		p("set link y2");
		p("set y2tics");
		p("set ytics scale 0");
		p("set ytics format \"\"");
		p("set y2label \"X [m]\" font \",15\" rotate by -90");
		p("set y2label offset -2, 0");
	}
	else {
		p("set xrange [0.0:0.22]");
		p("set ylabel \"X [m]\" font \",15\" rotate by 90");
		p("set ylabel offset 2, 0");
	}
	p("set xlabel \"Y [m]\" font \",15\"");
	
	p("set xtics nomirror");
	p("set ytics nomirror");
	p("set key bmargin center");
	p("set key spacing 1");
		
	p("set theta clockwise top");	
	p("set rtics scale 0");
	p("set rtics format \"\"");
	p("unset raxis");

	p("unset key");
}


void StepModifier::plotCR()
{
	fprintf(p.gp, "plot");

	vector<Input> temp = captureRegion;

	if (!temp.empty())
	{
		sort(temp.begin(), temp.end(), [](const Input & a, const Input & b) {
			return a.n > b.n;
		}); //descending

		int step_num = temp[0].n;

		while (step_num != 0)
		{
			fprintf(p.gp, "'-' t '%d' with points pointsize 1.5 pointtype 7 lc \"%s\",", step_num, p.int_color[step_num].c_str());
			step_num--;
		}

		fprintf(p.gp, "'-' t 'Capture Point' with points pointsize 3.0 pointtype 7 lc \"orange\",");
		fprintf(p.gp, "'-' t 'Current Swing Foot' with points pointsize 3.0 pointtype 7 lc \"green\"\n");

		size_t ind = 0;
		step_num = temp[0].n;

		while (step_num != 0)
		{
			while (temp[ind].n == step_num)
			{
				fprintf(p.gp, "%f %f\n", temp[ind].dsf.th, temp[ind].dsf.r);
				ind++;
			}
			fprintf(p.gp, "e\n");
			step_num--;
		}
	}
	else {
		fprintf(p.gp, "'-' t 'Capture Point' with points pointsize 3.0 pointtype 7 lc \"orange\",");
		fprintf(p.gp, "'-' t 'Current Swing Foot' with points pointsize 3.0 pointtype 7 lc \"green\"\n");
	}
	
	fprintf(p.gp, "%f %f\n", current_state.icp.th, current_state.icp.r);
	fprintf(p.gp, "e\n");
	fprintf(p.gp, "%f %f\n", current_state.swf.th, current_state.swf.r);
	fprintf(p.gp, "e\n");
	fflush(p.gp);
}

void StepModifier::setCurrent(State state, PolarCoord swft_position)
{
  current_state = state;
  current_swft_position = swft_position;
}

void StepModifier::readTable()
{
  FILE *fp;
  errno_t error;
  error = fopen_s(&fp, "gridsTable.csv", "r");
  if (error != 0)
  {
    cout << "gridsTable cannot be opened" << endl;
    exit(EXIT_FAILURE);
  }else{
    float buf[4];

    while (fscanf_s(fp, "%f, %f, %f, %f", &buf[0], &buf[1], &buf[2], &buf[3]) != EOF)
    {
      cp_r.push_back(buf[0]);
      cp_th.push_back(buf[1]);
      foot_r.push_back(buf[2]);
      foot_th.push_back(buf[3]);
    }
    cout << "table is loaded" << endl;
    fclose(fp);
  }
}

State StepModifier::closestGridfrom(State a)
{
  State grid;
  grid.icp.r = closest(cp_r, a.icp.r);
  grid.icp.th = closest(cp_th, a.icp.th);
  grid.swf.r = closest(foot_r, a.swf.r);
  grid.swf.th = closest(foot_th, a.swf.th);
  return grid;
  // cout << grid.icp.r << ", "
  //      << grid.icp.th << ", "
  //      << grid.swf.r << ", "
  //      << grid.swf.th << endl;
}

float StepModifier::closest(std::vector<float> const& vec, float val)
{
  float lower, upper;
  auto const it = lower_bound(vec.begin(), vec.end(), val);

  if (it == vec.begin()) { // no smaller value than val in vector
    lower = *it;
    upper = *it;
  }
  else if (it == vec.end()) {  // no bigger value than val in vector
    lower = *(it - 1);
    upper = *(it - 1);
  } else {
    lower = *(it - 1);
    upper = *it;
  }

  if (abs(lower - val) < abs(upper - val)) {
    return lower;
  }else{
    return upper;
  }
}

void StepModifier::findCaptureRegion(std::vector<Data> *d)
{
  State grid = closestGridfrom(current_state);
  captureRegion.clear();

  for (size_t i = 0; i < d->size(); i++) {
    if ((*d)[i].state == grid) {
      captureRegion.push_back((*d)[i].input);
      if ((*d)[i].state.swf.th != (*d)[i + 1].state.swf.th) {
		break;
      }
    }
  }

  //for (size_t i = 0; i < captureRegion.size(); i++)
  //{
  //  cout << captureRegion[i].dsf.r << ' '
  //       << captureRegion[i].dsf.th << ' '
  //       << captureRegion[i].n << endl;
  //}

}

float StepModifier::distTwoPolar(const PolarCoord &a, const PolarCoord &b)
{
  return (a.r)*(a.r) + (b.r)*(b.r) - 2*a.r*b.r*cos(a.th-b.th);
}

//bool StepModifier::cmpDistFromCurrentSwFt(Input a, Input b)
//{
//  float t1 = distTwoPolar(a.dsf, current_swft_position);
//  float t2 = distTwoPolar(b.dsf, current_swft_position);
//  return t1 < t2;
//}


Input StepModifier::modifier()
{
  vector<Input> temp;

  for (size_t n_step = 1; n_step < 4; n_step++) {
	  auto i = captureRegion.begin(), end = captureRegion.end();
	  while (i != end)
	  {
		  i = std::find_if(i, end, [n_step](const Input& d) {return d.n == n_step;});
		  if (i != end)
		  {
			  temp.push_back(*i);
			  i++;
		  }
	  }
	  if (!temp.empty()) {
		  break;
	  }
  }
  
  sort(temp.begin(), temp.end(), [this](const Input & a, const Input & b) {
	  float t_a = distTwoPolar(a.dsf, current_swft_position);
	  float t_b = distTwoPolar(b.dsf, current_swft_position);
	  return t_a < t_b;
  }); //ascending



  return temp[0];
  //for (size_t i = 0; i < temp.size(); i++)
  //{
  //  cout << temp[i].dsf.r << ' '
  //       << temp[i].dsf.th << ' '
  //       << temp[i].n << endl;
  //}
}
















//
