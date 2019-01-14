/*
    author G. KIM
*/

#include "../include/Gnuplot.h"

gnuplot::gnuplot()
{
	gp = popen("gnuplot -persist", "w");

	if (!gp)
	{
		cerr << ("gnuplot not found");
	}
	int_color = {
		{ 1, "grey80" },
		{ 2, "grey50" },
		{ 3, "grey20" },
		{ 4, "black" },
		{ 5, "gold" },
		{ 6, "cyan" },
		{ 7, "violet" },
		{ 8, "purple" }
	};

}

gnuplot::~gnuplot()
{
	fprintf(gp, "exit\n");
	pclose(gp);
}

void gnuplot::operator() (const string &command)
{
	fprintf(gp, "%s\n", command.c_str());
	fflush(gp);
}
