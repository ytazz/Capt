/*
    author G. KIM
*/

#include <Gnuplot.h>

gnuplot::gnuplot()
{
	#ifdef LINUX
	gp = popen("gnuplot -persist", "w");
	#endif
	#ifdef WINDOWS
	gp = _popen("gnuplot -persist", "w");
	#endif
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
	#ifdef LINUX
	pclose(gp);
	#endif
	#ifdef WINDOWS
	_pclose(gp);
	#endif
}

void gnuplot::operator() (const string &command)
{
	fprintf(gp, "%s\n", command.c_str());
	fflush(gp);
}
