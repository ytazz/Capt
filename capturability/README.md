# Tex

$ platex main.tex
$ dvipdfmx main.dvi

sudo cp /home/kuribayashi/study/capturability/lib/libCapturability.so /usr/local/lib/libCapturability.so

# Installation

## Gnuplot

    $ sudo apt-get install libgd2-dev

    $ cd ~/Downloads
    $ wget http://sourceforge.net/projects/gnuplot/files/gnuplot/5.2.7/gnuplot-5.2.7.tar.gz
    $ tar zxvf gnuplot-5.2.7.tar.gz
    $ cd gnuplot-5.2.7
    $ ls
    $ ./configure --with-gd
    $ make -j4
    $ sudo make install
