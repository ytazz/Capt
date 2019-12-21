/*

   Interpolation calss

   - 4 dim ( 5 parameters )
     - x0, xf, v0, vf, tf
   - 2 dim ( 3 parameters )
     - x0, xf, tf

 */

#ifndef __INTERPOLATION_H__
#define __INTERPOLATION_H__

namespace Capt {

class Interp3 {
public:
  Interp3();
  ~Interp3();

  void set(const double x0, const double xf, const double v0,
           const double vf, const double tf);
  double get(const double t);

private:
  double coef[4];
};

class Interp2 {
public:
  Interp2();
  ~Interp2();

  void   set(const double x0, const double xf, const double vf, const double vf, const double tf);
  double get(const double t);

private:
  double coef[3];
};

}

#endif