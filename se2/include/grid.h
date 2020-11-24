#pragma once

#include "base.h"
#include "state.h"
#include "base.h"
#include <iostream>
#include <string>
#include <vector>
#include <array>

namespace Capt {

struct Grid1D{
	real_t  min;
	real_t  max;
	real_t  stp;
	int     num;
	std::vector<real_t> val;

	void init ();
	int  round(real_t v);
	void indexRange(real_t fmin, real_t fmax, int& imin, int& imax);
};

struct Index2D : std::array<int, 2>{
	Index2D(){}
	Index2D(int i0, int i1){
		at(0) = i0;
		at(1) = i1;
	}
};

struct Index3D : std::array<int, 3>{
	Index3D(){}
	Index3D(int i0, int i1, int i2){
		at(0) = i0;
		at(1) = i1;
		at(2) = i2;
	}
};

struct Index4D : std::array<int, 4>{
	Index4D(){}
	Index4D(int i0, int i1, int i2, int i3){
		at(0) = i0;
		at(1) = i1;
		at(2) = i2;
		at(3) = i3;
	}
};

struct Grid2D{
	Grid1D* axis[2];

	int     num      ();
	int     toIndex  (Index2D idx2);
	void    fromIndex(int idx, Index2D& idx2);
	Index2D round(vec2_t v);

	vec2_t operator[](int idx);
	vec2_t operator[](Index2D idx2);
};

struct Grid3D{
	Grid1D* axis[3];

	int     num      ();
	int     toIndex  (Index3D idx3);
	void    fromIndex(int idx, Index3D& idx3);
	Index3D round(vec3_t v);

	vec3_t operator[](int idx);
	vec3_t operator[](Index3D idx3);
};

struct Grid4D{
	Grid1D* axis[4];

	int     num      ();
	int     toIndex  (Index4D idx4);
	void    fromIndex(int idx, Index4D& idx4);
	Index4D round(vec4_t v);

	vec4_t operator[](int idx);
	vec4_t operator[](Index4D idx4);
};

class Grid {
public:
	Grid1D x;
	Grid1D y;
	Grid1D z;
	Grid1D r;
	Grid1D t;

	Grid2D xy;
	Grid3D xyz;
	Grid3D xyt;
	Grid4D xyzr;

public:
	void Read(Scenebuilder::XMLNode* node);

	 Grid();
	~Grid();
};

} // namespace Capt
