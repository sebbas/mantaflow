/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2015 Kiwon Um, Nils Thuerey
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL)
 * http://www.gnu.org/licenses
 *
 * Matrix (3x3) and matrix (2x2) classes
 *
 ******************************************************************************/

#ifndef MATRIXBASE_H
#define MATRIXBASE_H

#include "general.h"
#include "vectorbase.h"

#if EIGEN==1
#	include "Eigen/Dense"
#endif

namespace Manta {

/**************************************************************************/
// Matrix3x3
/**************************************************************************/

template<typename T>
class Matrix3x3 {
public:
	// NOTE: default is the identity matrix!
	explicit Matrix3x3(const T &p00=1, const T &p01=0, const T &p02=0,
			   const T &p10=0, const T &p11=1, const T &p12=0,
			   const T &p20=0, const T &p21=0, const T &p22=1) {
		v[0][0]=p00; v[0][1]=p01; v[0][2]=p02;
		v[1][0]=p10; v[1][1]=p11; v[1][2]=p12;
		v[2][0]=p20; v[2][1]=p21; v[2][2]=p22;
	}

	explicit Matrix3x3(const Vector3D<T> &diag) {
		v[0][0]=diag.x; v[0][1]=0; v[0][2]=0;
		v[1][0]=0; v[1][1]=diag.y; v[1][2]=0;
		v[2][0]=0; v[2][1]=0; v[2][2]=diag.z;
	}

	Matrix3x3(const Vector3D<T> &c0, const Vector3D<T> &c1, const Vector3D<T> &c2) {
		v[0][0]=c0.x; v[0][1]=c1.x; v[0][2]=c2.x;
		v[1][0]=c0.y; v[1][1]=c1.y; v[1][2]=c2.y;
		v[2][0]=c0.z; v[2][1]=c1.z; v[2][2]=c2.z;
	}

	// assignment operators
	Matrix3x3& operator+=(const Matrix3x3 &m) {
		v00 += m.v00; v01 += m.v01; v02 += m.v02;
		v10 += m.v10; v11 += m.v11; v12 += m.v12;
		v20 += m.v20; v21 += m.v21; v22 += m.v22;
		return *this;
	}
	Matrix3x3& operator-=(const Matrix3x3 &m) {
		v00 -= m.v00; v01 -= m.v01; v02 -= m.v02;
		v10 -= m.v10; v11 -= m.v11; v12 -= m.v12;
		v20 -= m.v20; v21 -= m.v21; v22 -= m.v22;
		return *this;
	}
	Matrix3x3& operator*=(const T s) {
		v00 *= s; v01 *= s; v02 *= s;
		v10 *= s; v11 *= s; v12 *= s;
		v20 *= s; v21 *= s; v22 *= s;
		return *this;
	}
	Matrix3x3& operator/=(const T s) {
		v00 /= s; v01 /= s; v02 /= s;
		v10 /= s; v11 /= s; v12 /= s;
		v20 /= s; v21 /= s; v22 /= s;
		return *this;
	}
	Matrix3x3& operator+=(const T s) {
		v00 += s; v01 += s; v02 += s;
		v10 += s; v11 += s; v12 += s;
		v20 += s; v21 += s; v22 += s;
		return *this;
	}
	Matrix3x3& operator=(const Matrix3x3 &m) {
		v00 = m.v00; v01 = m.v01; v02 = m.v02;
		v10 = m.v10; v11 = m.v11; v12 = m.v12;
		v20 = m.v20; v21 = m.v21; v22 = m.v22;
		return *this;
	}
	Matrix3x3& operator=(const T s) {
		v00 = s; v01 = s; v02 = s;
		v10 = s; v11 = s; v12 = s;
		v20 = s; v21 = s; v22 = s;
		return *this;
	}
	// binary operators
	Matrix3x3 operator+(const T s) const { return Matrix3x3(*this)+=s; }
	Matrix3x3 operator+(const Matrix3x3 &m) const { return Matrix3x3(*this)+=m; }
	Matrix3x3 operator-(const Matrix3x3 &m) const { return Matrix3x3(*this)-=m; }
	Matrix3x3 operator*(const Matrix3x3 &m) const {
		return Matrix3x3(v00*m.v00 + v01*m.v10 + v02*m.v20,
				 v00*m.v01 + v01*m.v11 + v02*m.v21,
				 v00*m.v02 + v01*m.v12 + v02*m.v22,

				 v10*m.v00 + v11*m.v10 + v12*m.v20,
				 v10*m.v01 + v11*m.v11 + v12*m.v21,
				 v10*m.v02 + v11*m.v12 + v12*m.v22,

				 v20*m.v00 + v21*m.v10 + v22*m.v20,
				 v20*m.v01 + v21*m.v11 + v22*m.v21,
				 v20*m.v02 + v21*m.v12 + v22*m.v22);
	}
	Matrix3x3 operator*(const T s) const { return Matrix3x3(*this)*=s; }
	Vector3D<T> operator*(const Vector3D<T> &v) const {
		return Vector3D<T>(v00*v.x+v01*v.y+v02*v.z,
				   v10*v.x+v11*v.y+v12*v.z,
				   v20*v.x+v21*v.y+v22*v.z);
	}
	Matrix3x3& operator*=(const Matrix3x3 &m) {
		v00 = v00*m.v00+v01*m.v10+v02*m.v20;
		v01 = v00*m.v01+v01*m.v11+v02*m.v21;
		v02 = v00*m.v02+v01*m.v12+v02*m.v22;

		v10 = v10*m.v00+v11*m.v10+v12*m.v20;
		v11 = v10*m.v01+v11*m.v11+v12*m.v21;
		v12 = v10*m.v02+v11*m.v12+v12*m.v22;

		v20 = v20*m.v00+v21*m.v10+v22*m.v20;
		v21 = v20*m.v01+v21*m.v11+v22*m.v21;
		v22 = v20*m.v02+v21*m.v12+v22*m.v22;
		return *this;
	}
	Vector3D<T> transposedMul(const Vector3D<T> &v) const {
		// M^T*v
		return Vector3D<T>(v00*v.x+v10*v.y+v20*v.z,
				   v01*v.x+v11*v.y+v21*v.z,
				   v02*v.x+v12*v.y+v22*v.z);
	}
	Matrix3x3 transposedMul(const Matrix3x3 &m) const {
		// M^T*M
		return Matrix3x3(v00*m.v00 + v10*m.v10 + v20*m.v20,
				 v00*m.v01 + v10*m.v11 + v20*m.v21,
				 v00*m.v02 + v10*m.v12 + v20*m.v22,

				 v01*m.v00 + v11*m.v10 + v21*m.v20,
				 v01*m.v01 + v11*m.v11 + v21*m.v21,
				 v01*m.v02 + v11*m.v12 + v21*m.v22,

				 v02*m.v00 + v12*m.v10 + v22*m.v20,
				 v02*m.v01 + v12*m.v11 + v22*m.v21,
				 v02*m.v02 + v12*m.v12 + v22*m.v22);
	}
	Matrix3x3 mulTranspose(const Matrix3x3 &m) const {
		// M*m^T
		return Matrix3x3(v00*m.v00 + v01*m.v01 + v02*m.v02,
				 v00*m.v10 + v01*m.v11 + v02*m.v12,
				 v00*m.v20 + v01*m.v21 + v02*m.v22,

				 v10*m.v00 + v11*m.v01 + v12*m.v02,
				 v10*m.v10 + v11*m.v11 + v12*m.v12,
				 v10*m.v20 + v11*m.v21 + v12*m.v22,

				 v20*m.v00 + v21*m.v01 + v22*m.v02,
				 v20*m.v10 + v21*m.v11 + v22*m.v12,
				 v20*m.v20 + v21*m.v21 + v22*m.v22);
	}

	bool operator==(const Matrix3x3 &m) const {
		return (v00==m.v00 && v01==m.v01 && v02==m.v02 &&
			v10==m.v10 && v11==m.v11 && v12==m.v12 &&
			v20==m.v20 && v21==m.v21 && v22==m.v22);
	}

	const T& operator()(const int r, const int c) const { return v[r][c]; }
	T& operator()(const int r, const int c) { return const_cast<T &>(const_cast<const Matrix3x3 &>(*this)(r, c)); }

	T trace() const { return v00 + v11 + v22; }
	T sumSqr() const { return (v00*v00 + v01*v01 + v02*v02 + v10*v10 + v11*v11 + v12*v12 + v20*v20 + v21*v21 + v22*v22); }

	Real determinant() const { return (v00*v11*v22 - v00*v12*v21 + v01*v12*v20 - v01*v10*v22 + v02*v10*v21 - v02*v11*v20); }
	Matrix3x3& transpose() { return *this = transposed(); }
	Matrix3x3 transposed() const { return Matrix3x3(v00, v10, v20, v01, v11, v21, v02, v12, v22); }
	Matrix3x3& invert() { return *this = inverse(); }
	Matrix3x3 inverse() const {
		const Real det=determinant(); // FIXME: assert(det);
		const Real idet=1e0/det;
		return Matrix3x3(idet*(v11*v22-v12*v21), idet*(v02*v21-v01*v22), idet*(v01*v12-v02*v11),
				 idet*(v12*v20-v10*v22), idet*(v00*v22-v02*v20), idet*(v02*v10-v00*v12),
				 idet*(v10*v21-v11*v20), idet*(v01*v20-v00*v21), idet*(v00*v11-v01*v10));
	}
	bool getInverse(Matrix3x3 &inv) const {
		const Real det=determinant();
		if(det==0e0) return false; // FIXME: is it likely to happen the floating error?

		const Real idet=1e0/det;
		inv.v00=idet*(v11*v22-v12*v21);
		inv.v01=idet*(v02*v21-v01*v22);
		inv.v02=idet*(v01*v12-v02*v11);

		inv.v10=idet*(v12*v20-v10*v22);
		inv.v11=idet*(v00*v22-v02*v20);
		inv.v12=idet*(v02*v10-v00*v12);

		inv.v20=idet*(v10*v21-v11*v20);
		inv.v21=idet*(v01*v20-v00*v21);
		inv.v22=idet*(v00*v11-v01*v10);

		return true;
	}

	Real normOne() const {
		// the maximum absolute column sum of the matrix
		return max(std::fabs(v00)+std::fabs(v10)+std::fabs(v20),
			   std::fabs(v01)+std::fabs(v11)+std::fabs(v21),
			   std::fabs(v02)+std::fabs(v12)+std::fabs(v22));
	}
	Real normInf() const {
		// the maximum absolute row sum of the matrix
		return max(std::fabs(v00)+std::fabs(v01)+std::fabs(v02),
			   std::fabs(v10)+std::fabs(v11)+std::fabs(v12),
			   std::fabs(v20)+std::fabs(v21)+std::fabs(v22));
	}
	Real normEuclidean() const {
		// sqrt of the sum of all of the squares of the elements
		return std::sqrt(square(v00)+square(v01)+square(v02)
			   + square(v10)+square(v11)+square(v12)
			   + square(v20)+square(v21)+square(v22));
	}
	T* data() { return &v[0][0]; }

	Vector3D<T> eigenvalues() const {
		Vector3D<T> eigen;

		const Real b = - v00 - v11 - v22;
		const Real c = v00*(v11+v22) + v11*v22 - v12*v21 - v01*v10 - v02*v20;
		Real d =
			- v00*(v11*v22-v12*v21) - v20*(v01*v12-v11*v02) - v10*(v02*v21-v22*v01);
		const Real f = (3.0*c - b*b)/3.0;
		const Real g = (2.0*b*b*b - 9.0*b*c + 27.0*d)/27.0;
		const Real h = g*g/4.0 + f*f*f/27.0;

		Real sign;
		if(h>0) {
			Real r = -g/2.0 + std::sqrt(h);
			if(r<0) { r = -r; sign = -1.0; } else sign = 1.0;
			Real s = sign*std::pow(r, 1.0/3.0);
			Real t = -g/2.0-std::sqrt(h);
			if(t<0) { t = -t; sign = -1.0; } else sign = 1.0;
			Real u = sign*std::pow(t, 1.0/3.0);
			eigen[0] = (s + u) - b/3.0; eigen[1] = eigen[2] = 0;
		} else if(h==0) {
			if(d<0) { d = -d; sign = -1.0; } sign = 1.0;
			eigen[0] = -1.0*sign*std::pow(d, 1.0/3.0); eigen[1] = eigen[2] = 0;
		} else {
			const Real i = std::sqrt(g*g/4.0 - h);
			const Real j = std::pow(i, 1.0/3.0);
			const Real k = std::acos(-g/(2.0*i));
			const Real l = -j;
			const Real m = std::cos(k/3.0);
			const Real n = std::sqrt(3.0)*std::sin(k/3.0);
			const Real p = -b/3.0;
			eigen[0] = 2e0*j*m + p;
			eigen[1] = l*(m+n) + p;
			eigen[2] = l*(m-n) + p;
		}

		return eigen;
	}

	//! Outputs the object in human readable form as string
	std::string toString() const {
		char buf[1024];
		snprintf ( buf,1024,"[%+4.6f,%+4.6f,%+4.6f ; %+4.6f,%+4.6f,%+4.6f ; %+4.6f,%+4.6f,%+4.6f]",
			( double ) ( *this )(0,0), ( double ) ( *this )(0,1), ( double ) ( *this ) (0,2),
			( double ) ( *this )(1,0), ( double ) ( *this )(1,1), ( double ) ( *this ) (1,2),
			( double ) ( *this )(2,0), ( double ) ( *this )(2,1), ( double ) ( *this ) (2,2));
		return std::string ( buf );
	}

	static Matrix3x3 I() { return Matrix3x3(1,0,0, 0,1,0, 0,0,1); }

#ifdef _WIN32
#pragma warning(disable:4201)
#endif
	union {
		struct { T v00, v01, v02, v10, v11, v12, v20, v21, v22; };
		T v[3][3];
		T v1[9];
	};
#ifdef _WIN32
#pragma warning(default:4201)
#endif
};

/**************************************************************************/
// Matrix2x2
/**************************************************************************/

template<typename T>
class Matrix2x2 {
public:
	// NOTE: default is the identity matrix!
	explicit Matrix2x2(const T &p00=1, const T &p01=0, const T &p10=0, const T &p11=1) {
		v[0][0]=p00; v[0][1]=p01;
		v[1][0]=p10; v[1][1]=p11;
	}

	explicit Matrix2x2(const Vector3D<T> &diag) {
		v[0][0]=diag.x; v[0][1]=0;
		v[1][0]=0; v[1][1]=diag.y;
	}

	Matrix2x2(const Vector3D<T> &c0, const Vector3D<T> &c1) {
		v[0][0]=c0.x; v[0][1]=c1.x;
		v[1][0]=c0.y; v[1][1]=c1.y;
	}

	// assignment operators
	Matrix2x2& operator+=(const Matrix2x2 &m) {
		v00 += m.v00; v01 += m.v01;
		v10 += m.v10; v11 += m.v11;
		return *this;
	}
	Matrix2x2& operator-=(const Matrix2x2 &m) {
		v00 -= m.v00; v01 -= m.v01;
		v10 -= m.v10; v11 -= m.v11;
		return *this;
	}
	Matrix2x2& operator*=(const T s) {
		v00 *= s; v01 *= s;
		v10 *= s; v11 *= s;
		return *this;
	}
	Matrix2x2& operator/=(const T s) {
		v00 /= s; v01 /= s;
		v10 /= s; v11 /= s;
		return *this;
	}
	Matrix2x2& operator+=(const T s) {
		v00 += s; v01 += s;
		v10 += s; v11 += s;
		return *this;
	}
	Matrix2x2& operator=(const Matrix2x2 &m) {
		v00 = m.v00; v01 = m.v01;
		v10 = m.v10; v11 = m.v11;
		return *this;
	}
	Matrix2x2& operator=(const T s) {
		v00 = s; v01 = s;
		v10 = s; v11 = s;
		return *this;
	}
	// binary operators
	Matrix2x2 operator+(const T s) const { return Matrix2x2(*this)+=s; }
	Matrix2x2 operator+(const Matrix2x2 &m) const { return Matrix2x2(*this)+=m; }
	Matrix2x2 operator-(const Matrix2x2 &m) const { return Matrix2x2(*this)-=m; }
	Matrix2x2 operator*(const Matrix2x2 &m) const {
		return Matrix2x2(v00*m.v00 + v01*m.v10, v00*m.v01 + v01*m.v11,
						 v10*m.v00 + v11*m.v10, v10*m.v01 + v11*m.v11);
	}
	Matrix2x2 operator*(const T s) const { return Matrix2x2(*this)*=s; }
	Vector3D<T> operator*(const Vector3D<T> &v) const {
		return Vector3D<T>(v00*v.x+v01*v.y, v10*v.x+v11*v.y, 1.);
	}
	Matrix2x2& operator*=(const Matrix2x2 &m) {
		v00 = v00*m.v00+v01*m.v10;
		v01 = v00*m.v01+v01*m.v11;
		v10 = v10*m.v00+v11*m.v10;
		v11 = v10*m.v01+v11*m.v11;
		return *this;
	}
	Vector3D<T> transposedMul(const Vector3D<T> &v) const {
		// M^T*v
		return Vector3D<T>(v00*v.x+v10*v.y, v01*v.x+v11*v.y, 1.);
	}
	Matrix2x2 transposedMul(const Matrix2x2 &m) const {
		// M^T*M
		return Matrix2x2(v00*m.v00 + v10*m.v10,
				 v00*m.v01 + v10*m.v11,
				 v01*m.v00 + v11*m.v10,
				 v01*m.v01 + v11*m.v11);
	}
	Matrix2x2 mulTranspose(const Matrix2x2 &m) const {
		// M*m^T
		return Matrix2x2(v00*m.v00 + v01*m.v01,
				 v00*m.v10 + v01*m.v11,
				 v10*m.v00 + v11*m.v01,
				 v10*m.v10 + v11*m.v11);
	}

	bool operator==(const Matrix2x2 &m) const {
		return (v00==m.v00 && v01==m.v01 &&
				v10==m.v10 && v11==m.v11);
	}

	const T& operator()(const int r, const int c) const { return v[r][c]; }
	T& operator()(const int r, const int c) { return const_cast<T &>(const_cast<const Matrix2x2 &>(*this)(r, c)); }

	T trace() const { return v00 + v11; }
	T sumSqr() const { return (v00*v00 + v01*v01 + v10*v10 + v11*v11); }

	Real determinant() const { return (v00*v11 - v01*v10); }
	Matrix2x2& transpose() { return *this = transposed(); }
	Matrix2x2 transposed() const { return Matrix2x2(v00, v10, v01, v11); }
	Matrix2x2& invert() { return *this = inverse(); }
	Matrix2x2 inverse() const {
		const Real det=determinant(); // FIXME: assert(det);
		const Real idet=1e0/det;
		return Matrix2x2(idet*(v11),  idet*(-v01),
						 idet*(-v10), idet*(v00));
	}
	bool getInverse(Matrix2x2 &inv) const {
		const Real det=determinant();
		if(det==0e0) return false; // FIXME: is it likely to happen the floating error?

		const Real idet=1e0/det;
		inv.v00=idet*(v11);
		inv.v01=idet*(-v01);

		inv.v10=idet*(-v10);
		inv.v11=idet*(v00);
		return true;
	}

	Real normOne() const {
		// the maximum absolute column sum of the matrix
		return max(std::fabs(v00)+std::fabs(v10), std::fabs(v01)+std::fabs(v11));
	}
	Real normInf() const {
		// the maximum absolute row sum of the matrix
		return max(std::fabs(v00)+std::fabs(v01), std::fabs(v10)+std::fabs(v11));
	}
	Real normEuclidean() const {
		// sqrt of the sum of all of the squares of the elements
		return std::sqrt(square(v00)+square(v01)+square(v10)+square(v11));
	}
	T* data() { return &v[0][0]; }

	Vector3D<T> eigenvalues() const {
		Vector3D<T> eigen;
		// Based on formula: lam_1, lam_2 = m * sqrt(m*m - p)
		// where m is the mean of diagonal elements (i.e. mean of eigenvalues),
		// and p is the determinant (i.e. product of eigenvalues)
		const Real mean = trace() / 2;
		const Real prod = determinant();
		eigen[0] = mean + std::sqrt(square(mean) - prod);
		eigen[1] = mean - std::sqrt(square(mean) - prod);
		eigen[2] = 0.; // dont care, only two eigenvalues exist
		return eigen;
	}

	//! Outputs the object in human readable form as string
	std::string toString() const {
		char buf[1024];
		snprintf ( buf,1024,"[%+4.6f,%+4.6f ; %+4.6f,%+4.6f]",
			( double ) ( *this ) (0,0), ( double ) ( *this ) (0,1),
			( double ) ( *this ) (1,0), ( double ) ( *this ) (1,1));
		return std::string ( buf );
	}

	static Matrix2x2 I() { return Matrix2x2(1,0, 0,1); }

#ifdef _WIN32
#pragma warning(disable:4201)
#endif
	union {
		struct { T v00, v01, v10, v11; };
		T v[2][2];
		T v1[4];
	};
#ifdef _WIN32
#pragma warning(default:4201)
#endif
};

template<typename T1, typename T> inline Matrix3x3<T> operator*(const T1 s, const Matrix3x3<T> &m) {
	return m*static_cast<T>(s);
}
template<typename T1, typename T> inline Matrix2x2<T> operator*(const T1 s, const Matrix2x2<T> &m) {
	return m*static_cast<T>(s);
}

//! Outputs the object in human readable form to stream
template<class S>
std::ostream& operator<< (std::ostream& os, const Matrix3x3<S>& m) {
	os << m.toString();
	return os;
}
template<class S>
std::ostream& operator<< (std::ostream& os, const Matrix2x2<S>& m) {
	os << m.toString();
	return os;
}

//! Reads the contents of the object from a stream 
template<class S>
std::istream& operator>> (std::istream& is, Matrix3x3<S>& m) {
	char c;
	char dummy[9];
	is >> c >> m(0,0) >> dummy >> m(0,1) >> dummy >> m(0,2) >> 
				m(1,0) >> dummy >> m(1,1) >> dummy >> m(1,2) >>
				m(2,0) >> dummy >> m(2,1) >> dummy >> m(2,2) >> c; 
	return is;
}
template<class S>
std::istream& operator>> (std::istream& is, Matrix2x2<S>& m) {
	char c;
	char dummy[4];
	is >> c >> m(0,0) >> dummy >> m(0,1) >> dummy >> m(1,0) >> dummy >> m(1,1) >> c;
	return is;
}

template<typename T> inline void outerProduct(Matrix3x3<T> &R, const Vector3D<T> &a, const Vector3D<T> &b) {
	R(0, 0) = a.x*b.x; R(0, 1) = a.x*b.y; R(0, 2) = a.x*b.z;
	R(1, 0) = a.y*b.x; R(1, 1) = a.y*b.y; R(1, 2) = a.y*b.z;
	R(2, 0) = a.z*b.x; R(2, 1) = a.z*b.y; R(2, 2) = a.z*b.z;
}
template<typename T> inline void outerProduct(Matrix2x2<T> &R, const Vector3D<T> &a, const Vector3D<T> &b) {
	R(0, 0) = a.x*b.x; R(0, 1) = a.x*b.y;
	R(1, 0) = a.y*b.x; R(1, 1) = a.y*b.y; 
}

template<typename T> inline Matrix3x3<T> crossProductMatrix(const Vector3D<T> &v) {
	return Matrix3x3<T>(0, -v.z, v.y,  v.z, 0, -v.x,  -v.y, v.x, 0);
}

/**************************************************************************/
// Specializations for common math functions
/**************************************************************************/

typedef Matrix3x3<Real> Matrix3x3f;
typedef Matrix2x2<Real> Matrix2x2f;

template<> inline Matrix3x3f clamp<Matrix3x3f>(const Matrix3x3f& a, const Matrix3x3f& b, const Matrix3x3f& c) {
	return Matrix3x3f ( clamp(a(0,0), b(0,0), c(0,0)), clamp(a(0,1), b(0,1), c(0,1)), clamp(a(0,2), b(0,2), c(0,2)),
						clamp(a(1,0), b(1,0), c(1,0)), clamp(a(1,1), b(1,1), c(1,1)), clamp(a(1,2), b(1,2), c(1,2)),
						clamp(a(2,0), b(2,0), c(2,0)), clamp(a(2,1), b(2,1), c(2,1)), clamp(a(2,2), b(2,2), c(2,2)) );
}
template<> inline Matrix2x2f clamp<Matrix2x2f>(const Matrix2x2f& a, const Matrix2x2f& b, const Matrix2x2f& c) {
	return Matrix2x2f ( clamp(a(0,0), b(0,0), c(0,0)), clamp(a(0,1), b(0,1), c(0,1)),
						clamp(a(1,0), b(1,0), c(1,0)), clamp(a(1,1), b(1,1), c(1,1)));
}

template<> inline Matrix3x3f safeDivide<Matrix3x3f>(const Matrix3x3f &a, const Matrix3x3f& b) {
	return a*b.inverse();
}
template<> inline Matrix2x2f safeDivide<Matrix2x2f>(const Matrix2x2f &a, const Matrix2x2f& b) {
	return a*b.inverse();
}

/**************************************************************************/
// SVD and polar decomposition for (3x3) and (2x2) matrices
/**************************************************************************/

//! Perform polar decomposition with SVD matrices U, V, Sig
template<typename T> inline void polarDecomp(const T& srcU, const T& srcV, const T& srcSig, T& destR, T& destS) {
	// Polar decomposition A=RS with SVD matrices U, V, Sig: S=V*Sigma*V^T, R=U*V^T	
	destR = srcU * srcV.transposed();
	destS = srcV * srcSig * srcV.transposed();
}

//! Perform polar decomposition in 2D
template<typename T> inline void inlinePolarDecomp(const Matrix2x2<T>& A, Matrix2x2<T>& R, Matrix2x2<T>& S) {
	Real x = A(0, 0) + A(1, 1);
	Real y = A(1, 0) - A(0, 1);
	Real scale = 1.0f / std::sqrt(x * x + y * y);
	Real c = x * scale, s = y * scale;
	R(0, 0) = c;
	R(0, 1) = -s;
	R(1, 0) = s;
	R(1, 1) = c;
	S = R.transposed() * A;
}

template<typename T> void polarDecomposition(const Matrix3x3<T>& A, Matrix3x3<T>& R, Matrix3x3<T>& S) {
#if EIGEN==1
	Matrix3x3<T> U, V, Sig;
	eigenSVD3x3(A, U, V, Sig);
	polarDecomp(U, V, Sig, R, S);
#else
	debMsg("Cannot compute polar decomposition without Eigen lib", 1);
#endif
}

template<typename T> void polarDecomposition(const Matrix2x2<T>& A, Matrix2x2<T>& R, Matrix2x2<T>& S, bool usingEigen=false) {
	if (usingEigen) {
#if EIGEN==1
		Matrix2x2<T> U, V, Sig;
		eigenSVD2x2(A, U, V, Sig);
		polarDecomp(U, V, Sig, R, S);
#else
		debMsg("Cannot compute polar decomposition without Eigen lib. But can do with usingEigen=false", 1);
#endif
	} else {
		inlinePolarDecomp(A, R, S);
	}
}

//! Perform SVD with Eigen lib
#if EIGEN==1
template<typename T> void eigenSVD3x3(const Matrix3x3<T>& srcA, Matrix3x3<T>& destU, Matrix3x3<T>& destV, Matrix3x3<T>& destSig) {
	// Perform SVD with 3x3 Eigen matrices (using Eigen::Dynamic to construct matrices is kind of slow ..)
	const Eigen::Matrix3f AMat { {srcA(0,0), srcA(0,1), srcA(0,2)}, {srcA(1,0), srcA(1,1), srcA(1,2)}, {srcA(2,0), srcA(2,1), srcA(2,2)} };
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(AMat, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix<Real, 3, 3, Eigen::RowMajor> UMat = svd.matrixU();
	Eigen::Matrix<Real, 3, 3, Eigen::RowMajor> VMat = svd.matrixV();
	Eigen::Vector3f singularValues = svd.singularValues();

	// Fill manta matrices with data from Eigen SVD matrices
	destU(0,0) = UMat.coeff(0,0); destU(0,1) = UMat.coeff(0,1); destU(0,2) = UMat.coeff(0,2);
	destU(1,0) = UMat.coeff(1,0); destU(1,1) = UMat.coeff(1,1); destU(1,2) = UMat.coeff(1,2);
	destU(2,0) = UMat.coeff(2,0); destU(2,1) = UMat.coeff(2,1); destU(2,2) = UMat.coeff(2,2);
	destV(0,0) = VMat.coeff(0,0); destV(0,1) = VMat.coeff(0,1); destV(0,2) = VMat.coeff(0,2);
	destV(1,0) = VMat.coeff(1,0); destV(1,1) = VMat.coeff(1,1); destV(1,2) = VMat.coeff(1,2);
	destV(2,0) = VMat.coeff(2,0); destV(2,1) = VMat.coeff(2,1); destV(2,2) = VMat.coeff(2,2);
	destSig(0,0) = singularValues[0]; destSig(1,1) = singularValues[1]; destSig(2,2) = singularValues[2];
}

template<typename T> void eigenSVD2x2(const Matrix2x2<T>& srcA, Matrix2x2<T>& destU, Matrix2x2<T>& destV, Matrix2x2<T>& destSig) {
	// Perform SVD with 2x2 Eigen matrices (using Eigen::Dynamic to construct matrices is kind of slow ..)
	const Eigen::Matrix2f AMat { {srcA(0,0), srcA(0,1)}, {srcA(1,0), srcA(1,1)} };
	Eigen::JacobiSVD<Eigen::Matrix2f> svd(AMat, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix<Real, 2, 2, Eigen::RowMajor> UMat = svd.matrixU();
	Eigen::Matrix<Real, 2, 2, Eigen::RowMajor> VMat = svd.matrixV();
	Eigen::Vector2f singularValues = svd.singularValues();

	// Fill manta matrices with data from Eigen SVD matrices
	destU(0,0) = UMat.coeff(0,0); destU(0,1) = UMat.coeff(0,1);
	destU(1,0) = UMat.coeff(1,0); destU(1,1) = UMat.coeff(1,1);
	destV(0,0) = VMat.coeff(0,0); destV(0,1) = VMat.coeff(0,1);
	destV(1,0) = VMat.coeff(1,0); destV(1,1) = VMat.coeff(1,1);
	destSig(0,0) = singularValues[0]; destSig(1,1) = singularValues[1];
}
#endif

//! Perform SVD without libs in 2D (faster than when using Eigen, is based on http://www.seas.upenn.edu/~cffjiang/research/svd/svd.pdf)
template<typename T> inline void inlineSVD(const Matrix2x2<T>& srcA, Matrix2x2<T>& destU, Matrix2x2<T>& destV, Matrix2x2<T>& destSig) {
	Matrix2x2<T> S;
	polarDecomposition(srcA, destU, S, false);
	Real c, s;
	if (std::abs(S(0, 1)) < 1e-6) {
		destSig = S;
		c = 1;
		s = 0;
	} else {
		Real tao = 0.5f * (S(0, 0) - S(1, 1));
		Real w = std::sqrt(tao * tao + S(0, 1) * S(0, 1));
		Real t = tao > 0 ? S(0, 1) / (tao + w) : S(0, 1) / (tao - w);
		c = 1.0f / std::sqrt(t * t + 1);
		s = -t * c;
		destSig(0, 0) = square(c) * S(0, 0) - 2 * c * s * S(0, 1) + square(s) * S(1, 1);
		destSig(1, 1) = square(s) * S(0, 0) + 2 * c * s * S(0, 1) + square(c) * S(1, 1);
	}
	if (destSig(0, 0) < destSig(1, 1)) {
		std::swap(destSig(0, 0), destSig(1, 1));
		destV(0, 0) = -s;
		destV(0, 1) = -c;
		destV(1, 0) = c;
		destV(1, 1) = -s;
	} else {
		destV(0, 0) = c;
		destV(0, 1) = -s;
		destV(1, 0) = s;
		destV(1, 1) = c;
	}
	destV = destV.transposed();
	destU = destU * destV;
}

template<typename T> void svd(const Matrix3x3<T>& A, Matrix3x3<T>& U, Matrix3x3<T>& V, Matrix3x3<T>& Sig, bool usingEigen=true) {
	unusedParameter(usingEigen); // 3x3 SVD only possible with Eigen for now
#if EIGEN==1
	eigenSVD3x3(A, U, V, Sig);
#else
	debMsg("Cannot compute SVD without Eigen lib", 1);
#endif
}

template<typename T> void svd(const Matrix2x2<T>& A, Matrix2x2<T>& U, Matrix2x2<T>& V, Matrix2x2<T>& Sig, bool usingEigen=true) {
	if (usingEigen) {
#if EIGEN==1
		eigenSVD2x2(A, U, V, Sig);
#else
		debMsg("Cannot compute SVD without Eigen lib. But can do with usingEigen=false", 1);
#endif
	} else {
		inlineSVD(A, U, V, Sig);
	}
}

} // namespace

#endif	/* MATRIXBASE_H */
