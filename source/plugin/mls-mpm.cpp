/******************************************************************************
*
* MantaFlow fluid solver framework
* Copyright 2024 Sebastian Barschkis
*
* This program is free software, distributed under the terms of the
* GNU General Public License (GPL)
* http://www.gnu.org/licenses
*
* Moving Least Squares Material Point Method (MLS-MPM) plugin
*
*******************************************************************************
*
* Copyright 2018 Taichi MPM Authors
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
******************************************************************************/

/*
 * Algorithm and equations based on the "MPM course":
 * Jiang, Chenfanfu, et al.
 * "The material point method for simulating continuum materials."
 * Acm siggraph 2016 courses. 2016. 1-52.
 */

#include "particle.h"
#include "grid.h"
#include "commonkernels.h"

#include <chrono>
using namespace std::chrono;

namespace Manta {

//! Polar decomposition A=UP
KERNEL(pts, bnd=0) template<class T>
void knMpmPolarDecomposition(ParticleDataImpl<T>& srcA, ParticleDataImpl<T>& destU, ParticleDataImpl<T>& destP)
{
	polarDecomposition(srcA[idx], destU[idx], destP[idx]);
}

PYTHON() void polarDecomposition2D(ParticleDataImpl<Matrix2x2f>& A,
	ParticleDataImpl<Matrix2x2f>& U, ParticleDataImpl<Matrix2x2f>& P)
{
	knMpmPolarDecomposition<Matrix2x2f>(A, U, P);
}

PYTHON() void polarDecomposition3D(ParticleDataImpl<Matrix3x3f>& A,
	ParticleDataImpl<Matrix3x3f>& U, ParticleDataImpl<Matrix3x3f>& P)
{
	knMpmPolarDecomposition<Matrix3x3f>(A, U, P);
}

KERNEL(pts, single, bnd=0) template<class T>
void knMpmMapVec3ToMACGrid(
	const BasicParticleSystem& pp, MACGrid& vel, Grid<Real>& mass, const ParticleDataImpl<Vec3>& pvel,
	ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<T>& deformationGrad,
	ParticleDataImpl<T>& R, ParticleDataImpl<T>& S, ParticleDataImpl<T>& affineMomentum,
	Real hardening, Real E, Real nu, Real pmass, Real pvol)
{
	// Initial Lame parameters
	const Real mu_0 = E / (2 * (1 + nu));
	const Real lambda_0 = E * nu / ((1+nu) * (1 - 2 * nu));
	const bool is3D = vel.is3D();

	const Real dt = pp.getParent()->getDt();
	const Real dx = pp.getParent()->getDx();
	const Real inv_dx = 1.0 / dx;

	const Vec3 pos = pp.getPos(idx);
	const Vec3i base = toVec3i(pos - Vec3(0.5));
	const Vec3 fx = (pos - toVec3(base));

	// Quadratic kernels, MPM course Equation 123.
	Vec3 w[3] = {
		Vec3(0.5)  * square(Vec3(1.5) - fx), // x = fx
		Vec3(0.75) - square(fx - Vec3(1.0)), // x = fx - 1
		Vec3(0.5)  * square(fx - Vec3(0.5))  // x = fx - 2
	};

	// Lame parameters, MPM course Equation 86.
	Real e = std::exp(hardening * (1.0f - detDeformationGrad[idx]));
	Real mu = mu_0 * e;
	Real lambda = lambda_0 * e;
	Real Dinv = 4 * inv_dx * inv_dx;

	// Current volume
	Real J = deformationGrad[idx].determinant();

	T dR = deformationGrad[idx] - R[idx];
	T PF = (2 * mu * (dR) * deformationGrad[idx].transposed() + lambda * (J-1) * J);

	T stress = - (dt * pvol) * (Dinv * PF);
	T affine = stress + pmass * affineMomentum[idx];

	const int size = sizeof(w) / sizeof(w[0]);
	const int sizeK = (is3D) ? 3 : 1;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < sizeK; k++) {
				Vec3 dpos = (Vec3(i,j,k) - fx) * dx;
				if (!is3D) dpos.z = 0.;
				
				if (!vel.isInBounds(base + toVec3i(i,j,k), 1)) continue;

				Vec3 vel_x_mass = pvel[idx] * pmass;
				if (!is3D) vel_x_mass.z = 0;

				Vec3 aff = affine * dpos;
				if (!is3D) aff.z = 0;

				Real weight = w[i].x*w[j].y;
				if (is3D) weight *= w[k].z;

				vel(base + toVec3i(i,j,k)) += weight * (vel_x_mass + aff);
				mass(base + toVec3i(i,j,k)) += weight * pmass;				
			}
		}
	}
}

PYTHON() void mpmMapPartsToMACGrid2D(
	const BasicParticleSystem& pp, MACGrid& vel, Grid<Real>& mass, const ParticleDataImpl<Vec3>& pvel,
	ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<Matrix2x2f>& deformationGrad,
	ParticleDataImpl<Matrix2x2f>& affineMomentum, ParticleDataImpl<Matrix2x2f>& rotation, ParticleDataImpl<Matrix2x2f>& scale,
	Real hardening=10.0f, Real E=1e4f, Real nu=0.2f, Real pmass=1.0f, Real pvol=1.0f)
{
	vel.clear();
	mass.clear();
	knMpmMapVec3ToMACGrid<Matrix2x2f>(pp, vel, mass, pvel, detDeformationGrad, deformationGrad, rotation, scale, affineMomentum, hardening, E, nu, pmass, pvol);
}

PYTHON() void mpmMapPartsToMACGrid3D(
	const BasicParticleSystem& pp, MACGrid& vel, Grid<Real>& mass, const ParticleDataImpl<Vec3>& pvel,
	ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<Matrix3x3f>& deformationGrad,
	ParticleDataImpl<Matrix3x3f>& affineMomentum, ParticleDataImpl<Matrix3x3f>& rotation, ParticleDataImpl<Matrix3x3f>& scale,
	Real hardening=10.0f, Real E=1e4f, Real nu=0.2f, Real pmass=1.0f, Real pvol=1.0f)
{
	vel.clear();
	mass.clear();
	knMpmMapVec3ToMACGrid<Matrix3x3f>(pp, vel, mass, pvel, detDeformationGrad, deformationGrad, rotation, scale, affineMomentum, hardening, E, nu, pmass, pvol);
}

KERNEL(bnd=0)
void KnMpmUpdateGrid(const FlagGrid& flags, const BasicParticleSystem& pp, MACGrid& vel, Grid<Real>& mass, Vec3 gravity)
{
	const Real dt = pp.getParent()->getDt();
	const Vec3i size = pp.getParent()->getGridSize();
	const int bnd = 1;
	const bool is3D = vel.is3D();

	if (mass(i,j,k) <= 0) return;

	// Normalize by mass
	vel(i,j,k) /= mass(i,j,k);
	mass(i,j,k) /= mass(i,j,k);
	vel(i,j,k) += dt * gravity;

	// Assume solid no-slip boundaries at all walls
	if (flags.isObstacle(i,j,k)) {
		vel(i,j,k) = Vec3(0.);
		mass(i,j,k) = 0.;
	}
}

PYTHON() void mpmUpdateGrid(const FlagGrid& flags, const BasicParticleSystem& pp, MACGrid& vel, Grid<Real>& mass, Vec3 gravity)
{
	KnMpmUpdateGrid(flags, pp, vel, mass, gravity);
}

KERNEL(pts, bnd=0) template<class T>
void knMpmMapMACGridToVec3(
	BasicParticleSystem& pp, const MACGrid& vel, Grid<Real>& mass, ParticleDataImpl<Vec3>& pvel,
	ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<T>& deformationGrad,
	ParticleDataImpl<T>& affineMomentum, bool plastic)
{
	const Real dt = pp.getParent()->getDt();
	const Real dx = pp.getParent()->getDx();
	const Real inv_dx = 1.0 / dx;
	const bool is3D = vel.is3D();

	const Vec3 pos = pp.getPos(idx);
	const Vec3i base = toVec3i(pos - Vec3(0.5));
	const Vec3 fx = (pos - toVec3(base));

	// Quadratic kernels, MPM course Equation 123.
	Vec3 w[3] = {
		Vec3(0.5)  * square(Vec3(1.5) - fx), // x = fx
		Vec3(0.75) - square(fx - Vec3(1.0)), // x = fx - 1
		Vec3(0.5)  * square(fx - Vec3(0.5))  // x = fx - 2
	};

	affineMomentum[idx] = T(Vec3(0.));
	pvel[idx] = Vec3(0.);
	T outerProd(Vec3(0.));

	const int size = sizeof(w) / sizeof(w[0]);
	const int sizeK = (is3D) ? 3 : 1;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < sizeK; k++) {
				Vec3 dpos = (Vec3(i,j,k) - fx);
				if (!is3D) dpos.z = 0.;

				if (!vel.isInBounds(base + toVec3i(i,j,k), 1)) continue;

				Vec3 gridVel = vel(base + toVec3i(i,j,k));
				if (!is3D) gridVel.z = 0;
				
				Real weight = w[i].x * w[j].y;
				if (is3D) weight *= w[k].z;
				
				pvel[idx] += weight * gridVel;
				if (!is3D) pvel[idx].z = 0;

				outerProduct(outerProd, weight * gridVel , dpos);
				affineMomentum[idx] += 4 * outerProd;
			}
		}
	}

	// Advection
	pp[idx].pos += dt * pvel[idx];

	// Deformation update
	T F = (T(Vec3(1.)) + dt * affineMomentum[idx]) * deformationGrad[idx];

	// SVD
	T svd_U(Vec3(0.)), sig(Vec3(0.)), svd_V(Vec3(0.));
	svd(F, svd_U, sig, svd_V);

	// Snow plasticity
	const int sizeSig = (is3D) ? 3 : 2;
	for (int i = 0; i < sizeSig * int(plastic); i++) {
		sig(i,i) = clamp(sig(i,i), 1.0f - 2.5e-2f, 1.0f + 7.5e-3f);
	}

	Real oldJ = F.determinant();
	F = svd_U * sig * svd_V.transposed();
	Real Jp_new = clamp(detDeformationGrad[idx] * oldJ / F.determinant(), 0.6f, 20.0f);

	detDeformationGrad[idx] = Jp_new;
	deformationGrad[idx] = F;
}

PYTHON() void mpmMapMACGridToParts2D(
	BasicParticleSystem& pp, const MACGrid& vel, Grid<Real>& mass, ParticleDataImpl<Vec3>& pvel,
	ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<Matrix2x2f>& deformationGrad,
	ParticleDataImpl<Matrix2x2f>& affineMomentum, bool plastic=true)
{
	knMpmMapMACGridToVec3<Matrix2x2f>(pp, vel, mass, pvel, detDeformationGrad, deformationGrad, affineMomentum, plastic);
}

PYTHON() void mpmMapMACGridToParts3D(
	BasicParticleSystem& pp, const MACGrid& vel, Grid<Real>& mass, ParticleDataImpl<Vec3>& pvel,
	ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<Matrix3x3f>& deformationGrad,
	ParticleDataImpl<Matrix3x3f>& affineMomentum, bool plastic=true)
{
	knMpmMapMACGridToVec3<Matrix3x3f>(pp, vel, mass, pvel, detDeformationGrad, deformationGrad, affineMomentum, plastic);
}

} // namespace
