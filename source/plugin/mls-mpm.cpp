/******************************************************************************
*
* MantaFlow fluid solver framework
* Copyright 2024 Sebastian Barschkis
*
* This program is free software, distributed under the terms of the
* Apache License, Version 2.0
* http://www.apache.org/licenses/LICENSE-2.0
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
void knMpmPolarDecomposition(const ParticleDataImpl<T>& srcA, ParticleDataImpl<T>& destU, ParticleDataImpl<T>& destP)
{
	polarDecomposition(srcA[idx], destU[idx], destP[idx]);
}

PYTHON() void polarDecomposition2D(const ParticleDataImpl<Matrix2x2f>& A,
	ParticleDataImpl<Matrix2x2f>& U, ParticleDataImpl<Matrix2x2f>& P)
{
	knMpmPolarDecomposition<Matrix2x2f>(A, U, P);
}

PYTHON() void polarDecomposition3D(const ParticleDataImpl<Matrix3x3f>& A,
	ParticleDataImpl<Matrix3x3f>& U, ParticleDataImpl<Matrix3x3f>& P)
{
	knMpmPolarDecomposition<Matrix3x3f>(A, U, P);
}

KERNEL(pts, single, bnd=0) template<class T>
void knMpmMapVec3ToMACGrid(
	const BasicParticleSystem& pp, MACGrid& vel, Grid<Real>& mass,
	const ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<Real>& detDeformationGrad,
	const ParticleDataImpl<T>& deformationGrad, const ParticleDataImpl<T>& R, const ParticleDataImpl<T>& affineMomentum,
	const Real hardening, const Real E, const Real nu, const Real pmass, const Real pvol, Vec3i loopStart, const bool is3D)
{
	// Initial Lame parameters
	const Real mu_0 = E / (2 * (1 + nu));
	const Real lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

	// Lame parameters, MPM course Equation 86.
	const Real e = std::exp(hardening * (1.0f - detDeformationGrad[idx]));
	const Real mu = mu_0 * e;
	const Real lambda = lambda_0 * e;

	// Domain parameters
	const Real dt = pp.getParent()->getDt();
	const Real dx = pp.getParent()->getDx();
	const Real inv_dx = 1.0 / dx;
	const Real Dinv = 4 * inv_dx * inv_dx;

	const Vec3 pos = pp.getPos(idx);
	const Vec3i base = toVec3i(pos - Vec3(0.5));
	const Vec3 fx = (pos - toVec3(base));

	// Quadratic kernels, MPM course Equation 123.
	Vec3 w[3] = {
		Vec3(0.5)  * square(Vec3(1.5) - fx), // x = fx
		Vec3(0.75) - square(fx - Vec3(1.0)), // x = fx - 1
		Vec3(0.5)  * square(fx - Vec3(0.5))  // x = fx - 2
	};
	// In 2D the Z component is unused
	if (!is3D) {
		w[0].z = 1.;
		w[1].z = 1.;
		w[2].z = 1.;
	}

	// Current volume
	const Real J = deformationGrad[idx].determinant();

	const T dR = deformationGrad[idx] - R[idx];
	const T PF = (2 * mu * (dR) * deformationGrad[idx].transposed() + lambda * (J-1) * J);

	const T stress = - (dt * pvol) * (Dinv * PF);
	const T affine = stress + pmass * affineMomentum[idx];

	const int sizeFull = sizeof(w) / sizeof(w[0]);
	int sizeI, sizeJ, sizeK;

	// Explicit value given, disable full i/j/k for-loop (i.e. loop given dim once)
	if (loopStart.x >= 0) { sizeI = loopStart.x+1; }
	if (loopStart.y >= 0) { sizeJ = loopStart.y+1; }
	if (loopStart.z >= 0) { sizeK = loopStart.z+1; }
	// No i/j/k given, activate full i/j/k for-loop
	if (loopStart.x == -1) { sizeI = sizeFull; loopStart.x = 0; }
	if (loopStart.y == -1) { sizeJ = sizeFull; loopStart.y = 0; }
	if (loopStart.z == -1) { sizeK = sizeFull; loopStart.z = 0; }

	Vec3 dpos, aff;
	IndexInt targetPos;
	Real weight;

	const Vec3 vel_x_mass = pvel[idx] * pmass;
	// Factor to zero out the last dim in 2D mode
	const Vec3 dimFactor = (is3D) ? Vec3(1) : Vec3(1,1,0);

	// Check if current and outermost cell is in bounds (knowledge saves time in neighbor loop)
	bool upperInBounds = vel.isInBounds(base + toVec3i(sizeI-1,sizeJ-1,sizeK-1));
	bool thisInBounds = vel.isInBounds(base);

	// Neighbor loop: Iterate over neighboring cells of this particle
	for (int k = loopStart.z; k < sizeK; k++)
	for (int j = loopStart.y; j < sizeJ; j++)
	for (int i = loopStart.x; i < sizeI; i++)
	{
		// Only perform bounds check for current ijk if this or upper bound cell are not in bounds (saves time)
		if ((!upperInBounds || !thisInBounds) && !vel.isInBounds(base + toVec3i(i,j,k))) continue;

		targetPos = vel.index(base + toVec3i(i,j,k));
		dpos = (Vec3(i,j,k) - fx) * dx;

		aff = affine * dpos;
		weight = w[i].x * w[j].y * w[k].z;

		// Grid access is not cache optimal here (i.e. most expensive part of this function)
		vel(targetPos) += weight * (vel_x_mass + aff) * dimFactor;
		mass(targetPos) += weight * pmass;
	}
}

KERNEL() template<class T>
void KernelHelper3D(const Grid<Real>& kernelGrid, const BasicParticleSystem& pp,
	MACGrid* vel, MACGrid* velK1, MACGrid* velK2,
	Grid<Real>* mass, Grid<Real>* massK1, Grid<Real>* massK2,
	const ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<Real>& detDeformationGrad,
	const ParticleDataImpl<T>& deformationGrad, const ParticleDataImpl<T>& affineMomentum, const ParticleDataImpl<T>& rotation,
	const Real hardening, const Real E, const Real nu, const Real pmass, const Real pvol)
{
	// assertMsg(i <= 2 && j <= 2 && k <= 2, "Unsupported grid size. Ensure size kernelGrid is at most 3x3x3");
	MACGrid* velSelect = nullptr;
	Grid<Real>* massSelect = nullptr;
	if (k == 0) { velSelect = vel; massSelect = mass; }
	if (k == 1) { velSelect = velK1; massSelect = massK1; }
	if (k == 2) { velSelect = velK2; massSelect = massK2; }
	// assertMsg(vel != nullptr, "vel grid pointer must not be be null");
	// assertMsg(mass != nullptr, "mass grid pointer must not be be null");

	const int ii = -1, jj = -1; // Always enable looping in i and j directions
	Vec3i loopStart(ii, jj, k); // Only neighbor search in k direction runs in parallel
	knMpmMapVec3ToMACGrid<T>(pp, *velSelect, *massSelect, pvel, detDeformationGrad, deformationGrad, rotation, affineMomentum, hardening, E, nu, pmass, pvol, loopStart, true);
}

PYTHON() void mpmMapPartsToMACGrid2D(MACGrid& vel, Grid<Real>& mass,
	const BasicParticleSystem& pp, const ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<Real>& detDeformationGrad,
	const ParticleDataImpl<Matrix2x2f>& deformationGrad, const ParticleDataImpl<Matrix2x2f>& affineMomentum, const ParticleDataImpl<Matrix2x2f>& rotation,
	const Real hardening=10.0f, const Real E=1e4f, const Real nu=0.2f, const Real pmass=1.0f, const Real pvol=1.0f)
{
	vel.clear();
	mass.clear();

	Vec3i loopStart(-1,-1,0); // 0: disable dim (loop once), -1: full neighbor loop [0,2]
	knMpmMapVec3ToMACGrid<Matrix2x2f>(pp, vel, mass, pvel, detDeformationGrad, deformationGrad, rotation, affineMomentum, hardening, E, nu, pmass, pvol, loopStart, false);
}

PYTHON() void mpmMapPartsToMACGrid3D(
	const BasicParticleSystem& pp, const ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<Real>& detDeformationGrad,
	const ParticleDataImpl<Matrix3x3f>& deformationGrad, const ParticleDataImpl<Matrix3x3f>& affineMomentum, const ParticleDataImpl<Matrix3x3f>& rotation,
	MACGrid& vel, Grid<Real>& mass,
	Grid<Real>* kernelGrid=nullptr, MACGrid* velK1=nullptr, MACGrid* velK2=nullptr,
	Grid<Real>* massK1=nullptr, Grid<Real>* massK2=nullptr,
	const Real hardening=10.0f, const Real E=1e4f, const Real nu=0.2f, const Real pmass=1.0f, const Real pvol=1.0f)
{
	// assertMsg(vel != nullptr && mass != nullptr, "Missing one of the base grids, check function args");

	// Only have to clear vel and mass, optional kernel buffer grids are cleared in KnMpmUpdateGrid() (this improves performance)
	vel.clear();
	mass.clear();

	// Compute particle data in single thread
	if (kernelGrid == nullptr) {
		Vec3i loopStart(-1,-1,-1); // Enable loops in all 3 dims
		knMpmMapVec3ToMACGrid<Matrix3x3f>(pp, vel, mass, pvel, detDeformationGrad, deformationGrad, rotation, affineMomentum, hardening, E, nu, pmass, pvol, loopStart, true);
		return;
	}

	// Compute particle data with multiple threads
	KernelHelper3D helper3D (*kernelGrid, pp,
		&vel, velK1, velK2, &mass, massK1, massK2, 
		pvel, detDeformationGrad, deformationGrad, affineMomentum, rotation, hardening, E, nu, pmass, pvol);
}

KERNEL(bnd=0)
void KnMpmUpdateGrid(const FlagGrid& flags, const BasicParticleSystem& pp,  const Vec3& gravity,
	MACGrid& vel, Grid<Real>& mass, MACGrid* velK1, MACGrid* velK2, Grid<Real>* massK1, Grid<Real>* massK2)
{
	// No grid updates needed in cells with no mass
	if (velK1 && massK1 && mass(i,j,k) <= 0 && (*massK1)(i,j,k) <= 0 && (*massK2)(i,j,k) <= 0) return;
	if ((velK1 == nullptr || massK1 == nullptr) && mass(i,j,k) <= 0) return;

	// If using kernel helper grids, add them to global vel and mass grids
	if (velK1 && massK1) {
		mass(i,j,k) += (*massK1)(i,j,k) + (*massK2)(i,j,k);
		vel(i,j,k) += (*velK1)(i,j,k) + (*velK2)(i,j,k);

		// Vel and mass kernel buffers not needed anymore, clear here
		(*velK1)(i,j,k) = Vec3(0.);
		(*velK2)(i,j,k) = Vec3(0.);
		(*massK1)(i,j,k) = 0.;
		(*massK2)(i,j,k) = 0.;
	}

	// Assume solid no-slip boundaries at all walls
	if (flags.isObstacle(i,j,k)) {
		vel(i,j,k) = Vec3(0.);
		mass(i,j,k) = 0.;
		return;
	}

	// Normalize by mass
	vel(i,j,k) /= mass(i,j,k);
	mass(i,j,k) /= mass(i,j,k);

	const Real dt = pp.getParent()->getDt();
	vel(i,j,k) += dt * gravity;
}

PYTHON() void mpmUpdateGrid(const FlagGrid& flags, const BasicParticleSystem& pp, const Vec3& gravity,
	MACGrid& vel, Grid<Real>& mass, MACGrid* velK1=nullptr, MACGrid* velK2=nullptr,Grid<Real>* massK1=nullptr, Grid<Real>* massK2=nullptr)
{
	KnMpmUpdateGrid(flags, pp, gravity, vel, mass, velK1, velK2, massK1, massK2);
}

KERNEL(pts, bnd=0) template<class T>
void knMpmMapMACGridToVec3(
	BasicParticleSystem& pp, const MACGrid& vel, const Grid<Real>& mass, ParticleDataImpl<Vec3>& pvel,
	ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<T>& deformationGrad,
	ParticleDataImpl<T>& affineMomentum, const bool plastic, Vec3i loopStart, const bool is3D)
{
	// Domain parameters
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
	// In 2D the Z component is unused
	if (!is3D) {
		w[0].z = 1.;
		w[1].z = 1.;
		w[2].z = 1.;
	}

	Vec3 pvelNew = Vec3(0.); // Stores new pvel from for-loop
	T outerProd(Vec3(0.));
	T affineMomentumNew(Vec3(0.)); // Stores new affine momentum from for-loop

	const int sizeFull = sizeof(w) / sizeof(w[0]);
	int sizeI, sizeJ, sizeK;

	// Explicit value given, disable full i/j/k for-loop (i.e. loop given dim once)
	if (loopStart.x >= 0) { sizeI = loopStart.x+1; }
	if (loopStart.y >= 0) { sizeJ = loopStart.y+1; }
	if (loopStart.z >= 0) { sizeK = loopStart.z+1; }
	// No i/j/k given, activate full i/j/k for-loop
	if (loopStart.x == -1) { sizeI = sizeFull; loopStart.x = 0; }
	if (loopStart.y == -1) { sizeJ = sizeFull; loopStart.y = 0; }
	if (loopStart.z == -1) { sizeK = sizeFull; loopStart.z = 0; }

	Vec3 dpos, gridVel;
	IndexInt targetPos;
	Real weight;

	// Factor to zero out the last dim in 2D mode
	const Vec3 dimFactor = (is3D) ? Vec3(1) : Vec3(1,1,0);

	// Check if current and outermost cell is in bounds (knowledge saves time in neighbor loop)
	bool upperInBounds = vel.isInBounds(base + toVec3i(sizeI-1,sizeJ-1,sizeK-1));
	bool thisInBounds = vel.isInBounds(base);

	// Neighbor loop: Iterate over neighboring cells of this particle
	for (int k = loopStart.z; k < sizeK; k++)
	for (int j = loopStart.y; j < sizeJ; j++)
	for (int i = loopStart.x; i < sizeI; i++)
	{
		// Only perform bounds check for current ijk if this or upper bound cell are not in bounds
		if ((!upperInBounds || !thisInBounds) && !vel.isInBounds(base + toVec3i(i,j,k))) continue;

		targetPos = vel.index(base + toVec3i(i,j,k));
		dpos = (Vec3(i,j,k) - fx);

		gridVel = vel(targetPos) * dimFactor;
		weight = w[i].x * w[j].y * w[k].z;
		pvelNew += weight * gridVel;

		outerProduct(outerProd, weight * gridVel, dpos * dimFactor);
		affineMomentumNew += 4 * outerProd;
	}
	// Copy accumulated values into actual particle data-structures (outside of for-loop for better performance)
	pvel[idx] = pvelNew;
	affineMomentum[idx] = affineMomentumNew;

	// Advection
	pp[idx].pos += dt * pvel[idx];

	// Deformation update
	T F = (T(Vec3(1.)) + dt * affineMomentum[idx]) * deformationGrad[idx];

	// SVD
	T svdU(Vec3(0.)), Sig(Vec3(0.)), svdV(Vec3(0.));
	svd(F, svdU, svdV, Sig);

	// Snow plasticity
	const int sizeSig = (is3D) ? 3 : 2;
	for (int i = 0; i < sizeSig * int(plastic); i++) {
		Sig(i,i) = clamp(Sig(i,i), 1.0f - 2.5e-2f, 1.0f + 7.5e-3f);
	}

	Real oldJ = F.determinant();
	F = svdU * Sig * svdV.transposed();
	Real Jp_new = clamp(detDeformationGrad[idx] * oldJ / F.determinant(), 0.6f, 20.0f);

	detDeformationGrad[idx] = Jp_new;
	deformationGrad[idx] = F;
}

PYTHON() void mpmMapMACGridToParts2D(
	BasicParticleSystem& pp, const MACGrid& vel, const Grid<Real>& mass, ParticleDataImpl<Vec3>& pvel,
	ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<Matrix2x2f>& deformationGrad,
	ParticleDataImpl<Matrix2x2f>& affineMomentum, const bool plastic=true)
{
	Vec3i loopStart(-1,-1,0); // 0: disable dim (loop once), -1: full neighbor loop [0,2]
	knMpmMapMACGridToVec3<Matrix2x2f>(pp, vel, mass, pvel, detDeformationGrad, deformationGrad, affineMomentum, plastic, loopStart, false);
}

PYTHON() void mpmMapMACGridToParts3D(
	BasicParticleSystem& pp, const MACGrid& vel, const Grid<Real>& mass, ParticleDataImpl<Vec3>& pvel,
	ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<Matrix3x3f>& deformationGrad,
	ParticleDataImpl<Matrix3x3f>& affineMomentum, const bool plastic=true)
{
	Vec3i loopStart(-1,-1,-1); // Enable loops in all 3 dims
	knMpmMapMACGridToVec3<Matrix3x3f>(pp, vel, mass, pvel, detDeformationGrad, deformationGrad, affineMomentum, plastic, loopStart, true);
}

} // namespace
