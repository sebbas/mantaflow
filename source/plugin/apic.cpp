// ----------------------------------------------------------------------------
//
// MantaFlow fluid solver framework
// Copyright 2016-2020 Kiwon Um, Nils Thuerey
//
// This program is free software, distributed under the terms of the
// Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// Affine Particle-In-Cell
//
// ----------------------------------------------------------------------------

#include "particle.h"
#include "grid.h"

namespace Manta {

#define FOR_INT_IJK(num) \
for(int i=0; i<num; ++i) \
	for(int j=0; j<num; ++j) \
		for(int k=0; k<num; ++k)

static inline IndexInt indexUFace(const Vec3 &pos, const MACGrid &ref) {
	const Vec3i f = toVec3i(pos), c = toVec3i(pos-0.5);
	const IndexInt index = f.x * ref.getStrideX() + c.y * ref.getStrideY() + c.z * ref.getStrideZ();
	assertDeb(ref.isInBounds(index), "Grid index out of bounds for particle position [" << pos.x << ", " << pos.y << ", " << pos.z << "]");
	return (ref.isInBounds(index)) ? index : -1;
}

static inline IndexInt indexVFace(const Vec3 &pos, const MACGrid &ref) {
	const Vec3i f = toVec3i(pos), c = toVec3i(pos-0.5);
	const IndexInt index = c.x * ref.getStrideX() + f.y * ref.getStrideY() + c.z * ref.getStrideZ();
	assertDeb(ref.isInBounds(index), "Grid index out of bounds for particle position [" << pos.x << ", " << pos.y << ", " << pos.z << "]");
	return (ref.isInBounds(index)) ? index : -1;
}

static inline IndexInt indexWFace(const Vec3 &pos, const MACGrid &ref) {
	const Vec3i f = toVec3i(pos), c = toVec3i(pos-0.5);
	const IndexInt index = c.x * ref.getStrideX() + c.y * ref.getStrideY() + f.z * ref.getStrideZ();
	assertDeb(ref.isInBounds(index), "Grid index out of bounds for particle position [" << pos.x << ", " << pos.y << ", " << pos.z << "]");
	return (ref.isInBounds(index)) ? index : -1;
}

static inline IndexInt indexOffset(const IndexInt gidx, const int i, const int j, const int k, const MACGrid &ref) {
	const IndexInt dX[2] = { 0, ref.getStrideX() };
	const IndexInt dY[2] = { 0, ref.getStrideY() };
	const IndexInt dZ[2] = { 0, ref.getStrideZ() };
	const IndexInt index = gidx + dX[i] + dY[j] + dZ[k];
	assertDeb(ref.isInBounds(index), "Grid index out of bounds for particle position [" << pos.x << ", " << pos.y << ", " << pos.z << "]");
	return (ref.isInBounds(index)) ? index : -1;
}

KERNEL(pts, single)
void knApicMapLinearVec3ToMACGrid(
	const BasicParticleSystem &p, MACGrid &mg, MACGrid &vg, const ParticleDataImpl<Vec3> &vp,
	const ParticleDataImpl<Vec3> &cpx, const ParticleDataImpl<Vec3> &cpy, const ParticleDataImpl<Vec3> &cpz,
	const ParticleDataImpl<int> *ptype, const int exclude, const int boundaryWidth)
{
	if (!p.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) return;
	if (!vg.isInBounds(p.getPos(idx), boundaryWidth)) {
		debMsg("Skipping particle at index " << idx << ". Is out of bounds and cannot be applied to grid.", 1);
		return;
	}

	const Vec3 &pos = p.getPos(idx), &vel = vp[idx];
	const Vec3i f = toVec3i(pos);
	const Vec3i c = toVec3i(pos-0.5);
	const Vec3 wf = clamp(pos-toVec3(f), Vec3(0.), Vec3(1.));
	const Vec3 wc = clamp(pos-toVec3(c)-0.5, Vec3(0.), Vec3(1.));

	{ // u-face
		const IndexInt gidx = indexUFace(pos, vg);
		if (gidx < 0) return; // debug will fail before

		const Vec3 gpos(f.x, c.y+0.5, c.z+0.5);
		const Real wi[2] = { Real(1)-wf.x, wf.x };
		const Real wj[2] = { Real(1)-wc.y, wc.y };
		const Real wk[2] = { Real(1)-wc.z, wc.z };

		FOR_INT_IJK(2) {
			const Real w = wi[i] * wj[j] * wk[k];
			const IndexInt vidx = indexOffset(gidx, i, j, k, vg);
			if (vidx < 0) continue; // debug will fail before

			mg[vidx].x += w;
			vg[vidx].x += w * vel.x;
			vg[vidx].x += w*dot(cpx[idx], gpos + Vec3(i, j, k) - pos);
		}
	}
	{ // v-face
		const IndexInt gidx = indexVFace(pos, vg);
		if (gidx < 0) return;

		const Vec3 gpos(c.x+0.5, f.y, c.z+0.5);
		const Real wi[2] = { Real(1)-wc.x, wc.x };
		const Real wj[2] = { Real(1)-wf.y, wf.y };
		const Real wk[2] = { Real(1)-wc.z, wc.z };

		FOR_INT_IJK(2) {
			const Real w = wi[i] * wj[j] * wk[k];
			const IndexInt vidx = indexOffset(gidx, i, j, k, vg);
			if (vidx < 0) continue;

			mg[vidx].y += w;
			vg[vidx].y += w * vel.y;
			vg[vidx].y += w * dot(cpy[idx], gpos + Vec3(i, j, k) - pos);
		}
	}
	if(!vg.is3D()) return;
	{ // w-face
		const IndexInt gidx = indexWFace(pos, vg);
		if (gidx < 0) return;

		const Vec3 gpos(c.x+0.5, c.y+0.5, f.z);
		const Real wi[2] = { Real(1)-wc.x, wc.x };
		const Real wj[2] = { Real(1)-wc.y, wc.y };
		const Real wk[2] = { Real(1)-wf.z, wf.z };

		FOR_INT_IJK(2) {
			const Real w = wi[i] * wj[j] * wk[k];
			const IndexInt vidx = indexOffset(gidx, i, j, k, vg);
			if (vidx < 0) continue;

			mg[vidx].z += w;
			vg[vidx].z += w * vel.z;
			vg[vidx].z += w * dot(cpz[idx], gpos + Vec3(i, j, k) - pos);
		}
	}
}

PYTHON()
void apicMapPartsToMAC(
	const FlagGrid &flags, MACGrid &vel,
	const BasicParticleSystem &parts, const ParticleDataImpl<Vec3> &partVel,
	const ParticleDataImpl<Vec3> &cpx, const ParticleDataImpl<Vec3> &cpy, const ParticleDataImpl<Vec3> &cpz,
	MACGrid *mass=NULL, const ParticleDataImpl<int> *ptype=NULL, const int exclude=0, const int boundaryWidth=0)
{
	// affine map: let's assume that the particle mass is constant, 1.0
	if(!mass) {
		MACGrid tmpmass(vel.getParent());
		mass = &tmpmass;
	}

	mass->clear();
	vel.clear();

	knApicMapLinearVec3ToMACGrid(parts, *mass, vel, partVel, cpx, cpy, cpz, ptype, exclude, boundaryWidth);
	mass->stomp(VECTOR_EPSILON);
	vel.safeDivide(*mass);
}

KERNEL(pts)
void knApicMapLinearMACGridToVec3(
	ParticleDataImpl<Vec3> &vp, ParticleDataImpl<Vec3> &cpx, ParticleDataImpl<Vec3> &cpy, ParticleDataImpl<Vec3> &cpz,
	const BasicParticleSystem &p, const MACGrid &vg, const FlagGrid &flags,
	const ParticleDataImpl<int> *ptype, const int exclude, const int boundaryWidth)
{
	if (!p.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) return;
	if (!vg.isInBounds(p.getPos(idx), boundaryWidth)) {
		debMsg("Skipping particle at index " << idx << ". Is out of bounds and cannot get value from grid.", 1);
		return;
	}

	vp[idx] = cpx[idx] = cpy[idx] = cpz[idx] = Vec3(Real(0));
	const Real gw[2] = {-Real(1), Real(1)};

	const Vec3 &pos = p.getPos(idx);
	const Vec3i f = toVec3i(pos);
	const Vec3i c = toVec3i(pos-0.5);
	const Vec3 wf = clamp(pos-toVec3(f), Vec3(0.), Vec3(1.));
	const Vec3 wc = clamp(pos-toVec3(c)-0.5, Vec3(0.), Vec3(1.));

	{ // u-face
		const IndexInt gidx = indexUFace(pos, vg);
		if (gidx < 0) return; // debug will fail before

		const Real wx[2] = { Real(1)-wf.x, wf.x };
		const Real wy[2] = { Real(1)-wc.y, wc.y };
		const Real wz[2] = { Real(1)-wc.z, wc.z };

		FOR_INT_IJK(2) {
			const IndexInt vidx = indexOffset(gidx, i, j, k, vg);
			if (vidx < 0) continue; // debug will fail before

			const Real vgx = vg[vidx].x;
			vp[idx].x  += wx[i] * wy[j] * wz[k] * vgx;
			cpx[idx].x += gw[i] * wy[j] * wz[k] * vgx;
			cpx[idx].y += wx[i] * gw[j] * wz[k] * vgx;
			cpx[idx].z += wx[i] * wy[j] * gw[k] * vgx;
		}
	}
	{ // v-face
		const IndexInt gidx = indexVFace(pos, vg);
		if (gidx < 0) return;

		const Real wx[2] = { Real(1)-wc.x, wc.x };
		const Real wy[2] = { Real(1)-wf.y, wf.y };
		const Real wz[2] = { Real(1)-wc.z, wc.z };

		FOR_INT_IJK(2) {
			const IndexInt vidx = indexOffset(gidx, i, j, k, vg);
			if (vidx < 0) continue;

			const Real vgy = vg[vidx].y;
			vp[idx].y  += wx[i] * wy[j] * wz[k] * vgy;
			cpy[idx].x += gw[i] * wy[j] * wz[k] * vgy;
			cpy[idx].y += wx[i] * gw[j] * wz[k] * vgy;
			cpy[idx].z += wx[i] * wy[j] * gw[k] * vgy;
		}
	}
	if(!vg.is3D()) return;
	{ // w-face
		const IndexInt gidx = indexWFace(pos, vg);
		if (gidx < 0) return;

		const Real wx[2] = { Real(1)-wc.x, wc.x };
		const Real wy[2] = { Real(1)-wc.y, wc.y };
		const Real wz[2] = { Real(1)-wf.z, wf.z };

		FOR_INT_IJK(2) {
			const IndexInt vidx = indexOffset(gidx, i, j, k, vg);
			if (vidx < 0) continue;

			const Real vgz = vg[vidx].z;
			vp[idx].z  += wx[i] * wy[j] * wz[k] * vgz;
			cpz[idx].x += gw[i] * wy[j] * wz[k] * vgz;
			cpz[idx].y += wx[i] * gw[j] * wz[k] * vgz;
			cpz[idx].z += wx[i] * wy[j] * gw[k] * vgz;
		}
	}
}

PYTHON()
void apicMapMACGridToParts(
	ParticleDataImpl<Vec3> &partVel, ParticleDataImpl<Vec3> &cpx, ParticleDataImpl<Vec3> &cpy, ParticleDataImpl<Vec3> &cpz,
	const BasicParticleSystem &parts, const MACGrid &vel, const FlagGrid &flags,
	const ParticleDataImpl<int> *ptype=NULL, const int exclude=0, const int boundaryWidth=0)
{
	knApicMapLinearMACGridToVec3(partVel, cpx, cpy, cpz, parts, vel, flags, ptype, exclude, boundaryWidth);
}

} // namespace
