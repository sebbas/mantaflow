/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2013 Tobias Pfaff, Nils Thuerey
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Particle data functionality
 *
 ******************************************************************************/

#include <fstream>
#include <cstring>
#if NO_ZLIB!=1
#include <zlib.h>
#endif
#include "particle.h"
#include "levelset.h"
#include "mantaio.h"

using namespace std;
namespace Manta {

int ParticleBase::globalSeed = 9832;

ParticleBase::ParticleBase(FluidSolver* parent, int fixedSeed)
	: PbClass(parent), mMaxParticles(0), mAllowCompress(true), mFreePdata(false), mSeed(fixedSeed)
{
	// use global random seed if none is given
	if (fixedSeed==-1) {
		mSeed = globalSeed;
	}
}

ParticleBase::~ParticleBase()
{
	// make sure data fields now parent system is deleted
	for(IndexInt i=0; i<(IndexInt)mPartData.size(); ++i)
		mPartData[i]->setParticleSys(nullptr);

	if(mFreePdata) {
		for(IndexInt i=0; i<(IndexInt)mPartData.size(); ++i)
			delete mPartData[i];
	}

}

std::string ParticleBase::infoString() const
{
	return "ParticleSystem " + mName + " <no info>";
}

void ParticleBase::cloneParticleData(ParticleBase* nm)
{
	// clone additional data , and make sure the copied particle system deletes it
	nm->mFreePdata = true;
	for(IndexInt i=0; i<(IndexInt)mPartData.size(); ++i) {
		ParticleDataBase* pdata = mPartData[i]->clone();
		nm->registerPdata(pdata);
	}
}

void ParticleBase::deregister(ParticleDataBase* pdata)
{
	bool done = false;
	// remove pointer from particle data list
	for(IndexInt i=0; i<(IndexInt)mPartData.size(); ++i) {
		if(mPartData[i] == pdata) {
			if(i<(IndexInt)mPartData.size()-1)
				mPartData[i] = mPartData[mPartData.size()-1];
			mPartData.pop_back();
			done = true;
		}
	}
	if(!done)
		errMsg("Invalid pointer given, not registered!");
}

// create and attach a new pdata field to this particle system
PbClass* ParticleBase::create(PbType t, PbTypeVec T, const string& name)
{
#	if NOPYTHON!=1
	_args.add("nocheck",true);
	if(t.str() == "") errMsg("Specify particle data type to create");
	//debMsg( "Pdata creating '"<< t.str <<" with size "<< this->getSizeSlow(), 5 );

	PbClass* pyObj = PbClass::createPyObject(t.str() + T.str(), name, _args, this->getParent());

	ParticleDataBase* pdata = dynamic_cast<ParticleDataBase*>(pyObj);
	if(!pdata) {
		errMsg("Unable to get particle data pointer from newly created object. Only create ParticleData type with a ParticleSys.creat() call, eg, PdataReal, PdataVec3 etc.");
		delete pyObj;
		return nullptr;
	} else {
		this->registerPdata(pdata);
	}

	// directly init size of new pdata field:
	pdata->resize(this->getSizeSlow());
#	else
	PbClass* pyObj = nullptr;
#	endif
	return pyObj;
}

void ParticleBase::registerPdata(ParticleDataBase* pdata)
{
	pdata->setParticleSys(this);
	mPartData.push_back(pdata);

	if(pdata->getType() == ParticleDataBase::TypeReal) {
		ParticleDataImpl<Real>* pd = dynamic_cast< ParticleDataImpl<Real>* >(pdata);
		if(!pd) errMsg("Invalid pdata object posing as real!");
		this->registerPdataReal(pd);
	} else if(pdata->getType() == ParticleDataBase::TypeInt) {
		ParticleDataImpl<int>* pd = dynamic_cast< ParticleDataImpl<int>* >(pdata);
		if(!pd) errMsg("Invalid pdata object posing as int!");
		this->registerPdataInt(pd);
	} else if(pdata->getType() == ParticleDataBase::TypeVec3) {
		ParticleDataImpl<Vec3>* pd = dynamic_cast< ParticleDataImpl<Vec3>* >(pdata);
		if(!pd) errMsg("Invalid pdata object posing as vec3!");
		this->registerPdataVec3(pd);
	} else if(pdata->getType() == ParticleDataBase::TypeMat3) {
		ParticleDataImpl<Matrix3x3f>* pd = dynamic_cast< ParticleDataImpl<Matrix3x3f>* >(pdata);
		if(!pd) errMsg("Invalid pdata object posing as mat3!");
		this->registerPdataMat3(pd);
	} else if(pdata->getType() == ParticleDataBase::TypeMat2) {
		ParticleDataImpl<Matrix2x2f>* pd = dynamic_cast< ParticleDataImpl<Matrix2x2f>* >(pdata);
		if(!pd) errMsg("Invalid pdata object posing as mat2!");
		this->registerPdataMat2(pd);
	}
}
void ParticleBase::registerPdataReal(ParticleDataImpl<Real>* pd) { mPdataReal.push_back(pd); }
void ParticleBase::registerPdataVec3(ParticleDataImpl<Vec3>* pd) { mPdataVec3.push_back(pd); }
void ParticleBase::registerPdataInt (ParticleDataImpl<int >* pd) { mPdataInt .push_back(pd); }
void ParticleBase::registerPdataMat3(ParticleDataImpl<Matrix3x3f>* pd) { mPdataMat3.push_back(pd); }
void ParticleBase::registerPdataMat2(ParticleDataImpl<Matrix2x2f>* pd) { mPdataMat2.push_back(pd); }

void ParticleBase::addAllPdata()
{
	for(IndexInt i=0; i<(IndexInt)mPartData.size(); ++i) {
		mPartData[i]->addEntry();
	}
}


BasicParticleSystem::BasicParticleSystem(FluidSolver* parent)
	: ParticleSystem<BasicParticleData>(parent)
{
	this->mAllowCompress = false;
}

// file io

void BasicParticleSystem::writeParticlesText(const string name) const
{
	ofstream ofs(name.c_str());
	if(!ofs.good()) errMsg("can't open file!");
	ofs << this->size()<<", pdata: "<< mPartData.size()<<" ("<<mPdataInt.size()<<","<<mPdataReal.size()<<","<<mPdataVec3.size()<<") \n";
	for(IndexInt i=0; i<this->size(); ++i) {
		ofs << i<<": "<< this->getPos(i) <<" , "<< this->getStatus(i) <<". ";
		for(IndexInt pd=0; pd<(IndexInt)mPdataInt.size() ; ++pd) ofs << mPdataInt [pd]->get(i)<<" ";
		for(IndexInt pd=0; pd<(IndexInt)mPdataReal.size(); ++pd) ofs << mPdataReal[pd]->get(i)<<" ";
		for(IndexInt pd=0; pd<(IndexInt)mPdataVec3.size(); ++pd) ofs << mPdataVec3[pd]->get(i)<<" ";
		for(IndexInt pd=0; pd<(IndexInt)mPdataMat3.size(); ++pd) ofs << mPdataMat3[pd]->get(i)<<" ";
		for(IndexInt pd=0; pd<(IndexInt)mPdataMat2.size(); ++pd) ofs << mPdataMat2[pd]->get(i)<<" ";
		ofs << "\n";
	}
	ofs.close();
}

void BasicParticleSystem::writeParticlesRawPositionsGz(const string name) const
{
#	if NO_ZLIB!=1
	gzFile gzf = (gzFile) safeGzopen(name.c_str(), "wb1");
	if(!gzf) errMsg("can't open file "<<name);
	for(IndexInt i=0; i<this->size(); ++i) {
		Vector3D<float> p = toVec3f(this->getPos(i));
		gzwrite(gzf, &p, sizeof(float)*3);
	}
	gzclose(gzf);
#	else
	cout << "file format not supported without zlib" << endl;
#	endif
}

void BasicParticleSystem::writeParticlesRawVelocityGz(const string name) const
{
#	if NO_ZLIB!=1
	gzFile gzf = (gzFile) safeGzopen(name.c_str(), "wb1");
	if (!gzf) errMsg("can't open file "<<name);
	if( mPdataVec3.size() < 1 ) errMsg("no vec3 particle data channel found!");
	// note , assuming particle data vec3 0 is velocity! make optional...
	for(IndexInt i=0; i<this->size(); ++i) {
		Vector3D<float> p = toVec3f(mPdataVec3[0]->get(i));
		gzwrite(gzf, &p, sizeof(float)*3);
	}
	gzclose(gzf);
#	else
	cout << "file format not supported without zlib" << endl;
#	endif
}


int BasicParticleSystem::load(const string name)
{
	if(name.find_last_of('.') == string::npos)
		errMsg("file '" + name + "' does not have an extension");
	string ext = name.substr(name.find_last_of('.'));
	if(ext == ".uni")
		return readParticlesUni(name, this );
	else if (ext == ".vdb") {
		std::vector<PbClass*> parts;
		parts.push_back(this);
		return readObjectsVDB(name, &parts);
	} else if(ext == ".raw") // raw = uni for now
		return readParticlesUni(name, this );
	else
		errMsg("particle '" + name +"' filetype not supported for loading");
	return 0;
}

int BasicParticleSystem::save(const string name)
{
	if(name.find_last_of('.') == string::npos)
		errMsg("file '" + name + "' does not have an extension");
	string ext = name.substr(name.find_last_of('.'));
	if(ext == ".txt")
		this->writeParticlesText(name);
	else if(ext == ".uni")
		return writeParticlesUni(name, this);
	else if(ext == ".raw") // raw = uni for now
		return writeParticlesUni(name, this);
	else if (ext == ".vdb") {
		std::vector<PbClass*> parts;
		parts.push_back(this);
		return writeObjectsVDB(name, &parts);
	// raw data formats, very basic for simple data transfer to other programs
	} else if(ext == ".posgz")
		this->writeParticlesRawPositionsGz(name);
	else if(ext == ".velgz")
		this->writeParticlesRawVelocityGz(name);
	else
		errMsg("particle '" + name +"' filetype not supported for saving");
	return 0;
}

void BasicParticleSystem::printParts(IndexInt start, IndexInt stop, bool printIndex)
{
	std::ostringstream sstr;
	IndexInt s = (start>0 ? start : 0                      );
	IndexInt e = (stop>0  ? stop  : (IndexInt)mData.size() );
	s = Manta::clamp(s, (IndexInt)0, (IndexInt)mData.size());
	e = Manta::clamp(e, (IndexInt)0, (IndexInt)mData.size());

	for(IndexInt i=s; i<e; ++i) {
		if(printIndex) sstr << i<<": ";
		sstr<<mData[i].pos<<" "<<mData[i].flag<<"\n";
	}
	debMsg( sstr.str() , 1 );
}

 std::vector<BasicParticleData>* BasicParticleSystem::getDataPointer() {
	return &mData;
}

void BasicParticleSystem::readParticles(BasicParticleSystem* from) {
	// re-allocate all data
	this->resizeAll(from->size());
	assertMsg(from->size() == this->size() , "particle size doesn't match");

	for(int i=0; i<this->size(); ++i) {
		(*this)[i].pos  = (*from)[i].pos;
		(*this)[i].flag = (*from)[i].flag;
	}
	this->transformPositions(from->getParent()->getGridSize(), this->getParent()->getGridSize());
}


// particle data

ParticleDataBase::ParticleDataBase(FluidSolver* parent)
	: PbClass(parent), mpParticleSys(nullptr)
{
}

ParticleDataBase::~ParticleDataBase()
{
	// notify parent of deletion
	if(mpParticleSys)
		mpParticleSys->deregister(this);
}


// actual data implementation

template<class T>
ParticleDataImpl<T>::ParticleDataImpl(FluidSolver* parent)
	: ParticleDataBase(parent), mpGridSource(nullptr), mGridSourceMAC(false)
{
}

template<class T>
ParticleDataImpl<T>::ParticleDataImpl(FluidSolver* parent, ParticleDataImpl<T>* other)
	: ParticleDataBase(parent), mpGridSource(nullptr), mGridSourceMAC(false)
{
	this->mData = other->mData;
	setName(other->getName());
}

template<class T>
ParticleDataImpl<T>::~ParticleDataImpl()
{
}

template<class T>
IndexInt ParticleDataImpl<T>::getSizeSlow() const
{
	return mData.size();
}
template<class T>
void ParticleDataImpl<T>::addEntry()
{
	// add zero'ed entry
	T tmp = T(0.);
	// for debugging, force init:
	//tmp = T(0.02 * mData.size()); // increasing
	//tmp = T(1.); // constant 1
	return mData.push_back(tmp);
}
template<class T>
void ParticleDataImpl<T>::resize(IndexInt s)
{
	mData.resize(s);
}
template<class T>
void ParticleDataImpl<T>::copyValueSlow(IndexInt from, IndexInt to)
{
	this->copyValue(from,to);
}
template<class T>
ParticleDataBase* ParticleDataImpl<T>::clone()
{
	ParticleDataImpl<T>* npd = new ParticleDataImpl<T>(getParent(), this);
	return npd;
}

template<class T>
void ParticleDataImpl<T>::setSource(GridBase* grid, bool isMAC)
{
	mpGridSource = (Grid<T>*) grid;
	mGridSourceMAC = isMAC;
	if (grid && isMAC) assertMsg( grid->getType() & GridBase::TypeMAC , "Given grid is not a valid MAC grid" );
}

template<class T>
void ParticleDataImpl<T>::initNewValue(IndexInt idx, Vec3 pos)
{
	if(!mpGridSource)
		mData[idx] = 0;
	else {
		mData[idx] = mpGridSource->getInterpolated(pos);
	}
}
// special handling needed for velocities
template<>
void ParticleDataImpl<Vec3>::initNewValue(IndexInt idx, Vec3 pos)
{
	if(!mpGridSource)
		mData[idx] = 0;
	else {
		if(!mGridSourceMAC)
			mData[idx] = mpGridSource->getInterpolated(pos);
		else
			mData[idx] = ((MACGrid*)mpGridSource)->getInterpolated(pos);
	}
}
// special handling needed for matrices
template<>
void ParticleDataImpl<Matrix3x3f>::initNewValue(IndexInt idx, Vec3 pos)
{
	mData[idx] = Matrix3x3f();
}
template<>
void ParticleDataImpl<Matrix2x2f>::initNewValue(IndexInt idx, Vec3 pos)
{
	mData[idx] = Matrix2x2f();
}

template<typename T>
int ParticleDataImpl<T>::load(string name)
{
	if(name.find_last_of('.') == string::npos)
		errMsg("file '" + name + "' does not have an extension");
	string ext = name.substr(name.find_last_of('.'));
	if(ext == ".uni")
		return readPdataUni<T>(name, this);
	else if (ext == ".vdb") {
		std::vector<PbClass*> parts;
		parts.push_back(this);
		return readObjectsVDB(name, &parts);
	}
	else if(ext == ".raw") // raw = uni for now
		return readPdataUni<T>(name, this);
	else
		errMsg("particle data '" + name +"' filetype not supported for loading");
	return 0;
}

template<typename T>
int ParticleDataImpl<T>::save(string name)
{
	if(name.find_last_of('.') == string::npos)
		errMsg("file '" + name + "' does not have an extension");
	string ext = name.substr(name.find_last_of('.'));
	if(ext == ".uni")
		return writePdataUni<T>(name, this);
	else if (ext == ".vdb") {
		std::vector<PbClass*> parts;
		parts.push_back(this);
		return writeObjectsVDB(name, &parts);
	}
	else if(ext == ".raw") // raw = uni for now
		return writePdataUni<T>(name, this);
	else
		errMsg("particle data '" + name +"' filetype not supported for saving");
	return 0;
}

// specializations

template<>
ParticleDataBase::PdataType ParticleDataImpl<Real>::getType() const
{
	return ParticleDataBase::TypeReal;
}
template<>
ParticleDataBase::PdataType ParticleDataImpl<int>::getType() const
{
	return ParticleDataBase::TypeInt;
}
template<>
ParticleDataBase::PdataType ParticleDataImpl<Vec3>::getType() const
{
	return ParticleDataBase::TypeVec3;
}
template<>
ParticleDataBase::PdataType ParticleDataImpl<Matrix3x3f>::getType() const
{
	return ParticleDataBase::TypeMat3;
}
template<>
ParticleDataBase::PdataType ParticleDataImpl<Matrix2x2f>::getType() const
{
	return ParticleDataBase::TypeMat2;
}

// note, we need a flag value for functions such as advection
// ideally, this value should never be modified
int ParticleIndexData::flag = 0;
Vec3 ParticleIndexData::pos = Vec3(0.,0.,0.);

KERNEL(pts) template<class T, class S> void knPdataAdd    (ParticleDataImpl<T>& me, const ParticleDataImpl<S>& other) { me[idx] += other[idx]; }
KERNEL(pts) template<class T, class S> void knPdataSub    (ParticleDataImpl<T>& me, const ParticleDataImpl<S>& other) { me[idx] -= other[idx]; }
KERNEL(pts) template<class T, class S> void knPdataMult   (ParticleDataImpl<T>& me, const ParticleDataImpl<S>& other) { me[idx] *= other[idx]; }
KERNEL(pts) template<class T, class S> void knPdataDiv    (ParticleDataImpl<T>& me, const ParticleDataImpl<S>& other) { me[idx] /= other[idx]; }
KERNEL(pts) template<class T>          void knPdataSafeDiv(ParticleDataImpl<T>& me, const ParticleDataImpl<T>& other) { me[idx] = safeDivide(me[idx], other[idx]); }

KERNEL(pts) template<class T, class S> void knPdataSetScalar    (ParticleDataImpl<T>& me, const S& other) { me[idx]  = other; }
KERNEL(pts) template<class T, class S> void knPdataAddScalar    (ParticleDataImpl<T>& me, const S& other) { me[idx] += other; }
KERNEL(pts) template<class T, class S> void knPdataMultScalar   (ParticleDataImpl<T>& me, const S& other) { me[idx] *= other; }
KERNEL(pts) template<class T, class S> void knPdataScaledAdd    (ParticleDataImpl<T>& me, const ParticleDataImpl<T>& other, const S& factor) { me[idx] += factor * other[idx]; }

KERNEL(pts) template<class T> void knPdataClamp(ParticleDataImpl<T>& me, const T vmin, const T vmax) { me[idx] = clamp(me[idx], vmin, vmax); }
KERNEL(pts) template<class T> void knPdataClampMin(ParticleDataImpl<T>& me, const T vmin)            { me[idx] = std::max(vmin, me[idx]); }
KERNEL(pts) template<class T> void knPdataClampMax(ParticleDataImpl<T>& me, const T vmax)            { me[idx] = std::min(vmax, me[idx]); }
KERNEL(pts)                   void knPdataClampMinVec3(ParticleDataImpl<Vec3>& me, const Real vmin)
{
	me[idx].x = std::max(vmin, me[idx].x);
	me[idx].y = std::max(vmin, me[idx].y);
	me[idx].z = std::max(vmin, me[idx].z);
}
KERNEL(pts)                   void knPdataClampMaxVec3(ParticleDataImpl<Vec3>& me, const Real vmax)
{
	me[idx].x = std::min(vmax, me[idx].x);
	me[idx].y = std::min(vmax, me[idx].y);
	me[idx].z = std::min(vmax, me[idx].z);
}
KERNEL(pts)                   void knPdataClampMinMat3(ParticleDataImpl<Matrix3x3f>& me, const Real vmin)
{
	me[idx].v00 = std::max(vmin, me[idx].v00);
	me[idx].v01 = std::max(vmin, me[idx].v01);
	me[idx].v02 = std::max(vmin, me[idx].v02);

	me[idx].v10 = std::max(vmin, me[idx].v10);
	me[idx].v11 = std::max(vmin, me[idx].v11);
	me[idx].v12 = std::max(vmin, me[idx].v12);

	me[idx].v20 = std::max(vmin, me[idx].v20);
	me[idx].v21 = std::max(vmin, me[idx].v21);
	me[idx].v22 = std::max(vmin, me[idx].v22);
}
KERNEL(pts)                   void knPdataClampMaxMat3(ParticleDataImpl<Matrix3x3f>& me, const Real vmax)
{
	me[idx].v00 = std::min(vmax, me[idx].v00);
	me[idx].v01 = std::min(vmax, me[idx].v01);
	me[idx].v02 = std::min(vmax, me[idx].v02);

	me[idx].v10 = std::min(vmax, me[idx].v10);
	me[idx].v11 = std::min(vmax, me[idx].v11);
	me[idx].v12 = std::min(vmax, me[idx].v12);

	me[idx].v20 = std::min(vmax, me[idx].v20);
	me[idx].v21 = std::min(vmax, me[idx].v21);
	me[idx].v22 = std::min(vmax, me[idx].v22);
}
KERNEL(pts)                   void knPdataClampMinMat2(ParticleDataImpl<Matrix2x2f>& me, const Real vmin)
{
	me[idx].v00 = std::max(vmin, me[idx].v00);
	me[idx].v01 = std::max(vmin, me[idx].v01);

	me[idx].v10 = std::max(vmin, me[idx].v10);
	me[idx].v11 = std::max(vmin, me[idx].v11);
}
KERNEL(pts)                   void knPdataClampMaxMat2(ParticleDataImpl<Matrix2x2f>& me, const Real vmax)
{
	me[idx].v00 = std::min(vmax, me[idx].v00);
	me[idx].v01 = std::min(vmax, me[idx].v01);

	me[idx].v10 = std::min(vmax, me[idx].v10);
	me[idx].v11 = std::min(vmax, me[idx].v11);
}

// python operators


template<typename T>
ParticleDataImpl<T>& ParticleDataImpl<T>::copyFrom(const ParticleDataImpl<T>& a)
{
	assertMsg(a.mData.size() == mData.size() , "different pdata size "<<a.mData.size()<<" vs "<<this->mData.size());
	mData = a.mData;
	return *this;
}

template<typename T>
void ParticleDataImpl<T>::setConst(const T &s)
{
	knPdataSetScalar<T,T> op( *this, s );
}
template<>
void ParticleDataImpl<Matrix3x3f>::setConst(const Matrix3x3f &s)
{
	knPdataSetScalar<Matrix3x3f, Matrix3x3f> op( *this, s );
}
template<>
void ParticleDataImpl<Matrix2x2f>::setConst(const Matrix2x2f &s)
{
	knPdataSetScalar<Matrix2x2f, Matrix2x2f> op( *this, s );
}

template<typename T>
void ParticleDataImpl<T>::setConstRange(const T &s, const int begin, const int end)
{
	for(int i=begin; i<end; ++i) (*this)[i] = s;
}

// special set by flag
KERNEL(pts) template<class T, class S> void knPdataSetScalarIntFlag(ParticleDataImpl<T> &me, const S &other, const ParticleDataImpl<int> &t, const int itype)
{
	if(t[idx]&itype) me[idx] = other;
}
template<typename T>
void ParticleDataImpl<T>::setConstIntFlag(const T &s, const ParticleDataImpl<int> &t, const int itype)
{
	knPdataSetScalarIntFlag<T,T> op(*this, s, t, itype);
}

template<typename T>
void ParticleDataImpl<T>::add(const ParticleDataImpl<T>& a)
{
	knPdataAdd<T,T> op( *this, a );
}
template<typename T>
void ParticleDataImpl<T>::sub(const ParticleDataImpl<T>& a)
{
	knPdataSub<T,T> op( *this, a );
}

template<typename T>
void ParticleDataImpl<T>::addConst(const T &s)
{
	knPdataAddScalar<T,T> op( *this, s );
}

template<typename T>
void ParticleDataImpl<T>::addScaled(const ParticleDataImpl<T>& a, const T& factor)
{
	knPdataScaledAdd<T,T> op( *this, a, factor );
}

template<typename T>
void ParticleDataImpl<T>::mult(const ParticleDataImpl<T>& a)
{
	knPdataMult<T,T> op( *this, a );
}

template<typename T>
void ParticleDataImpl<T>::safeDiv(const ParticleDataImpl<T>& a)
{
	knPdataSafeDiv<T> op( *this, a );
}

template<typename T>
void ParticleDataImpl<T>::multConst(const T &s)
{
	knPdataMultScalar<T,T> op( *this, s );
}
template<>
void ParticleDataImpl<Matrix3x3f>::multConst(const Matrix3x3f &s)
{
	knPdataMultScalar<Matrix3x3f,Matrix3x3f> op( *this, s );
}
template<>
void ParticleDataImpl<Matrix2x2f>::multConst(const Matrix2x2f &s)
{
	knPdataMultScalar<Matrix2x2f,Matrix2x2f> op( *this, s );
}

template<typename T>
void ParticleDataImpl<T>::clamp(const Real vmin, const Real vmax)
{
	knPdataClamp<T> op( *this, T(vmin), T(vmax) );
}

template<typename T>
void ParticleDataImpl<T>::clampMin(const Real vmin)
{
	knPdataClampMin<T> op( *this, T(vmin) );
}
template<typename T>
void ParticleDataImpl<T>::clampMax(const Real vmax)
{
	knPdataClampMax<T> op( *this, T(vmax) );
}

template<>
void ParticleDataImpl<Vec3>::clampMin(const Real vmin)
{
	knPdataClampMinVec3 op( *this, vmin );
}
template<>
void ParticleDataImpl<Vec3>::clampMax(const Real vmax)
{
	knPdataClampMaxVec3 op( *this, vmax );
}

template<>
void ParticleDataImpl<Matrix3x3f>::clampMin(const Real vmin)
{
	knPdataClampMinMat3 op( *this, vmin );
}
template<>
void ParticleDataImpl<Matrix3x3f>::clampMax(const Real vmax)
{
	knPdataClampMaxMat3 op( *this, vmax );
}

template<>
void ParticleDataImpl<Matrix2x2f>::clampMin(const Real vmin)
{
	knPdataClampMinMat2 op( *this, vmin );
}
template<>
void ParticleDataImpl<Matrix2x2f>::clampMax(const Real vmax)
{
	knPdataClampMaxMat2 op( *this, vmax );
}

template<typename T> KERNEL(pts, reduce=+) returns(T result=T(0.)) T    KnPtsSum(const ParticleDataImpl<T>& val, const ParticleDataImpl<int> *t, const int itype) { if(t && !((*t)[idx]&itype)) return; result += val[idx]; }
template<typename T> KERNEL(pts, reduce=+) returns(Real result=0.) Real KnPtsSumSquare(const ParticleDataImpl<T>& val)    { result += normSquare(val[idx]); }
template<typename T> KERNEL(pts, reduce=+) returns(Real result=0.) Real KnPtsSumMagnitude(const ParticleDataImpl<T>& val) { result += norm(val[idx]); }

template<typename T>
T ParticleDataImpl<T>::sum(const ParticleDataImpl<int> *t, const int itype) const
{
	return KnPtsSum<T>(*this, t, itype);
}
template<typename T>
Real ParticleDataImpl<T>::sumSquare() const
{
	return KnPtsSumSquare<T>(*this);
}
template<typename T>
Real ParticleDataImpl<T>::sumMagnitude() const
{
	return KnPtsSumMagnitude<T>(*this);
}

template<typename T>
KERNEL(pts, reduce=min) returns(Real minVal=std::numeric_limits<Real>::max())
Real CompPdata_Min(const ParticleDataImpl<T>& val)
{
	if(val[idx] < minVal) minVal = val[idx];
}

template<typename T>
KERNEL(pts, reduce=max) returns(Real maxVal=-std::numeric_limits<Real>::max())
Real CompPdata_Max(const ParticleDataImpl<T>& val)
{
	if(val[idx] > maxVal) maxVal = val[idx];
}

template<typename T>
Real ParticleDataImpl<T>::getMin() const
{
	return CompPdata_Min<T>(*this);
}

template<typename T>
Real ParticleDataImpl<T>::getMaxAbs() const
{
	Real amin = CompPdata_Min<T>(*this);
	Real amax = CompPdata_Max<T>(*this);
	return max(fabs(amin), fabs(amax));
}

template<typename T>
Real ParticleDataImpl<T>::getMax() const
{
	return CompPdata_Max<T>(*this);
}

template<typename T>
void ParticleDataImpl<T>::printPdata(IndexInt start, IndexInt stop, bool printIndex)
{
	std::ostringstream sstr;
	IndexInt s = (start>0 ? start : 0                      );
	IndexInt e = (stop>0  ? stop  : (IndexInt)mData.size() );
	s = Manta::clamp(s, (IndexInt)0, (IndexInt)mData.size());
	e = Manta::clamp(e, (IndexInt)0, (IndexInt)mData.size());

	for(IndexInt i=s; i<e; ++i) {
		if(printIndex) sstr << i<<": ";
		sstr<<mData[i]<<" "<<"\n";
	}
	debMsg( sstr.str() , 1 );
}
template<>
void ParticleDataImpl<Matrix3x3f>::printPdata(IndexInt start, IndexInt stop, bool printIndex)
{
	std::ostringstream sstr;
	IndexInt s = (start>0 ? start : 0                      );
	IndexInt e = (stop>0  ? stop  : (IndexInt)mData.size() );
	s = Manta::clamp(s, (IndexInt)0, (IndexInt)mData.size());
	e = Manta::clamp(e, (IndexInt)0, (IndexInt)mData.size());

	for(IndexInt i=s; i<e; ++i) {
		Matrix3x3f mat = mData[i];
		if(printIndex) sstr << i<<": ";
		sstr << "[" << mat(0,0) << "," << mat(0,1) << "," << mat(0,2);
		sstr << "[" << mat(1,0) << "," << mat(1,1) << "," << mat(1,2);
		sstr << "[" << mat(2,0) << "," << mat(2,1) << "," << mat(2,2);
		sstr <<" ]\n";
	}
	debMsg( sstr.str() , 1 );
}
template<>
void ParticleDataImpl<Matrix2x2f>::printPdata(IndexInt start, IndexInt stop, bool printIndex)
{
	std::ostringstream sstr;
	IndexInt s = (start>0 ? start : 0                      );
	IndexInt e = (stop>0  ? stop  : (IndexInt)mData.size() );
	s = Manta::clamp(s, (IndexInt)0, (IndexInt)mData.size());
	e = Manta::clamp(e, (IndexInt)0, (IndexInt)mData.size());

	for(IndexInt i=s; i<e; ++i) {
		Matrix2x2f mat = mData[i];
		if(printIndex) sstr << i<<": ";
		sstr << "[" << mat(0,0) << "," << mat(0,1);
		sstr << "[" << mat(1,0) << "," << mat(1,1);
		sstr <<" ]\n";
	}
	debMsg( sstr.str() , 1 );
}

template<class T> std::vector<T>* ParticleDataImpl<T>::getDataPointer() {
	return &mData;
}

// specials for vec3
// work on length values, ie, always positive (in contrast to scalar versions above)

KERNEL(pts, reduce=min) returns(Real minVal=std::numeric_limits<Real>::max())
Real CompPdata_MinVec3(const ParticleDataImpl<Vec3>& val)
{
	const Real s = normSquare(val[idx]);
	if(s < minVal) minVal = s;
}

KERNEL(pts, reduce=max) returns(Real maxVal=-std::numeric_limits<Real>::max())
Real CompPdata_MaxVec3(const ParticleDataImpl<Vec3>& val)
{
	const Real s = normSquare(val[idx]);
	if(s > maxVal) maxVal = s;
}

template<>
Real ParticleDataImpl<Vec3>::getMin() const
{
	return sqrt(CompPdata_MinVec3(*this));
}

template<>
Real ParticleDataImpl<Vec3>::getMaxAbs() const
{
	return sqrt(CompPdata_MaxVec3(*this));  // no minimum necessary here
}

template<>
Real ParticleDataImpl<Vec3>::getMax() const
{
	return sqrt(CompPdata_MaxVec3(*this));
}

// specials for mat3

KERNEL(pts, reduce=min) returns(Real minVal=std::numeric_limits<Real>::max())
Real CompPdata_MinMat3(const ParticleDataImpl<Matrix3x3f>& val)
{
	const Real s = val[idx].normEuclidean();
	if(s < minVal) minVal = s;
}

KERNEL(pts, reduce=max) returns(Real maxVal=-std::numeric_limits<Real>::max())
Real CompPdata_MaxMat3(const ParticleDataImpl<Matrix3x3f>& val)
{
	const Real s = val[idx].normEuclidean();
	if(s > maxVal) maxVal = s;
}

KERNEL(pts, reduce=+) returns(Real result=0.)
Real KnPtsSumSquareMat3(const ParticleDataImpl<Matrix3x3f>& val)
{
	result += square(val[idx].normEuclidean());

}
KERNEL(pts, reduce=+) returns(Real result=0.)
Real KnPtsSumMagnitudeMat3(const ParticleDataImpl<Matrix3x3f>& val)
{
	result += val[idx].normEuclidean();
}

template<>
Real ParticleDataImpl<Matrix3x3f>::getMin() const
{
	return sqrt(CompPdata_MinMat3(*this));
}

template<>
Real ParticleDataImpl<Matrix3x3f>::getMaxAbs() const
{
	return sqrt(CompPdata_MaxMat3(*this));  // no minimum necessary here
}

template<>
Real ParticleDataImpl<Matrix3x3f>::getMax() const
{
	return sqrt(CompPdata_MaxMat3(*this));
}
template<>
Real ParticleDataImpl<Matrix3x3f>::sumSquare() const
{
	return KnPtsSumSquareMat3(*this);
}
template<>
Real ParticleDataImpl<Matrix3x3f>::sumMagnitude() const
{
	return KnPtsSumMagnitudeMat3(*this);
}

// specials for mat2

KERNEL(pts, reduce=min) returns(Real minVal=std::numeric_limits<Real>::max())
Real CompPdata_MinMat2(const ParticleDataImpl<Matrix2x2f>& val)
{
	const Real s = val[idx].normEuclidean();
	if(s < minVal) minVal = s;
}

KERNEL(pts, reduce=max) returns(Real maxVal=-std::numeric_limits<Real>::max())
Real CompPdata_MaxMat2(const ParticleDataImpl<Matrix2x2f>& val)
{
	const Real s = val[idx].normEuclidean();
	if(s > maxVal) maxVal = s;
}

KERNEL(pts, reduce=+) returns(Real result=0.)
Real KnPtsSumSquareMat2(const ParticleDataImpl<Matrix2x2f>& val)
{
	result += square(val[idx].normEuclidean());

}
KERNEL(pts, reduce=+) returns(Real result=0.)
Real KnPtsSumMagnitudeMat2(const ParticleDataImpl<Matrix2x2f>& val)
{
	result += val[idx].normEuclidean();
}

template<>
Real ParticleDataImpl<Matrix2x2f>::getMin() const
{
	return sqrt(CompPdata_MinMat2(*this));
}

template<>
Real ParticleDataImpl<Matrix2x2f>::getMaxAbs() const
{
	return sqrt(CompPdata_MaxMat2(*this));  // no minimum necessary here
}

template<>
Real ParticleDataImpl<Matrix2x2f>::getMax() const
{
	return sqrt(CompPdata_MaxMat2(*this));
}
template<>
Real ParticleDataImpl<Matrix2x2f>::sumSquare() const
{
	return KnPtsSumSquareMat2(*this);
}
template<>
Real ParticleDataImpl<Matrix2x2f>::sumMagnitude() const
{
	return KnPtsSumMagnitudeMat2(*this);
}

// explicit instantiation
template class ParticleDataImpl<int>;
template class ParticleDataImpl<Real>;
template class ParticleDataImpl<Vec3>;
template class ParticleDataImpl<Matrix3x3f>;
template class ParticleDataImpl<Matrix2x2f>;

} // namespace
