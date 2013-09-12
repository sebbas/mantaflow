/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 * Base class for particle systems
 *
 ******************************************************************************/

#ifndef _PARTICLE_H
#define _PARTICLE_H

#include <vector>
#include "grid.h"
#include "vectorbase.h"
#include "integrator.h"
#include "randomstream.h"
namespace Manta {

// fwd decl
template<class T> class Grid;
class ParticleDataBase;

//! Baseclass for particle systems. Does not implement any data
PYTHON class ParticleBase : public PbClass {
public:
    enum SystemType { BASE=0, PARTICLE, VORTEX, FILAMENT, FLIP, TURBULENCE };
    enum ParticleType {
        PNONE         = 0,
        PNEW          = (1<<1),  // particles newly created in this step
        PDELETE       = (1<<10), // mark as deleted, will be deleted in next compress() step
        PINVALID      = (1<<30), // unused
    };
    
    PYTHON ParticleBase(FluidSolver* parent) : PbClass(parent) {}
    virtual ~ParticleBase();

	//! copy all the particle data thats registered with the other particle system to this one
    virtual void cloneParticleData(ParticleBase* nm);

    virtual SystemType getType() const { return BASE; }
    virtual std::string infoString() const { return "ParticleSystem " + mName + " <no info>"; };
    virtual ParticleBase* clone() { assertMsg( false , "Dont use, override..."); return NULL; } 

	// slow virtual function to query size, do not use in kernels! use size() instead
	virtual int getSizeSlow() const { assertMsg( false , "Dont use, override..."); return 0; } 

	// particle data functions

    //! create a particle data object
    PYTHON PbClass* create(PbType type, const std::string& name = "");
	//! add a particle data field, set its parent particle-system pointer
	void addParticleData(ParticleDataBase* pdata);
	//! remove a particle data entry
	void deregister(ParticleDataBase* pdata);
	//! add one zero entry to all data fields
	void addAllPdata();
	// note - deletion is handled in compress function

	//! how many are there?
	int getNumPdata() const { return mPartData.size(); }
	//! access one of the fields
	ParticleDataBase* getPdata(int i) { return mPartData[i]; }

	//! debug info about pdata
	std::string debugInfoPdata();

protected:  
	//! store particle data 
	std::vector<ParticleDataBase*> mPartData;
};


//! Main class for particle systems
/*! Basetype S must at least contain flag, pos fields */
PYTHON template<class S> class ParticleSystem : public ParticleBase {
public:    
    PYTHON ParticleSystem(FluidSolver* parent) : ParticleBase(parent), mDeletes(0), mDeleteChunk(0) {}
    virtual ~ParticleSystem() {};
    
    virtual SystemType getType() const { return S::getType(); };
    
    // accessors
    inline S& operator[](int i) { return mData[i]; }
    inline const S& operator[](int i) const { return mData[i]; }
    PYTHON inline int size() const { return mData.size(); }
	// slow virtual function of base class
	virtual int getSizeSlow() const { return size(); }
    std::vector<S>& getData() { return mData; }
    
    // adding and deleting 
    inline void kill(int i);
    inline bool isActive(int i);
    int add(const S& data);
    void clear();
    
    //! safe accessor for python
    PYTHON void setPos(int idx, const Vec3& pos);
    //! safe accessor for python
    PYTHON Vec3 getPos(int idx);
            
    //! Advect particle in grid velocity field
    PYTHON void advectInGrid(FlagGrid& flaggrid, MACGrid& vel, int integrationMode);
    
    //! Project particles outside obstacles
    PYTHON void projectOutside(Grid<Vec3>& gradient);
    
    virtual ParticleBase* clone();
    virtual std::string infoString() const;
    
protected:  
    virtual void compress();
    
    int mDeletes, mDeleteChunk;    
    std::vector<S> mData;    
};

//! Simplest data class for particle systems
struct BasicParticleData {
public:
    BasicParticleData() : pos(0.), flag(0) {}
    BasicParticleData(const Vec3& p) : pos(p), flag(0) {}
    static ParticleBase::SystemType getType() { return ParticleBase::PARTICLE; }

	//! data
    Vec3 pos;
    int flag;
};

PYTHON class BasicParticleSystem : public ParticleSystem<BasicParticleData> {
public:
    PYTHON BasicParticleSystem(FluidSolver* parent) : ParticleSystem<BasicParticleData>(parent) {}
    
    virtual std::string infoString() const;

    PYTHON void addParticle(Vec3 pos) { add(BasicParticleData(pos)); }
};




//! Particle set with connectivity
PYTHON template<class DATA, class CON> 
class ConnectedParticleSystem : public ParticleSystem<DATA> {
public:
    PYTHON ConnectedParticleSystem(FluidSolver* parent) : ParticleSystem<DATA>(parent) {}
    
    // accessors
    inline bool isSegActive(int i) { return (mSegments[i].flag & ParticleBase::PDELETE) == 0; }    
    inline int segSize() const { return mSegments.size(); }    
    inline CON& seg(int i) { return mSegments[i]; }
    inline const CON& seg(int i) const { return mSegments[i]; }
        
    virtual ParticleBase* clone();
    
protected:
    std::vector<CON> mSegments;
    virtual void compress();    
};


//! abstract interface for particle data
PYTHON class ParticleDataBase : public PbClass {
public:
    PYTHON ParticleDataBase(FluidSolver* parent);
	virtual ~ParticleDataBase(); 

    enum PdataType { UNKNOWN=0, DATA_INT, DATA_REAL, DATA_VEC3 };

	// interface functions, using assert instead of pure virtual for python compatibility
	virtual int  size() const { assertMsg( false , "Dont use, override..."); return 0; } 
	virtual void add()        { assertMsg( false , "Dont use, override..."); return;   }
	virtual void kill(int i)  { assertMsg( false , "Dont use, override..."); return;   }
    virtual ParticleDataBase* clone() { assertMsg( false , "Dont use, override..."); return NULL; }
	virtual PdataType getType() const { assertMsg( false , "Dont use, override..."); return UNKNOWN; } 

	//! set base pointer
	void setParticleSys(ParticleBase* set) { mpParticleSys = set; }

	//! debugging
	inline void checkPartIndex(int idx) const;

protected:
	ParticleBase* mpParticleSys;
};

//! abstract interface for particle data
PYTHON template<class T>
class ParticleDataImpl : public ParticleDataBase {
public:
	PYTHON ParticleDataImpl(FluidSolver* parent);
	ParticleDataImpl(FluidSolver* parent, ParticleDataImpl<T>* other);
	virtual ~ParticleDataImpl();

    //! access data
    inline T get(int idx) const { DEBUG_ONLY(checkPartIndex(idx)); return mData[idx]; }
    //! access data
    inline T& operator[](int idx) { DEBUG_ONLY(checkPartIndex(idx)); return mData[idx]; }
    //! access data
    inline const T operator[](int idx) const { DEBUG_ONLY(checkPartIndex(idx)); return mData[idx]; }

	// particle data base interface
	virtual int  size() const;
	virtual void add();
	virtual void kill(int i);
    virtual ParticleDataBase* clone();
	virtual PdataType getType() const;

protected:
	std::vector<T> mData; // todo
};

PYTHON alias ParticleDataImpl<int>  PdataInt;
PYTHON alias ParticleDataImpl<Real> PdataReal;
PYTHON alias ParticleDataImpl<Vec3> PdataVec3;


//******************************************************************************
// Implementation
//******************************************************************************

const int DELETE_PART = 20; // chunk size for compression
   
template<class S>
void ParticleSystem<S>::clear() {
    mDeleteChunk = mDeletes = 0;
    mData.clear();
}

template<class S>
int ParticleSystem<S>::add(const S& data) {
    mData.push_back(data); 
    mDeleteChunk = mData.size() / DELETE_PART;
	this->addAllPdata();
    return mData.size()-1;
}

template<class S>
inline void ParticleSystem<S>::kill(int idx)     { 
    assertMsg(idx>=0 && idx<size(), "Index out of bounds");
	mData[idx].flag |= PDELETE; 
	if (++mDeletes > mDeleteChunk) compress(); 
}

template<class S>
inline bool ParticleSystem<S>::isActive(int idx) { 
    assertMsg(idx>=0 && idx<size(), "Index out of bounds");
	return (mData[idx].flag & PDELETE) == 0; 
}  

template<class S> Vec3 ParticleSystem<S>::getPos(int idx) {
    assertMsg(idx>=0 && idx<size(), "Index out of bounds");
    return mData[idx].pos;
}

template<class S> void ParticleSystem<S>::setPos(int idx, const Vec3& pos) {
    assertMsg(idx>=0 && idx<size(), "Index out of bounds");
    mData[idx].pos = pos;
}
KERNEL(pts) template<class S> returns(std::vector<Vec3> u(size))
std::vector<Vec3> GridAdvectKernel (std::vector<S>& p, const MACGrid& vel, const FlagGrid& flaggrid, Real dt)
{
    if (p[i].flag & ParticleBase::PDELETE) 
        u[i] =_0;
    else if (!flaggrid.isInBounds(p[i].pos,1) || flaggrid.isObstacle(p[i].pos)) {
        p[i].flag |= ParticleBase::PDELETE;
        u[i] = _0;
    }        
    else 
        u[i] = vel.getInterpolated(p[i].pos) * dt;
};

// advection plugin
template<class S>
void ParticleSystem<S>::advectInGrid(FlagGrid& flaggrid, MACGrid& vel, int integrationMode) {
    GridAdvectKernel<S> kernel(mData, vel, flaggrid, getParent()->getDt());
    integratePointSet(kernel, integrationMode);
}

KERNEL(pts, single) // no thread-safe random gen yet
template<class S>
void KnProjectParticles(ParticleSystem<S>& part, Grid<Vec3>& gradient) {
    static RandomStream rand (3123984);
    const double jlen = 0.1;
    
    if (part.isActive(i)) {
        // project along levelset gradient
        Vec3 p = part[i].pos;
        if (gradient.isInBounds(p)) {
            Vec3 n = gradient.getInterpolated(p);
            Real dist = normalize(n);
            Vec3 dx = n * (-dist + jlen * (1 + rand.getReal()));
            p += dx;            
        }
        // clamp to outer boundaries (+jitter)
        const double jlen = 0.1;
        Vec3 jitter = jlen * rand.getVec3();
        part[i].pos = clamp(p, Vec3(1,1,1)+jitter, toVec3(gradient.getSize()-1)-jitter);
    }
}

template<class S>
void ParticleSystem<S>::projectOutside(Grid<Vec3>& gradient) {
    KnProjectParticles<S>(*this, gradient);
}

template<class S>
void ParticleSystem<S>::compress() {
    int nextRead = mData.size();
    for (int i=0; i<(int)mData.size(); i++) {
        while ((mData[i].flag & PDELETE) != 0) {
            nextRead--;
            mData[i] = mData[nextRead];
            mData[nextRead].flag = 0;           
			// NT_DEBUG , handle data
        }
    }
    mData.resize(nextRead);
    mDeletes = 0;
    mDeleteChunk = mData.size() / DELETE_PART;
}

template<class DATA, class CON>
void ConnectedParticleSystem<DATA,CON>::compress() {
    const int sz = ParticleSystem<DATA>::size();
    int *renumber_back = new int[sz];
    int *renumber = new int[sz];
    for (int i=0; i<sz; i++)
        renumber[i] = renumber_back[i] = -1;
        
    // reorder elements
    std::vector<DATA>& data = ParticleSystem<DATA>::mData;
    int nextRead = sz;
    for (int i=0; i<nextRead; i++) {
        if ((data[i].flag & ParticleBase::PDELETE) != 0) {
            nextRead--;
            data[i] = data[nextRead];
            data[nextRead].flag = 0;           
            renumber_back[i] = nextRead;
        } else 
            renumber_back[i] = i;
    }
    
    // acceleration structure
    for (int i=0; i<nextRead; i++)
        renumber[renumber_back[i]] = i;
    
    // rename indices in filaments
    for (int i=0; i<(int)mSegments.size(); i++)
        mSegments[i].renumber(renumber);
        
    ParticleSystem<DATA>::mData.resize(nextRead);
    ParticleSystem<DATA>::mDeletes = 0;
    ParticleSystem<DATA>::mDeleteChunk = ParticleSystem<DATA>::size() / DELETE_PART;
    
    delete[] renumber;
    delete[] renumber_back;
}

template<class S>
ParticleBase* ParticleSystem<S>::clone() {
    ParticleSystem<S>* nm = new ParticleSystem<S>(getParent());
    compress();
    
    nm->mData = mData;
    nm->setName(getName());
	this->cloneParticleData(nm);
    return nm;
}

template<class DATA,class CON>
ParticleBase* ConnectedParticleSystem<DATA,CON>::clone() {
    ConnectedParticleSystem<DATA,CON>* nm = new ConnectedParticleSystem<DATA,CON>(this->getParent());
    compress();
    
    nm->mData = this->mData;
    nm->mSegments = mSegments;
    nm->setName(this->getName());
	this->cloneParticleData(nm);
    return nm;
}

template<class S>
std::string ParticleSystem<S>::infoString() const { 
    std::stringstream s;
    s << "ParticleSystem '" << getName() << "' [" << size() << " parts]";
    return s.str();
}
    
    
inline void ParticleDataBase::checkPartIndex(int idx) const {
	int mySize = this->size();
    if (idx<0 || idx > mySize ) {
        errMsg( "ParticleData " << " size " << mySize << " : index " << idx << " out of bound " );
    }
    if ( mpParticleSys && mpParticleSys->getSizeSlow()!=mySize ) {
        errMsg( "ParticleData " << " size " << mySize << " does not match parent! (" << mpParticleSys->getSizeSlow() << ") " );
    }
}

} // namespace

#endif

