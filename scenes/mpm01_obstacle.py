#
# MLS-MPM scene with obstacle
#
from manta import *
import time

# solver params
dim = 2
particleNumber = 2
res = 200
withMesh = 1
withObs = 1
withPseudo3D = 0 # simulate with 3D grids but 2D MPM functions
withParallelParts = 1 # enable parallel "parts-to-grid" mapping in 3D (noticeably faster when using lots of particles)

gs = vec3(res,res,res/4)
if (dim==2):
	gs.z=1
	particleNumber = 3 # use more particles in 2d

# Gui solver - only used to update GUI
sGui = Solver(name='gui', gridSize = gs, dim=dim)
sGui.timestep = 1e-2

# MPM solver - uses small timestep and thus not used for GUI updates
s = Solver(name='main', gridSize = gs, dim=dim)
s.timestep = 1e-4
s.updateGui = False

timings = Timings()

gravity   = vec3(0,-2e3,0)
hardening = 10.0 # Hardening factor
E         = 1e4 # Young's modulus
nu        = 0.2 # Poisson ratio
pmass     = 1.0 # Particle mass
pvol      = 1.0 # Particle volume
plastic   = bool(1) # plastic=0 behave more like soft body

# prepare grids and particles
flags    = s.create(FlagGrid)
tmpVec3  = s.create(VecGrid)
vel      = s.create(MACGrid)
mesh     = s.create(Mesh)
pp       = s.create(BasicParticleSystem)
pp.maxParticles = int(5e3) if (dim == 2) else int(1e5) # 2D still runs in realtime on notebook
# add velocity data to particles
pVel     = pp.create(PdataVec3)
# mpm part
dummy    = s.create(RealGrid) # just for nicer grid view in GUI
mass     = s.create(RealGrid)

# placeholders for acceleration grids
kernelGrid = None
velK1  = None
velK2  = None
massK1 = None
massK2 = None

if dim == 3 and withParallelParts:
	gsKernel       = vec3(1,1,3)
	sAcc           = Solver(name='kernelHelper', gridSize=gsKernel, dim=dim)
	sAcc.updateGui = False
	kernelGrid     = sAcc.create(RealGrid)
	velK1  = s.create(MACGrid)  # k=1
	velK2  = s.create(MACGrid)  # k=2
	massK1 = s.create(RealGrid) # k=1
	massK2 = s.create(RealGrid) # k=2

# acceleration data for particle nbs
pindex   = s.create(ParticleIndexSystem)
gpi      = s.create(IntGrid)
counter  = s.create(IntGrid)
phiParts = s.create(LevelsetGrid)

# obstacle
phiInit  = s.create(LevelsetGrid)
phiInit.setConst(999.)
phiObs   = s.create(LevelsetGrid)

# determinant of deformation grad
Jp = pp.create(PdataReal)

# matrix data structures per particle
F = pp.create(PdataMat2) if (dim == 2 or withPseudo3D) else pp.create(PdataMat3) # deformation grad
C = pp.create(PdataMat2) if (dim == 2 or withPseudo3D) else pp.create(PdataMat3) # affine momentum
R = pp.create(PdataMat2) if (dim == 2 or withPseudo3D) else pp.create(PdataMat3) # rotation mat from F=RS
S = pp.create(PdataMat2) if (dim == 2 or withPseudo3D) else pp.create(PdataMat3) # scale mat from F=RS

flags.initDomain(boundaryWidth=0, phiWalls=phiObs)

# Setup fluid and obstacle geometries
fluidbox  = Box(parent=s, p0=gs*vec3(0.45,0.75,0.4), p1=gs*vec3(0.65,0.95,0.6))
fluidbox2 = Box(parent=s, p0=gs*vec3(0.4,0.5,0.4), p1=gs*vec3(0.6,0.7,0.6))
phiInit = fluidbox.computeLevelset()
phiInit.join(fluidbox2.computeLevelset())
if withObs:
	obs = Box(parent=s, p0=gs*vec3(0.49,0,0), p1=gs*vec3(0.51,0.4,1))
	phiObs.join(obs.computeLevelset())

flags.updateFromLevelset(phiInit)
phiInit.subtract(phiObs)

sampleFlagsWithParticles(flags=flags, parts=pp, discretization=particleNumber, randomness=0.1)
setObstacleFlags(flags=flags, phiObs=phiObs)

# Initialize particle data structures (after sampling particles!)
Jp.setConst(1.0) # initialize with 0.0 for more rigidity
if not plastic: Jp.setConst(1.0)
identity = mat2() if (dim == 2 or withPseudo3D) else mat3()
zeros = mat2(0) if (dim == 2 or withPseudo3D) else mat3(0)
F.setConst(identity)
C.setConst(zeros)
R.setConst(zeros)
S.setConst(zeros)

if (GUI):
	gui = Gui()
	gui.show()
	gui.pause()

# main loop
startTime = time.time()
runtime = int(1e5) if (dim == 2 or withPseudo3D) else int(5e2)

for t in xrange(runtime):
	#mantaMsg('\nFrame %i, simulation time %f' % (s.frame, s.timeTotal))

	if (dim == 2 or withPseudo3D):
		polarDecomposition2D(A=F, U=R, P=S)
		mpmMapPartsToMACGrid2D(vel=vel, mass=mass, pp=pp, pvel=pVel, detDeformationGrad=Jp, deformationGrad=F,
			affineMomentum=C, rotation=R, hardening=hardening, E=E, nu=nu, pmass=pmass, pvol=pvol)
	else:
		polarDecomposition3D(A=F, U=R, P=S)
		mpmMapPartsToMACGrid3D(vel=vel, mass=mass, pp=pp, pvel=pVel, detDeformationGrad=Jp, deformationGrad=F,
			affineMomentum=C, rotation=R, kernelGrid=kernelGrid, velK1=velK1, velK2=velK2, massK1=massK1, massK2=massK2,
			hardening=hardening, E=E, nu=nu, pmass=pmass, pvol=pvol)

	mpmUpdateGrid(flags=flags, pp=pp, gravity=gravity, vel=vel, mass=mass, velK1=velK1, velK2=velK2, massK1=massK1, massK2=massK2)
	
	if (dim == 2 or withPseudo3D):
		mpmMapMACGridToParts2D(pp=pp, vel=vel, mass=mass, pvel=pVel, detDeformationGrad=Jp, deformationGrad=F, affineMomentum=C, plastic=plastic)
	else:
		mpmMapMACGridToParts3D(pp=pp, vel=vel, mass=mass, pvel=pVel, detDeformationGrad=Jp, deformationGrad=F, affineMomentum=C, plastic=plastic)

	s.step()

	# Gui uses larger timestep
	if (t % int(sGui.timestep / s.timestep) == 0):
		if 0:
			gui.screenshot('test_%02d_imgs/test_00_%04d.png' % (int(test), t));
		if withMesh and dim==3 and t>2000:
			gridParticleIndex(parts=pp , flags=flags, indexSys=pindex, index=gpi, counter=counter)
			#unionParticleLevelset(pp, pindex, flags, gpi, phiParts)
			improvedParticleLevelset(pp, pindex, flags, gpi, phiParts , 1 , 1, 1)
			phiParts.createMesh(mesh)
		sGui.step()

	timings.display()

print('Time taken : %s seconds ---' % (time.time() - startTime))
