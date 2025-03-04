################################################################################
#
# MantaFlow fluid solver framework
# Copyright 2011 Tobias Pfaff, Nils Thuerey 
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0 
# http://www.apache.org/licenses/LICENSE-2.0
#
# Defines some constants for use in python subprograms
#
#################################################################################

# mantaflow conventions
Real = float

# some defines to make C code and scripts more alike...
false = False
true  = True
Vec3  = vec3
Vec4  = vec4
Vec3Grid = VecGrid
Matrix3x3 = mat3
Matrix2x2 = mat2

# grid flags
FlagFluid    = 1
FlagObstacle = 2
FlagEmpty    = 4
FlagInflow   = 8
FlagOutflow  = 16
FlagStick    = 64
FlagReserved = 256
# and same for FlagGrid::CellType enum names:
TypeFluid    = 1
TypeObstacle = 2
TypeEmpty    = 4
TypeInflow   = 8
TypeOutflow  = 16
TypeStick    = 64
TypeReserved = 256

# index grid types
IndexLinear = 0
IndexBlock  = 1

# integration mode
IntEuler = 0
IntRK2   = 1
IntRK4   = 2

# CG preconditioner
PcNone      = 0
PcMIC       = 1
PcMGDynamic = 2
PcMGStatic  = 3

# particles
PtypeSpray   = 2
PtypeBubble  = 4
PtypeFoam    = 8
PtypeTracer  = 16

# OpenVDB export flags
Compression_None = 0
Compression_Zip = 1
Compression_Blosc = 2

# OpenVDB precision flags
Precision_Full = 0
Precision_Half = 1
Precision_Mini = 2
