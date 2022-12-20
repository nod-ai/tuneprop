# Copyright 2022 Nod Labs
from abc import ABC, abstractmethod
from enum import Enum
from numbers import Integral
from typing import List, Union
import numpy as np
class TensorShape(ABC):
    @property
    @abstractmethod
    def dims(self) -> List[Integral]:
        pass
    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        pass
Shape = Union[TensorShape, List["Shape"]]
DeviceMeshIndex = List[Integral]
"""The index of a device in a device mesh.
In the case of 2D mesh an example index is [1, 2]."""
InstructionId = Integral
class TensorMeshSharding(ABC):
    @property
    @abstractmethod
    def tensor_split_dims_mapping(self) -> List[DeviceMeshIndex]:
        """Returns the a correspondence that maps each
        tensor dimension to a device mesh axis.
        See:
        tensorflow.compiler.xla.experimental.xla_sharding.xla_sharding.mesh_split_sharding
        """
        pass
Sharding = Union[None, TensorMeshSharding, List["Sharding"]]
class OpCode(Enum):
    Abs = 0
    Add = 1
    AddDependency = 2
    AfterAll = 3
    AllGather = 4
    AllGatherStart = 5
    AllGatherDone = 6
    AllReduce = 7
    AllReduceStart = 8
    AllReduceDone = 9
    AllToAll = 10
    AsyncStart = 11
    AsyncUpdate = 12
    AsyncDone = 13
    Atan2 = 14
    BatchNormGrad = 15
    BatchNormInference = 16
    BatchNormTraining = 17
    Bitcast = 18
    BitcastConvert = 19
    Broadcast = 20
    Call = 21
    Ceil = 22
    Cholesky = 23
    Clamp = 24
    CollectivePermute = 25
    CollectivePermuteStart = 26
    CollectivePermuteDone = 27
    Clz = 28
    Compare = 29
    Complex = 30
    Concatenate = 31
    Conditional = 32
    Constant = 33
    Convert = 34
    Convolution = 35
    Copy = 36
    CopyDone = 37
    CopyStart = 38
    Cos = 39
    CustomCall = 40
    Divide = 41
    Domain = 42
    Dot = 43
    DynamicSlice = 44
    DynamicUpdateSlice = 45
    Exp = 46
    Expm1 = 47
    Fft = 48
    Floor = 49
    Fusion = 50
    Gather = 51
    GetDimensionSize = 52
    SetDimensionSize = 53
    GetTupleElement = 54
    Imag = 55
    Infeed = 56
    Iota = 57
    IsFinite = 58
    Log = 59
    Log1p = 60
    Logistic = 61
    And = 62
    Not = 63
    OptimizationBarrier = 64
    Or = 65
    Xor = 66
    Map = 67
    Maximum = 68
    Minimum = 69
    Multiply = 70
    Negate = 71
    Outfeed = 72
    Pad = 73
    Parameter = 74
    PartitionId = 75
    PopulationCount = 76
    Power = 77
    Real = 78
    Recv = 79
    RecvDone = 80
    Reduce = 81
    ReducePrecision = 82
    ReduceScatter = 83
    ReduceWindow = 84
    Remainder = 85
    ReplicaId = 86
    Reshape = 87
    DynamicReshape = 88
    Reverse = 89
    Rng = 90
    RngGetAndUpdateState = 91
    RngBitGenerator = 92
    RoundNearestAfz = 93
    Rsqrt = 94
    Scatter = 95
    Select = 96
    SelectAndScatter = 97
    Send = 98
    SendDone = 99
    ShiftLeft = 100
    ShiftRightArithmetic = 101
    ShiftRightLogical = 102
    Sign = 103
    Sin = 104
    Slice = 105
    Sort = 106
    Sqrt = 107
    Cbrt = 108
    Subtract = 109
    Tanh = 110
    Transpose = 111
    TriangularSolve = 112
    Tuple = 113
    TupleSelect = 114
    While = 115
class Operand(ABC):
    @property
    @abstractmethod
    def instruction(self) -> "Instruction":
        pass
    @property
    @abstractmethod
    def sharding(self) -> Sharding:
        """Input sharding of the operand."""
        pass
class Instruction(ABC):
    @property
    @abstractmethod
    def id(self) -> InstructionId:
        pass
    @property
    @abstractmethod
    def op_code(self) -> OpCode:
        pass
    @property
    @abstractmethod
    def shape(self) -> Shape:
        pass
    @property
    @abstractmethod
    def operands(self) -> List[Operand]:
        pass
    @property
    @abstractmethod
    def sharding(self) -> Sharding:
        """Output sharding."""
        pass
class ComputationalGraph(ABC):
    @property
    @abstractmethod
    def instructions(self) -> List[Instruction]:
        pass