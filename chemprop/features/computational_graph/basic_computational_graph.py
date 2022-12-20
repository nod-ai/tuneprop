# Copyright 2022 Nod Labs
from numbers import Integral
from typing import List
import numpy as np
from chemprop.features.computational_graph.computational_graph import (
    ComputationalGraph,
    DeviceMeshIndex,
    Instruction,
    InstructionId,
    OpCode,
    Operand,
    Shape,
    Sharding,
    TensorMeshSharding,
    TensorShape,
)
class BasicTensorShape(TensorShape):
    def __init__(self, dims: List[Integral], dtype: np.dtype):
        self.__dims = dims
        self.__dtype = dtype
    @property
    def dims(self) -> List[Integral]:
        return self.__dims
    @property
    def dtype(self) -> np.dtype:
        return self.__dtype
class BasicTensorMeshSharding(TensorMeshSharding):
    def __init__(self, tensor_split_dims_mapping: List[DeviceMeshIndex]):
        self.__dims_mapping = tensor_split_dims_mapping
    @property
    def tensor_split_dims_mapping(self) -> List[DeviceMeshIndex]:
        return self.__dims_mapping
class BasicOperand(Operand):
    def __init__(self, instruction: Instruction, sharding: Sharding):
        self.__instruction = instruction
        self.__sharding = sharding
    @property
    def instruction(self) -> Instruction:
        return self.__instruction
    @property
    def sharding(self) -> Sharding:
        return self.__sharding
class BasicInstruction(Instruction):
    def __init__(
        self,
        instruction_id: InstructionId,
        op_code: OpCode,
        shape: Shape,
        operands: List[Operand],
        sharding: Sharding,
    ):
        self.__id = instruction_id
        self.__op_code = op_code
        self.__shape = shape
        self.__operands = operands
        self.__sharding = sharding
    @property
    def id(self) -> InstructionId:
        return self.__id
    @property
    def op_code(self) -> OpCode:
        return self.__op_code
    @property
    def shape(self) -> Shape:
        return self.__shape
    @property
    def operands(self) -> List[Operand]:
        return self.__operands
    @property
    def sharding(self) -> Sharding:
        return self.__sharding
class BasicComputationalGraph(ComputationalGraph):
    def __init__(self, instructions: List[Instruction]):
        self.__instructions = instructions
    @property
    def instructions(self) -> List[Instruction]:
        return self.__instructions