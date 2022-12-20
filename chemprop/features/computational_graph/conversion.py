# Copyright 2022 Nod Labs
from numbers import Integral
from typing import Dict, List
import numpy as np
from chemprop.features.computational_graph.basic_computational_graph import (
    BasicComputationalGraph,
    BasicInstruction,
    BasicOperand,
    BasicTensorMeshSharding,
    BasicTensorShape,
)
from chemprop.features.computational_graph.computational_graph import (
    ComputationalGraph,
    Instruction,
    OpCode,
    Shape,
    Sharding,
)
from tensorflow.compiler.xla.python_api.types import MAP_XLA_TYPE_TO_RECORD
from tensorflow.compiler.xla.service.hlo_pb2 import HloInstructionProto, HloModuleProto
from tensorflow.compiler.xla.xla_data_pb2 import OpSharding, PrimitiveType, ShapeProto
HLO_PRIMITIVE_TYPE_TO_DTYPE_MAP: Dict["PrimitiveType", np.dtype] = {
    v.primitive_type: v.numpy_dtype for k, v in MAP_XLA_TYPE_TO_RECORD.items()
}
# Source:
# https://github.com/nod-ai/tensorflow-alpa/blob/master/tensorflow/compiler/xla/service/hlo_opcode.h # noqa: E501
HLO_OPCODE_MAP: Dict[str, OpCode] = {
    "abs": OpCode.Abs,
    "add": OpCode.Add,
    "add-dependency": OpCode.AddDependency,
    "after-all": OpCode.AfterAll,
    "all-gather": OpCode.AllGather,
    "all-gather-start": OpCode.AllGatherStart,
    "all-gather-done": OpCode.AllGatherDone,
    "all-reduce": OpCode.AllReduce,
    "all-reduce-start": OpCode.AllReduceStart,
    "all-reduce-done": OpCode.AllReduceDone,
    "all-to-all": OpCode.AllToAll,
    "async-start": OpCode.AsyncStart,
    "async-update": OpCode.AsyncUpdate,
    "async-done": OpCode.AsyncDone,
    "atan2": OpCode.Atan2,
    "batch-norm-grad": OpCode.BatchNormGrad,
    "batch-norm-inference": OpCode.BatchNormInference,
    "batch-norm-training": OpCode.BatchNormTraining,
    "bitcast": OpCode.Bitcast,
    "bitcast-convert": OpCode.BitcastConvert,
    "broadcast": OpCode.Broadcast,
    "call": OpCode.Call,
    "ceil": OpCode.Ceil,
    "cholesky": OpCode.Cholesky,
    "clamp": OpCode.Clamp,
    "collective-permute": OpCode.CollectivePermute,
    "collective-permute-start": OpCode.CollectivePermuteStart,
    "collective-permute-done": OpCode.CollectivePermuteDone,
    "count-leading-zeros": OpCode.Clz,
    "compare": OpCode.Compare,
    "complex": OpCode.Complex,
    "concatenate": OpCode.Concatenate,
    "conditional": OpCode.Conditional,
    "constant": OpCode.Constant,
    "convert": OpCode.Convert,
    "convolution": OpCode.Convolution,
    "copy": OpCode.Copy,
    "copy-done": OpCode.CopyDone,
    "copy-start": OpCode.CopyStart,
    "cosine": OpCode.Cos,
    "custom-call": OpCode.CustomCall,
    "divide": OpCode.Divide,
    "domain": OpCode.Domain,
    "dot": OpCode.Dot,
    "dynamic-slice": OpCode.DynamicSlice,
    "dynamic-update-slice": OpCode.DynamicUpdateSlice,
    "exponential": OpCode.Exp,
    "exponential-minus-one": OpCode.Expm1,
    "fft": OpCode.Fft,
    "floor": OpCode.Floor,
    "fusion": OpCode.Fusion,
    "gather": OpCode.Gather,
    "get-dimension-size": OpCode.GetDimensionSize,
    "set-dimension-size": OpCode.SetDimensionSize,
    "get-tuple-element": OpCode.GetTupleElement,
    "imag": OpCode.Imag,
    "infeed": OpCode.Infeed,
    "iota": OpCode.Iota,
    "is-finite": OpCode.IsFinite,
    "log": OpCode.Log,
    "log-plus-one": OpCode.Log1p,
    "logistic": OpCode.Logistic,
    "and": OpCode.And,
    "not": OpCode.Not,
    "opt-barrier": OpCode.OptimizationBarrier,
    "or": OpCode.Or,
    "xor": OpCode.Xor,
    "map": OpCode.Map,
    "maximum": OpCode.Maximum,
    "minimum": OpCode.Minimum,
    "multiply": OpCode.Multiply,
    "negate": OpCode.Negate,
    "outfeed": OpCode.Outfeed,
    "pad": OpCode.Pad,
    "parameter": OpCode.Parameter,
    "partition-id": OpCode.PartitionId,
    "popcnt": OpCode.PopulationCount,
    "power": OpCode.Power,
    "real": OpCode.Real,
    "recv": OpCode.Recv,
    "recv-done": OpCode.RecvDone,
    "reduce": OpCode.Reduce,
    "reduce-precision": OpCode.ReducePrecision,
    "reduce-scatter": OpCode.ReduceScatter,
    "reduce-window": OpCode.ReduceWindow,
    "remainder": OpCode.Remainder,
    "replica-id": OpCode.ReplicaId,
    "reshape": OpCode.Reshape,
    "dynamic-reshape": OpCode.DynamicReshape,
    "reverse": OpCode.Reverse,
    "rng": OpCode.Rng,
    "rng-get-and-update-state": OpCode.RngGetAndUpdateState,
    "rng-bit-generator": OpCode.RngBitGenerator,
    "round-nearest-afz": OpCode.RoundNearestAfz,
    "rsqrt": OpCode.Rsqrt,
    "scatter": OpCode.Scatter,
    "select": OpCode.Select,
    "select-and-scatter": OpCode.SelectAndScatter,
    "send": OpCode.Send,
    "send-done": OpCode.SendDone,
    "shift-left": OpCode.ShiftLeft,
    "shift-right-arithmetic": OpCode.ShiftRightArithmetic,
    "shift-right-logical": OpCode.ShiftRightLogical,
    "sign": OpCode.Sign,
    "sine": OpCode.Sin,
    "slice": OpCode.Slice,
    "sort": OpCode.Sort,
    "sqrt": OpCode.Sqrt,
    "cbrt": OpCode.Cbrt,
    "subtract": OpCode.Subtract,
    "tanh": OpCode.Tanh,
    "transpose": OpCode.Transpose,
    "triangular-solve": OpCode.TriangularSolve,
    "tuple": OpCode.Tuple,
    "tuple-select": OpCode.TupleSelect,
    "while": OpCode.While,
}
DeviceId = Integral
MeshIndex = np.ndarray
DeviceIdMeshIndexMap = Dict[DeviceId, MeshIndex]
def recover_2d_device_mesh_dimension(
    device_id_mesh_index_map: DeviceIdMeshIndexMap, device_ids: List[Integral]
) -> Integral:
    assert len(device_ids) > 1
    mesh_indices = np.array(
        [device_id_mesh_index_map[device_id] for device_id in device_ids]
    )
    same_dims = np.array(
        [
            np.where(mesh_indices[0] == mesh_indices[i])
            for i in range(1, len(mesh_indices))
        ],
        dtype=int,
    )
    assert np.all(same_dims == same_dims[0])
    return same_dims[0]
def recover_2d_device_mesh_restricted_sharding(
    hlo_sharding: OpSharding, device_id_mesh_index_map: DeviceIdMeshIndexMap
) -> List[Integral]:
    tile_assignment = np.array(hlo_sharding.tile_assignment_devices).reshape(
        hlo_sharding.tile_assignment_dimensions
    )
    tensor_split_dims_mapping = np.empty(
        len(tile_assignment.shape)
        - (1 if hlo_sharding.replicate_on_last_tile_dim else 0),
        dtype=int,
    )
    for i in range(len(tensor_split_dims_mapping)):
        if tile_assignment.shape[i] == 1:
            tensor_split_dims_mapping[i] = -1
        else:
            device_ids = np.take(tile_assignment, indices=[0], axis=i).flatten()
            tensor_split_dims_mapping[i] = recover_2d_device_mesh_dimension(
                device_id_mesh_index_map, device_ids
            )
    return tensor_split_dims_mapping
class FromHloConversionContext:
    def __init__(self, device_mesh: np.ndarray):
        self.id_instruction_map: Dict[Integral, Instruction] = {}
        self.device_mesh = device_mesh
        self.device_id_mesh_index_map = {
            device_id: mesh_index
            for mesh_index, device_id in np.ndenumerate(device_mesh)
        }
def convert_hlo_sharding(
    hlo_sharding: OpSharding, device_id_mesh_index_map: DeviceIdMeshIndexMap
) -> Sharding:
    if hlo_sharding.type == OpSharding.Type.TUPLE:
        return [convert_hlo_sharding(s) for s in hlo_sharding.tuple_shardings]
    elif hlo_sharding.type == OpSharding.Type.OTHER:
        return BasicTensorMeshSharding(
            tensor_split_dims_mapping=recover_2d_device_mesh_restricted_sharding(
                hlo_sharding, device_id_mesh_index_map
            )
        )
    else:
        raise NotImplementedError()
def convert_hlo_shape(hlo_shape: ShapeProto) -> Shape:
    if len(hlo_shape.tuple_shapes) > 0:
        return (convert_hlo_shape(s) for s in hlo_shape.tuple_shapes)
    else:
        return BasicTensorShape(
            dims=hlo_shape.dimensions,
            dtype=HLO_PRIMITIVE_TYPE_TO_DTYPE_MAP[hlo_shape.element_type],
        )
def convert_hlo_instruction(
    hlo_instruction: HloInstructionProto, ctx: FromHloConversionContext
) -> Instruction:
    res = BasicInstruction(
        instruction_id=hlo_instruction.id,
        op_code=HLO_OPCODE_MAP[hlo_instruction.opcode],
        shape=convert_hlo_shape(hlo_instruction.shape),
        operands=[
            BasicOperand(instruction=ctx.id_instruction_map[operand_id], sharding=None)
            for operand_id in hlo_instruction.operand_ids
        ],
        sharding=convert_hlo_sharding(
            hlo_instruction.sharding, ctx.device_id_mesh_index_map
        )
        if hlo_instruction.HasField("sharding")
        else None,
    )
    ctx.id_instruction_map[hlo_instruction.id] = res
    return res
def convert_hlo_module(
    hlo_module: HloModuleProto, device_mesh: np.ndarray
) -> ComputationalGraph:
    if len(hlo_module.computations) != 1:
        raise NotImplementedError()
    ctx = FromHloConversionContext(device_mesh)
    instructions = [
        convert_hlo_instruction(inst, ctx)
        for inst in hlo_module.computations[0].instructions
    ]
    return BasicComputationalGraph(instructions)