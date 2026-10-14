# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import extensibility

from max.gpu.host import DeviceContext

from extensibility import InputTensor, OutputTensor, foreach

from std.utils.coord import Coord


@extensibility.register("my_kernel")
struct MyKernel:
    @staticmethod
    def execute[
        target: StaticString
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def add_one[width: Int](idx: Coord) {var} -> SIMD[x.dtype, width]:
            return x.load[width](idx) + 1

        foreach[target=target](add_one, output, ctx)
