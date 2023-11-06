import benchmark
from tensor import Tensor, TensorShape
from random import rand
from algorithm import vectorize, parallelize

alias workers = 6
alias nelts = 4 * simdwidthof[DType.float32]()
alias TensorF32 = Tensor[DType.float32]
alias BufferPtrFloat32 = DTypePointer[DType.float32]


struct Config:
    var dim: Int
    var kv_dim: Int
    var hidden_dim: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var kv_mul: Int
    var vocab_size: Int
    var seq_len: Int
    var head_size: Int

    fn __init__(inout self):
        self.dim = 0
        self.hidden_dim = 0
        self.n_layers = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.vocab_size = 0
        self.seq_len = 0
        self.kv_dim = 0
        self.kv_mul = 0
        self.head_size = 0


struct TensorSlice:
    # Provides a view into a tensor representing a 1D slice on its first or first 2 dimensions.
    # Same function signatures as Tensor but without owning the data.
    var _data: BufferPtrFloat32
    var _shape: TensorShape

    fn __init__(inout self, t: TensorF32, layer: Int) raises:
        let elements_per_layer = t.num_elements() // t.dim(0)
        self._data = t.data().offset(layer * elements_per_layer)
        if t.rank() == 2:
            self._shape = TensorShape(t.dim(1))
        elif t.rank() == 3:
            self._shape = TensorShape(t.dim(1), t.dim(2))
        else:
            # Compiler complains if _shape not defined
            self._shape = TensorShape(1)
            raise Error("TensorSlice: rank greater than 3 not implemented.")

    fn __init__(inout self, t: TensorF32, layer: Int, row: Int) raises:
        let elements_per_layer = t.num_elements() // t.dim(0)
        let elements_per_row = elements_per_layer // t.dim(1)
        self._data = t.data().offset(
            layer * elements_per_layer + row * elements_per_row
        )
        if t.rank() == 3:
            self._shape = TensorShape(t.dim(2))
        elif t.rank() == 1:
            # Compiler complains if _shape not defined
            self._shape = TensorShape(1)
            raise Error(
                "Trying to slice a 1D Tensor by layer and row.  This requires a 3D"
                " Tensor."
            )
        else:
            # Compiler complains if _shape not defined
            self._shape = TensorShape(1)
            raise Error("TensorSlice: rank greater than 3 not implemented.")

    fn data(self) -> BufferPtrFloat32:
        return self._data

    fn shape(self) -> TensorShape:
        return self._shape

    fn num_elements(self) -> Int:
        return self._shape.num_elements()

    fn dim(self, idx: Int) -> Int:
        return self._shape[idx]

    fn rank(self) -> Int:
        return self._shape.rank()

    fn simd_load[nelts: Int](self, idx: Int) -> SIMD[DType.float32, nelts]:
        return self._data.simd_load[nelts](idx)

    fn simd_load[nelts: Int](self, *indices: Int) -> SIMD[DType.float32, nelts]:
        if len(VariadicList(indices)) > 2:
            print(
                "Warning: TensorSlice only supports 1D and 2D indexing.  Results are"
                " unlikely to be correct."
            )
        return self.simd_load[nelts](indices[0] * self._shape[1] + indices[1])

    fn simd_load[
        nelts: Int
    ](self, indices: StaticIntTuple[2]) -> SIMD[DType.float32, nelts]:
        return self._data.simd_load[nelts](indices[0] * self._shape[1] + indices[1])

    fn __getitem__(self, idx: Int) -> SIMD[DType.float32, 1]:
        return self._data.simd_load[1](idx)

    fn simd_store[nelts: Int](self, idx: Int, val: SIMD[DType.float32, nelts]):
        return self._data.simd_store[nelts](idx, val)

    fn __setitem__(self, idx: Int, val: SIMD[DType.float32, 1]):
        return self.simd_store[1](idx, val)


# Apply RoPE rotation to the q and k vectors for each head
# rotate odd and even dim
@always_inline
fn rope_vectorize_parallelize(
    inout q: TensorF32,
    inout k: TensorSlice,
    freq_cis_real_row: TensorSlice,
    freq_cis_imag_row: TensorSlice,
    config: Config,
    /,
) -> None:
    # stories model, llama2
    let head_size = config.head_size

    @parameter
    fn head_loop(i: Int):
        let outer_offset = i * head_size

        @parameter
        fn inner_loop[_nelts: Int](kk: Int):
            let j = kk * 2
            # Frequency dim is half the head size so index is half of other tensors
            # Draw one frequency for each pair to be rotated
            let fcr = freq_cis_real_row.simd_load[_nelts](kk)
            let fci = freq_cis_imag_row.simd_load[_nelts](kk)
            let offset = outer_offset + j
            # Make two strided draws offset by 1 for 'nelts' pairs
            let q0 = q.data().offset(offset).simd_strided_load[_nelts](2)
            let q1 = q.data().offset(offset + 1).simd_strided_load[_nelts](2)
            q.data().offset(offset).simd_strided_store[_nelts](q0 * fcr - q1 * fci, 2)
            q.data().offset(offset + 1).simd_strided_store[_nelts](
                q0 * fci + q1 * fcr, 2
            )
            if i < config.n_kv_heads:
                let k0 = k.data().offset(offset).simd_strided_load[_nelts](2)
                let k1 = k.data().offset(offset + 1).simd_strided_load[_nelts](2)
                k.data().offset(offset).simd_strided_store[_nelts](
                    k0 * fcr - k1 * fci, 2
                )
                k.data().offset(offset + 1).simd_strided_store[_nelts](
                    k0 * fci + k1 * fcr, 2
                )

        vectorize[nelts, inner_loop](head_size // 2)

    parallelize[head_loop](config.n_heads, workers)


# Apply RoPE rotation to the q and k vectors for each head
# rotate odd and even dim
@always_inline
fn rope_vectorize(
    inout q: TensorF32,
    inout k: TensorSlice,
    freq_cis_real_row: TensorSlice,
    freq_cis_imag_row: TensorSlice,
    config: Config,
    /,
) -> None:
    # stories model, llama2
    let head_size = config.head_size

    for i in range(config.n_heads):
        let outer_offset = i * head_size

        @parameter
        fn inner_loop[_nelts: Int](kk: Int):
            let j = kk * 2
            # Frequency dim is half the head size so index is half of other tensors
            # Draw one frequency for each pair to be rotated
            let fcr = freq_cis_real_row.simd_load[_nelts](kk)
            let fci = freq_cis_imag_row.simd_load[_nelts](kk)
            let offset = outer_offset + j
            # Make two strided draws offset by 1 for 'nelts' pairs
            let q0 = q.data().offset(offset).simd_strided_load[_nelts](2)
            let q1 = q.data().offset(offset + 1).simd_strided_load[_nelts](2)
            q.data().offset(offset).simd_strided_store[_nelts](q0 * fcr - q1 * fci, 2)
            q.data().offset(offset + 1).simd_strided_store[_nelts](
                q0 * fci + q1 * fcr, 2
            )
            if i < config.n_kv_heads:
                let k0 = k.data().offset(offset).simd_strided_load[_nelts](2)
                let k1 = k.data().offset(offset + 1).simd_strided_load[_nelts](2)
                k.data().offset(offset).simd_strided_store[_nelts](
                    k0 * fcr - k1 * fci, 2
                )
                k.data().offset(offset + 1).simd_strided_store[_nelts](
                    k0 * fci + k1 * fcr, 2
                )

        vectorize[nelts, inner_loop](head_size // 2)


# Apply RoPE rotation to the q and k vectors for each head
# rotate odd and even dim
@always_inline
fn rope_parallelize(
    inout q: TensorF32,
    inout k: TensorSlice,
    freq_cis_real_row: TensorSlice,
    freq_cis_imag_row: TensorSlice,
    config: Config,
    /,
) -> None:
    # stories model, llama2
    let head_size = config.head_size

    @parameter
    fn head_loop(i: Int):
        let outer_offset = i * head_size

        for j in range(0, head_size, 2):
            # Frequency dim is half the head size so index is half of other tensors
            # Draw one frequency for each pair to be rotated
            let fcr = freq_cis_real_row[j // 2]
            let fci = freq_cis_imag_row[j // 2]
            let offset = outer_offset + j
            # Make two strided draws offset by 1 for 'nelts' pairs
            let q0 = q[offset]
            let q1 = q[offset + 1]
            q[offset] = q0 * fcr - q1 * fci
            q[offset + 1] = q0 * fci + q1 * fcr

            if i < config.n_kv_heads:
                let k0 = k[offset]
                let k1 = k[offset + 1]
                k[offset] = k0 * fcr - k1 * fci
                k[offset + 1] = k0 * fci + k1 * fcr

    parallelize[head_loop](config.n_heads, workers)


# Apply RoPE rotation to the q and k vectors for each head
# rotate odd and even dim
@always_inline
fn rope(
    inout q: TensorF32,
    inout k: TensorSlice,
    freq_cis_real_row: TensorSlice,
    freq_cis_imag_row: TensorSlice,
    config: Config,
    /,
) -> None:
    # stories model, llama2
    let head_size = config.head_size

    for i in range(config.n_heads):
        let outer_offset = i * head_size

        for j in range(0, head_size, 2):
            # Frequency dim is half the head size so index is half of other tensors
            # Draw one frequency for each pair to be rotated
            let fcr = freq_cis_real_row[j // 2]
            let fci = freq_cis_imag_row[j // 2]
            let offset = outer_offset + j
            # Make two strided draws offset by 1 for 'nelts' pairs
            let q0 = q[offset]
            let q1 = q[offset + 1]
            q[offset] = q0 * fcr - q1 * fci
            q[offset + 1] = q0 * fci + q1 * fcr

            if i < config.n_kv_heads:
                let k0 = k[offset]
                let k1 = k[offset + 1]
                k[offset] = k0 * fcr - k1 * fci
                k[offset + 1] = k0 * fci + k1 * fcr


alias func_sig_type = fn (
    inout TensorF32, inout TensorSlice, TensorSlice, TensorSlice, Config
) -> None


fn benchmark_rope[
    func: func_sig_type
](dim: Int, n_heads: Int, head_size: Int, n_kv_heads: Int, name: String) raises:
    var config = Config()
    config.dim = dim
    config.hidden_dim = dim * 4
    config.n_heads = n_heads
    config.n_kv_heads = n_kv_heads
    config.head_size = head_size

    let q = rand[DType.float32](dim)
    let k = TensorSlice(rand[DType.float32](1, dim), 0)

    let freq1 = TensorSlice(rand[DType.float32](1, head_size // 2), 0)
    let freq2 = TensorSlice(rand[DType.float32](1, head_size // 2), 0)

    @parameter
    fn wrapper():
        func(q, k, freq1, freq2, config)

    let report = benchmark.run[wrapper]()
    let time = report.mean["ms"]()

    print(
        "RoPE",
        name,
        "time:",
        time,
        "ms",
    )


fn main() raises:
    print(
        "stories15M size",
        "dims-heads-size:",
        String(288) + " - " + String(6) + " - " + String(48),
    )
    benchmark_rope[rope_vectorize_parallelize](288, 6, 48, 6, "vectorize_parallelize")
    benchmark_rope[rope_parallelize](288, 6, 48, 6, "parallelize (current)")
    benchmark_rope[rope_vectorize](288, 6, 48, 6, "vectorize")
    benchmark_rope[rope](288, 6, 48, 6, "vanilla for loops")
    print()
    print(
        "stories110M size",
        "dims-heads-size:",
        String(768) + " - " + String(12) + " - " + String(64),
    )
    benchmark_rope[rope_vectorize_parallelize](768, 12, 64, 12, "vectorize_parallelize")
    benchmark_rope[rope_parallelize](768, 12, 64, 12, "parallelize (current)")
    benchmark_rope[rope_vectorize](768, 12, 64, 12, "vectorize")
    benchmark_rope[rope](768, 12, 64, 12, "vanilla for loops")
    print()
    print(
        "TinyLlama-1B size",
        "dims-heads-size:",
        String(2048) + " - " + String(32) + " - " + String(64),
    )
    benchmark_rope[rope_vectorize_parallelize](2048, 32, 64, 4, "vectorize_parallelize")
    benchmark_rope[rope_parallelize](2048, 32, 64, 4, "parallelize (current)")
    benchmark_rope[rope_vectorize](2048, 32, 64, 4, "vectorize")
    benchmark_rope[rope](2048, 32, 64, 4, "vanilla for loops")
    print()
    print(
        "imaginary huge network size",
        "dims-heads-size:",
        String(512 * 512) + " - " + String(128) + " - " + String(512),
    )
    benchmark_rope[rope_vectorize_parallelize](
        512 * 512, 128, 512, 32, "vectorize_parallelize"
    )
    benchmark_rope[rope_parallelize](512 * 512, 128, 512, 32, "parallelize (current)")
    benchmark_rope[rope_vectorize](512 * 512, 128, 512, 32, "vectorize")
    benchmark_rope[rope](512 * 512, 128, 512, 32, "vanilla for loops")
