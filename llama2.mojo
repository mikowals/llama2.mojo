from algorithm import sum
from algorithm import vectorize, parallelize
from builtin import string
from math import round
from memory import memset_zero, memcpy
from memory.buffer import NDBuffer, Buffer
from memory.unsafe import DTypePointer
from python import Python
from random import rand
from read import BufReader, File
from runtime.llcl import num_cores, Runtime
from sys import argv
from testing import assert_equal, assert_true

# The SIMD vector width.
from sys.info import simdwidthof
import math
import os
import random
import time

alias nelts = (2 * simdwidthof[DType.float32]())

alias PointerString = Pointer[UInt8]
alias BufferPtrType = DTypePointer[DType.uint8]
alias BufferPtrFloat32 = DTypePointer[DType.float32]
alias PointerStrings = Pointer[PointerString]


fn read_val_int(inout buf: FileBuf) raises -> Int:
    # DTypePointer[DType.ui8](buf.data).bitcast[DType.ui8]()
    let data = buf.data.offset(buf.get_offset()).bitcast[DType.int32]()
    let result = data.load(0)
    buf.move_offset(4)
    return result.to_int()


fn read_val_float32(inout buf: FileBuf) raises -> Float32:
    # DTypePointer[DType.ui8](buf.data).bitcast[DType.ui8]()
    let val = buf.data.offset(buf.get_offset()).bitcast[DType.float32]().load(0)
    buf.move_offset(4)
    return val


fn read_val_str(inout buf: FileBuf, slen: Int) raises -> PointerString:
    let str = PointerString.alloc(slen + 1)
    for i in range(slen):
        str.store(i, buf.data.load(buf.get_offset()))
        buf.move_offset(1)
    str.store(slen, 0)

    return str


fn str_len(s: PointerString) -> Int:
    var len = 0
    while s[len] != 0:
        len += 1
    return len


# not optimal concat
fn str_concat(s1: PointerString, s2: PointerString) -> PointerString:
    let l1 = str_len(s1)
    let l2 = str_len(s2)
    let str = PointerString.alloc(l1 + l2 + 1)
    memcpy[UInt8](str, s1, l1)
    memcpy[UInt8](str.offset(l1), s2, l2)
    str.store(l1 + l2, 0)
    return str


fn str_to_ptr(s: String) -> PointerString:
    let ret = PointerString.alloc(len(s) + 1)
    for i in range(len(s)):
        ret.store(i, ord(s[i]))
    ret.store(len(s), 0)
    return ret


fn string_compare(a: PointerString, b: PointerString) -> Int:
    var index = 0
    while a[index] != 0 and b[index] != 0:
        if a[index] < b[index]:
            return -1
        if a[index] > b[index]:
            return 1

        index += 1

    if a[index] != 0 and b[index] == 0:
        return 1

    if a[index] == 0 and b[index] != 0:
        return -1

    return 0


# Quicksort helper function to find the partition position
fn partition(
    inout array: PointerStrings, inout indices: DynamicVector[Int], low: Int, high: Int
) -> Int:
    let pivot = array[high]
    var ii = low - 1
    for jj in range(low, high):
        if string_compare(pivot, array[jj]) == 1:
            # If element smaller than pivot, swap
            ii = ii + 1

            let tmp = array[ii]
            let tmp_idx = indices[ii]
            array.store(ii, array[jj])
            indices[ii] = indices[jj]
            array.store(jj, tmp)
            indices[jj] = tmp_idx

    # Swap the pivot element
    let tmp = array[ii + 1]
    let tmp_idx = indices[ii + 1]
    array.store(ii + 1, array[high])
    indices[ii + 1] = indices[high]
    array.store(high, tmp)
    indices[high] = tmp_idx

    return ii + 1


fn quicksort(
    inout array: PointerStrings, inout indices: DynamicVector[Int], low: Int, high: Int
):
    if low < high:
        let pi = partition(array, indices, low, high)
        quicksort(array, indices, low, pi - 1)
        quicksort(array, indices, pi + 1, high)


struct FileBuf:
    var data: BufferPtrType
    var offset: Int
    var size: Int

    @always_inline
    fn __init__(inout self):
        self.data = BufferPtrType()
        self.offset = 0
        self.size = 0

    @always_inline
    fn move_offset(inout self, size: Int) raises:
        let new_offset = self.offset + size
        if new_offset > self.size:
            raise Error("Resulting offset will be past the end of the FileBuf")
        if new_offset < 0:
            raise Error("Resulting offset will be before the beginning of the FileBuf")
        self.offset = new_offset

    @always_inline
    fn bitcast_offset_f32(inout self, size: Int) raises -> BufferPtrFloat32:
        let ret = self.data.offset(self.offset).bitcast[DType.float32]()
        self.move_offset(size * sizeof[DType.float32]())
        return ret

    @always_inline
    fn get_offset(self) raises -> Int:
        if self.offset > self.size:
            raise Error("Offset is past the end of the FileBuf")
        if self.offset < 0:
            raise Error("Offset is before the beginning of the FileBuf")
        return self.offset


@always_inline
fn wrap(token: PointerString) -> PointerString:
    if string_compare(token, str_to_ptr("\\n")) == 0:
        return str_to_ptr("<0x0A>")
    if string_compare(token, str_to_ptr("\\t")) == 0:
        return str_to_ptr("<0x09>")
    if string_compare(token, str_to_ptr("'")) == 0:
        return str_to_ptr("<0x27>")
    elif string_compare(token, str_to_ptr('"')) == 0:
        return str_to_ptr("<0x22>")
    return token


struct Tokenizer:
    var vocab: PointerStrings
    var vocab_scores: BufferPtrFloat32
    var max_token_length: Int
    var vocab_size: Int
    var sorted_vocab: PointerStrings
    var sorted_indices: DynamicVector[Int]

    fn __init__(inout self, vocab_size: Int, inout buf: FileBuf) raises -> None:
        self.vocab_size = vocab_size
        self.max_token_length = read_val_int(buf)
        self.vocab_scores = BufferPtrFloat32.alloc(self.vocab_size)
        self.vocab = PointerStrings.alloc(self.vocab_size)
        # lazy load sorted vocab
        self.sorted_vocab = PointerStrings.alloc(0)
        self.sorted_indices = DynamicVector[Int](0)

        # read vocab_scores & vocab values (tokens)
        for i in range(0, self.vocab_size):
            self.vocab_scores.store(i, read_val_float32(buf))
            let slen = read_val_int(buf)
            self.vocab.store(i, read_val_str(buf, slen))

        return None

    # sort vocab by string_compare
    @always_inline
    fn sort(inout self) -> None:
        if len(self.sorted_indices) < self.vocab_size:
            self.sorted_indices = DynamicVector[Int](self.vocab_size)
            self.sorted_vocab = PointerStrings.alloc(self.vocab_size)
            for ii in range(self.vocab_size):
                self.sorted_vocab.store(ii, self.vocab[ii])
                self.sorted_indices.push_back(ii)

        let n = self.vocab_size
        quicksort(self.sorted_vocab, self.sorted_indices, 0, n - 1)
        return None

    # Binary search that returns -1 if string is not found
    @always_inline
    fn find(inout self, token_o: PointerString) -> Int:
        let token = wrap(token_o)
        let n = self.vocab_size
        if len(self.sorted_indices) < n:
            self.sort()
        var left = 0
        var right = n - 1
        while left <= right:
            let mid = left + (right - left) // 2
            let comparison = string_compare(self.sorted_vocab[mid], token)
            if comparison == 0:
                return self.sorted_indices[mid]
            if comparison < 0:
                left = mid + 1
            else:
                right = mid - 1
        return -1


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

    @always_inline
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


struct RunState:
    # activation at current time stamp (dim,)
    var x: NDBuffer[1, DimList(), DType.float32]
    # same, but inside a residual branch (dim,)
    var xb: NDBuffer[1, DimList(), DType.float32]
    # an additional buffer just for convenience (dim,)
    var xb2: NDBuffer[1, DimList(), DType.float32]
    # buffer for hidden dimension in the ffn (hidden_dim,)
    var hb: NDBuffer[1, DimList(), DType.float32]
    # buffer for hidden dimension in the ffn (hidden_dim,)
    var hb2: NDBuffer[1, DimList(), DType.float32]
    var q: NDBuffer[1, DimList(), DType.float32]  # query (dim,)
    var k: NDBuffer[1, DimList(), DType.float32]  # key (kv_dim,)
    var v: NDBuffer[1, DimList(), DType.float32]  # value (kv_dim,)
    # buffer for scores/attention values (n_heads, seq_len)
    var att: NDBuffer[2, DimList(), DType.float32]
    var logits: NDBuffer[1, DimList(), DType.float32]  # output logits
    var key_cache: NDBuffer[3, DimList(), DType.float32]  # (layer, seq_len, dim)
    var value_cache: NDBuffer[3, DimList(), DType.float32]  # (layer, seq_len, dim)
    var rt: Runtime

    @always_inline
    fn __init__(
        inout self,
        config: Config,
    ) raises:
        fn create_weight[
            rank: Int
        ](*dims: Dim) raises -> NDBuffer[rank, DimList(), DType.float32]:
            let shape = DimList(dims)
            let num_elements = shape.product[rank]().get()
            return NDBuffer[rank, DimList(), DType.float32](
                DTypePointer[DType.float32].alloc(num_elements), shape
            )

        self.x = create_weight[1](config.dim)
        self.xb = create_weight[1](config.dim)
        self.xb2 = create_weight[1](config.dim)
        self.hb = create_weight[1](config.hidden_dim)
        self.hb2 = create_weight[1](config.hidden_dim)
        self.q = create_weight[1](config.dim)
        self.att = create_weight[2](config.n_heads, config.seq_len)
        self.logits = create_weight[1](config.vocab_size)
        self.key_cache = create_weight[3](
            config.n_layers, config.seq_len, config.kv_dim
        )
        self.value_cache = create_weight[3](
            config.n_layers, config.seq_len, config.kv_dim
        )
        # So their updates flow to the caches, k and v are slices with shared memory.
        # Initialize with placeholders. The real tensors reference layer and position during forward pass.
        self.k = create_weight[1](config.kv_dim)
        self.v = create_weight[1](config.kv_dim)
        self.rt = Runtime(num_cores())


struct TransformerWeights:
    var token_embedding_table: NDBuffer[2, DimList(), DType.float32]
    var freq_cis_real: NDBuffer[2, DimList(), DType.float32]
    var freq_cis_imag: NDBuffer[2, DimList(), DType.float32]
    var rms_att_weight: NDBuffer[2, DimList(), DType.float32]
    var wq: NDBuffer[3, DimList(), DType.float32]
    var wk: NDBuffer[3, DimList(), DType.float32]
    var wv: NDBuffer[3, DimList(), DType.float32]
    var wo: NDBuffer[3, DimList(), DType.float32]
    var rms_ffn_weight: NDBuffer[2, DimList(), DType.float32]
    var w1: NDBuffer[3, DimList(), DType.float32]
    var w3: NDBuffer[3, DimList(), DType.float32]
    var w2: NDBuffer[3, DimList(), DType.float32]
    var rms_final_weight: NDBuffer[1, DimList(), DType.float32]
    var wcls: NDBuffer[2, DimList(), DType.float32]

    @always_inline
    fn __init__(
        inout self, config: Config, shared_weights: Int, inout buf: FileBuf
    ) raises:
        fn load_weights[
            rank: Int
        ](inout buf: FileBuf, *dims: Dim) raises -> NDBuffer[
            rank, DimList(), DType.float32
        ]:
            let shape = DimList(dims)
            let num_elements = shape.product[rank]().get()
            return NDBuffer[rank, DimList(), DType.float32](
                buf.bitcast_offset_f32(num_elements), shape
            )

        self.token_embedding_table = load_weights[2](buf, config.vocab_size, config.dim)

        self.rms_att_weight = load_weights[2](buf, config.n_layers, config.dim)
        self.wq = load_weights[3](buf, config.n_layers, config.dim, config.dim)
        self.wk = load_weights[3](buf, config.n_layers, config.kv_dim, config.dim)
        self.wv = load_weights[3](buf, config.n_layers, config.kv_dim, config.dim)
        self.wo = load_weights[3](buf, config.n_layers, config.dim, config.dim)
        self.rms_ffn_weight = load_weights[2](buf, config.n_layers, config.dim)
        self.w1 = load_weights[3](buf, config.n_layers, config.hidden_dim, config.dim)
        self.w2 = load_weights[3](buf, config.n_layers, config.dim, config.hidden_dim)
        self.w3 = load_weights[3](buf, config.n_layers, config.hidden_dim, config.dim)
        self.rms_final_weight = load_weights[1](buf, config.dim)
        # maybe need modifying for different model
        # config.head_size // 2 for stories and tinyllama-1.1
        self.freq_cis_real = load_weights[2](buf, config.seq_len, config.head_size // 2)
        self.freq_cis_imag = load_weights[2](buf, config.seq_len, config.head_size // 2)
        if shared_weights:
            self.wcls = self.token_embedding_table
        else:
            self.wcls = load_weights[2](buf, config.vocab_size, config.dim)


@always_inline
fn read_file(file_name: String, inout buf: FileBuf) raises:
    let _os = Python.import_module("os")
    let ff_size = _os.path.getsize(file_name)
    let cp_size = string.atol(ff_size.to_string())
    let cp_buf: BufferPtrType = BufferPtrType.alloc(cp_size)
    # set window buffer to read binary data from file
    let f = File(file_name)
    var reader = BufReader[4096](f ^)
    var bytes_read = 1
    var offset = 0

    while bytes_read > 0:
        let buf = Buffer[4096, DType.uint8](cp_buf.offset(offset))
        bytes_read = reader.read(buf)
        offset += bytes_read
    reader.do_nothing()  # keeps lifetimes working
    buf.data = cp_buf
    buf.size = cp_size
    buf.offset = 0
    return None


@always_inline
fn config_init(inout config: Config, inout buf: FileBuf) raises:
    config.dim = read_val_int(buf)
    config.hidden_dim = read_val_int(buf)
    config.n_layers = read_val_int(buf)
    config.n_heads = read_val_int(buf)
    config.n_kv_heads = read_val_int(buf)
    config.vocab_size = read_val_int(buf)
    config.seq_len = read_val_int(buf)
    config.head_size = config.dim // config.n_heads
    config.kv_dim = (config.n_kv_heads * config.dim) // config.n_heads
    config.kv_mul = config.n_heads // config.n_kv_heads
    return None


@always_inline
fn accum(
    inout a: NDBuffer[1, DimList(), DType.float32],
    b: NDBuffer[1, DimList(), DType.float32],
) -> None:
    let size = a.dim(0)

    @parameter
    fn _acc[_nelts: Int](j: Int):
        a.simd_store[_nelts](
            StaticIntTuple[1](j), a.simd_load[_nelts](j) + b.simd_load[_nelts](j)
        )

    vectorize[nelts, _acc](size)


@always_inline
fn rmsnorm(
    o: NDBuffer[1, DimList(), DType.float32],
    x: NDBuffer[1, DimList(), DType.float32],
    weight: NDBuffer[1, DimList(), DType.float32],
) -> None:
    let size = x.dim(0)
    # Calculate sum of squares
    var tmp = SIMD[DType.float32, nelts](0)

    @parameter
    fn _sum2[_nelts: Int](j: Int):
        if _nelts < nelts:
            tmp[0] += (x.simd_load[_nelts](j) ** 2).reduce_add()
        else:
            tmp += x.simd_load[nelts](j) ** 2

    vectorize[nelts, _sum2](size)

    var ss: Float32 = tmp.reduce_add()
    ss = ss / size + 1e-5
    ss = 1.0 / math.sqrt(ss)

    # Normalize and scale
    @parameter
    fn _norm[_nelts: Int](j: Int):
        let val = weight.simd_load[_nelts](j) * ss * x.simd_load[_nelts](j)
        o.simd_store[_nelts](StaticIntTuple[1](j), val)

    vectorize[nelts, _norm](size)


@always_inline
fn softmax(x: NDBuffer[1, DimList(), DType.float32]) -> None:
    softmax(x, 0, x.dim(0))


@always_inline
fn softmax[
    rank: Int
](x: NDBuffer[rank, DimList(), DType.float32], start: Int, end: Int):
    var max_val: Float32 = -1e9

    @parameter
    fn _max[_nelts: Int](ii: Int):
        let val = x.data.simd_load[_nelts](start + ii).reduce_max()
        if val > max_val:
            max_val = val

    vectorize[nelts, _max](end - start)

    var ssum: Float32 = 0.0

    @parameter
    fn _exp[_nelts: Int](ii: Int):
        x.data.simd_store[_nelts](
            start + ii, math.exp(x.data.simd_load[_nelts](start + ii) - max_val)
        )
        ssum += x.data.simd_load[_nelts](start + ii).reduce_add()

    vectorize[nelts, _exp](end - start)

    @parameter
    fn _norm[_nelts: Int](ii: Int):
        x.data.simd_store[_nelts](
            start + ii, x.data.simd_load[_nelts](start + ii) / ssum
        )

    vectorize[nelts, _norm](end - start)


@always_inline
fn matmul(
    C: NDBuffer[1, DimList(), DType.float32],
    A: NDBuffer[1, DimList(), DType.float32],
    B: NDBuffer[2, DimList(), DType.float32],
    rt: Runtime,
):
    let rows = B.dim(0)
    let cols = B.dim(1)

    @parameter
    fn compute_row(i: Int):
        var tmp = SIMD[DType.float32, nelts](0)

        @parameter
        fn dot[_nelts: Int](j: Int):
            if _nelts < nelts:  # take care of tail array elements with length <  nelts
                tmp[0] += (
                    A.simd_load[_nelts](j) * B.simd_load[_nelts](i, j)
                ).reduce_add()
            else:
                tmp += A.simd_load[nelts](j) * B.simd_load[nelts](i, j)

        vectorize[nelts, dot](cols)
        C[i] = tmp.reduce_add()

    parallelize[compute_row](rows, rt.parallelism_level())


# Apply RoPE rotation to the q and k vectors for each head
# rotate odd and even dim
@always_inline
fn rope_rotation_llama(
    inout state: RunState,
    freq_cis_real_row: NDBuffer[1, DimList(), DType.float32],
    freq_cis_imag_row: NDBuffer[1, DimList(), DType.float32],
    config: Config,
) -> None:
    # stories model, llama2
    let head_size = config.head_size

    @parameter
    fn head_loop(i: Int):
        # Simple vectorization with (head_size // 2) steps gave junk transformer output.
        # Maybe because the nelt ranges end up overlapping between the steps.
        for j in range(0, config.head_size, 2):
            let fcr = freq_cis_real_row[j // 2]
            let fci = freq_cis_imag_row[j // 2]
            let q0 = state.q[i * head_size + j]
            let q1 = state.q[i * head_size + j + 1]
            state.q[i * head_size + j] = q0 * fcr - q1 * fci
            state.q[i * head_size + j + 1] = q0 * fci + q1 * fcr
            if i < config.n_kv_heads:
                let k0 = state.k[i * head_size + j]
                let k1 = state.k[i * head_size + j + 1]
                state.k[i * head_size + j] = k0 * fcr - k1 * fci
                state.k[i * head_size + j + 1] = k0 * fci + k1 * fcr

    parallelize[head_loop](config.n_heads, state.rt.parallelism_level())


fn slice[
    rank: Int, slice_dims: Int
](buf: NDBuffer[rank, DimList(), DType.float32], *dims: Int) raises -> NDBuffer[
    rank - slice_dims, DimList(), DType.float32
]:
    var offset_per_index = buf.num_elements()
    var offset = 0
    let shape: DimList
    for ii in range(slice_dims):
        offset_per_index /= buf.dim(ii)
        offset += dims[ii] * offset_per_index
    if rank - slice_dims == 1:
        shape = DimList(buf.dim(rank - 1))
    elif rank - slice_dims == 2:
        shape = DimList(buf.dim(rank - 2), buf.dim(rank - 1))
    else:
        shape = DimList()
        raise Error("slice not implemente for rank higher than 3")
    return NDBuffer[rank - slice_dims, DimList(), DType.float32](
        buf.data.offset(offset), shape
    )


@always_inline
fn transformer(
    token: Int,
    pos: Int,
    config: Config,
    inout state: RunState,
    weights: TransformerWeights,
) raises -> None:
    # A few convenience variables
    let dim = config.dim
    let hidden_dim = config.hidden_dim
    let head_size = config.head_size
    let kv_dim = config.kv_dim
    let kv_mul = config.kv_mul

    # Copy the token embedding into x
    let content_row = weights.token_embedding_table.data.offset(token * dim)
    memcpy[DType.float32](state.x.data, content_row, dim)

    # Pluck out the "pos" row of freq_cis_real and freq_cis_imag
    let freq_cis_real_row = slice[2, 1](weights.freq_cis_real, pos)
    let freq_cis_imag_row = slice[2, 1](weights.freq_cis_imag, pos)

    # Forward all the layers
    for l in range(config.n_layers):
        # Attention rmsnorm
        rmsnorm(state.xb, state.x, slice[2, 1](weights.rms_att_weight, l))
        # QKV matmuls for this position
        matmul(state.q, state.xb, slice[3, 1](weights.wq, l), state.rt)

        let loff = l * config.seq_len * config.kv_dim
        state.k = slice[3, 2](state.key_cache, l, pos)
        matmul(state.k, state.xb, slice[3, 1](weights.wk, l), state.rt)

        state.v = slice[3, 2](state.value_cache, l, pos)
        matmul(state.v, state.xb, slice[3, 1](weights.wv, l), state.rt)

        # Apply RoPE rotation to the q and k vectors for each head
        rope_rotation_llama(state, freq_cis_real_row, freq_cis_imag_row, config)

        memset_zero(state.xb.data, state.xb.num_elements())

        # Multihead attention. Iterate over all heads in parallel.
        @parameter
        fn loop_over_heads(h: Int):
            # Get the query vector for this head
            let q_offset = h * head_size

            # Index of attention scores for this head
            let att_offset = h * config.seq_len

            # Iterate over all timesteps, including the current one
            for t in range(pos + 1):
                # Starting index of the key vector for this head and at this timestep
                let k_offset = (h // kv_mul) * head_size
                # Calculate the attention score as the dot product of q and k
                var score: Float32 = 0.0

                @parameter
                fn score_fn[_nelts: Int](i: Int):
                    score += (
                        state.q.simd_load[_nelts](q_offset + i)
                        * state.key_cache.simd_load[_nelts](l, t, k_offset + i)
                    ).reduce_add()

                vectorize[nelts, score_fn](head_size)
                score /= math.sqrt[DType.float32, 1](head_size)

                # Save the score to the attention buffer
                state.att[StaticIntTuple[2](h, t)] = score

            # Softmax the scores to get attention weights, from 0..pos inclusively
            softmax[2](state.att, att_offset, att_offset + pos + 1)
            # Weighted sum of the values, store back into xb
            let xb_offset = h * head_size
            for t in range(pos + 1):
                # Starting index of the value vector for this head and at this timestep
                let v_offset = (h // kv_mul) * head_size

                # Get the attention weight for this timestep
                let a = state.att[h, t]
                # Accumulate the weighted value into xb

                @parameter
                fn xb_accumulate[_nelts: Int](i: Int):
                    let xbi = state.xb.simd_load[_nelts](
                        xb_offset + i
                    ) + a * state.value_cache.simd_load[_nelts](l, t, v_offset + i)
                    state.xb.simd_store[_nelts](StaticIntTuple[1](xb_offset + i), xbi)

                vectorize[nelts, xb_accumulate](head_size)

        parallelize[loop_over_heads](config.n_heads, state.rt.parallelism_level())
        # Final matrix multiplication to get the output of the attention
        matmul(state.xb2, state.xb, slice[3, 1](weights.wo, l), state.rt)
        # Residual connection back into x
        accum(state.x, state.xb2)
        # FFN rmsnorm
        rmsnorm(state.xb, state.x, slice[2, 1](weights.rms_ffn_weight, l))

        # Calculate self.w1(x) and self.w3(x) for FFN
        matmul(state.hb, state.xb, slice[3, 1](weights.w1, l), state.rt)

        matmul(state.hb2, state.xb, slice[3, 1](weights.w3, l), state.rt)

        @parameter
        fn silu[_nelts: Int](i: Int):
            let initial_hb = state.hb.simd_load[_nelts](i)
            # Apply SiLU activation function (silu(x) = x * sigmoid(x))
            let hbi = initial_hb * (1.0 / (1.0 + math.exp(-initial_hb)))
            # Elementwise multiply with w3(x)
            state.hb.simd_store[_nelts](
                StaticIntTuple[1](i), hbi * state.hb2.simd_load[_nelts](i)
            )

        vectorize[nelts, silu](hidden_dim)
        # Final matrix multiplication to get the output of the FFN
        matmul(state.xb, state.hb, slice[3, 1](weights.w2, l), state.rt)

        # Residual connection
        accum(state.x, state.xb)

    # Final rmsnorm
    rmsnorm(state.x, state.x, weights.rms_final_weight)

    # Classifier into logits
    matmul(state.logits, state.x, weights.wcls, state.rt)


@always_inline
fn argmax(v: NDBuffer[1, DimList(), DType.float32]) -> Int:
    # return argmax of v
    var max_i: Int = 0
    var max_p: Float32 = v[0]
    for i in range(v.dim(0)):
        if v[i] > max_p:
            max_i = i
            max_p = v[i]
    return max_i


@always_inline
fn sample(probabilities: NDBuffer[1, DimList(), DType.float32]) -> Int:
    let n = probabilities.dim(0)
    # Sample index from probabilities, they must sum to 1
    # get random value within (min, max) float32 range
    let r = DTypePointer[DType.float32].alloc(1)
    rand[DType.float32](r, 1)
    var cdf: Float32 = 0.0
    for i in range(n):
        cdf += probabilities[i]
        if r.load(0) < cdf:
            return i
    return n - 1  # In case of rounding errors


@always_inline
fn bpe_encode(inout tokens: DynamicVector[Int], text: String, inout tok: Tokenizer):
    for pos in range(len(text)):
        let char = str_to_ptr(text[pos])
        let id = tok.find(char)

        if id == -1:
            print("Not a good prompt token at pos ", pos)
            return
        tokens.push_back(id)

    while True:
        var best_score = Float32(-1e10)
        var best_id = -1
        var best_idx = -1

        for i in range(len(tokens) - 1):
            # Check if we can merge the pair (tokens[i], tokens[i+1])
            let str = str_concat(tok.vocab[tokens[i]], tok.vocab[tokens[i + 1]])
            let id = tok.find(str)
            if id != -1 and tok.vocab_scores.load(id) > best_score:
                best_score = tok.vocab_scores.load(id)
                best_id = id
                best_idx = i

        if best_idx == -1:
            # We couldn't find any more pairs to merge, so we're done
            break

        # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id
        # Delete token at position best_idx+1, shift the entire sequence back 1
        var _tokens = DynamicVector[Int]()
        for i in range(0, best_idx + 1):
            _tokens.push_back(tokens[i])
        for i in range(best_idx + 2, len(tokens)):
            _tokens.push_back(tokens[i])
        tokens = _tokens


@always_inline
fn str2num(d: Int) -> Int:
    # covert Hex to decimal
    if d >= ord("A"):
        return d - ord("A") + 10
    return d - ord("0")


@always_inline
fn print_str(s: PointerString):
    # print raw byte like <0x0A>
    if (s[1].to_int() == ord("0")) and (s[2].to_int() == ord("x")):
        let d1: Int = s[3].to_int()
        let d2: Int = s[4].to_int()
        print_no_newline(chr(str2num(d1) * 16 + str2num(d2)))
        return
    # print all chars till null character
    var p: Int = 0
    while s[p].to_int() != 0:
        print_no_newline(chr(s[p].to_int()))
        p += 1


@always_inline
fn time_in_ms() -> Int:
    # Returns time in milliseconds for benchmarking the model speed
    return time.now() // 1_000_000


@always_inline
fn print_usage():
    print("Usage: mojo llama2.mojo <checkpoint> [options]")
    print(
        'Example: mojo llama2.mojo stories15M.bin -s 99 -n 256 -t 0.5 -i "Llama is an'
        ' animal"'
    )
    print("Options:")
    print("  -s <int>    random seed, default time.now()")
    print("  -t <float>  temperature in [0,1.0], default 1.0")
    print("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len")
    print("  -i <string> input prompt")
    print("  -z          tokenizer path")


@always_inline
fn main() raises:
    print("num hardware threads: ", num_cores())
    print("SIMD vector width: ", nelts)
    var tokenizer = StringRef("tokenizer.bin")
    var checkpoint = StringRef("stories110M.bin")
    var temperature = 0.9
    var steps = 256
    var prompt = String("")
    var rng_seed: Int = time.now()

    @parameter
    fn argparse() raises -> Int:
        let args = argv()
        if len(args) < 2:
            return 0
        checkpoint = args[1]
        for i in range(2, len(args), 2):
            if args[i] == "-p":
                print("Option not supported: ", args[i])
            if args[i] == "-n":
                steps = atol(args[i + 1])
            if args[i] == "-z":
                tokenizer = args[i + 1]
            if args[i] == "-s":
                rng_seed = atol(args[i + 1])
            if args[i] == "-i":
                prompt = args[i + 1]
            if args[i] == "-t":
                let val = args[i + 1]
                temperature = 0.0
                # hacky parse float, keep only 1 digit
                for c in range(0, len(val)):
                    if val[c] == ".":
                        temperature += atol(val[c + 1]) * 0.1
                        break
                    else:
                        temperature = atol(val[c])
                if temperature < -1e9 or temperature > (1 + 1e9):
                    print("Wrong temperature value", temperature)
                    return 0
        return 1

    let res = argparse()
    if res == 0:
        print_usage()
        return

    random.seed(rng_seed)
    var fbuf: FileBuf = FileBuf()
    var tbuf: FileBuf = FileBuf()
    var config: Config = Config()

    read_file(checkpoint, fbuf)
    print("checkpoint size: ", fbuf.size, "[", fbuf.size // 1024 // 1024, "MB ]")
    config_init(config, fbuf)

    # negative vocab size is hacky way of signaling unshared weights. bit yikes.
    let shared_weights = 1 if config.vocab_size > 0 else 0
    config.vocab_size = (
        -config.vocab_size if config.vocab_size < 0 else config.vocab_size
    )

    let weights: TransformerWeights = TransformerWeights(config, shared_weights, fbuf)

    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len

    # Read in the tokenizer.bin file
    read_file(tokenizer, tbuf)
    var tok = Tokenizer(config.vocab_size, tbuf)

    # print the layers number and vocab size
    print("n layers: ", config.n_layers)
    print("vocab size: ", tok.vocab_size)

    # Create and initialize the application RunState
    var state = RunState(config)

    # Process the prompt, if any
    var prompt_tokens = DynamicVector[Int]()

    if prompt:
        bpe_encode(prompt_tokens, prompt, tok)

    # Start the main loop
    var start = 0  # Used to time our code, only initialized after the first iteration
    var next_token = 0  # Will store the next token in the sequence
    # Initialize with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    var token = 1

    # Position in the sequence
    var pos = 0
    while pos < steps:
        # Forward the transformer to get logits for the next token
        transformer(token, pos, config, state, weights)

        if pos < len(prompt_tokens):
            next_token = prompt_tokens[pos]
        else:
            # Sample the next token
            if temperature == 0.0:
                # Greedy argmax sampling: take the token with the highest probability
                next_token = argmax(state.logits)
            else:
                # Apply the temperature to the logits
                for q in range(config.vocab_size):
                    state.logits[q] = state.logits[q] / temperature
                # Apply softmax to the logits to get the probabilities for the next token
                softmax(state.logits)
                # Sample from this distribution to get the next token
                next_token = sample(state.logits)

            # Finish generating when EOS, BOS appear
            if next_token == 1 or next_token == 2:
                break
        var token_str: PointerString = tok.vocab[next_token]
        if token == 1 and token_str[0] == ord(" "):
            token_str = token_str.offset(1)

        print_str(token_str)

        # Advance forward
        token = next_token
        pos += 1

        if start == 0:
            start = time_in_ms()

    let end = time_in_ms()
    print("\nachieved tok/s: ", (pos - 1) / (end - start) * 1000)
