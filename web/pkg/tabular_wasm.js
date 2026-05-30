export class TabularModel {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TabularModelFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_tabularmodel_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    metadata() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.tabularmodel_metadata(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {Uint8Array} bytes
     */
    constructor(bytes) {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.tabularmodel_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        TabularModelFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {string}
     */
    norm_encoded() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.tabularmodel_norm_encoded(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {Float32Array} values
     * @returns {string}
     */
    predict(values) {
        let deferred3_0;
        let deferred3_1;
        try {
            const ptr0 = passArrayF32ToWasm0(values, wasm.__wbindgen_malloc);
            const len0 = WASM_VECTOR_LEN;
            const ret = wasm.tabularmodel_predict(this.__wbg_ptr, ptr0, len0);
            var ptr2 = ret[0];
            var len2 = ret[1];
            if (ret[3]) {
                ptr2 = 0; len2 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
        }
    }
}
if (Symbol.dispose) TabularModel.prototype[Symbol.dispose] = TabularModel.prototype.free;

/**
 * Edge Small Language Model — causal Transformer running in WASM.
 *
 * JavaScript usage:
 * ```js
 * const resp = await fetch('model.bin');
 * const bytes = new Uint8Array(await resp.arrayBuffer());
 * const slm = new TransformerSLMModel(bytes);
 *
 * const meta = JSON.parse(slm.metadata());
 * const vocab = meta.class_names;           // idx → char mapping
 * const contextLen = meta.input_dim;
 *
 * // Feed a context of integer token IDs
 * const context = new Float32Array([12, 5, 3, ...]);
 * const probs = slm.predict_next(context);  // Float32Array length vocab_size
 *
 * // After inference, read attention weights for visualization
 * const attn = slm.get_last_attention_weights(); // Float32Array [heads × T × T]
 * const numHeads = slm.num_heads();
 * const contextLength = slm.context_len();
 * ```
 */
export class TransformerSLMModel {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TransformerSLMModelFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_transformerslmmodel_free(ptr, 0);
    }
    /**
     * Context length (number of characters the model sees at once).
     * @returns {number}
     */
    context_len() {
        const ret = wasm.transformerslmmodel_context_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Compute Shannon entropy of a probability distribution (0 = certain, ln(V) = uniform).
     * @param {Float32Array} probs
     * @returns {number}
     */
    entropy(probs) {
        const ptr0 = passArrayF32ToWasm0(probs, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.transformerslmmodel_entropy(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * Return the self-attention weights from the LAST Transformer block's last forward pass.
     *
     * Returns a flat Float32Array of shape [num_heads × context_len × context_len].
     * For `num_heads=4` and `context_len=32`, this is 4096 floats.
     *
     * In JavaScript, index as: `attn[h * T * T + i * T + j]` where h=head, i=query pos, j=key pos.
     * @returns {Float32Array}
     */
    get_last_attention_weights() {
        const ret = wasm.transformerslmmodel_get_last_attention_weights(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Return model metadata JSON (vocab, context length, architecture details).
     * @returns {string}
     */
    metadata() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.transformerslmmodel_metadata(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Construct from FINF v4 bytes.
     * @param {Uint8Array} bytes
     */
    constructor(bytes) {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.transformerslmmodel_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        TransformerSLMModelFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Number of attention heads.
     * @returns {number}
     */
    num_heads() {
        const ret = wasm.transformerslmmodel_num_heads(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Number of Transformer blocks.
     * @returns {number}
     */
    num_layers() {
        const ret = wasm.transformerslmmodel_num_layers(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Run inference on a context of token indices.
     *
     * `context` must have exactly `context_len` elements (Float32Array of usize-as-f32).
     * Returns a Float32Array of length `vocab_size` with next-token probabilities.
     * @param {Float32Array} context
     * @returns {Float32Array}
     */
    predict_next(context) {
        const ptr0 = passArrayF32ToWasm0(context, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.transformerslmmodel_predict_next(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Sample a next token index from probabilities using temperature scaling.
     * `probs` = output of `predict_next`. `temperature` = 1.0 is neutral.
     * Lower temperature → more deterministic. Higher → more random.
     * @param {Float32Array} probs
     * @param {number} temperature
     * @param {number} random_value
     * @returns {number}
     */
    sample_from_probs(probs, temperature, random_value) {
        const ptr0 = passArrayF32ToWasm0(probs, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.transformerslmmodel_sample_from_probs(this.__wbg_ptr, ptr0, len0, temperature, random_value);
        return ret >>> 0;
    }
    /**
     * Return the top-k token indices sorted by probability (descending).
     * @param {Float32Array} probs
     * @param {number} k
     * @returns {Uint32Array}
     */
    top_k_indices(probs, k) {
        const ptr0 = passArrayF32ToWasm0(probs, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.transformerslmmodel_top_k_indices(this.__wbg_ptr, ptr0, len0, k);
        var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
}
if (Symbol.dispose) TransformerSLMModel.prototype[Symbol.dispose] = TransformerSLMModel.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_1506f2235d1bdba0: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./tabular_wasm_bg.js": import0,
    };
}

const TabularModelFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_tabularmodel_free(ptr, 1));
const TransformerSLMModelFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_transformerslmmodel_free(ptr, 1));

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('tabular_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
