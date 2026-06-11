import init, { TabularModel } from '../pkg/tabular_wasm.js';

let wasmInitialized = false;

// Initialize WASM instance
export async function initializeWasm() {
    if (!wasmInitialized) {
        await init();
        wasmInitialized = true;
        console.log("WASM Initialized successfully.");
    }
}

// Translate a character into a lowercase hex string
export function charToHex(ch) {
    return ch.charCodeAt(0).toString(16).toLowerCase();
}

// Decode a hex string back to a character representation
export function hexToChar(hex) {
    const code = parseInt(hex, 16);
    if (isNaN(code)) return ' ';
    return String.fromCharCode(code);
}

// Load a specific model binary file
export async function loadModel(modelPath) {
    await initializeWasm();
    try {
        console.log(`Fetching model from ${modelPath}...`);
        const response = await fetch(modelPath);
        if (!response.ok) {
            throw new Error(`Failed to fetch model from ${modelPath}`);
        }
        const buffer = await response.arrayBuffer();
        const bytes = new Uint8Array(buffer);
        const model = new TabularModel(bytes);
        const metadata = JSON.parse(model.metadata());
        console.log(`Loaded model: ${metadata.dataset_name}`);
        return { model, metadata };
    } catch (err) {
        console.error("Error loading model:", err);
        throw err;
    }
}

// Autoregressive character generation loop in Javascript using TabularModel WASM bindings
export async function generateAutoregressive(
    modelObj, 
    seedText, 
    numCharsToGenerate, 
    temperature, 
    onCharCallback
) {
    const { model, metadata } = modelObj;
    const vocab = metadata.class_names; // Array of hex strings representing chars
    const vocabSize = vocab.length;
    const inputDim = metadata.input_dim;
    const contextLen = Math.floor(inputDim / vocabSize);
    
    // Build char-to-index mapping for fast lookup
    const vocabToIdx = {};
    vocab.forEach((hex, idx) => {
        vocabToIdx[hex] = idx;
    });

    let generated = seedText;
    
    // Slide the context window forward
    for (let step = 0; step < numCharsToGenerate; step++) {
        const currentLen = generated.length;
        
        const contextStr = generated.slice(-contextLen);
        const contextIndices = new Float32Array(inputDim);
        
        let paddedContext = contextStr;
        if (contextStr.length < contextLen) {
            paddedContext = " ".repeat(contextLen - contextStr.length) + contextStr;
        }
        
        for (let i = 0; i < contextLen; i++) {
            const hex = charToHex(paddedContext[i]);
            const idx = vocabToIdx[hex] !== undefined ? vocabToIdx[hex] : 0;
            // One-hot encode position i
            for (let j = 0; j < vocabSize; j++) {
                contextIndices[i * vocabSize + j] = (j === idx) ? 1.0 : 0.0;
            }
        }

        // Run inference in WASM
        const jsonResult = JSON.parse(model.predict(contextIndices));
        if (jsonResult.type !== "classification") {
            throw new Error("Generative SLM expects a classification task output");
        }
        
        const probs = jsonResult.probabilities;
        const nextIdx = sampleWithTemperature(probs, temperature);
        const nextHex = vocab[nextIdx];
        const nextChar = hexToChar(nextHex);

        generated += nextChar;

        // Callback to stream characters to the UI
        if (onCharCallback) {
            const stop = onCharCallback(nextChar, probs, vocab);
            if (stop === true) break;
        }
    }
    
    return generated;
}

// Temperature sampling algorithm
function sampleWithTemperature(probs, temperature) {
    const t = Math.max(temperature, 0.01);
    
    // Apply temperature scaling
    let expProbs = probs.map(p => Math.exp(Math.log(p + 1e-10) / t));
    const sum = expProbs.reduce((a, b) => a + b, 0);
    expProbs = expProbs.map(p => p / sum);
    
    // Random sampling
    const r = Math.random();
    let cumulative = 0;
    for (let i = 0; i < expProbs.length; i++) {
        cumulative += expProbs[i];
        if (r <= cumulative) return i;
    }
    return expProbs.length - 1;
}

// Shannon Entropy utility
export function computeEntropy(probs) {
    return probs.reduce((sum, p) => {
        if (p > 1e-10) {
            return sum - p * Math.log(p);
        }
        return sum;
    }, 0);
}
