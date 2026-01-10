// worker.js
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

// Limit the cache size to prevent the browser from running out of disk/mem space
env.allowLocalModels = false;

let generator = null;

self.onmessage = async (e) => {
    const { type, text } = e.data;

    if (type === 'load') {
        try {
            generator = await pipeline('text-generation', 'onnx-community/Qwen2.5-0.5B-Instruct', {
                device: 'webgpu',
                dtype: 'q4', // Forced 4-bit quantization
            });
            self.postMessage({ type: 'ready' });
        } catch (err) {
            self.postMessage({ type: 'error', error: err.message });
        }
    }

    if (type === 'generate') {
        await generator(text, {
            max_new_tokens: 128,
            temperature: 0.7,
            max_window_size: 256, // Keep memory footprint small
            on_token_callback: (tokens) => {
                const output = generator.tokenizer.decode(tokens, { skip_special_tokens: true });
                self.postMessage({ type: 'update', output });
            }
        });
        self.postMessage({ type: 'done' });
    }
};
