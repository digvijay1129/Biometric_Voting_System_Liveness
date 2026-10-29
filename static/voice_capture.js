/* Shared microphone capture + browser speech transcription for the voice stage. */
(function () {
    let current = null;

    function makeWavBlob(floatSamples, inputRate) {
        const outputRate = 16000;
        const inputLength = floatSamples.length;
        const outputLength = Math.max(1, Math.round(inputLength * outputRate / inputRate));
        const samples = new Float32Array(outputLength);
        for (let i = 0; i < outputLength; i++) {
            const sourcePosition = i * inputRate / outputRate;
            const left = Math.floor(sourcePosition);
            const right = Math.min(inputLength - 1, left + 1);
            const blend = sourcePosition - left;
            samples[i] = (floatSamples[Math.min(left, inputLength - 1)] * (1 - blend))
                + (floatSamples[right] * blend);
        }

        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        function writeText(offset, text) {
            for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
        }
        writeText(0, "RIFF");
        view.setUint32(4, 36 + samples.length * 2, true);
        writeText(8, "WAVE");
        writeText(12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, outputRate, true);
        view.setUint32(28, outputRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeText(36, "data");
        view.setUint32(40, samples.length * 2, true);
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            const clamped = Math.max(-1, Math.min(0.999969, samples[i]));
            view.setInt16(offset, clamped < 0 ? clamped * 32768 : clamped * 32767, true);
            offset += 2;
        }
        return new Blob([buffer], { type: "audio/wav" });
    }

    function cleanupCapture(capture) {
        if (!capture || capture.cleaned) return;
        capture.cleaned = true;
        try { if (capture.processor) capture.processor.disconnect(); } catch (_) { }
        try { if (capture.source) capture.source.disconnect(); } catch (_) { }
        try { if (capture.mute) capture.mute.disconnect(); } catch (_) { }
        if (capture.stream) capture.stream.getTracks().forEach(track => track.stop());
    }

    async function finalizeCapture(capture) {
        if (!capture || capture.finalized) return;
        capture.finalized = true;
        if (capture.fallbackTimer) clearTimeout(capture.fallbackTimer);
        cleanupCapture(capture);
        const length = capture.chunks.reduce((total, chunk) => total + chunk.length, 0);
        const joined = new Float32Array(length);
        let offset = 0;
        for (const chunk of capture.chunks) {
            joined.set(chunk, offset);
            offset += chunk.length;
        }
        try {
            if (capture.audioContext && capture.audioContext.state !== "closed") {
                await capture.audioContext.close();
            }
        } catch (_) { }
        const blob = makeWavBlob(joined, capture.inputRate || 48000);
        const result = { blob, transcript: (capture.transcript || "").trim() };
        if (capture.resolveStop) capture.resolveStop(result);
        if (current === capture) current = null;
    }

    window.VoiceCapture = {
        async start(callbacks = {}) {
            if (current) throw new Error("A recording is already in progress. Stop it before starting again.");
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                throw new Error("Live sentence recognition is not supported by this browser. Please use Google Chrome or Microsoft Edge on localhost.");
            }
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error("Microphone access is unavailable. Open the project on http://127.0.0.1:5000/ and allow microphone permission.");
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
            });
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            const audioContext = new AudioContextClass();
            await audioContext.resume();
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            const mute = audioContext.createGain();
            mute.gain.value = 0;

            const capture = {
                stream, audioContext, source, processor, mute,
                inputRate: audioContext.sampleRate,
                chunks: [], transcript: "", recognitionEnded: false,
                stopping: false, finalized: false, cleaned: false
            };
            current = capture;
            processor.onaudioprocess = event => {
                if (!capture.stopping && !capture.finalized) {
                    capture.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
                }
            };
            source.connect(processor);
            processor.connect(mute);
            mute.connect(audioContext.destination);

            const recognition = new SpeechRecognition();
            capture.recognition = recognition;
            recognition.lang = "en-IN";
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.maxAlternatives = 3;
            recognition.onresult = event => {
                const finalParts = [];
                let interim = "";
                for (let i = 0; i < event.results.length; i++) {
                    const item = event.results[i];
                    if (item.isFinal && item[0]) finalParts.push(item[0].transcript);
                    else if (item[0]) interim = item[0].transcript;
                }
                if (finalParts.length) capture.transcript = finalParts.join(" ").trim();
                const shown = capture.transcript || interim;
                if (callbacks.onTranscript) callbacks.onTranscript(shown);
            };
            recognition.onerror = event => {
                const readable = event.error === "not-allowed"
                    ? "Microphone permission was denied. Allow microphone access and retry."
                    : event.error === "no-speech"
                        ? "No speech was detected yet. Speak the full sentence clearly."
                        : `Speech recognition issue: ${event.error}. You can try again.`;
                if (callbacks.onError) callbacks.onError(readable);
            };
            recognition.onend = () => {
                capture.recognitionEnded = true;
                if (callbacks.onStatus && !capture.stopping) {
                    callbacks.onStatus("Speech capture finished. Click Stop & Verify to submit the recording.");
                }
                if (capture.stopping) finalizeCapture(capture);
            };

            try {
                recognition.start();
            } catch (error) {
                cleanupCapture(capture);
                try { await audioContext.close(); } catch (_) { }
                current = null;
                throw new Error("The browser could not start speech recognition. Reload the page and try again.");
            }
            if (callbacks.onStatus) callbacks.onStatus("Listening… say the displayed sentence once, in a normal voice.");
            return true;
        },

        stop() {
            return new Promise((resolve, reject) => {
                const capture = current;
                if (!capture) {
                    reject(new Error("Start a recording first."));
                    return;
                }
                capture.stopping = true;
                capture.resolveStop = resolve;
                capture.rejectStop = reject;
                try {
                    if (!capture.recognitionEnded && capture.recognition) capture.recognition.stop();
                } catch (_) { }
                if (capture.recognitionEnded) finalizeCapture(capture);
                else capture.fallbackTimer = setTimeout(() => finalizeCapture(capture), 1500);
            });
        },

        isRecording() {
            return !!current;
        }
    };
})();
