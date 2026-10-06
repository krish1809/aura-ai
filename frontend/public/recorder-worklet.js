// AudioWorklet that forwards raw microphone samples to the main thread.
class AuraRecorder extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0]
    if (channel) this.port.postMessage(channel.slice(0))
    return true
  }
}
registerProcessor('aura-recorder', AuraRecorder)
