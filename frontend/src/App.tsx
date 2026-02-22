import { UploadPanel } from './components/UploadPanel';
import { ResultsTimeline } from './components/ResultsTimeline';
import type { TimelineEvent } from './components/ResultsTimeline';
import { ShieldAlert, Mic, Loader2 } from 'lucide-react';
import { useState, useRef } from 'react';
import { uploadFile } from './api';
import './App.css';

function App() {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [status, setStatus] = useState<'safe' | 'violence'>('safe');
  const [appState, setAppState] = useState<'idle' | 'processing' | 'results'>('idle');

  const [isStreaming, setIsStreaming] = useState(false);
  const streamRefs = useRef<{
    ws: WebSocket | null;
    context: AudioContext | null;
    stream: MediaStream | null;
    processor: ScriptProcessorNode | null;
    source: MediaStreamAudioSourceNode | null;
  }>({ ws: null, context: null, stream: null, processor: null, source: null });

  const startStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      const context = new AudioContextClass({ sampleRate: 16000 });
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/analyze/stream`;
      const ws = new WebSocket(wsUrl);

      streamRefs.current = { ws, context, stream, processor, source };

      ws.onopen = () => {
        setIsStreaming(true);
        setAppState('results');
        setEvents([]);
        setStatus('safe');
      };

      ws.onmessage = (event) => {
        const result = JSON.parse(event.data);
        const newEvent: TimelineEvent = {
          timestamp: result.chunk_id * 2.5,
          score: result.fused_score,
          alertLevel: result.alert.toLowerCase() as 'safe' | 'violence',
          details: {
            acoustic: result.acoustic_score || 0,
            nlp: result.nlp_score || 0,
            emotion: result.emotion_score || 0
          }
        };

        setEvents(prev => [...prev, newEvent]);
        setStatus(result.alert.toLowerCase() as 'safe' | 'violence');
      };

      ws.onclose = () => {
        stopStream();
      };

      ws.onerror = () => {
        stopStream();
      };

      processor.onaudioprocess = (e) => {
        if (ws.readyState === WebSocket.OPEN) {
          const float32Array = e.inputBuffer.getChannelData(0);
          const int16Array = new Int16Array(float32Array.length);
          for (let i = 0; i < float32Array.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          ws.send(int16Array.buffer);
        }
      };

      source.connect(processor);
      processor.connect(context.destination);

    } catch (error) {
      console.error("Error starting stream:", error);
      alert("Microphone access denied or error occurred.");
    }
  };

  const stopStream = () => {
    const refs = streamRefs.current;
    if (refs.ws) refs.ws.close();
    if (refs.processor) refs.processor.disconnect();
    if (refs.source) refs.source.disconnect();
    if (refs.context) refs.context.close();
    if (refs.stream) refs.stream.getTracks().forEach(track => track.stop());

    streamRefs.current = { ws: null, context: null, stream: null, processor: null, source: null };
    setIsStreaming(false);
  };


  const handleFileSelect = async (file: File) => {
    setAppState('processing');
    setEvents([]);
    setStatus('safe');

    try {
      console.log("Sending file to backend:", file.name);
      const result = await uploadFile(file);

      // Map API response to TimelineEvent format
      const transformedEvents: TimelineEvent[] = result.chunks.map((chunk: any) => ({
        timestamp: chunk.start,
        score: chunk.fused_score,
        alertLevel: chunk.alert.toLowerCase() as 'safe' | 'violence',
        details: {
          acoustic: chunk.acoustic_score || 0,
          nlp: chunk.nlp_score || 0,
          emotion: chunk.emotion_score || 0
        }
      }));

      setEvents(transformedEvents);
      setStatus(result.overall_alert.toLowerCase() as 'safe' | 'violence');
      setAppState('results');

    } catch (error) {
      console.error("Analysis failed:", error);
      alert("Analysis failed. Make sure the backend server is running.");
      setAppState('idle');
    }
  };

  return (
    <div className="min-h-screen p-8 flex flex-col items-center">
      <header className="w-full max-w-4xl mb-12 flex flex-col items-center justify-center gap-4 animate-in slide-in-from-top-4 fade-in duration-500 text-center">
        <div className="p-4 bg-dark-accent/10 rounded-2xl border border-dark-accent/30 shadow-glow-safe hover:scale-105 transition-transform">
          <ShieldAlert className="w-12 h-12 text-dark-accent" />
        </div>
        <div>
          <h1 className="text-4xl font-bold text-white tracking-tight mb-2">Violence Detection</h1>
          <p className="text-slate-400 text-lg">Multimodal Neural Analysis Pipeline</p>
        </div>
      </header>

      <main className="w-full max-w-4xl flex flex-col items-center justify-center gap-6">
        {appState === 'idle' && (
          <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-8 animate-in zoom-in-95 duration-500">
            {/* Upload Panel */}
            <div className="w-full">
              <UploadPanel onFileSelect={handleFileSelect} />
            </div>

            {/* Mic Panel */}
            <div className="glass-panel p-8 flex flex-col items-center justify-center relative overflow-hidden group border-dashed border-2 border-dark-border cursor-pointer hover:border-alert-critical/50 h-[256px]" onClick={startStream}>
              <div className="absolute inset-0 bg-gradient-to-br from-alert-critical/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

              <div className="p-5 bg-dark-card rounded-full mb-4 border border-dark-border group-hover:border-alert-critical/30 group-hover:bg-alert-critical/10 transition-all shadow-lg">
                <Mic className="w-10 h-10 text-slate-400 group-hover:text-alert-critical transition-colors" />
              </div>

              <p className="text-white font-medium text-lg mb-2">Live Microphone</p>
              <p className="text-slate-400 text-sm text-center mb-6">Stream real-time audio analysis directly from your browser</p>

              <button className="btn-danger w-[80%] py-2 text-sm group-hover:shadow-glow-critical opacity-0 group-hover:opacity-100 transition-all translate-y-4 group-hover:translate-y-0">
                Start Stream
              </button>
            </div>
          </div>
        )}

        {appState === 'processing' && (
          <div className="glass-panel w-full p-12 flex flex-col items-center justify-center relative shadow-inner animate-in fade-in zoom-in duration-300 min-h-[400px]">
            <Loader2 className="w-20 h-20 text-dark-accent animate-spin mb-6" />
            <p className="text-white text-2xl font-medium mb-2">Processing Audio...</p>
            <p className="text-slate-400">Processing the audio...</p>
          </div>
        )}

        {appState === 'results' && (
          <div className="flex flex-col gap-6 w-full animate-in slide-in-from-bottom-8 fade-in duration-700">
            {/* Main Status Banner */}
            <div className={`glass-panel p-6 flex items-center justify-between border-${status === 'safe' ? 'alert-safe' : 'alert-critical'}/30 bg-${status === 'safe' ? 'alert-safe' : 'alert-critical'}/10 transition-colors duration-500 shadow-glow-${status === 'safe' ? 'safe' : 'critical'}`}>
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-full bg-${status === 'safe' ? 'alert-safe' : 'alert-critical'}/20`}>
                  <ShieldAlert className={`w-8 h-8 text-${status === 'safe' ? 'alert-safe' : 'alert-critical'}`} />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white mb-1">
                    {isStreaming ? 'Live Monitoring Active' : 'Analysis Complete'}
                  </h2>
                  <p className="text-slate-300">
                    {isStreaming ? 'Streaming real-time audio to neural pipeline...' : 'File processed successfully'}
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-end">
                <span className={`status-badge-${status === 'safe' ? 'safe' : 'critical'} font-bold text-lg px-6 py-2 uppercase tracking-wider`}>
                  {status}
                </span>
              </div>
            </div>

            {/* Timeline & Graph */}
            <div className="glass-panel p-6 flex flex-col w-full relative shadow-inner">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h3 className="text-xl font-semibold text-white">Escalation Graph & Detection Timeline</h3>
                  <p className="text-slate-400 text-sm mt-1">Temporal modeling of violence probability across the file duration</p>
                </div>
                {isStreaming ? (
                  <button onClick={() => { stopStream(); setAppState('idle'); }} className="btn-danger text-sm px-4 py-1.5 shadow-glow-critical animate-pulse">
                    Stop Stream
                  </button>
                ) : (
                  <button onClick={() => setAppState('idle')} className="btn-primary text-sm px-4 py-1.5 bg-dark-card hover:bg-dark-border">
                    Analyze Another
                  </button>
                )}
              </div>
              <ResultsTimeline events={events} duration={events.length > 0 ? Math.max(...events.map(e => e.timestamp)) + 5 : 30} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
