import clsx from 'clsx';
export interface TimelineEvent {
    timestamp: number;
    score: number;
    alertLevel: 'safe' | 'violence';
    details: {
        acoustic: number;
        nlp: number;
        emotion: number;
    };
}

interface ResultsTimelineProps {
    events: TimelineEvent[];
    duration: number; // in seconds
}

export function ResultsTimeline({ events, duration }: ResultsTimelineProps) {

    // Calculate percentage position for a given timestamp
    const getPosition = (time: number) => {
        return Math.min(Math.max((time / duration) * 100, 0), 100);
    };

    const formatTime = (seconds: number) => {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    };

    if (events.length === 0) {
        return (
            <div className="w-full h-full flex flex-col items-center justify-center text-slate-500 min-h-[200px]">
                <p>No analysis data available yet.</p>
                <p className="text-sm mt-1">Upload a file or start the microphone to begin.</p>
            </div>
        );
    }

    return (
        <div className="w-full flex flex-col gap-6">

            {/* Visual Timeline Track */}
            <div className="relative w-full h-24 bg-dark-bg/50 rounded-xl border border-dark-border p-4 overflow-hidden">
                {/* The Track Line */}
                <div className="absolute top-1/2 left-4 right-4 h-1 bg-slate-700/50 -translate-y-1/2 rounded-full overflow-hidden">
                    {/* Render colored segments based on high severity events */}
                    {events.map((ev, i) => {
                        const nextEv = events[i + 1];
                        const width = nextEv ? getPosition(nextEv.timestamp) - getPosition(ev.timestamp) : 100 - getPosition(ev.timestamp);

                        let bgColor = "bg-transparent";
                        if (ev.alertLevel === 'violence') bgColor = "bg-alert-critical/60";

                        return (
                            <div
                                key={`seg-${i}`}
                                className={`absolute h-full ${bgColor} transition-all`}
                                style={{ left: `${getPosition(ev.timestamp)}%`, width: `${width}%` }}
                            />
                        );
                    })}
                </div>

                {/* The Event Nodes */}
                {events.map((ev, i) => {
                    let colorClass = "text-alert-safe bg-alert-safe/20 border-alert-safe/50";

                    if (ev.alertLevel === 'violence') {
                        colorClass = "text-alert-critical bg-alert-critical/20 border-alert-critical/50 shadow-glow-critical";
                    }

                    return (
                        <div
                            key={`node-${i}`}
                            className={clsx(
                                "absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full border-2 flex items-center justify-center cursor-pointer hover:scale-125 transition-transform z-10",
                                colorClass
                            )}
                            style={{ left: `calc(1rem + (100% - 2rem) * ${getPosition(ev.timestamp) / 100})` }}
                            title={`Time: ${formatTime(ev.timestamp)} | Score: ${(ev.score * 100).toFixed(1)}%`}
                        >
                            {ev.alertLevel === 'violence' && <div className="absolute inset-0 bg-alert-critical rounded-full animate-ping opacity-20" />}
                        </div>
                    );
                })}
            </div>

            {/* Detail Table */}
            <div className="w-full bg-dark-bg/50 rounded-xl border border-dark-border overflow-hidden">
                <div className="grid grid-cols-5 gap-4 p-3 border-b border-dark-border bg-dark-card/50 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    <div>Timestamp</div>
                    <div>Violence Status</div>
                    <div>Audio Score</div>
                    <div>NLP Score</div>
                    <div>Fused Likelihood</div>
                </div>

                <div className="max-h-48 overflow-y-auto">
                    {events.map((ev, i) => (
                        <div key={`row-${i}`} className="grid grid-cols-5 gap-4 p-3 border-b border-dark-border/50 text-sm hover:bg-white/5 transition-colors">
                            <div className="text-slate-300 font-mono">{formatTime(ev.timestamp)}</div>
                            <div>
                                <span className={clsx(
                                    "px-2 py-0.5 rounded text-xs font-medium uppercase",
                                    ev.alertLevel === 'safe' ? "text-alert-safe bg-alert-safe/10" :
                                        "text-alert-critical bg-alert-critical/10 animate-pulse"
                                )}>
                                    {ev.alertLevel}
                                </span>
                            </div>
                            <div className="text-slate-400">{(ev.details.acoustic * 100).toFixed(1)}%</div>
                            <div className="text-slate-400">{(ev.details.nlp * 100).toFixed(1)}%</div>
                            <div className="font-semibold text-white">{(ev.score * 100).toFixed(1)}%</div>
                        </div>
                    ))}
                </div>
            </div>

        </div>
    );
}
