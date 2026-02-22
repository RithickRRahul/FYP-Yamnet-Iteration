import { UploadCloud, FileAudio, FileVideo, X } from 'lucide-react';
import { useState, useCallback } from 'react';
import clsx from 'clsx';

interface UploadPanelProps {
    onFileSelect: (file: File) => void;
}

export function UploadPanel({ onFileSelect }: UploadPanelProps) {
    const [isDragging, setIsDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setIsDragging(true);
        } else if (e.type === "dragleave") {
            setIsDragging(false);
        }
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('audio/') || file.type.startsWith('video/')) {
                setSelectedFile(file);
                onFileSelect(file);
            } else {
                alert("Please upload an audio or video file.");
            }
        }
    }, [onFileSelect]);

    const handleChange = function (e: React.ChangeEvent<HTMLInputElement>) {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            setSelectedFile(e.target.files[0]);
            onFileSelect(e.target.files[0]);
        }
    };

    const clearFile = () => {
        setSelectedFile(null);
    };

    return (
        <div
            className={clsx(
                "glass-panel p-6 h-64 flex flex-col items-center justify-center border-dashed border-2 transition-all relative overflow-hidden",
                isDragging ? "border-dark-accent bg-dark-accent/5 backdrop-blur-xl" : "border-dark-border",
                selectedFile ? "border-alert-safe/50 bg-alert-safe/5" : ""
            )}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
        >
            {selectedFile ? (
                <div className="flex flex-col items-center animate-in fade-in zoom-in duration-300">
                    <div className="absolute top-3 right-3">
                        <button onClick={clearFile} className="p-1 hover:bg-dark-border rounded-full text-slate-400 hover:text-white transition-colors">
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    <div className="p-4 bg-dark-card rounded-full mb-3 shadow-lg border border-dark-border">
                        {selectedFile.type.startsWith('video/') ? (
                            <FileVideo className="w-10 h-10 text-dark-accent" />
                        ) : (
                            <FileAudio className="w-10 h-10 text-dark-accent" />
                        )}
                    </div>
                    <p className="font-medium text-white text-center max-w-[200px] truncate">{selectedFile.name}</p>
                    <p className="text-xs text-slate-400 mt-1">
                        {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                    </p>

                    <button className="btn-primary mt-4 w-full bg-dark-accent/20 border-dark-accent/50">
                        Analyze File
                    </button>
                </div>
            ) : (
                <>
                    <div className={clsx(
                        "p-3 rounded-xl mb-4 transition-all duration-300",
                        isDragging ? "bg-dark-accent/20 text-dark-accent scale-110" : "bg-dark-border/50 text-slate-400"
                    )}>
                        <UploadCloud className="w-8 h-8" />
                    </div>
                    <p className="text-slate-300 font-medium mb-1">Drag and drop file here</p>
                    <p className="text-sm text-slate-500 mb-4 text-center">Supports MP4, WAV, MP3, FLAC (Max 50MB)</p>

                    <label className="btn-primary cursor-pointer w-[60%] text-sm">
                        Browse Files
                        <input
                            type="file"
                            className="hidden"
                            accept="audio/*,video/*"
                            onChange={handleChange}
                        />
                    </label>
                </>
            )}
        </div>
    );
}
