import React from 'react';
import { OperationStep } from '../types';
import { Play, Pause, SkipBack, SkipForward, ChevronLeft, ChevronRight, RotateCcw } from 'lucide-react';

interface TimelineControlsProps {
    steps: OperationStep[];
    currentStepIndex: number;
    isPlaying: boolean;
    onPlayPause: () => void;
    onStepChange: (index: number) => void;
    onReplay?: () => void;
}

const TimelineControls: React.FC<TimelineControlsProps> = ({
    steps,
    currentStepIndex,
    isPlaying,
    onPlayPause,
    onStepChange,
    onReplay,
}) => {
    const currentStep = steps[currentStepIndex];
    const progress = steps.length > 0 ? ((currentStepIndex + 1) / steps.length) * 100 : 0;
    const isAtEnd = currentStepIndex >= steps.length - 1;

    const handleReplay = () => {
        if (onReplay) {
            onReplay();
        } else {
            // Default behavior: go to start and play
            onStepChange(0);
            // Small delay to ensure state updates before playing
            setTimeout(() => onPlayPause(), 50);
        }
    };

    return (
        <div className="w-full bg-gray-900/95 backdrop-blur p-4 border-t border-gray-800 flex flex-col gap-3 shadow-2xl">
            {/* Info Display */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
                <div className="flex-1 min-w-0">
                    <h4 className="text-xs uppercase tracking-wider text-gray-500 font-semibold mb-1">Current Operation</h4>
                    {currentStep ? (
                        <div className="text-base sm:text-lg font-mono text-amber-400 truncate" title={currentStep.formula}>
                            {currentStep.formula}
                        </div>
                    ) : (
                        <div className="text-gray-500 italic text-sm">Select a vectorization strategy to begin...</div>
                    )}
                </div>
                <div className="text-right shrink-0">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wide">Step</div>
                    <div className="text-lg font-bold tabular-nums">
                        <span className="text-white">{steps.length > 0 ? currentStepIndex + 1 : 0}</span>
                        <span className="text-gray-600"> / {steps.length}</span>
                    </div>
                </div>
            </div>

            {/* Progress Bar */}
            <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div
                    className="h-full bg-gradient-to-r from-amber-500 to-orange-500 transition-all duration-200 ease-out rounded-full"
                    style={{ width: `${progress}%` }}
                />
            </div>

            {/* Controls */}
            <div className="flex items-center justify-center gap-2 sm:gap-3">
                <button
                    onClick={() => onStepChange(0)}
                    disabled={steps.length === 0}
                    className="p-2 hover:bg-gray-800 rounded-full text-gray-400 hover:text-white disabled:opacity-30 transition-colors"
                    title="First step"
                >
                    <SkipBack size={18} />
                </button>
                <button
                    onClick={() => onStepChange(Math.max(0, currentStepIndex - 1))}
                    disabled={steps.length === 0}
                    className="p-2 hover:bg-gray-800 rounded-full text-gray-400 hover:text-white disabled:opacity-30 transition-colors"
                    title="Previous step"
                >
                    <ChevronLeft size={22} />
                </button>

                <button
                    onClick={onPlayPause}
                    disabled={steps.length === 0}
                    className="w-11 h-11 flex items-center justify-center bg-gradient-to-br from-amber-500 to-orange-600 hover:from-amber-400 hover:to-orange-500 text-white rounded-full shadow-lg shadow-orange-500/20 transition-all active:scale-95 disabled:opacity-50 disabled:grayscale"
                    title={isPlaying ? "Pause" : "Play"}
                >
                    {isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" className="ml-0.5" />}
                </button>

                <button
                    onClick={() => onStepChange(Math.min(steps.length - 1, currentStepIndex + 1))}
                    disabled={steps.length === 0}
                    className="p-2 hover:bg-gray-800 rounded-full text-gray-400 hover:text-white disabled:opacity-30 transition-colors"
                    title="Next step"
                >
                    <ChevronRight size={22} />
                </button>
                <button
                    onClick={() => onStepChange(steps.length - 1)}
                    disabled={steps.length === 0}
                    className="p-2 hover:bg-gray-800 rounded-full text-gray-400 hover:text-white disabled:opacity-30 transition-colors"
                    title="Last step"
                >
                    <SkipForward size={18} />
                </button>

                {/* Divider */}
                <div className="h-6 w-px bg-gray-700 mx-1" />

                {/* Replay Button */}
                <button
                    onClick={handleReplay}
                    disabled={steps.length === 0}
                    className={`p-2.5 rounded-full transition-all disabled:opacity-30 ${isAtEnd && !isPlaying
                            ? 'bg-green-600 hover:bg-green-500 text-white shadow-lg shadow-green-500/20'
                            : 'hover:bg-gray-800 text-gray-400 hover:text-white'
                        }`}
                    title="Replay from start"
                >
                    <RotateCcw size={18} />
                </button>
            </div>
        </div>
    );
};

export default TimelineControls;
