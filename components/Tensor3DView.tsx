import React, { useState, useRef, useCallback, useMemo } from 'react';
import { TensorSpec, DimensionSizes, OperationStep, TensorData } from '../types';
import { Maximize2, Check, X as XIcon, RotateCcw } from 'lucide-react';
import { getFlatIndex, formatTensorValue } from '../utils/tensorData';

interface Tensor3DViewProps {
    tensor: TensorSpec;
    sizes: DimensionSizes;
    highlight?: OperationStep['highlights'][string] | null;
    label: string;
    data?: TensorData | null;
    computedData?: TensorData | null;
    expectedData?: TensorData | null;
    showValues?: boolean;
}

/**
 * Render a 3D tensor as an interactive rotatable cube visualization
 * Uses CSS 3D transforms for the rotation effect
 */
const Tensor3DView: React.FC<Tensor3DViewProps> = ({
    tensor,
    sizes,
    highlight,
    label,
    data,
    computedData,
    expectedData,
    showValues = true
}) => {
    const dims = tensor.indices.map(idx => sizes[idx] || 4);
    const [rotateX, setRotateX] = useState(-20);
    const [rotateY, setRotateY] = useState(25);
    const [isDragging, setIsDragging] = useState(false);
    const lastMousePos = useRef({ x: 0, y: 0 });
    const containerRef = useRef<HTMLDivElement>(null);

    // Get shape for flat index calculation
    const shape = dims;

    // Cell size based on dimensions (smaller cells for larger tensors)
    const maxDim = Math.max(...dims);
    const cellSize = maxDim > 4 ? 28 : maxDim > 2 ? 36 : 44;
    const cellGap = 2;

    // Determine if a cell is highlighted
    const isHighlighted = (indexValues: number[]): 'vector' | 'scalar-broadcast' | 'output' | null => {
        if (!highlight) return null;

        for (let d = 0; d < tensor.indices.length; d++) {
            const idx = tensor.indices[d];
            const val = indexValues[d];
            const range = highlight.indices[idx];

            if (range === undefined) continue;

            if (Array.isArray(range)) {
                if (val < range[0] || val > range[1]) return null;
            } else {
                if (val !== range) return null;
            }
        }

        return highlight.type;
    };

    // Get value at index
    const getValue = (indexValues: number[], useData: TensorData | null | undefined): number | null => {
        if (!useData) return null;
        const flatIdx = getFlatIndex(shape, indexValues);
        return useData[flatIdx] ?? null;
    };

    // Check if computed matches expected
    const getVerificationStatus = (indexValues: number[]): 'match' | 'mismatch' | null => {
        if (!computedData || !expectedData) return null;
        const flatIdx = getFlatIndex(shape, indexValues);
        const computed = computedData[flatIdx] ?? 0;
        const expected = expectedData[flatIdx] ?? 0;
        return Math.abs(computed - expected) < 0.001 ? 'match' : 'mismatch';
    };

    // Get cell style based on highlight state
    const getCellStyle = (state: 'vector' | 'scalar-broadcast' | 'output' | null, verification: 'match' | 'mismatch' | null) => {
        let bgColor = 'rgba(31, 41, 55, 0.7)'; // gray-800/70
        let borderColor = 'rgba(55, 65, 81, 0.5)'; // gray-700/50
        let textColor = 'rgb(209, 213, 219)'; // gray-300
        let boxShadow = 'none';
        let zOffset = 0;

        if (verification === 'match' && state === null) {
            bgColor = 'rgba(20, 83, 45, 0.5)'; // green-900/50
            borderColor = 'rgba(21, 128, 61, 0.5)'; // green-700/50
            textColor = 'rgb(134, 239, 172)'; // green-300
        } else if (verification === 'mismatch' && state === null) {
            bgColor = 'rgba(127, 29, 29, 0.5)'; // red-900/50
            borderColor = 'rgba(185, 28, 28, 0.5)'; // red-700/50
            textColor = 'rgb(252, 165, 165)'; // red-300
        }

        switch (state) {
            case 'scalar-broadcast':
                bgColor = 'rgb(245, 158, 11)'; // amber-500
                borderColor = 'rgb(253, 224, 71)'; // amber-300
                boxShadow = '0 0 12px rgba(245,158,11,0.7)';
                zOffset = 8;
                break;
            case 'vector':
                bgColor = 'rgb(14, 165, 233)'; // sky-500
                borderColor = 'rgb(125, 211, 252)'; // sky-300
                boxShadow = '0 0 10px rgba(14,165,233,0.6)';
                zOffset = 4;
                break;
            case 'output':
                bgColor = 'rgb(139, 92, 246)'; // violet-500
                borderColor = 'rgb(196, 181, 253)'; // violet-300
                boxShadow = '0 0 12px rgba(139,92,246,0.6)';
                zOffset = 4;
                break;
        }

        return { bgColor, borderColor, textColor, boxShadow, zOffset };
    };

    // Mouse handlers for rotation
    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        // Prevent default to avoid text selection
        e.preventDefault();
        setIsDragging(true);
        lastMousePos.current = { x: e.clientX, y: e.clientY };
    }, []);

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
        if (!isDragging) return;

        const deltaX = e.clientX - lastMousePos.current.x;
        const deltaY = e.clientY - lastMousePos.current.y;

        setRotateY(prev => prev + deltaX * 0.5);
        setRotateX(prev => Math.max(-80, Math.min(80, prev - deltaY * 0.5)));

        lastMousePos.current = { x: e.clientX, y: e.clientY };
    }, [isDragging]);

    const handleMouseUp = useCallback(() => {
        setIsDragging(false);
    }, []);

    const handleMouseLeave = useCallback(() => {
        setIsDragging(false);
    }, []);

    const resetRotation = useCallback(() => {
        setRotateX(-20);
        setRotateY(25);
    }, []);

    // Dimensions: [depth, rows, cols] for 3D
    const depth = dims[0];
    const rows = dims[1];
    const cols = dims[2];

    // Calculate total size of the 3D structure
    const totalWidth = cols * (cellSize + cellGap);
    const totalHeight = rows * (cellSize + cellGap);
    const totalDepth = depth * (cellSize + cellGap);

    // Generate all cells in 3D space
    const cells = useMemo(() => {
        const result: React.ReactNode[] = [];

        for (let d = 0; d < depth; d++) {
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const indexValues = [d, r, c];
                    const state = isHighlighted(indexValues);
                    const verification = getVerificationStatus(indexValues);
                    const style = getCellStyle(state, verification);
                    const displayData = computedData || data;
                    const value = getValue(indexValues, displayData);

                    // Position in 3D space
                    const x = c * (cellSize + cellGap) - totalWidth / 2 + cellSize / 2;
                    const y = r * (cellSize + cellGap) - totalHeight / 2 + cellSize / 2;
                    const z = (depth - 1 - d) * (cellSize + cellGap) - totalDepth / 2 + cellSize / 2;

                    result.push(
                        <div
                            key={`${d}-${r}-${c}`}
                            className="absolute flex flex-col items-center justify-center rounded text-xs font-mono transition-all duration-200"
                            style={{
                                width: cellSize,
                                height: cellSize,
                                transform: `translate3d(${x}px, ${y}px, ${z + style.zOffset}px)`,
                                background: style.bgColor,
                                border: `1px solid ${style.borderColor}`,
                                color: style.textColor,
                                boxShadow: style.boxShadow,
                                backfaceVisibility: 'hidden',
                            }}
                        >
                            {state === 'scalar-broadcast' ? (
                                <>
                                    <div className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-amber-300 rounded-full animate-ping" />
                                    <Maximize2 size={12} className="text-white" />
                                </>
                            ) : (
                                <>
                                    <span className="text-[10px] font-bold leading-none">
                                        {showValues && value !== null ? formatTensorValue(value) : '–'}
                                    </span>
                                    <span className="text-[6px] opacity-50 mt-0.5 leading-none">
                                        [{indexValues.join(',')}]
                                    </span>
                                </>
                            )}
                            {verification && state === null && (
                                <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full flex items-center justify-center ${verification === 'match' ? 'bg-green-500' : 'bg-red-500'}`}>
                                    {verification === 'match' ? <Check size={7} className="text-white" /> : <XIcon size={7} className="text-white" />}
                                </div>
                            )}
                        </div>
                    );
                }
            }
        }

        return result;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [dims, highlight, data, computedData, expectedData, showValues, cellSize]);

    // Get label color
    const getLabelColor = () => {
        if (!highlight) return 'text-gray-400';
        switch (highlight.type) {
            case 'scalar-broadcast': return 'text-amber-400';
            case 'vector': return 'text-sky-400';
            case 'output': return 'text-violet-400';
            default: return 'text-gray-400';
        }
    };

    // Container size for perspective
    const containerSize = Math.max(totalWidth, totalHeight, totalDepth) + 100;

    return (
        <div className="flex flex-col items-center">
            <h3 className={`font-bold text-sm mb-2 flex items-center gap-2 ${getLabelColor()}`}>
                {label}
                <span className="text-xs text-gray-500 font-normal">
                    [{tensor.indices.join(',')}] = ({dims.join('×')})
                </span>
            </h3>

            {/* Instructions and reset button */}
            <div className="flex items-center gap-3 mb-2">
                <span className="text-[10px] text-gray-500">
                    Drag to rotate
                </span>
                <button
                    onClick={resetRotation}
                    className="flex items-center gap-1 px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-[10px] text-gray-400 transition-colors"
                    title="Reset rotation"
                >
                    <RotateCcw size={10} />
                    Reset
                </button>
            </div>

            {/* 3D Container */}
            <div
                ref={containerRef}
                className="relative cursor-grab active:cursor-grabbing border border-gray-700/30 rounded-xl bg-gray-900/50 backdrop-blur"
                style={{
                    width: containerSize,
                    height: containerSize,
                    perspective: 800,
                    perspectiveOrigin: '50% 50%',
                }}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseLeave}
            >
                {/* 3D Scene */}
                <div
                    className="absolute inset-0 flex items-center justify-center"
                    style={{
                        transformStyle: 'preserve-3d',
                        transform: `rotateX(${rotateX}deg) rotateY(${rotateY}deg)`,
                        transition: isDragging ? 'none' : 'transform 0.1s ease-out',
                    }}
                >
                    {cells}

                    {/* Axis indicators */}
                    <div
                        className="absolute text-[9px] font-bold text-sky-400 pointer-events-none"
                        style={{ transform: `translate3d(${totalWidth / 2 + 20}px, 0, 0)` }}
                    >
                        {tensor.indices[2]} →
                    </div>
                    <div
                        className="absolute text-[9px] font-bold text-violet-400 pointer-events-none"
                        style={{ transform: `translate3d(0, ${totalHeight / 2 + 20}px, 0)` }}
                    >
                        {tensor.indices[1]} ↓
                    </div>
                    <div
                        className="absolute text-[9px] font-bold text-amber-400 pointer-events-none"
                        style={{ transform: `translate3d(0, 0, ${totalDepth / 2 + 20}px)` }}
                    >
                        {tensor.indices[0]} ⬤
                    </div>
                </div>
            </div>

            {/* Type indicator */}
            {highlight && (
                <div className={`mt-2 text-[10px] font-medium uppercase tracking-wide ${getLabelColor()}`}>
                    {highlight.type === 'scalar-broadcast' ? '⟨ Broadcast ⟩' :
                        highlight.type === 'output' ? '⟨ Output ⟩' : '⟨ Vector ⟩'}
                </div>
            )}
        </div>
    );
};

export default Tensor3DView;
